from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from .schedule import make_beta_schedule
from .timestep import extract_into_tensor

TensorCallback = Callable[[int, int, torch.Tensor], None]


def _make_sampling_timesteps(num_timesteps: int, steps: int | None, device: torch.device) -> torch.Tensor:
    if steps is None or steps >= num_timesteps:
        return torch.arange(num_timesteps - 1, -1, -1, device=device, dtype=torch.long)
    if steps <= 0:
        raise ValueError("steps must be a positive integer.")
    return torch.linspace(0, num_timesteps - 1, steps, device=device).long().unique().flip(0)


def _maybe_progress(iterable, enabled: bool, total: int | None = None, desc: str = "Sampling"):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


class GaussianDiffusion(nn.Module):
    """DDPM objective and reverse process for latent SDF diffusion."""

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        linear_start: float = 1.0e-4,
        linear_end: float = 2.0e-2,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_timesteps = int(timesteps)
        betas = torch.tensor(
            make_beta_schedule(beta_schedule, self.num_timesteps, linear_start, linear_end),
            dtype=torch.float32,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp_min(1.0e-20)))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def apply_model(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: object = None,
        guidance_scale: float = 1.0,
        unconditional_cond: object = None,
    ) -> torch.Tensor:
        """Run the denoiser, optionally applying classifier-free guidance."""
        if guidance_scale == 1.0 or unconditional_cond is None:
            return self.model(x, t, cond)
        cond_eps = self.model(x, t, cond)
        uncond_eps = self.model(x, t, unconditional_cond)
        return uncond_eps + guidance_scale * (cond_eps - uncond_eps)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Add forward-process Gaussian noise to clean latents at timestep t."""
        noise = torch.randn_like(x_start) if noise is None else noise
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Recover the predicted clean latent x0 from noisy latent xt and predicted noise."""
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, cond: object = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Train the denoiser to predict the sampled Gaussian noise."""
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.apply_model(x_noisy, t, cond)
        loss_simple = F.mse_loss(predicted_noise, noise)
        return loss_simple, {"loss_total": loss_simple.detach(), "loss_simple": loss_simple.detach()}

    def p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: object = None,
        guidance_scale: float = 1.0,
        unconditional_cond: object = None,
        clip_denoised: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute DDPM posterior mean/variance for the previous timestep."""
        eps = self.apply_model(x, t, cond, guidance_scale=guidance_scale, unconditional_cond=unconditional_cond)
        x_recon = self.predict_start_from_noise(x, t, eps)
        if clip_denoised:
            x_recon = x_recon.clamp(-1.0, 1.0)
        model_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * x_recon
            + extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x
        )
        log_variance = extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        return model_mean, log_variance, x_recon

    def p_mean_variance_to_prev(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        prev_t: torch.Tensor,
        cond: object = None,
        guidance_scale: float = 1.0,
        unconditional_cond: object = None,
        clip_denoised: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute posterior parameters for a skipped reverse step."""
        eps = self.apply_model(x, t, cond, guidance_scale=guidance_scale, unconditional_cond=unconditional_cond)
        x_recon = self.predict_start_from_noise(x, t, eps)
        if clip_denoised:
            x_recon = x_recon.clamp(-1.0, 1.0)

        alpha_t = extract_into_tensor(self.alphas_cumprod, t, x.shape)
        safe_prev_t = prev_t.clamp_min(0)
        alpha_prev = extract_into_tensor(self.alphas_cumprod, safe_prev_t, x.shape)
        alpha_prev = torch.where((prev_t < 0).view(-1, *([1] * (x.ndim - 1))), torch.ones_like(alpha_prev), alpha_prev)
        alpha_ratio = (alpha_t / alpha_prev).clamp(max=1.0)
        variance = ((1.0 - alpha_prev) * (1.0 - alpha_ratio) / (1.0 - alpha_t).clamp_min(1.0e-20)).clamp_min(0.0)
        coef_x0 = torch.sqrt(alpha_prev) * (1.0 - alpha_ratio) / (1.0 - alpha_t).clamp_min(1.0e-20)
        coef_xt = torch.sqrt(alpha_ratio) * (1.0 - alpha_prev) / (1.0 - alpha_t).clamp_min(1.0e-20)
        mean = coef_x0 * x_recon + coef_xt * x
        final = (prev_t < 0).view(-1, *([1] * (x.ndim - 1)))
        mean = torch.where(final, x_recon, mean)
        log_variance = torch.log(variance.clamp_min(1.0e-20))
        return mean, log_variance, x_recon

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        prev_t: torch.Tensor | None = None,
        cond: object = None,
        guidance_scale: float = 1.0,
        unconditional_cond: object = None,
        clip_denoised: bool = False,
        temperature: float = 1.0,
        noise_dropout: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Take one stochastic DDPM reverse step and return the new latent plus predicted x0."""
        if prev_t is None:
            mean, log_variance, pred_x0 = self.p_mean_variance(
                x,
                t,
                cond,
                guidance_scale=guidance_scale,
                unconditional_cond=unconditional_cond,
                clip_denoised=clip_denoised,
            )
            nonzero_mask = (t != 0).float().view(-1, *([1] * (x.ndim - 1)))
        else:
            mean, log_variance, pred_x0 = self.p_mean_variance_to_prev(
                x,
                t,
                prev_t,
                cond,
                guidance_scale=guidance_scale,
                unconditional_cond=unconditional_cond,
                clip_denoised=clip_denoised,
            )
            nonzero_mask = (prev_t >= 0).float().view(-1, *([1] * (x.ndim - 1)))
        noise = torch.randn_like(x) * temperature
        if noise_dropout > 0.0:
            noise = F.dropout(noise, p=noise_dropout)
        sample = mean + nonzero_mask * (0.5 * log_variance).exp() * noise
        return sample, pred_x0

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        cond: object = None,
        guidance_scale: float = 1.0,
        unconditional_cond: object = None,
        steps: int | None = None,
        x_T: torch.Tensor | None = None,
        x0: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        clip_denoised: bool = False,
        temperature: float = 1.0,
        noise_dropout: float = 0.0,
        return_intermediates: bool = False,
        log_every_t: int = 100,
        callback: TensorCallback | None = None,
        img_callback: TensorCallback | None = None,
        progress: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
        """Run the DDPM reverse chain, optionally with skipped steps and intermediate logging."""
        device = torch.device(device)
        x = torch.randn(shape, device=device) if x_T is None else x_T.to(device)
        timesteps = _make_sampling_timesteps(self.num_timesteps, steps, device)
        intermediates: dict[str, list[torch.Tensor]] = {"x_inter": [x.detach().cpu()], "pred_x0": []}
        iterator = _maybe_progress(list(enumerate(timesteps)), progress, total=len(timesteps), desc="DDPM")
        for index, step in iterator:
            t = torch.full((shape[0],), int(step.item()), device=device, dtype=torch.long)
            if index == len(timesteps) - 1:
                prev_value = -1
            else:
                prev_value = int(timesteps[index + 1].item())
            prev_t = torch.full((shape[0],), prev_value, device=device, dtype=torch.long)
            if mask is not None and x0 is not None:
                x_orig = self.q_sample(x0.to(device), t)
                x = x_orig * mask.to(device) + (1.0 - mask.to(device)) * x
            x, pred_x0 = self.p_sample(
                x,
                t,
                prev_t if steps is not None and steps < self.num_timesteps else None,
                cond,
                guidance_scale=guidance_scale,
                unconditional_cond=unconditional_cond,
                clip_denoised=clip_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
            )
            if callback is not None:
                callback(index, int(step.item()), x)
            if img_callback is not None:
                img_callback(index, int(step.item()), pred_x0)
            if return_intermediates and (index % log_every_t == 0 or index == len(timesteps) - 1):
                intermediates["x_inter"].append(x.detach().cpu())
                intermediates["pred_x0"].append(pred_x0.detach().cpu())
        if return_intermediates:
            return x, intermediates
        return x
