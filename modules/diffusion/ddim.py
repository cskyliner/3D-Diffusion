from __future__ import annotations

from typing import Callable

import torch

from .timestep import extract_into_tensor

TensorCallback = Callable[[int, int, torch.Tensor], None]


def make_ddim_timesteps(
    ddim_discretize: str,
    num_ddim_timesteps: int,
    num_ddpm_timesteps: int,
    device: torch.device,
) -> torch.Tensor:
    if num_ddim_timesteps <= 0:
        raise ValueError("steps must be a positive integer.")
    if num_ddim_timesteps > num_ddpm_timesteps:
        raise ValueError("DDIM steps cannot exceed diffusion.num_timesteps.")
    if ddim_discretize == "uniform":
        c = max(num_ddpm_timesteps // num_ddim_timesteps, 1)
        steps = torch.arange(0, num_ddpm_timesteps, c, device=device, dtype=torch.long)[:num_ddim_timesteps]
    elif ddim_discretize == "quad":
        steps = (
            torch.linspace(0, (num_ddpm_timesteps * 0.8) ** 0.5, num_ddim_timesteps, device=device) ** 2
        ).long()
        steps = steps.clamp(max=num_ddpm_timesteps - 1).unique()
        if len(steps) < num_ddim_timesteps:
            fill = torch.linspace(0, num_ddpm_timesteps - 1, num_ddim_timesteps, device=device).long()
            steps = torch.cat([steps, fill]).unique()[:num_ddim_timesteps]
    else:
        raise ValueError(f"Unknown DDIM discretization '{ddim_discretize}'. Use 'uniform' or 'quad'.")
    return steps.sort().values


def make_ddim_sampling_parameters(
    alphas_cumprod: torch.Tensor,
    timesteps: torch.Tensor,
    eta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    alphas = alphas_cumprod[timesteps]
    alphas_prev = torch.cat([alphas_cumprod.new_ones(1), alphas[:-1]], dim=0)
    sigmas = eta * (((1.0 - alphas_prev) / (1.0 - alphas)) * (1.0 - alphas / alphas_prev)).clamp_min(0.0).sqrt()
    return sigmas, alphas, alphas_prev


def _maybe_progress(iterable, enabled: bool, total: int | None = None, desc: str = "DDIM"):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


class DDIMSampler:
    def __init__(self, diffusion) -> None:
        self.diffusion = diffusion

    def make_schedule(
        self,
        steps: int,
        ddim_discretize: str = "uniform",
        eta: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        device = torch.device(device)
        timesteps = make_ddim_timesteps(ddim_discretize, steps, self.diffusion.num_timesteps, device)
        sigmas, alphas, alphas_prev = make_ddim_sampling_parameters(self.diffusion.alphas_cumprod.to(device), timesteps, eta)
        return {
            "timesteps": timesteps,
            "sigmas": sigmas,
            "alphas": alphas,
            "alphas_prev": alphas_prev,
            "sqrt_one_minus_alphas": (1.0 - alphas).sqrt(),
        }

    def _get_model_eps(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: object,
        guidance_scale: float,
        unconditional_cond: object,
        score_corrector: Callable | None,
        corrector_kwargs: dict | None,
    ) -> torch.Tensor:
        eps = self.diffusion.apply_model(
            x,
            t,
            cond,
            guidance_scale=guidance_scale,
            unconditional_cond=unconditional_cond,
        )
        if score_corrector is not None:
            corrector_kwargs = corrector_kwargs or {}
            eps = score_corrector.modify_score(self.diffusion, eps, x, t, cond, **corrector_kwargs)
        return eps

    def p_sample_ddim(
        self,
        x: torch.Tensor,
        cond: object,
        t: torch.Tensor,
        index: int,
        schedule: dict[str, torch.Tensor],
        guidance_scale: float = 1.0,
        unconditional_cond: object = None,
        clip_denoised: bool = False,
        quantize_denoised: Callable[[torch.Tensor], torch.Tensor] | None = None,
        temperature: float = 1.0,
        noise_dropout: float = 0.0,
        score_corrector: Callable | None = None,
        corrector_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps = self._get_model_eps(x, t, cond, guidance_scale, unconditional_cond, score_corrector, corrector_kwargs)
        alpha = schedule["alphas"][index].view(1, *([1] * (x.ndim - 1))).to(x)
        alpha_prev = schedule["alphas_prev"][index].view(1, *([1] * (x.ndim - 1))).to(x)
        sigma = schedule["sigmas"][index].view(1, *([1] * (x.ndim - 1))).to(x)
        sqrt_one_minus_alpha = schedule["sqrt_one_minus_alphas"][index].view(1, *([1] * (x.ndim - 1))).to(x)

        pred_x0 = (x - sqrt_one_minus_alpha * eps) / alpha.sqrt()
        if clip_denoised:
            pred_x0 = pred_x0.clamp(-1.0, 1.0)
        if quantize_denoised is not None:
            pred_x0 = quantize_denoised(pred_x0)
        direction = (1.0 - alpha_prev - sigma**2).clamp_min(0.0).sqrt() * eps
        noise = sigma * torch.randn_like(x) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = alpha_prev.sqrt() * pred_x0 + direction + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        steps: int = 100,
        eta: float = 0.0,
        cond: object = None,
        guidance_scale: float = 1.0,
        unconditional_cond: object = None,
        device: torch.device | str = "cpu",
        ddim_discretize: str = "uniform",
        x_T: torch.Tensor | None = None,
        x0: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        clip_denoised: bool = False,
        quantize_denoised: Callable[[torch.Tensor], torch.Tensor] | None = None,
        temperature: float = 1.0,
        noise_dropout: float = 0.0,
        score_corrector: Callable | None = None,
        corrector_kwargs: dict | None = None,
        return_intermediates: bool = False,
        log_every_t: int = 100,
        callback: TensorCallback | None = None,
        img_callback: TensorCallback | None = None,
        progress: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
        diffusion = self.diffusion
        device = torch.device(device)
        schedule = self.make_schedule(steps, ddim_discretize=ddim_discretize, eta=eta, device=device)
        timesteps = schedule["timesteps"].flip(0)
        x = torch.randn(shape, device=device) if x_T is None else x_T.to(device)
        intermediates: dict[str, list[torch.Tensor]] = {"x_inter": [x.detach().cpu()], "pred_x0": []}
        iterator = _maybe_progress(list(enumerate(timesteps)), progress, total=len(timesteps), desc="DDIM")
        for loop_index, step in iterator:
            schedule_index = len(timesteps) - loop_index - 1
            t = torch.full((shape[0],), int(step.item()), device=device, dtype=torch.long)
            if mask is not None and x0 is not None:
                img_orig = diffusion.q_sample(x0.to(device), t)
                x = img_orig * mask.to(device) + (1.0 - mask.to(device)) * x
            x, pred_x0 = self.p_sample_ddim(
                x,
                cond,
                t,
                schedule_index,
                schedule,
                guidance_scale=guidance_scale,
                unconditional_cond=unconditional_cond,
                clip_denoised=clip_denoised,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
            )
            if callback is not None:
                callback(loop_index, int(step.item()), x)
            if img_callback is not None:
                img_callback(loop_index, int(step.item()), pred_x0)
            if return_intermediates and (loop_index % log_every_t == 0 or loop_index == len(timesteps) - 1):
                intermediates["x_inter"].append(x.detach().cpu())
                intermediates["pred_x0"].append(pred_x0.detach().cpu())
        if return_intermediates:
            return x, intermediates
        return x
