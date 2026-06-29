from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .schedule import make_beta_schedule
from .timestep import extract_into_tensor


class GaussianDiffusion(nn.Module):
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

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        noise = torch.randn_like(x_start) if noise is None else noise
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, cond: object = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t, cond)
        loss_simple = F.mse_loss(predicted_noise, noise)
        return loss_simple, {"loss_total": loss_simple.detach(), "loss_simple": loss_simple.detach()}

    def p_mean_variance(self, x: torch.Tensor, t: torch.Tensor, cond: object = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = self.model(x, t, cond)
        x_recon = self.predict_start_from_noise(x, t, eps)
        model_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * x_recon
            + extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x
        )
        log_variance = extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        return model_mean, log_variance, x_recon

    @torch.no_grad()
    def p_sample_loop(self, shape: tuple[int, ...], device: torch.device, cond: object = None) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            mean, log_variance, _ = self.p_mean_variance(x, t, cond)
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = mean + (0.5 * log_variance).exp() * noise
        return x
