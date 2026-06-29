from __future__ import annotations

import torch

from .timestep import extract_into_tensor


class PLMSSampler:
    def __init__(self, diffusion) -> None:
        self.diffusion = diffusion

    def _transfer(
        self,
        x: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
        next_t: torch.Tensor,
        eta: float,
    ) -> torch.Tensor:
        diffusion = self.diffusion
        alpha = extract_into_tensor(diffusion.alphas_cumprod, t, x.shape)
        alpha_prev = extract_into_tensor(diffusion.alphas_cumprod, next_t, x.shape)
        pred_x0 = (x - (1.0 - alpha).sqrt() * eps) / alpha.sqrt()
        sigma = eta * (((1 - alpha_prev) / (1 - alpha)) * (1 - alpha / alpha_prev)).clamp_min(0).sqrt()
        direction = (1.0 - alpha_prev - sigma**2).clamp_min(0).sqrt() * eps
        noise = sigma * torch.randn_like(x) if eta > 0 else 0.0
        return alpha_prev.sqrt() * pred_x0 + direction + noise

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
    ) -> torch.Tensor:
        diffusion = self.diffusion
        device = torch.device(device)
        times = torch.linspace(0, diffusion.num_timesteps - 1, steps, device=device).long().flip(0)
        x = torch.randn(shape, device=device)
        old_eps: list[torch.Tensor] = []
        for index, step in enumerate(times):
            t = torch.full((shape[0],), int(step.item()), device=device, dtype=torch.long)
            eps = diffusion.apply_model(
                x,
                t,
                cond,
                guidance_scale=guidance_scale,
                unconditional_cond=unconditional_cond,
            )
            if len(old_eps) == 0:
                eps_prime = eps
            elif len(old_eps) == 1:
                eps_prime = (3 * eps - old_eps[-1]) / 2
            elif len(old_eps) == 2:
                eps_prime = (23 * eps - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
            else:
                eps_prime = (55 * eps - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

            if index == len(times) - 1:
                next_t = torch.zeros_like(t)
            else:
                next_t = torch.full((shape[0],), int(times[index + 1].item()), device=device, dtype=torch.long)
            x = self._transfer(x, eps_prime, t, next_t, eta)
            old_eps.append(eps.detach())
            old_eps = old_eps[-3:]
        return x
