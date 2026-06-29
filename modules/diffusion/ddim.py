from __future__ import annotations

import torch

from .timestep import extract_into_tensor


class DDIMSampler:
    def __init__(self, diffusion) -> None:
        self.diffusion = diffusion

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        steps: int = 100,
        eta: float = 0.0,
        cond: object = None,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        diffusion = self.diffusion
        device = torch.device(device)
        times = torch.linspace(0, diffusion.num_timesteps - 1, steps, device=device).long()
        times = torch.flip(times, dims=[0])
        x = torch.randn(shape, device=device)
        for index, step in enumerate(times):
            t = torch.full((shape[0],), int(step.item()), device=device, dtype=torch.long)
            eps = diffusion.model(x, t, cond)
            alpha = extract_into_tensor(diffusion.alphas_cumprod, t, x.shape)
            pred_x0 = (x - (1.0 - alpha).sqrt() * eps) / alpha.sqrt()
            if index == len(times) - 1:
                x = pred_x0
                continue
            next_t = torch.full((shape[0],), int(times[index + 1].item()), device=device, dtype=torch.long)
            alpha_prev = extract_into_tensor(diffusion.alphas_cumprod, next_t, x.shape)
            sigma = eta * (((1 - alpha_prev) / (1 - alpha)) * (1 - alpha / alpha_prev)).clamp_min(0).sqrt()
            direction = (1.0 - alpha_prev - sigma**2).clamp_min(0).sqrt() * eps
            noise = sigma * torch.randn_like(x) if eta > 0 else 0.0
            x = alpha_prev.sqrt() * pred_x0 + direction + noise
        return x
