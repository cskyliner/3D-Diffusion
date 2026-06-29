from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from sdfusion.models.vqvae import SDFVQVAE
from sdfusion.modules.diffusion import DDIMSampler, GaussianDiffusion, UNet3D


class BaseSDFusionSystem(nn.Module):
    def __init__(
        self,
        vqvae: SDFVQVAE,
        latent_channels: int = 256,
        latent_size: int = 8,
        unet_base_channels: int = 128,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        linear_start: float = 1.0e-4,
        linear_end: float = 2.0e-2,
        scale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.vqvae = vqvae
        self.scale_factor = float(scale_factor)
        self.latent_channels = int(latent_channels)
        self.latent_size = int(latent_size)
        self.denoiser = UNet3D(in_channels=latent_channels, base_channels=unet_base_channels)
        self.diffusion = GaussianDiffusion(
            self.denoiser,
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            linear_start=linear_start,
            linear_end=linear_end,
        )
        for parameter in self.vqvae.parameters():
            parameter.requires_grad = False

    def encode_sdf_to_latent(self, sdf: torch.Tensor) -> torch.Tensor:
        self.vqvae.eval()
        return self.vqvae.encode(sdf) * self.scale_factor

    def decode_latent_to_sdf(self, latent: torch.Tensor) -> torch.Tensor:
        self.vqvae.eval()
        return self.vqvae.decode(latent / self.scale_factor)

    def get_condition(self, batch: dict) -> object:
        raise NotImplementedError

    def diffusion_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sdf = batch["sdf"]
        with torch.no_grad():
            z = self.encode_sdf_to_latent(sdf)
        t = torch.randint(0, self.diffusion.num_timesteps, (z.shape[0],), device=z.device).long()
        return self.diffusion.p_losses(z, t, self.get_condition(batch))

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.diffusion_loss(batch)

    @torch.no_grad()
    def sample(self, num_samples: int, sampler: str = "ddim", steps: int = 100, eta: float = 0.0) -> torch.Tensor:
        device = next(self.parameters()).device
        shape = (num_samples, self.latent_channels, self.latent_size, self.latent_size, self.latent_size)
        if sampler == "ddpm":
            latent = self.diffusion.p_sample_loop(shape, device=device, cond=None)
        elif sampler == "ddim":
            latent = DDIMSampler(self.diffusion).sample(shape, steps=steps, eta=eta, cond=None, device=device)
        else:
            raise ValueError(f"Unknown sampler '{sampler}'. Use 'ddim' or 'ddpm'.")
        return self.decode_latent_to_sdf(latent)

    def save_checkpoint(self, path: str | Path, **extra: Any) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.state_dict(), **extra}, path)

    def load_checkpoint(self, path: str | Path, strict: bool = True) -> dict[str, Any]:
        state = torch.load(path, map_location="cpu")
        if "model" in state:
            self.load_state_dict(state["model"], strict=strict)
        elif "df" in state:
            self.load_state_dict(state["df"], strict=False)
        else:
            self.load_state_dict(state, strict=strict)
        return state if isinstance(state, dict) else {}
