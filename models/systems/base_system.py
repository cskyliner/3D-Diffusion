from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from models.vqvae import SDFVQVAE
from modules.diffusion import DDIMSampler, DiffusionUNet, GaussianDiffusion, PLMSSampler, UNet3D


class BaseSDFusionSystem(nn.Module):
    """Base system that connects frozen SDF VQ-VAE latents with a trainable diffusion denoiser."""

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
        conditioning_key: str | None = None,
        concat_channels: int = 0,
        context_dim: int = 0,
        unet_architecture: str = "legacy_openai",
        unet_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.vqvae = vqvae
        self.scale_factor = float(scale_factor)
        self.latent_channels = int(latent_channels)
        self.latent_size = int(latent_size)
        self.conditioning_key = conditioning_key
        self.unet_architecture = unet_architecture
        if unet_architecture == "legacy_openai":
            params = {
                "image_size": latent_size,
                "in_channels": latent_channels + (concat_channels if conditioning_key in {"concat", "hybrid"} else 0),
                "out_channels": latent_channels,
                "model_channels": unet_base_channels,
                "num_res_blocks": 2,
                "attention_resolutions": [1, 2, 4],
                "dropout": 0.0,
                "channel_mult": [1, 2, 4, 4],
                "conv_resample": True,
                "dims": 3,
                "num_heads": 6,
                "num_head_channels": -1,
            }
            if unet_params:
                params.update(unet_params)
            self.denoiser = DiffusionUNet(params, conditioning_key=conditioning_key)
        elif unet_architecture == "compact":
            self.denoiser = UNet3D(
                in_channels=latent_channels,
                base_channels=unet_base_channels,
                conditioning_key=conditioning_key,
                concat_channels=concat_channels,
                context_dim=context_dim,
            )
        else:
            raise ValueError(f"Unknown unet_architecture '{unet_architecture}'. Use 'legacy_openai' or 'compact'.")
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
        """Encode SDF grids into continuous, scaled latents used by diffusion training."""
        self.vqvae.eval()
        z = self.vqvae.encode(sdf)
        return z * self.scale_factor

    def decode_latent_to_sdf(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode scaled diffusion latents through VQ quantization back into SDF grids."""
        self.vqvae.eval()
        z = latent / self.scale_factor
        return self.vqvae.decode_no_quant(z)

    def get_condition(self, batch: dict) -> object:
        raise NotImplementedError

    def diffusion_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute one latent denoising loss from a batch of SDF grids."""
        sdf = batch["sdf"]
        with torch.no_grad():
            z = self.encode_sdf_to_latent(sdf)
        t = torch.randint(0, self.diffusion.num_timesteps, (z.shape[0],), device=z.device).long()
        return self.diffusion.p_losses(z, t, self.get_condition(batch))

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Alias the training forward pass to diffusion_loss."""
        return self.diffusion_loss(batch)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        sampler: str = "ddim",
        steps: int = 100,
        eta: float = 0.0,
        cond: object = None,
        guidance_scale: float = 1.0,
        unconditional_cond: object = None,
        x_T: torch.Tensor | None = None,
        x0: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        clip_denoised: bool = False,
        temperature: float = 1.0,
        noise_dropout: float = 0.0,
        ddim_discretize: str = "uniform",
        return_intermediates: bool = False,
        log_every_t: int = 100,
        callback=None,
        img_callback=None,
        progress: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
        """Sample latent grids with DDPM/DDIM/PLMS and decode them into SDF outputs."""
        device = next(self.parameters()).device
        shape = (num_samples, self.latent_channels, self.latent_size, self.latent_size, self.latent_size)
        if sampler == "ddpm":
            latent = self.diffusion.p_sample_loop(
                shape,
                device=device,
                cond=cond,
                guidance_scale=guidance_scale,
                unconditional_cond=unconditional_cond,
                steps=steps,
                x_T=x_T,
                x0=x0,
                mask=mask,
                clip_denoised=clip_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                return_intermediates=return_intermediates,
                log_every_t=log_every_t,
                callback=callback,
                img_callback=img_callback,
                progress=progress,
            )
        elif sampler == "ddim":
            latent = DDIMSampler(self.diffusion).sample(
                shape,
                steps=steps,
                eta=eta,
                cond=cond,
                guidance_scale=guidance_scale,
                unconditional_cond=unconditional_cond,
                device=device,
                ddim_discretize=ddim_discretize,
                x_T=x_T,
                x0=x0,
                mask=mask,
                clip_denoised=clip_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                return_intermediates=return_intermediates,
                log_every_t=log_every_t,
                callback=callback,
                img_callback=img_callback,
                progress=progress,
            )
        elif sampler == "plms":
            latent = PLMSSampler(self.diffusion).sample(
                shape,
                steps=steps,
                eta=eta,
                cond=cond,
                guidance_scale=guidance_scale,
                unconditional_cond=unconditional_cond,
                device=device,
                ddim_discretize=ddim_discretize,
                x_T=x_T,
                x0=x0,
                mask=mask,
                clip_denoised=clip_denoised,
                return_intermediates=return_intermediates,
                log_every_t=log_every_t,
                callback=callback,
                img_callback=img_callback,
                progress=progress,
            )
        else:
            raise ValueError(f"Unknown sampler '{sampler}'. Use 'ddim', 'ddpm', or 'plms'.")
        if return_intermediates:
            latent_tensor, intermediates = latent
            return self.decode_latent_to_sdf(latent_tensor), intermediates
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
