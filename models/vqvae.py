from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from modules.vqvae import Decoder3D, Encoder3D, LegacyDecoder3D, LegacyEncoder3D, VectorQuantizer


class SDFVQVAE(nn.Module):
    """3D SDF VQ-VAE that maps voxel SDFs to quantized latents and reconstructs SDF grids."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        resolution: int = 64,
        base_channels: int = 64,
        channel_multipliers: list[int] | tuple[int, ...] = (1, 2, 4),
        z_channels: int = 256,
        embed_dim: int = 256,
        n_embed: int = 1024,
        codebook_weight: float | None = None,
        architecture: str = "simple",
        ddconfig: dict[str, Any] | None = None,
        legacy_quantizer_loss: bool = False,
    ) -> None:
        super().__init__()
        del codebook_weight
        self.architecture = architecture
        if architecture == "legacy":
            legacy_config = {
                "ch": base_channels,
                "out_ch": out_channels,
                "ch_mult": tuple(channel_multipliers),
                "num_res_blocks": 1,
                "attn_resolutions": (),
                "dropout": 0.0,
                "resamp_with_conv": True,
                "in_channels": in_channels,
                "resolution": resolution,
                "z_channels": z_channels,
                "double_z": False,
                "activ": "gelu",
            }
            if ddconfig:
                legacy_config.update(ddconfig)
            z_channels = int(legacy_config["z_channels"])
            self.encoder = LegacyEncoder3D(**legacy_config)
            self.decoder = LegacyDecoder3D(**legacy_config)
        elif architecture == "simple":
            self.encoder = Encoder3D(in_channels, base_channels, channel_multipliers, z_channels)
            self.decoder = Decoder3D(out_channels, base_channels, channel_multipliers, z_channels)
        else:
            raise ValueError(f"Unknown VQ-VAE architecture '{architecture}'. Use 'simple' or 'legacy'.")
        self.quant_conv = nn.Conv3d(z_channels, embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(embed_dim, z_channels, kernel_size=1)
        self.quantize = VectorQuantizer(n_embed=n_embed, embed_dim=embed_dim, beta=1.0, legacy=legacy_quantizer_loss)
        self.embed_dim = embed_dim

    @property
    def quantizer(self) -> VectorQuantizer:
        return self.quantize

    def encode(self, sdf: torch.Tensor) -> torch.Tensor:
        """Encode SDF grids into continuous latent features before vector quantization."""
        return self.quant_conv(self.encoder(sdf))

    def quantize_latent(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vector-quantize latent features and return quantized latents, loss, and code indices."""
        return self.quantize(z)

    def encode_quantized(self, sdf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode and quantize SDF grids in one call."""
        return self.quantize_latent(self.encode(sdf))

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latent grids back into SDF voxel grids."""
        return self.decoder(self.post_quant_conv(z_q))

    def decode_no_quant(self, z: torch.Tensor, force_not_quantize: bool = False) -> torch.Tensor:
        """Decode latents, optionally applying quantization before the decoder."""
        if not force_not_quantize:
            z, _, _ = self.quantize_latent(z)
        return self.decode(z)

    def forward(self, sdf: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the full VQ-VAE path and return latents, codebook loss, indices, and reconstruction."""
        z = self.encode(sdf)
        z_q, codebook_loss, indices = self.quantize_latent(z)
        reconstruction = self.decode(z_q)
        return {
            "z": z,
            "z_q": z_q,
            "indices": indices,
            "codebook_loss": codebook_loss,
            "reconstruction": reconstruction,
        }

    def save_checkpoint(self, path: str | Path, **extra: Any) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.state_dict(), **extra}, path)

    def load_checkpoint(self, path: str | Path, strict: bool = True) -> dict[str, Any]:
        state = torch.load(path, map_location="cpu")
        if "model" in state:
            self.load_state_dict(state["model"], strict=strict)
        elif "vqvae" in state:
            self.load_state_dict(state["vqvae"], strict=strict)
        else:
            self.load_state_dict(state, strict=strict)
        return state if isinstance(state, dict) else {}
