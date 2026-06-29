from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from sdfusion.modules.vqvae import Decoder3D, Encoder3D, VectorQuantizer


class SDFVQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: list[int] | tuple[int, ...] = (1, 2, 4),
        z_channels: int = 256,
        embed_dim: int = 256,
        n_embed: int = 1024,
    ) -> None:
        super().__init__()
        self.encoder = Encoder3D(in_channels, base_channels, channel_multipliers, z_channels)
        self.decoder = Decoder3D(out_channels, base_channels, channel_multipliers, z_channels)
        self.quant_conv = nn.Conv3d(z_channels, embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(embed_dim, z_channels, kernel_size=1)
        self.quantizer = VectorQuantizer(n_embed=n_embed, embed_dim=embed_dim, beta=1.0)
        self.embed_dim = embed_dim

    def encode(self, sdf: torch.Tensor) -> torch.Tensor:
        return self.quant_conv(self.encoder(sdf))

    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.quantizer(z)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(z_q))

    def forward(self, sdf: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(sdf)
        z_q, codebook_loss, indices = self.quantize(z)
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
