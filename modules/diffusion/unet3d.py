from __future__ import annotations

import torch
from torch import nn

from .timestep import timestep_embedding


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 256, base_channels: int = 128, out_channels: int | None = None) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.time_embed_dim = base_channels
        self.time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.time_proj1 = nn.Linear(self.time_dim, base_channels)
        self.time_proj2 = nn.Linear(self.time_dim, base_channels * 2)
        self.in_conv = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down = nn.Conv3d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.mid = nn.Sequential(
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
            nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
        )
        self.up = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, cond: object = None) -> torch.Tensor:
        emb = timestep_embedding(timesteps, self.time_embed_dim)
        temb = self.time_mlp(emb)
        h1 = self.in_conv(x) + self.time_proj1(temb)[:, :, None, None, None]
        h1 = torch.nn.functional.silu(h1)
        h2 = self.down(h1) + self.time_proj2(temb)[:, :, None, None, None]
        h2 = self.mid(h2)
        h = self.up(h2)
        if h.shape[-3:] == h1.shape[-3:]:
            h = h + h1
        return self.out(h)
