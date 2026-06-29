from __future__ import annotations

import torch
from torch import nn

from .timestep import timestep_embedding


def _as_condition_dict(cond: object) -> dict[str, object]:
    if cond is None:
        return {}
    if isinstance(cond, dict):
        return cond
    return {"c_crossattn": [cond]}


def _first_tensor(items: object) -> torch.Tensor | None:
    if items is None:
        return None
    if torch.is_tensor(items):
        return items
    if isinstance(items, (list, tuple)) and items:
        return _first_tensor(items[0])
    return None


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        base_channels: int = 128,
        out_channels: int | None = None,
        conditioning_key: str | None = None,
        concat_channels: int = 0,
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.conditioning_key = conditioning_key
        self.concat_channels = int(concat_channels)
        self.context_dim = int(context_dim)
        self.time_embed_dim = base_channels
        self.time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.time_proj1 = nn.Linear(self.time_dim, base_channels)
        self.time_proj2 = nn.Linear(self.time_dim, base_channels * 2)
        effective_in_channels = in_channels + (self.concat_channels if conditioning_key in {"concat", "hybrid"} else 0)
        self.in_conv = nn.Conv3d(effective_in_channels, base_channels, kernel_size=3, padding=1)
        if conditioning_key in {"crossattn", "hybrid"}:
            if self.context_dim <= 0:
                self.context_proj = nn.LazyLinear(self.time_dim)
            else:
                self.context_proj = nn.Linear(self.context_dim, self.time_dim)
            self.context_gate = nn.Sequential(nn.SiLU(), nn.Linear(self.time_dim, self.time_dim))
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

    def _prepare_concat(self, x: torch.Tensor, cond: dict[str, object]) -> torch.Tensor:
        concat = _first_tensor(cond.get("c_concat"))
        if concat is None:
            if self.conditioning_key in {"concat", "hybrid"} and self.concat_channels > 0:
                zeros = x.new_zeros((x.shape[0], self.concat_channels, *x.shape[2:]))
                return torch.cat([x, zeros], dim=1)
            return x
        concat = concat.to(device=x.device, dtype=x.dtype)
        if concat.shape[-3:] != x.shape[-3:]:
            concat = torch.nn.functional.interpolate(concat, size=x.shape[-3:], mode="trilinear", align_corners=False)
        return torch.cat([x, concat], dim=1)

    def _prepare_context(self, x: torch.Tensor, cond: dict[str, object]) -> torch.Tensor | None:
        context = _first_tensor(cond.get("c_crossattn"))
        if context is None:
            return None
        context = context.to(device=x.device, dtype=x.dtype)
        if context.dim() == 2:
            pooled = context
        elif context.dim() == 3:
            pooled = context.mean(dim=1)
        else:
            pooled = context.flatten(1)
        return self.context_gate(self.context_proj(pooled))

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, cond: object = None) -> torch.Tensor:
        cond_dict = _as_condition_dict(cond)
        if self.conditioning_key in {"concat", "hybrid"}:
            x = self._prepare_concat(x, cond_dict)
        emb = timestep_embedding(timesteps, self.time_embed_dim)
        temb = self.time_mlp(emb)
        if self.conditioning_key in {"crossattn", "hybrid"}:
            context = self._prepare_context(x, cond_dict)
            if context is not None:
                temb = temb + context
        h1 = self.in_conv(x) + self.time_proj1(temb)[:, :, None, None, None]
        h1 = torch.nn.functional.silu(h1)
        h2 = self.down(h1) + self.time_proj2(temb)[:, :, None, None, None]
        h2 = self.mid(h2)
        h = self.up(h2)
        if h.shape[-3:] == h1.shape[-3:]:
            h = h + h1
        return self.out(h)
