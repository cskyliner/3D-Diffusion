from __future__ import annotations

import math
from inspect import isfunction

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn

from .ldm_util import checkpoint


def exists(value) -> bool:
    return value is not None


def default(value, fallback):
    if exists(value):
        return value
    return fallback() if isfunction(fallback) else fallback


def zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


def normalize(in_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1.0e-6, affine=True)


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int | None = None, mult: int = 4, glu: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.norm = normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, _ = q.shape
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        weight = torch.einsum("bij,bjk->bik", q, k) * (int(c) ** -0.5)
        weight = torch.softmax(weight, dim=2)
        v = rearrange(v, "b c h w -> b c (h w)")
        weight = rearrange(weight, "b i j -> b j i")
        h_ = torch.einsum("bij,bjk->bik", v, weight)
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        return x + self.proj_out(h_)


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int | None = None, heads: int = 8, dim_head: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda tensor: rearrange(tensor, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)
        attn = sim.softmax(dim=-1)
        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, d_head: int, dropout: float = 0.0, context_dim: int | None = None, gated_ff: bool = True, checkpoint_enabled: bool = True) -> None:
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint_enabled

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, in_channels: int, n_heads: int, d_head: int, depth: int = 1, dropout: float = 0.0, context_dim: int | None = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = normalize(in_channels)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim) for _ in range(depth)]
        )
        self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1))

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        b, _, h, w = x.shape
        x_in = x
        x = self.proj_in(self.norm(x))
        x = rearrange(x, "b c h w -> b (h w) c")
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return self.proj_out(x) + x_in


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv3d):
        nn.init.xavier_normal_(module.weight)


class SpatialTransformer3D(nn.Module):
    def __init__(self, in_channels: int, n_heads: int, d_head: int, depth: int = 1, dropout: float = 0.0, context_dim: int | None = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = normalize(in_channels)
        self.proj_in = nn.Conv3d(in_channels, inner_dim, kernel_size=1)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim) for _ in range(depth)]
        )
        self.proj_out = zero_module(nn.Conv3d(inner_dim, in_channels, kernel_size=1))
        self.apply(init_weights)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        _, _, depth, height, width = x.shape
        x_in = x
        x = self.proj_in(self.norm(x))
        x = rearrange(x, "b c d h w -> b (d h w) c")
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, "b (d h w) c -> b c d h w", d=depth, h=height, w=width)
        return self.proj_out(x) + x_in
