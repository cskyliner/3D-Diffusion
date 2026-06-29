from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def make_norm(channels: int, requested_groups: int = 32) -> nn.GroupNorm:
    groups = requested_groups
    if channels <= 32:
        groups = max(1, channels // 4)
    elif channels % groups != 0:
        groups = math.gcd(channels, groups)
    return nn.GroupNorm(num_groups=max(1, groups), num_channels=channels, eps=1.0e-6, affine=True)


def make_activation(name: str) -> nn.Module:
    if name == "lrelu":
        return nn.LeakyReLU()
    if name == "swish":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported VQ-VAE activation: {name}")


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = bool(with_conv)
        if self.with_conv:
            self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = bool(with_conv)
        if self.with_conv:
            self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1, 0, 1), mode="constant", value=0)
            return self.conv(x)
        return F.avg_pool3d(x, kernel_size=2, stride=2)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 0,
    ) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = make_norm(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = make_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            if conv_shortcut:
                self.conv_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, temb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(swish(self.norm1(x)))
        if temb is not None and hasattr(self, "temb_proj"):
            h = h + self.temb_proj(swish(temb))[:, :, None, None, None]
        h = self.conv2(self.dropout(swish(self.norm2(h))))
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.norm = make_norm(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        b, c, d, height, width = q.shape
        q = q.reshape(b, c, d * height * width).permute(0, 2, 1)
        k = k.reshape(b, c, d * height * width)
        weight = torch.bmm(q, k) * (c ** -0.5)
        weight = torch.softmax(weight, dim=2).permute(0, 2, 1)
        v = v.reshape(b, c, d * height * width)
        h = torch.bmm(v, weight).reshape(b, c, d, height, width)
        return x + self.proj_out(h)


class LegacyEncoder3D(nn.Module):
    def __init__(
        self,
        *,
        ch: int = 64,
        out_ch: int = 1,
        ch_mult: list[int] | tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 1,
        attn_resolutions: list[int] | tuple[int, ...] = (),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int = 1,
        resolution: int = 64,
        z_channels: int = 3,
        double_z: bool = False,
        activ: str = "gelu",
        **_: object,
    ) -> None:
        super().__init__()
        del out_ch
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.nonlinearity = make_activation(activ)
        self.conv_in = nn.Conv3d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.norm_out = make_norm(block_in)
        self.conv_out = nn.Conv3d(block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for level in self.down:
            for i_block, block in enumerate(level.block):
                h = block(h, None)
                if len(level.attn) > 0:
                    h = level.attn[i_block](h)
            if hasattr(level, "downsample"):
                h = level.downsample(h)
        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)
        return self.conv_out(self.nonlinearity(self.norm_out(h)))


class LegacyDecoder3D(nn.Module):
    def __init__(
        self,
        *,
        ch: int = 64,
        out_ch: int = 1,
        ch_mult: list[int] | tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 1,
        attn_resolutions: list[int] | tuple[int, ...] = (),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int = 1,
        resolution: int = 64,
        z_channels: int = 3,
        give_pre_end: bool = False,
        activ: str = "gelu",
        **_: object,
    ) -> None:
        super().__init__()
        self.nonlinearity = make_activation(activ)
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res, curr_res)
        self.conv_in = nn.Conv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = make_norm(block_in)
        self.conv_out = nn.Conv3d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)
        for i_level in reversed(range(self.num_resolutions)):
            level = self.up[i_level]
            for i_block, block in enumerate(level.block):
                h = block(h, None)
                if len(level.attn) > 0:
                    h = level.attn[i_block](h)
            if hasattr(level, "upsample"):
                h = level.upsample(h)
        if self.give_pre_end:
            return h
        return self.conv_out(self.nonlinearity(self.norm_out(h)))
