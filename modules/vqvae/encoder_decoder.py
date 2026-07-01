from __future__ import annotations

import torch
from torch import nn

from .legacy import AttnBlock, Downsample, ResnetBlock, Upsample, make_activation, make_norm


def _as_tuple(values: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


class Encoder3D(nn.Module):
    """Compact residual 3D encoder used by the non-legacy VQ-VAE path."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: list[int] | tuple[int, ...] = (1, 2, 4),
        z_channels: int = 256,
        num_res_blocks: int = 1,
        resamp_with_conv: bool = True,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        multipliers = _as_tuple(channel_multipliers)
        self.num_resolutions = len(multipliers)
        self.num_res_blocks = int(num_res_blocks)
        self.nonlinearity = make_activation(activation)
        self.conv_in = nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        current = base_channels
        self.down = nn.ModuleList()
        for level, multiplier in enumerate(multipliers):
            block_out = base_channels * multiplier
            blocks = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                blocks.append(ResnetBlock(in_channels=current, out_channels=block_out, dropout=0.0, temb_channels=0))
                current = block_out
            stage = nn.Module()
            stage.block = blocks
            if level != self.num_resolutions - 1:
                stage.downsample = Downsample(current, resamp_with_conv)
            self.down.append(stage)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=current, out_channels=current, dropout=0.0, temb_channels=0)
        self.mid.attn_1 = AttnBlock(current)
        self.mid.block_2 = ResnetBlock(in_channels=current, out_channels=current, dropout=0.0, temb_channels=0)
        self.norm_out = make_norm(current)
        self.conv_out = nn.Conv3d(current, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for stage in self.down:
            for block in stage.block:
                h = block(h, None)
            if hasattr(stage, "downsample"):
                h = stage.downsample(h)
        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)
        return self.conv_out(self.nonlinearity(self.norm_out(h)))


class Decoder3D(nn.Module):
    """Compact residual 3D decoder with nearest-neighbor upsample plus convolution."""

    def __init__(
        self,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: list[int] | tuple[int, ...] = (1, 2, 4),
        z_channels: int = 256,
        num_res_blocks: int = 1,
        resamp_with_conv: bool = True,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        multipliers = _as_tuple(channel_multipliers)
        self.num_resolutions = len(multipliers)
        self.num_res_blocks = int(num_res_blocks)
        self.nonlinearity = make_activation(activation)
        current = base_channels * multipliers[-1]
        self.conv_in = nn.Conv3d(z_channels, current, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=current, out_channels=current, dropout=0.0, temb_channels=0)
        self.mid.attn_1 = AttnBlock(current)
        self.mid.block_2 = ResnetBlock(in_channels=current, out_channels=current, dropout=0.0, temb_channels=0)

        self.up = nn.ModuleList()
        for level in reversed(range(self.num_resolutions)):
            block_out = base_channels * multipliers[level]
            blocks = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                blocks.append(ResnetBlock(in_channels=current, out_channels=block_out, dropout=0.0, temb_channels=0))
                current = block_out
            stage = nn.Module()
            stage.block = blocks
            if level != 0:
                stage.upsample = Upsample(current, resamp_with_conv)
            self.up.insert(0, stage)

        self.norm_out = make_norm(current)
        self.conv_out = nn.Conv3d(current, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)
        for level in reversed(range(self.num_resolutions)):
            stage = self.up[level]
            for block in stage.block:
                h = block(h, None)
            if hasattr(stage, "upsample"):
                h = stage.upsample(h)
        return self.conv_out(self.nonlinearity(self.norm_out(h)))
