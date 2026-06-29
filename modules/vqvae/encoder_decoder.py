from __future__ import annotations

import torch
from torch import nn


def _block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        nn.GroupNorm(num_groups=max(1, min(8, out_channels // 4)), num_channels=out_channels),
        nn.SiLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups=max(1, min(8, out_channels // 4)), num_channels=out_channels),
        nn.SiLU(inplace=True),
    )


class Encoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: list[int] | tuple[int, ...] = (1, 2, 4),
        z_channels: int = 256,
    ) -> None:
        super().__init__()
        channels = [base_channels * mult for mult in channel_multipliers]
        layers: list[nn.Module] = []
        current = in_channels
        for channel in channels:
            layers.append(_block(current, channel, stride=2))
            current = channel
        layers.append(nn.Conv3d(current, z_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder3D(nn.Module):
    def __init__(
        self,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: list[int] | tuple[int, ...] = (1, 2, 4),
        z_channels: int = 256,
    ) -> None:
        super().__init__()
        channels = [base_channels * mult for mult in channel_multipliers]
        current = z_channels
        layers: list[nn.Module] = []
        for channel in reversed(channels):
            layers.append(nn.ConvTranspose3d(current, channel, kernel_size=4, stride=2, padding=1))
            layers.append(nn.GroupNorm(num_groups=max(1, min(8, channel // 4)), num_channels=channel))
            layers.append(nn.SiLU(inplace=True))
            current = channel
        layers.append(nn.Conv3d(current, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
