from __future__ import annotations

import torch
from torch import nn

from .base import BaseConditioner


class ImageConditioner(BaseConditioner):
    condition_type = "image"

    def __init__(self, context_dim: int = 128) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, self.context_dim),
            nn.LayerNorm(self.context_dim),
        )

    def encode(self, batch: dict) -> dict:
        image = batch.get("image")
        if image is None:
            batch_size = int(batch["sdf"].shape[0])
            device = batch["sdf"].device
            image = torch.zeros((batch_size, 3, 1, 1), device=device, dtype=batch["sdf"].dtype)
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        image = image[:, :3].to(next(self.parameters()).device)
        context = self.encoder(image).unsqueeze(1)
        return {"c_crossattn": [context]}
