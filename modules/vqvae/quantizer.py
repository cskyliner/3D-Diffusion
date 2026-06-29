from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, n_embed: int, embed_dim: int, beta: float = 1.0) -> None:
        super().__init__()
        self.n_embed = int(n_embed)
        self.embed_dim = int(embed_dim)
        self.beta = float(beta)
        self.embedding = nn.Embedding(self.n_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_embed, 1.0 / self.n_embed)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_perm = z.permute(0, 2, 3, 4, 1).contiguous()
        flat = z_perm.view(-1, self.embed_dim)
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2.0 * flat @ self.embedding.weight.t()
        )
        indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(indices).view_as(z_perm)
        loss = self.beta * F.mse_loss(quantized.detach(), z_perm) + F.mse_loss(quantized, z_perm.detach())
        quantized = z_perm + (quantized - z_perm).detach()
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()
        indices = indices.view(z.shape[0], z.shape[2], z.shape[3], z.shape[4])
        return quantized, loss, indices
