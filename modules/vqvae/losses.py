from __future__ import annotations

import torch
from torch.nn import functional as F


def vqvae_loss(
    sdf: torch.Tensor,
    reconstruction: torch.Tensor,
    codebook_loss: torch.Tensor,
    codebook_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reconstruction_loss = F.l1_loss(reconstruction, sdf)
    total = reconstruction_loss + float(codebook_weight) * codebook_loss.mean()
    return total, {
        "loss_total": total.detach(),
        "loss_rec": reconstruction_loss.detach(),
        "loss_nll": reconstruction_loss.detach(),
        "loss_codebook": codebook_loss.detach().mean(),
    }


def occupancy_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    pred_occ = pred <= threshold
    target_occ = target <= threshold
    intersection = (pred_occ & target_occ).float().sum(dim=(1, 2, 3, 4))
    union = (pred_occ | target_occ).float().sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
    return intersection / union
