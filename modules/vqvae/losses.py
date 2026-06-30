from __future__ import annotations

import torch
from torch.nn import functional as F


def _gradient_3d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dz = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
    dz = F.pad(dz, (0, 0, 0, 0, 0, 1))
    dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
    dx = F.pad(dx, (0, 1, 0, 0, 0, 0))
    return dz, dy, dx


def _normal_tensor(x: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    grad = torch.cat(_gradient_3d(x), dim=1)
    return F.normalize(grad, dim=1, eps=eps)


def _multiscale_l1(reconstruction: torch.Tensor, target: torch.Tensor, levels: int) -> torch.Tensor:
    if levels <= 1:
        return reconstruction.new_tensor(0.0)
    losses: list[torch.Tensor] = []
    current_recon = reconstruction
    current_target = target
    for _ in range(1, levels):
        if min(current_recon.shape[-3:]) < 2:
            break
        current_recon = F.avg_pool3d(current_recon, kernel_size=2, stride=2)
        current_target = F.avg_pool3d(current_target, kernel_size=2, stride=2)
        losses.append(F.l1_loss(current_recon, current_target))
    if not losses:
        return reconstruction.new_tensor(0.0)
    return torch.stack(losses).mean()


def vqvae_loss(
    sdf: torch.Tensor,
    reconstruction: torch.Tensor,
    codebook_loss: torch.Tensor,
    codebook_weight: float = 1.0,
    occupancy_weight: float = 0.0,
    surface_weight: float = 0.0,
    normal_weight: float = 0.0,
    multiscale_weight: float = 0.0,
    surface_band: float = 0.02,
    occupancy_temperature: float = 0.02,
    multiscale_levels: int = 1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reconstruction_loss = F.l1_loss(reconstruction, sdf)
    codebook = codebook_loss.mean()
    occupancy_loss = reconstruction.new_tensor(0.0)
    if occupancy_weight > 0.0:
        target_occ = (sdf <= 0.0).float()
        logits = -reconstruction / max(float(occupancy_temperature), 1.0e-6)
        occupancy_loss = F.binary_cross_entropy_with_logits(logits, target_occ)

    surface_loss = reconstruction.new_tensor(0.0)
    if surface_weight > 0.0:
        band = max(float(surface_band), 1.0e-6)
        weights = torch.exp(-torch.abs(sdf) / band)
        surface_loss = (weights * torch.abs(reconstruction - sdf)).sum() / weights.sum().clamp_min(1.0)

    normal_loss = reconstruction.new_tensor(0.0)
    if normal_weight > 0.0:
        pred_normals = _normal_tensor(reconstruction)
        target_normals = _normal_tensor(sdf)
        band = max(float(surface_band), 1.0e-6)
        normal_weights = torch.exp(-torch.abs(sdf) / band).expand_as(pred_normals)
        cosine_distance = 1.0 - (pred_normals * target_normals).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        normal_loss = (normal_weights[:, :1] * cosine_distance).sum() / normal_weights[:, :1].sum().clamp_min(1.0)

    multiscale_loss = _multiscale_l1(reconstruction, sdf, int(multiscale_levels)) if multiscale_weight > 0.0 else reconstruction.new_tensor(0.0)
    total = (
        reconstruction_loss
        + float(codebook_weight) * codebook
        + float(occupancy_weight) * occupancy_loss
        + float(surface_weight) * surface_loss
        + float(normal_weight) * normal_loss
        + float(multiscale_weight) * multiscale_loss
    )
    return total, {
        "loss_total": total.detach(),
        "loss_rec": reconstruction_loss.detach(),
        "loss_nll": reconstruction_loss.detach(),
        "loss_codebook": codebook.detach(),
        "loss_occupancy": occupancy_loss.detach(),
        "loss_surface": surface_loss.detach(),
        "loss_normal": normal_loss.detach(),
        "loss_multiscale": multiscale_loss.detach(),
    }


def occupancy_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    pred_occ = pred <= threshold
    target_occ = target <= threshold
    intersection = (pred_occ & target_occ).float().sum(dim=(1, 2, 3, 4))
    union = (pred_occ | target_occ).float().sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
    return intersection / union
