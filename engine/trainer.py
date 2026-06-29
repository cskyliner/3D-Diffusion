from __future__ import annotations

import json
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.registry import get_dataset
from engine.checkpoint import load_model_checkpoint, save_checkpoint
from models.systems.uncond_system import UncondSDFusionSystem
from models.vqvae import SDFVQVAE
from modules.vqvae.losses import occupancy_iou, vqvae_loss
from utils.mesh import sdf_to_mesh
from utils.sdf_io import save_sdf_npy
from utils.seed import seed_everything


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def build_dataloader(config: dict[str, Any], split: str | None = None, shuffle: bool = True) -> DataLoader:
    data_cfg = dict(config.get("data", {}))
    if split is not None:
        data_cfg["split"] = split
    dataset_name = data_cfg.pop("dataset", "shapenet_sdf")
    train_cfg = config.get("train", {})
    dataset = get_dataset(dataset_name)(**data_cfg)
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 4)),
        shuffle=shuffle,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=bool(train_cfg.get("pin_memory", False)),
        drop_last=bool(train_cfg.get("drop_last", False)),
    )


def build_vqvae(config: dict[str, Any]) -> SDFVQVAE:
    vq_cfg = dict(config.get("vqvae", {}))
    return SDFVQVAE(**vq_cfg)


def build_uncond_system(config: dict[str, Any], vqvae: SDFVQVAE) -> UncondSDFusionSystem:
    diffusion_cfg = dict(config.get("diffusion", {}))
    return UncondSDFusionSystem(
        vqvae=vqvae,
        latent_channels=int(diffusion_cfg.get("latent_channels", config.get("vqvae", {}).get("embed_dim", 3))),
        latent_size=int(diffusion_cfg.get("latent_size", 16)),
        unet_base_channels=int(diffusion_cfg.get("unet_base_channels", 128)),
        timesteps=int(diffusion_cfg.get("timesteps", 1000)),
        beta_schedule=str(diffusion_cfg.get("beta_schedule", "linear")),
        linear_start=float(diffusion_cfg.get("linear_start", 1.0e-4)),
        linear_end=float(diffusion_cfg.get("linear_end", 2.0e-2)),
        scale_factor=float(diffusion_cfg.get("scale_factor", 1.0)),
        conditioning_key=config.get("conditioning_key"),
        concat_channels=int(diffusion_cfg.get("concat_channels", 0)),
        context_dim=int(diffusion_cfg.get("context_dim", 0)),
    )


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


@torch.no_grad()
def export_vqvae_reconstructions(
    model: SDFVQVAE,
    batch: dict[str, Any],
    out_dir: str | Path,
    max_items: int = 4,
) -> list[dict[str, Any]]:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    sdf = batch["sdf"]
    result = model(sdf)
    recon = result["reconstruction"]
    records: list[dict[str, Any]] = []
    for index in range(min(max_items, sdf.shape[0])):
        model_id = batch.get("model_id", [str(index)])
        model_name = model_id[index] if isinstance(model_id, list) else str(index)
        stem = f"{index:03d}_{model_name}"
        recon_path = save_sdf_npy(recon[index].detach().cpu(), output / f"{stem}_recon.sdf.npy")
        target_path = save_sdf_npy(sdf[index].detach().cpu(), output / f"{stem}_target.sdf.npy")
        mesh_meta = sdf_to_mesh(recon[index].detach().cpu().numpy(), output / f"{stem}_recon.ply")
        records.append({"model_id": model_name, "recon_sdf": str(recon_path), "target_sdf": str(target_path), "mesh": mesh_meta})
    (output / "reconstructions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


@torch.no_grad()
def evaluate_vqvae(model: SDFVQVAE, loader: DataLoader, device: torch.device, max_batches: int | None = None) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    ious: list[torch.Tensor] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_to_device(batch, device)
        output = model(batch["sdf"])
        loss, _ = vqvae_loss(batch["sdf"], output["reconstruction"], output["codebook_loss"])
        losses.append(float(loss.detach().cpu()))
        ious.append(occupancy_iou(output["reconstruction"], batch["sdf"]).detach().cpu())
    if not losses:
        return {"loss_total": 0.0, "iou": 0.0}
    return {"loss_total": float(sum(losses) / len(losses)), "iou": float(torch.cat(ious).mean())}


def train_vqvae(config: dict[str, Any], out_dir: str | Path, resume: str | None = None) -> Path:
    train_cfg = config.get("train", {})
    seed_everything(int(train_cfg.get("seed", 0)))
    device = resolve_device(str(train_cfg.get("device", "cuda")))
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output / "metrics.jsonl")
    loader = build_dataloader(config, split=config.get("data", {}).get("split", "train"), shuffle=True)
    try:
        val_loader = build_dataloader(config, split="test", shuffle=False)
    except FileNotFoundError:
        val_loader = build_dataloader(config, split=config.get("data", {}).get("split", "train"), shuffle=False)
    model = build_vqvae(config).to(device)
    if resume:
        load_model_checkpoint(model, resume, component="vqvae", strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("lr", 1.0e-4)), betas=(0.5, 0.9))
    max_steps = int(train_cfg.get("max_steps", 10000))
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every", 1000))
    codebook_weight = float(config.get("vqvae", {}).get("codebook_weight", 1.0))

    last_ckpt = output / "checkpoints" / "vqvae_last.pt"
    for step, batch in zip(range(1, max_steps + 1), cycle(loader)):
        model.train()
        batch = move_to_device(batch, device)
        output_dict = model(batch["sdf"])
        loss, loss_dict = vqvae_loss(
            batch["sdf"],
            output_dict["reconstruction"],
            output_dict["codebook_loss"],
            codebook_weight=codebook_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step == 1:
            row = {"step": step, **{key: float(value.cpu()) for key, value in loss_dict.items()}}
            logger.write(row)
            print(row)
        if step % save_every == 0 or step == max_steps:
            metrics = evaluate_vqvae(model, val_loader, device, max_batches=int(train_cfg.get("eval_batches", 8)))
            save_checkpoint(last_ckpt, vqvae=model.state_dict(), optimizer=optimizer.state_dict(), step=step, metrics=metrics)
            logger.write({"step": step, "split": "test", **metrics})
            first_batch = move_to_device(next(iter(val_loader)), device)
            export_vqvae_reconstructions(model, first_batch, output / "reconstructions" / f"step_{step:07d}")
    return last_ckpt


def train_diffusion(config: dict[str, Any], out_dir: str | Path, vqvae_ckpt: str | None = None, resume: str | None = None) -> Path:
    train_cfg = config.get("train", {})
    seed_everything(int(train_cfg.get("seed", 0)))
    device = resolve_device(str(train_cfg.get("device", "cuda")))
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output / "metrics.jsonl")
    loader = build_dataloader(config, split=config.get("data", {}).get("split", "train"), shuffle=True)
    vqvae = build_vqvae(config)
    if vqvae_ckpt:
        load_model_checkpoint(vqvae, vqvae_ckpt, component="vqvae", strict=False)
    system = build_uncond_system(config, vqvae).to(device)
    if resume:
        load_model_checkpoint(system, resume, component="model", strict=False)
    optimizer = torch.optim.AdamW([p for p in system.denoiser.parameters() if p.requires_grad], lr=float(train_cfg.get("lr", 1.0e-4)))
    max_steps = int(train_cfg.get("max_steps", 10000))
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every", 1000))
    last_ckpt = output / "checkpoints" / "diffusion_last.pt"

    for step, batch in zip(range(1, max_steps + 1), cycle(loader)):
        system.train()
        batch = move_to_device(batch, device)
        loss, loss_dict = system(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step == 1:
            row = {"step": step, **{key: float(value.cpu()) for key, value in loss_dict.items()}}
            logger.write(row)
            print(row)
        if step % save_every == 0 or step == max_steps:
            save_checkpoint(last_ckpt, model=system.state_dict(), optimizer=optimizer.state_dict(), step=step)
    return last_ckpt
