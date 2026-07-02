from __future__ import annotations

import json
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.registry import get_dataset
from engine.checkpoint import load_model_checkpoint, save_checkpoint
from models.systems.uncond_system import UncondSDFusionSystem
from models.vqvae import SDFVQVAE
from modules.vqvae.losses import occupancy_iou, vqvae_loss
from utils.mesh import sdf_to_mesh
from utils.metrics import diversity_l1, sdf_stats
from utils.sdf_io import save_sdf_npy
from utils.seed import seed_everything


def resolve_device(name: str) -> torch.device:
    """Return the requested torch device, falling back to CPU when CUDA is unavailable."""
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a dataloader batch to the training/evaluation device."""
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def build_dataloader(config: dict[str, Any], split: str | None = None, shuffle: bool = True) -> DataLoader:
    """Create a dataset and dataloader from the config data/train sections."""
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
    """Instantiate the SDF VQ-VAE from the config vqvae section."""
    vq_cfg = dict(config.get("vqvae", {}))
    return SDFVQVAE(**vq_cfg)


def build_uncond_system(config: dict[str, Any], vqvae: SDFVQVAE) -> UncondSDFusionSystem:
    """Wrap a frozen VQ-VAE with the configured unconditional latent diffusion system."""
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
        unet_architecture=str(diffusion_cfg.get("unet_architecture", "legacy_openai")),
        unet_params=dict(diffusion_cfg.get("unet_params", {})),
    )


def vqvae_loss_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Extract optional VQ-VAE geometry loss weights from config."""
    loss_cfg = config.get("vqvae_loss", {})
    return {
        "codebook_weight": float(loss_cfg.get("codebook_weight", 1.0)),
        "occupancy_weight": float(loss_cfg.get("occupancy_weight", 0.0)),
        "surface_weight": float(loss_cfg.get("surface_weight", 0.0)),
        "normal_weight": float(loss_cfg.get("normal_weight", 0.0)),
        "multiscale_weight": float(loss_cfg.get("multiscale_weight", 0.0)),
        "surface_band": float(loss_cfg.get("surface_band", 0.02)),
        "occupancy_temperature": float(loss_cfg.get("occupancy_temperature", 0.02)),
        "multiscale_levels": int(loss_cfg.get("multiscale_levels", 1)),
    }


def codebook_metrics(indices: torch.Tensor, n_embed: int) -> dict[str, torch.Tensor]:
    """Return batch-level codebook utilization metrics for VQ-VAE diagnostics."""
    flat = indices.reshape(-1).to(torch.long)
    counts = torch.bincount(flat, minlength=int(n_embed)).float()
    total = counts.sum().clamp_min(1.0)
    probs = counts / total
    nonzero = probs > 0
    entropy = -(probs[nonzero] * torch.log(probs[nonzero])).sum()
    used = (counts > 0).float().sum()
    return {
        "codebook_used": used.detach(),
        "codebook_usage": (used / float(n_embed)).detach(),
        "codebook_perplexity": torch.exp(entropy).detach(),
    }


class JsonlLogger:
    """Append one JSON metrics row per line for long-running training jobs."""

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
    """Save target/reconstructed SDFs and reconstructed meshes for VQ-VAE debugging."""
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
def evaluate_vqvae(
    model: SDFVQVAE,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    loss_kwargs: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Evaluate VQ-VAE reconstruction losses and occupancy IoU over a dataloader."""
    model.eval()
    losses: list[float] = []
    ious: list[torch.Tensor] = []
    loss_sums: dict[str, float] = {}
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_to_device(batch, device)
        output = model(batch["sdf"])
        loss, loss_dict = vqvae_loss(batch["sdf"], output["reconstruction"], output["codebook_loss"], **(loss_kwargs or {}))
        losses.append(float(loss.detach().cpu()))
        for key, value in loss_dict.items():
            loss_sums[key] = loss_sums.get(key, 0.0) + float(value.detach().cpu())
        for key, value in codebook_metrics(output["indices"], model.quantize.n_embed).items():
            loss_sums[key] = loss_sums.get(key, 0.0) + float(value.detach().cpu())
        ious.append(occupancy_iou(output["reconstruction"], batch["sdf"]).detach().cpu())
    if not losses:
        return {"loss_total": 0.0, "iou": 0.0}
    metrics = {key: value / len(losses) for key, value in loss_sums.items()}
    metrics["loss_total"] = float(sum(losses) / len(losses))
    metrics["iou"] = float(torch.cat(ious).mean())
    return metrics


@torch.no_grad()
def evaluate_diffusion_loss(system: UncondSDFusionSystem, loader: DataLoader, device: torch.device, max_batches: int | None = None) -> dict[str, float]:
    """Average denoising losses on validation SDF batches without optimizer updates."""
    system.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_to_device(batch, device)
        _, loss_dict = system(batch)
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        count += 1
    if count == 0:
        return {"loss_total": 0.0}
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def export_diffusion_snapshot(
    system: UncondSDFusionSystem,
    out_dir: str | Path,
    num_samples: int = 4,
    steps: int = 100,
    sampler: str = "ddim",
) -> dict[str, Any]:
    """Sample SDFs during diffusion training and export meshes plus aggregate sample metrics."""
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    system.eval()
    sdf_batch = system.sample(num_samples=num_samples, sampler=sampler, steps=steps)
    sdfs: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    mesh_success = 0
    for index, sdf in enumerate(sdf_batch):
        sdf_np = sdf.detach().cpu().numpy().astype(np.float32)
        sdfs.append(sdf_np)
        stem = f"sample_{index:04d}"
        sdf_path = save_sdf_npy(sdf_np, output / f"{stem}.sdf.npy")
        mesh_meta = sdf_to_mesh(sdf_np, output / f"{stem}.ply")
        mesh_success += int(bool(mesh_meta.get("success", False)))
        metadata = {
            "sample": index,
            "sdf_path": str(sdf_path),
            "mesh": mesh_meta,
            "stats": sdf_stats(sdf_np),
        }
        (output / f"{stem}.metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        records.append(metadata)
    success_rate = float(mesh_success / len(records)) if records else 0.0
    aggregate_stats = sdf_stats(np.stack(sdfs)) if sdfs else {"min": 0.0, "max": 0.0, "mean": 0.0, "occupancy_ratio": 0.0}
    report = {
        "num_samples": len(records),
        "sampler": sampler,
        "steps": steps,
        "mesh_success": mesh_success,
        "mesh_failed": len(records) - mesh_success,
        "mesh_success_rate": success_rate,
        "sdf_stats": aggregate_stats,
        "diversity_l1": diversity_l1(sdfs),
        "samples": records,
    }
    (output / "snapshot_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def train_vqvae(config: dict[str, Any], out_dir: str | Path, resume: str | None = None) -> Path:
    """Train the SDF VQ-VAE and periodically save checkpoints, metrics, and reconstructions."""
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
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    loss_kwargs = vqvae_loss_kwargs(config)

    last_ckpt = output / "checkpoints" / "vqvae_last.pt"
    for step, batch in zip(range(1, max_steps + 1), cycle(loader)):
        model.train()
        batch = move_to_device(batch, device)
        output_dict = model(batch["sdf"])
        loss, loss_dict = vqvae_loss(
            batch["sdf"],
            output_dict["reconstruction"],
            output_dict["codebook_loss"],
            **loss_kwargs,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = None
        if grad_clip_norm > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        if step % log_every == 0 or step == 1:
            row = {
                "step": step,
                **{key: float(value.cpu()) for key, value in loss_dict.items()},
                **{key: float(value.cpu()) for key, value in codebook_metrics(output_dict["indices"], model.quantize.n_embed).items()},
            }
            if grad_norm is not None:
                row["grad_norm"] = float(grad_norm.detach().cpu())
            logger.write(row)
            print(row)
        if step % save_every == 0 or step == max_steps:
            metrics = evaluate_vqvae(
                model,
                val_loader,
                device,
                max_batches=int(train_cfg.get("eval_batches", 8)),
                loss_kwargs=loss_kwargs,
            )
            save_checkpoint(last_ckpt, vqvae=model.state_dict(), optimizer=optimizer.state_dict(), step=step, metrics=metrics)
            logger.write({"step": step, "split": "test", **metrics})
            first_batch = move_to_device(next(iter(val_loader)), device)
            export_vqvae_reconstructions(model, first_batch, output / "reconstructions" / f"step_{step:07d}")
    return last_ckpt


def train_diffusion(config: dict[str, Any], out_dir: str | Path, vqvae_ckpt: str, resume: str | None = None) -> Path:
    """Train unconditional latent diffusion using a frozen, checkpoint-loaded VQ-VAE encoder/decoder."""
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
    vqvae = build_vqvae(config)
    if not vqvae_ckpt:
        raise ValueError("train_diffusion requires a trained VQ-VAE checkpoint.")
    vqvae_report = load_model_checkpoint(vqvae, vqvae_ckpt, component="vqvae", strict=False)
    vqvae_load_row = {
        "event": "vqvae_load",
        "checkpoint": str(vqvae_ckpt),
        "matched_keys": int(vqvae_report.get("matched_keys", 0)),
        "matched_param_ratio": float(vqvae_report.get("matched_param_ratio", 0.0)),
        "missing_keys": len(vqvae_report.get("missing_keys", [])),
        "unexpected_keys": len(vqvae_report.get("unexpected_keys", [])),
    }
    logger.write(vqvae_load_row)
    print(vqvae_load_row)
    system = build_uncond_system(config, vqvae).to(device)
    if resume:
        load_model_checkpoint(system, resume, component="model", strict=False)
    optimizer = torch.optim.AdamW([p for p in system.denoiser.parameters() if p.requires_grad], lr=float(train_cfg.get("lr", 1.0e-4)))
    max_steps = int(train_cfg.get("max_steps", 10000))
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every", 1000))
    eval_every = int(train_cfg.get("eval_every", save_every))
    sample_every = int(train_cfg.get("sample_every", save_every))
    eval_batches = int(train_cfg.get("eval_batches", 8))
    sample_num = int(train_cfg.get("sample_num", 4))
    sample_steps = int(train_cfg.get("sample_steps", 100))
    sample_sampler = str(train_cfg.get("sample_sampler", "ddim"))
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
        if eval_every > 0 and (step % eval_every == 0 or step == max_steps):
            metrics = evaluate_diffusion_loss(system, val_loader, device, max_batches=eval_batches)
            row = {"step": step, "split": "test", **metrics}
            logger.write(row)
            print(row)
        if sample_every > 0 and (step % sample_every == 0 or step == max_steps):
            snapshot_dir = output / "samples" / f"step_{step:07d}"
            sample_report = export_diffusion_snapshot(
                system,
                snapshot_dir,
                num_samples=sample_num,
                steps=sample_steps,
                sampler=sample_sampler,
            )
            logger.write(
                {
                    "step": step,
                    "split": "sample",
                    "sample_dir": str(snapshot_dir),
                    "mesh_success_rate": sample_report["mesh_success_rate"],
                    "diversity_l1": sample_report["diversity_l1"],
                    "sdf_min": sample_report["sdf_stats"]["min"],
                    "sdf_max": sample_report["sdf_stats"]["max"],
                    "sdf_mean": sample_report["sdf_stats"]["mean"],
                    "occupancy_ratio": sample_report["sdf_stats"]["occupancy_ratio"],
                }
            )
            system.train()
        if step % save_every == 0 or step == max_steps:
            save_checkpoint(last_ckpt, model=system.state_dict(), optimizer=optimizer.state_dict(), step=step)
    return last_ckpt
