from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from utils.mesh import sdf_to_mesh
from utils.metrics import sdf_stats
from utils.sdf_io import save_sdf_npy


@torch.no_grad()
def generate_unconditional(
    system,
    num_samples: int,
    out_dir: str | Path,
    ddim_steps: int = 100,
    sampler: str = "ddim",
    eta: float = 0.0,
    guidance_scale: float = 1.0,
    ddim_discretize: str = "uniform",
    clip_denoised: bool = False,
    temperature: float = 1.0,
    progress: bool = False,
) -> list[dict]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sdf_batch = system.sample(
        num_samples=num_samples,
        sampler=sampler,
        steps=ddim_steps,
        eta=eta,
        guidance_scale=guidance_scale,
        ddim_discretize=ddim_discretize,
        clip_denoised=clip_denoised,
        temperature=temperature,
        progress=progress,
    )
    results: list[dict] = []
    for index, sdf in enumerate(sdf_batch):
        sdf_np = sdf.detach().cpu().numpy().astype(np.float32)
        stem = f"sample_{index:04d}"
        sdf_path = save_sdf_npy(sdf_np, output_dir / f"{stem}.sdf.npy")
        mesh_meta = sdf_to_mesh(sdf_np, output_dir / f"{stem}.ply")
        metadata = {"sample": index, "sdf_path": str(sdf_path), "mesh": mesh_meta, "stats": sdf_stats(sdf_np)}
        meta_path = output_dir / f"{stem}.metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        results.append(metadata)
    return results
