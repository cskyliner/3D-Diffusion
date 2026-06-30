#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from config.load import load_config
from data.shapenet_sdf import ShapeNetSDFDataset
from engine.trainer import build_vqvae, build_uncond_system
from modules.diffusion import DDIMSampler, GaussianDiffusion, PLMSSampler, UNet3D

cfg = load_config("config/defaults/diffusion_snet_chair.yaml", ["train.device=cpu"])
vqvae = build_vqvae(cfg)
system = build_uncond_system(cfg, vqvae)
print("imports_ok")
print(type(system).__name__)
PY
