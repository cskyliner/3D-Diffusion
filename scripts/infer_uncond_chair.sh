#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-data}
OUT_DIR=${OUT_DIR:-outputs}
CATEGORY=${CATEGORY:-chair}
NUM_SAMPLES=${NUM_SAMPLES:-4}
DDIM_STEPS=${DDIM_STEPS:-100}
VQVAE_CKPT=${VQVAE_CKPT:-${OUT_DIR}/vqvae_${CATEGORY}/checkpoints/vqvae_last.pt}
DIFFUSION_CKPT=${DIFFUSION_CKPT:-${OUT_DIR}/diffusion_${CATEGORY}/checkpoints/diffusion_last.pt}

python tools/infer_uncond.py \
  --config config/defaults/diffusion_snet_chair.yaml \
  --out_dir "${OUT_DIR}/samples_${CATEGORY}" \
  --vqvae_ckpt "${VQVAE_CKPT}" \
  --diffusion_ckpt "${DIFFUSION_CKPT}" \
  --num_samples "${NUM_SAMPLES}" \
  --ddim_steps "${DDIM_STEPS}" \
  --override "data.data_root=${DATA_ROOT}" \
  --override "data.category=${CATEGORY}"
