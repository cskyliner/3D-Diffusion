#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-data}
OUT_DIR=${OUT_DIR:-outputs}
CATEGORY=${CATEGORY:-chair}

python tools/train_vqvae.py \
  --config config/defaults/vqvae_snet_chair.yaml \
  --out_dir "${OUT_DIR}/vqvae_${CATEGORY}" \
  --override "data.data_root=${DATA_ROOT}" \
  --override "data.category=${CATEGORY}"
