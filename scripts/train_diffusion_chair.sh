#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-data}
OUT_DIR=${OUT_DIR:-outputs}
CATEGORY=${CATEGORY:-chair}
VQVAE_CKPT=${VQVAE_CKPT:-${OUT_DIR}/vqvae_${CATEGORY}/checkpoints/vqvae_last.pt}
LATENT_STATS=${LATENT_STATS:-}

SCALE_OVERRIDE=()
if [[ -n "${LATENT_STATS}" && -f "${LATENT_STATS}" ]]; then
  SCALE_FACTOR=$(python - <<PY
import json
print(json.load(open("${LATENT_STATS}", "r", encoding="utf-8"))["scale_factor"])
PY
)
  SCALE_OVERRIDE=(--override "diffusion.scale_factor=${SCALE_FACTOR}")
fi

python tools/train_diffusion.py \
  --config config/defaults/diffusion_snet_chair.yaml \
  --out_dir "${OUT_DIR}/diffusion_${CATEGORY}" \
  --vqvae_ckpt "${VQVAE_CKPT}" \
  --override "data.data_root=${DATA_ROOT}" \
  --override "data.category=${CATEGORY}" \
  "${SCALE_OVERRIDE[@]}"
