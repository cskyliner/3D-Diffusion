#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-data}
OUT_DIR=${OUT_DIR:-outputs}
CATEGORY=${CATEGORY:-chair}
SAMPLE_DIR=${SAMPLE_DIR:-${OUT_DIR}/samples_${CATEGORY}}

python tools/inspect_dataset.py \
  --data_root "${DATA_ROOT}" \
  --category "${CATEGORY}" \
  --res 64 \
  --split train \
  --max_samples 2

python tools/evaluate_generation.py \
  --sample_dir "${SAMPLE_DIR}"
