#!/usr/bin/env bash
set -euo pipefail

SHAPENET_ROOT=${SHAPENET_ROOT:-data/ShapeNetCore.v1}
DATA_ROOT=${DATA_ROOT:-data}
CATEGORY=${CATEGORY:-chair}
BACKEND=${BACKEND:-auto}
RES=${RES:-64}
NUM_WORKERS=${NUM_WORKERS:-1}

SDFGEN_ARGS=()
if [[ -n "${SDFGEN:-}" ]]; then
  SDFGEN_ARGS=(--sdfgen "${SDFGEN}")
fi

python tools/preprocess_shapenet_obj_to_sdf.py \
  --shapenet_root "${SHAPENET_ROOT}" \
  --data_root "${DATA_ROOT}" \
  --category "${CATEGORY}" \
  --backend "${BACKEND}" \
  --res "${RES}" \
  --num_workers "${NUM_WORKERS}" \
  --write_filelist \
  "${SDFGEN_ARGS[@]}" \
  "$@"
