from __future__ import annotations

from .shapenet_sdf import ShapeNetSDFDataset

DATASETS = {
    "shapenet_sdf": ShapeNetSDFDataset,
}


def get_dataset(name: str):
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(DATASETS)}")
    return DATASETS[name]
