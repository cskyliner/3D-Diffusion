from __future__ import annotations

import itertools

import numpy as np


def sdf_stats(sdf: np.ndarray, threshold: float = 0.0) -> dict[str, float]:
    array = np.asarray(sdf, dtype=np.float32)
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "occupancy_ratio": float((array <= threshold).mean()),
    }


def diversity_l1(sdfs: list[np.ndarray]) -> float:
    if len(sdfs) < 2:
        return 0.0
    values = [float(np.mean(np.abs(a - b))) for a, b in itertools.combinations(sdfs, 2)]
    return float(np.mean(values)) if values else 0.0
