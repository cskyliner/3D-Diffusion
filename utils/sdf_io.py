from __future__ import annotations

from pathlib import Path

import numpy as np


def save_sdf_npy(sdf, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    array = sdf.detach().cpu().numpy() if hasattr(sdf, "detach") else np.asarray(sdf)
    np.save(output, array.astype(np.float32))
    return output


def load_sdf_npy(path: str | Path) -> np.ndarray:
    return np.load(Path(path))
