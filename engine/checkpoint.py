from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, **payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu"):
    return torch.load(path, map_location=map_location)
