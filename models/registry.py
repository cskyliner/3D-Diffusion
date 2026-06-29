from __future__ import annotations

from .vqvae import SDFVQVAE

MODELS = {
    "vqvae": SDFVQVAE,
}


def get_model(name: str):
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODELS)}")
    return MODELS[name]
