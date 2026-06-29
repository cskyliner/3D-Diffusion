from __future__ import annotations

import math

import numpy as np


def make_beta_schedule(
    schedule: str = "linear",
    timesteps: int = 1000,
    linear_start: float = 1.0e-4,
    linear_end: float = 2.0e-2,
    cosine_s: float = 8.0e-3,
) -> np.ndarray:
    if schedule == "linear":
        return np.linspace(linear_start, linear_end, timesteps, dtype=np.float64)
    if schedule == "cosine":
        steps = timesteps + 1
        x = np.linspace(0, timesteps, steps, dtype=np.float64)
        alphas_cumprod = np.cos(((x / timesteps) + cosine_s) / (1 + cosine_s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0, 0.999)
    raise NotImplementedError(f"Unknown beta schedule: {schedule}")
