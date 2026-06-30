from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn


STATE_KEYS = ("model", "state_dict", "vqvae", "df", "denoiser")
PREFIXES = ("module.", "model.", "vqvae_module.", "df_module.")


def save_checkpoint(path: str | Path, **payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    return torch.load(path, map_location=map_location)


def strip_known_prefixes(state_dict: dict[str, torch.Tensor], prefixes: Iterable[str] = PREFIXES) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def extract_state_dict(payload: Any, preferred_keys: Iterable[str] = STATE_KEYS) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, dict) and all(isinstance(k, str) for k in value):
                return value
        if all(isinstance(k, str) for k in payload):
            tensor_values = [v for v in payload.values() if torch.is_tensor(v)]
            if tensor_values:
                return payload
    raise ValueError("Checkpoint does not contain a recognizable PyTorch state_dict.")


def convert_legacy_vqvae_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted = strip_known_prefixes(state_dict)
    output: dict[str, torch.Tensor] = {}
    for key, value in converted.items():
        if key.startswith("quantizer."):
            key = "quantize." + key[len("quantizer.") :]
        output[key] = value
    return output


def convert_legacy_diffusion_state(state_dict: dict[str, torch.Tensor], target_prefix: str | None = None) -> dict[str, torch.Tensor]:
    converted = strip_known_prefixes(state_dict)
    output: dict[str, torch.Tensor] = {}
    for key, value in converted.items():
        if key.startswith("diffusion_net.") and target_prefix is not None:
            output[target_prefix + key] = value
        else:
            output[key] = value
    return output


def adapt_state_dict_to_model(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    model_keys = set(model.state_dict().keys())
    if state_dict and all(key.startswith("diffusion_net.") for key in state_dict) and any(
        key.startswith("denoiser.diffusion_net.") for key in model_keys
    ):
        return {f"denoiser.{key}": value for key, value in state_dict.items()}
    if state_dict and all(not key.startswith("denoiser.") for key in state_dict) and any(
        key.startswith("denoiser.diffusion_net.") for key in model_keys
    ):
        diffusion_keys = {f"denoiser.diffusion_net.{key}": value for key, value in state_dict.items()}
        if any(key in model_keys for key in diffusion_keys):
            return diffusion_keys
    return state_dict


def load_model_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    component: str = "model",
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = load_checkpoint(path, map_location=map_location)
    keys = (component,) + tuple(key for key in STATE_KEYS if key != component)
    state_dict = extract_state_dict(payload, preferred_keys=keys)
    if component == "vqvae":
        state_dict = convert_legacy_vqvae_state(state_dict)
    elif component in {"df", "denoiser", "diffusion"}:
        state_dict = convert_legacy_diffusion_state(state_dict, target_prefix=None)
    else:
        state_dict = strip_known_prefixes(state_dict)
    state_dict = adapt_state_dict_to_model(model, state_dict)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return {
        "payload": payload if isinstance(payload, dict) else {},
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
