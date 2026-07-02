from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _common import ROOT  # noqa: F401
from config.load import load_config
from engine.checkpoint import load_model_checkpoint, save_checkpoint
from engine.trainer import build_vqvae


def json_safe(value: Any) -> Any:
    """Convert lightweight checkpoint metadata to JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Return lightweight checkpoint metadata without serializing tensors."""
    return {
        "payload_keys": sorted(str(key) for key in payload.keys()),
        "global_step": json_safe(payload.get("global_step")),
        "step": json_safe(payload.get("step")),
        "metadata": json_safe(payload.get("metadata")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check or convert a VQ-VAE checkpoint for this refactor.")
    parser.add_argument("--config", required=True, help="Config whose vqvae section defines the target architecture.")
    parser.add_argument("--ckpt", required=True, help="VQ-VAE checkpoint, including SDFusion vqvae-snet-all.pth.")
    parser.add_argument("--out", default=None, help="Optional path to save a converted refactored checkpoint.")
    parser.add_argument("--strict", action="store_true", help="Require every checkpoint key to match exactly.")
    parser.add_argument("--override", action="append", default=[], help="Dotted config override, for example vqvae.n_embed=8192.")
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    model = build_vqvae(config)
    report = load_model_checkpoint(model, args.ckpt, component="vqvae", strict=args.strict, map_location="cpu")
    payload = report.pop("payload", {})
    result: dict[str, Any] = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "config": str(Path(args.config).resolve()),
        "strict": bool(args.strict),
        "vqvae": dict(config.get("vqvae", {})),
        **payload_summary(payload),
        **report,
    }

    if args.out:
        metadata = {
            "source": result["checkpoint"],
            "source_payload_keys": result["payload_keys"],
            "matched_keys": result["matched_keys"],
            "matched_params": result["matched_params"],
            "model_params": result["model_params"],
            "matched_param_ratio": result["matched_param_ratio"],
            "note": "Converted from a compatible VQ-VAE checkpoint for this refactored codebase.",
        }
        save_checkpoint(args.out, vqvae=model.state_dict(), metadata=metadata)
        result["converted_checkpoint"] = str(Path(args.out).resolve())

    print(json.dumps(json_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
