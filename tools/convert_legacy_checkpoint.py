from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _common import ROOT  # noqa: F401
from engine.checkpoint import (
    convert_legacy_diffusion_state,
    convert_legacy_vqvae_state,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert/inspect SDFusion legacy checkpoints for 3D-Diffusion.")
    parser.add_argument("--input", required=True, help="Legacy .pth/.pt checkpoint.")
    parser.add_argument("--output", required=True, help="Converted checkpoint path.")
    parser.add_argument("--component", choices=["vqvae", "diffusion"], default="vqvae")
    args = parser.parse_args()

    payload = load_checkpoint(args.input, map_location="cpu")
    preferred = ("vqvae", "df", "model", "state_dict") if args.component == "vqvae" else ("df", "model", "state_dict")
    state = extract_state_dict(payload, preferred_keys=preferred)
    converted = convert_legacy_vqvae_state(state) if args.component == "vqvae" else convert_legacy_diffusion_state(state, target_prefix=None)
    metadata = {
        "source": str(Path(args.input).resolve()),
        "component": args.component,
        "num_tensors": len(converted),
        "num_parameters": int(sum(v.numel() for v in converted.values() if torch.is_tensor(v))),
        "note": "VQ-VAE legacy keys are preserved for architecture=legacy. Diffusion conversion is best-effort for the refactored UNet.",
    }
    save_checkpoint(args.output, **{args.component if args.component == "vqvae" else "df": converted}, metadata=metadata)
    report_path = Path(args.output).with_suffix(".json")
    report_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(Path(args.output).resolve()), "report": str(report_path), **metadata}, indent=2))


if __name__ == "__main__":
    main()
