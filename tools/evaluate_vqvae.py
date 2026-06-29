from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _common import add_config_args, load_run_config
from engine.checkpoint import load_model_checkpoint
from engine.trainer import build_dataloader, build_vqvae, evaluate_vqvae, export_vqvae_reconstructions, move_to_device, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a VQ-VAE checkpoint and export reconstructions.")
    add_config_args(parser)
    parser.add_argument("--ckpt", required=True, help="VQ-VAE checkpoint.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate.")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--export_items", type=int, default=8)
    args = parser.parse_args()
    config = load_run_config(args)
    device = resolve_device(str(config.get("train", {}).get("device", "cuda")))
    loader = build_dataloader(config, split=args.split, shuffle=False)
    model = build_vqvae(config).to(device)
    report = load_model_checkpoint(model, args.ckpt, component="vqvae", strict=False)
    metrics = evaluate_vqvae(model, loader, device, max_batches=args.max_batches)
    first_batch = move_to_device(next(iter(loader)), device)
    exports = export_vqvae_reconstructions(model, first_batch, Path(args.out_dir) / "reconstructions", max_items=args.export_items)
    result = {"metrics": metrics, "load_report": {k: v for k, v in report.items() if k != "payload"}, "exports": exports}
    result_path = Path(args.out_dir) / "evaluation.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
