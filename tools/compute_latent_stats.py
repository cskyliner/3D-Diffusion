from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _common import add_config_args, load_run_config
from engine.checkpoint import load_model_checkpoint
from engine.trainer import build_dataloader, build_vqvae, move_to_device, resolve_device


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Compute VQ-VAE latent statistics and recommended scale_factor.")
    add_config_args(parser)
    parser.add_argument("--vqvae_ckpt", required=True, help="Path to trained VQ-VAE checkpoint.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument(
        "--latent_mode",
        choices=["continuous", "quantized"],
        default="continuous",
        help="Use continuous encoder latents for SDFusion-style diffusion, or quantized codebook latents.",
    )
    args = parser.parse_args()

    config = load_run_config(args)
    device = resolve_device(str(config.get("train", {}).get("device", "cuda")))
    loader = build_dataloader(config, split=args.split, shuffle=False)
    model = build_vqvae(config).to(device)
    load_model_checkpoint(model, args.vqvae_ckpt, component="vqvae", strict=False)
    model.eval()

    total = 0
    sum_value = torch.tensor(0.0, device=device)
    sum_sq_value = torch.tensor(0.0, device=device)
    channel_sum = None
    channel_sum_sq = None
    channel_count = 0
    for batch_index, batch in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        batch = move_to_device(batch, device)
        z = model.encode(batch["sdf"])
        latent = z
        if args.latent_mode == "quantized":
            latent, _, _ = model.quantize_latent(z)
        total += latent.numel()
        sum_value += latent.sum()
        sum_sq_value += (latent * latent).sum()
        reduce_dims = (0, 2, 3, 4)
        if channel_sum is None:
            channel_sum = latent.sum(dim=reduce_dims)
            channel_sum_sq = (latent * latent).sum(dim=reduce_dims)
        else:
            channel_sum += latent.sum(dim=reduce_dims)
            channel_sum_sq += (latent * latent).sum(dim=reduce_dims)
        channel_count += latent.shape[0] * latent.shape[2] * latent.shape[3] * latent.shape[4]

    if total == 0:
        raise RuntimeError("No latent tensors were processed.")
    mean = sum_value / total
    variance = (sum_sq_value / total - mean * mean).clamp_min(0.0)
    std = variance.sqrt()
    scale_factor = 1.0 / std.clamp_min(1.0e-12)
    channel_mean = channel_sum / channel_count
    channel_std = (channel_sum_sq / channel_count - channel_mean * channel_mean).clamp_min(0.0).sqrt()
    report = {
        "split": args.split,
        "latent_mode": args.latent_mode,
        "num_values": int(total),
        "mean": float(mean.detach().cpu()),
        "std": float(std.detach().cpu()),
        "scale_factor": float(scale_factor.detach().cpu()),
        "channel_mean": [float(x) for x in channel_mean.detach().cpu()],
        "channel_std": [float(x) for x in channel_std.detach().cpu()],
    }
    out_dir = Path(args.out_dir)
    pt_path = out_dir / "latent_stats.pt"
    json_path = out_dir / "latent_stats.json"
    torch.save({**report, "channel_mean_tensor": channel_mean.detach().cpu(), "channel_std_tensor": channel_std.detach().cpu()}, pt_path)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"pt": str(pt_path), "json": str(json_path), **report}, indent=2))


if __name__ == "__main__":
    main()
