from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import add_config_args, load_run_config
from engine.checkpoint import load_model_checkpoint
from engine.inference import generate_unconditional
from engine.trainer import build_uncond_system, build_vqvae, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unconditional SDF samples from trained checkpoints.")
    add_config_args(parser)
    parser.add_argument("--vqvae_ckpt", required=True, help="Path to trained VQ-VAE checkpoint.")
    parser.add_argument("--diffusion_ckpt", required=True, help="Path to trained diffusion checkpoint.")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--ddim_steps", type=int, default=100)
    parser.add_argument("--sampler", choices=["ddim", "ddpm", "plms"], default="ddim")
    parser.add_argument("--ddim_discretize", choices=["uniform", "quad"], default="uniform")
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--clip_denoised", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    config = load_run_config(args)
    device = resolve_device(str(config.get("train", {}).get("device", "cuda")))
    vqvae = build_vqvae(config)
    load_model_checkpoint(vqvae, args.vqvae_ckpt, component="vqvae", strict=False)
    system = build_uncond_system(config, vqvae).to(device)
    report = load_model_checkpoint(system, args.diffusion_ckpt, component="model", strict=False, map_location=device)
    system.eval()
    diffusion_cfg = config.get("diffusion", {})
    eta = float(diffusion_cfg.get("ddim_eta", 0.0)) if args.eta is None else args.eta
    guidance_scale = float(diffusion_cfg.get("guidance_scale", 1.0)) if args.guidance_scale is None else args.guidance_scale
    results = generate_unconditional(
        system,
        num_samples=args.num_samples,
        out_dir=args.out_dir,
        ddim_steps=args.ddim_steps,
        sampler=args.sampler,
        eta=eta,
        guidance_scale=guidance_scale,
        ddim_discretize=args.ddim_discretize,
        clip_denoised=args.clip_denoised,
        temperature=args.temperature,
        progress=args.progress,
    )
    summary = {
        "num_samples": len(results),
        "sampler": args.sampler,
        "steps": args.ddim_steps,
        "ddim_discretize": args.ddim_discretize,
        "eta": eta,
        "guidance_scale": guidance_scale,
        "clip_denoised": args.clip_denoised,
        "temperature": args.temperature,
        "load_report": {key: value for key, value in report.items() if key != "payload"},
        "samples": results,
    }
    summary_path = Path(args.out_dir) / "inference_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
