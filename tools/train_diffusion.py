from __future__ import annotations

import argparse

from _common import add_config_args, load_run_config
from engine.trainer import train_diffusion


def main() -> None:
    parser = argparse.ArgumentParser(description="Train latent SDF diffusion.")
    add_config_args(parser)
    parser.add_argument("--vqvae_ckpt", required=True, help="Path to trained VQ-VAE checkpoint.")
    parser.add_argument("--resume", default=None, help="Optional diffusion/system checkpoint to resume/load.")
    args = parser.parse_args()
    config = load_run_config(args)
    ckpt = train_diffusion(config, args.out_dir, vqvae_ckpt=args.vqvae_ckpt, resume=args.resume)
    print(f"saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
