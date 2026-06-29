from __future__ import annotations

import argparse

from _common import add_config_args, load_run_config
from engine.trainer import train_vqvae


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SDFusion-compatible VQ-VAE.")
    add_config_args(parser)
    parser.add_argument("--resume", default=None, help="Optional VQ-VAE checkpoint to resume/load.")
    args = parser.parse_args()
    config = load_run_config(args)
    ckpt = train_vqvae(config, args.out_dir, resume=args.resume)
    print(f"saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
