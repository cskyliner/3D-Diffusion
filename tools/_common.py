from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.load import load_config, save_config  # noqa: E402


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--out_dir", required=True, help="Directory for logs, checkpoints, and exports.")
    parser.add_argument("--override", action="append", default=[], help="Dotted config override, for example train.max_steps=10.")


def load_run_config(args: argparse.Namespace) -> dict:
    config = load_config(args.config, args.override)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, out_dir / "resolved_config.yaml")
    return config
