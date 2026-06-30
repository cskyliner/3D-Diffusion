from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT  # noqa: F401
from data.shapenet_sdf import DEFAULT_CATEGORY_IDS, ShapeNetSDFDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ShapeNet SDF dataset paths, split files, and tensor ranges.")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--category", default="chair")
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--filelist", default=None)
    parser.add_argument("--split_file_root", default=None)
    parser.add_argument("--trunc_thres", type=float, default=0.0)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    dataset = ShapeNetSDFDataset(
        data_root=args.data_root,
        category=args.category,
        split=args.split,
        res=args.res,
        max_samples=args.max_samples,
        filelist=args.filelist,
        split_file_root=args.split_file_root,
        trunc_thres=args.trunc_thres,
    )
    sample = dataset[0]
    sdf = sample["sdf"]
    report = {
        "category": args.category,
        "synset": DEFAULT_CATEGORY_IDS[args.category],
        "split": args.split,
        "data_root": str(Path(args.data_root)),
        "base_dir": str(dataset.base_dir),
        "filelist": args.filelist,
        "max_samples": args.max_samples,
        "num_samples": len(dataset),
        "total_candidates_before_max_samples": dataset.total_candidates,
        "sample_0_path": sample["path"],
        "sample_0_model_id": sample["model_id"],
        "sdf_shape": list(sdf.shape),
        "sdf_min": float(sdf.min()),
        "sdf_max": float(sdf.max()),
        "sdf_mean": float(sdf.mean()),
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"category: {report['category']}")
    print(f"synset: {report['synset']}")
    print(f"split: {report['split']}")
    print(f"data_root: {report['data_root']}")
    print(f"base_dir: {report['base_dir']}")
    print(f"filelist: {report['filelist']}")
    print(f"max_samples: {report['max_samples']}")
    print(f"num_samples: {report['num_samples']}")
    if report["total_candidates_before_max_samples"] is not None:
        print(f"total_candidates_before_max_samples: {report['total_candidates_before_max_samples']}")
    print(f"sample[0] path: {report['sample_0_path']}")
    print(f"sample[0] model_id: {report['sample_0_model_id']}")
    print(f"sdf shape: {report['sdf_shape']}")
    print(f"sdf min/max/mean: {report['sdf_min']:.6f} / {report['sdf_max']:.6f} / {report['sdf_mean']:.6f}")


if __name__ == "__main__":
    main()
