from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import ROOT  # noqa: F401
from utils.metrics import diversity_l1, sdf_stats


def _metadata_success(path: Path) -> bool | None:
    meta_path = path.with_name(path.name.replace(".sdf.npy", ".metadata.json"))
    if not meta_path.exists():
        return None
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    mesh = metadata.get("mesh", {})
    if isinstance(mesh, dict) and "success" in mesh:
        return bool(mesh["success"])
    return None


def _ply_success(path: Path) -> bool:
    ply_path = path.with_name(path.name.replace(".sdf.npy", ".ply"))
    if not ply_path.exists():
        return False
    vertices = 0
    faces = 0
    with ply_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("element vertex"):
                vertices = int(line.split()[-1])
            elif line.startswith("element face"):
                faces = int(line.split()[-1])
            elif line == "end_header":
                break
    return vertices > 0 and faces > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated SDF samples in a directory.")
    parser.add_argument("--sample_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    sdf_paths = sorted(sample_dir.glob("sample_*.sdf.npy"))
    sdfs: list[np.ndarray] = []
    per_sample: list[dict] = []
    mesh_success = 0
    failed = 0
    for path in sdf_paths:
        sdf = np.asarray(np.load(path), dtype=np.float32)
        sdfs.append(sdf)
        success = _metadata_success(path)
        if success is None:
            success = _ply_success(path)
        mesh_success += int(success)
        failed += int(not success)
        per_sample.append({"path": str(path), "mesh_success": bool(success), "stats": sdf_stats(sdf, threshold=args.threshold)})

    aggregate_stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "occupancy_ratio": 0.0}
    if sdfs:
        stacked = np.stack(sdfs)
        aggregate_stats = sdf_stats(stacked, threshold=args.threshold)
    total = len(sdf_paths)
    success_rate = float(mesh_success / total) if total else 0.0
    report = {
        "sample_dir": str(sample_dir),
        "num_samples": total,
        "mesh_success": mesh_success,
        "mesh_failed": failed,
        "mesh_success_rate": success_rate,
        "sdf_stats": aggregate_stats,
        "diversity_l1": diversity_l1(sdfs),
        "per_sample": per_sample,
        "summary": f"Generated {total} samples; {mesh_success} meshes extracted successfully ({success_rate * 100:.1f}%).",
    }
    output_path = sample_dir / "evaluation.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
