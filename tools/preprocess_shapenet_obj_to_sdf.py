from __future__ import annotations

import argparse
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np


CATEGORY_IDS = {
    "chair": "03001627",
    "airplane": "02691156",
    "car": "02958343",
    "table": "04379243",
    "rifle": "04090263",
}


def require_trimesh():
    """Import trimesh lazily so non-preprocess commands do not require mesh dependencies."""
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "Preprocessing requires trimesh. Install it with `python -m pip install -e '.[preprocess]'`."
        ) from exc
    return trimesh


def resolve_category(category: str) -> str:
    """Map a friendly category name such as chair to its ShapeNet synset id."""
    return CATEGORY_IDS.get(category, category)


def as_mesh(scene_or_mesh):
    """Convert a trimesh Scene or Trimesh into one triangular mesh."""
    trimesh = require_trimesh()
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            raise ValueError("Loaded mesh scene is empty.")
        geometries = [
            trimesh.Trimesh(vertices=geom.vertices, faces=geom.faces, process=False)
            for geom in scene_or_mesh.geometry.values()
            if len(geom.vertices) > 0 and len(geom.faces) > 0
        ]
        if not geometries:
            raise ValueError("Loaded mesh scene contains no triangular geometry.")
        return trimesh.util.concatenate(geometries)
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        if len(scene_or_mesh.vertices) == 0 or len(scene_or_mesh.faces) == 0:
            raise ValueError("Loaded mesh has no vertices or faces.")
        return trimesh.Trimesh(vertices=scene_or_mesh.vertices, faces=scene_or_mesh.faces, process=False)
    raise TypeError(f"Unsupported mesh type: {type(scene_or_mesh)!r}")


def load_normalized_mesh(obj_path: Path, surface_samples: int) -> tuple[Any, np.ndarray, float]:
    """Load model.obj and normalize it to a centered unit-scale mesh."""
    trimesh = require_trimesh()
    loaded = trimesh.load_mesh(obj_path, process=False)
    mesh = as_mesh(loaded)
    if len(mesh.faces) > 0 and surface_samples > 0:
        try:
            points, _ = trimesh.sample.sample_surface(mesh, surface_samples)
        except Exception:
            points = np.asarray(mesh.vertices)
    else:
        points = np.asarray(mesh.vertices)
    centroid = np.asarray(points, dtype=np.float32).mean(axis=0)
    centered_points = np.asarray(points, dtype=np.float32) - centroid
    scale = float(np.linalg.norm(centered_points, axis=1).max())
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid normalization scale for {obj_path}: {scale}")
    mesh.vertices = (np.asarray(mesh.vertices, dtype=np.float32) - centroid) / scale
    return mesh, centroid.astype(np.float32), scale


def make_grid_points(resolution: int, extent: float) -> np.ndarray:
    """Create query points for a cubic SDF grid in normalized mesh coordinates."""
    coords = np.linspace(-extent, extent, resolution, dtype=np.float32)
    z_vals, y_vals, x_vals = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.stack([x_vals, y_vals, z_vals], axis=-1).reshape(-1, 3)


def compute_trimesh_sdf(mesh, resolution: int, extent: float, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute a signed distance grid with trimesh as the pure-Python fallback backend."""
    trimesh = require_trimesh()
    points = make_grid_points(resolution, extent)
    values: list[np.ndarray] = []
    for start in range(0, len(points), chunk_size):
        chunk = points[start : start + chunk_size]
        try:
            signed = trimesh.proximity.signed_distance(mesh, chunk)
        except Exception as exc:
            raise RuntimeError(
                "trimesh signed distance failed. Install `rtree` with "
                "`python -m pip install -e '.[preprocess]'`, or use `--backend sdfgen`."
            ) from exc
        # trimesh reports positive values inside for watertight meshes; training expects inside <= 0.
        values.append((-signed).astype(np.float32))
    sdf = np.concatenate(values, axis=0).reshape(resolution, resolution, resolution)
    sdf_params = np.asarray([-extent, -extent, -extent, extent, extent, extent], dtype=np.float32)
    return sdf, sdf_params


def read_sdfgen_dist(dist_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read SDFGen's binary .dist output into a dense grid and bounding parameters."""
    int_size = 4
    float_size = 8
    raw = dist_path.read_bytes()
    if len(raw) < int_size * 3 + float_size * 6:
        raise ValueError(f"SDFGen output is too small: {dist_path}")
    dims = np.frombuffer(raw[: int_size * 3], dtype=np.int32)
    params = np.frombuffer(raw[int_size * 3 : int_size * 3 + float_size * 6], dtype=np.float64).astype(np.float32)
    values = np.frombuffer(raw[int_size * 3 + float_size * 6 :], dtype=np.float32)
    side = round(values.size ** (1.0 / 3.0))
    if side**3 != values.size:
        raise ValueError(f"SDFGen output value count is not cubic: {values.size}")
    grid = values.reshape(side, side, side)
    expected = abs(int(dims[0]))
    if side not in {expected, expected + 1}:
        raise ValueError(f"SDFGen grid side {side} does not match header resolution {dims.tolist()}")
    return grid.astype(np.float32), params


def downsample_sdf_grid(grid: np.ndarray, output_res: int) -> np.ndarray:
    """Downsample SDFGen's dense grid to the training resolution expected by the dataset."""
    if grid.shape == (output_res, output_res, output_res):
        return grid.astype(np.float32)
    side = int(grid.shape[0])
    if grid.shape[0] != grid.shape[1] or grid.shape[1] != grid.shape[2]:
        raise ValueError(f"SDF grid must be cubic, got {grid.shape}")
    if (side - 1) % output_res == 0:
        stride = (side - 1) // output_res
    elif side % output_res == 0:
        stride = side // output_res
    else:
        raise ValueError(f"Cannot downsample grid side {side} to resolution {output_res}")
    return grid[::stride, ::stride, ::stride][:output_res, :output_res, :output_res].astype(np.float32)


def compute_sdfgen_sdf(
    mesh,
    model_dir: Path,
    output_res: int,
    source_res: int,
    sdfgen: Path,
    expand_rate: float,
    index: int,
    keep_intermediate: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Run an external SDFGen binary and convert its output to the training SDF grid."""
    work_dir = model_dir / "_preprocess"
    work_dir.mkdir(parents=True, exist_ok=True)
    norm_obj = work_dir / "pc_norm.obj"
    dist_name = f"{index:08d}.dist"
    dist_path = work_dir / dist_name
    mesh.export(norm_obj)
    cmd = [
        str(sdfgen),
        str(norm_obj),
        str(source_res),
        str(source_res),
        str(source_res),
        "-s",
        "-e",
        str(expand_rate),
        "-o",
        dist_name,
        "-m",
        "1",
        "-c",
    ]
    subprocess.run(cmd, cwd=work_dir, check=True)
    grid, params = read_sdfgen_dist(dist_path)
    sdf = downsample_sdf_grid(grid, output_res)
    if not keep_intermediate:
        for path in (norm_obj, dist_path):
            if path.exists():
                path.unlink()
        try:
            work_dir.rmdir()
        except OSError:
            pass
    return sdf, params


def write_sdf_h5(
    h5_path: Path,
    sdf: np.ndarray,
    sdf_params: np.ndarray,
    centroid: np.ndarray,
    scale: float,
    source_obj: Path,
    backend: str,
) -> None:
    """Write an SDFusion-compatible ori_sample_grid.h5 file for ShapeNetSDFDataset."""
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    norm_params = np.concatenate([centroid.astype(np.float32), np.asarray([scale], dtype=np.float32)])
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset("pc_sdf_original", data=np.zeros((1, 3), dtype=np.float32), compression="gzip", compression_opts=4)
        h5_file.create_dataset("pc_sdf_sample", data=sdf.reshape(-1, 1).astype(np.float32), compression="gzip", compression_opts=4)
        h5_file.create_dataset("norm_params", data=norm_params, compression="gzip", compression_opts=4)
        h5_file.create_dataset("sdf_params", data=sdf_params.astype(np.float32), compression="gzip", compression_opts=4)
        h5_file.attrs["source_obj"] = str(source_obj)
        h5_file.attrs["backend"] = backend


def collect_obj_paths(shapenet_root: Path, cat_id: str, model_ids: list[str] | None, max_models: int | None) -> list[Path]:
    """Collect ShapeNetCore model.obj paths for a category and optional model id subset."""
    category_root = shapenet_root / cat_id
    if model_ids:
        paths = [category_root / model_id / "model.obj" for model_id in model_ids]
    else:
        paths = sorted(category_root.glob("*/model.obj"))
    if max_models is not None:
        paths = paths[:max_models]
    return paths


def read_model_ids(path: Path | None) -> list[str] | None:
    """Read a plain text model-id list, or return None when no list is provided."""
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def process_obj(obj_path: Path, args: argparse.Namespace, cat_id: str, backend: str, index: int) -> dict[str, Any]:
    """Convert one ShapeNet model.obj into the h5 file consumed by training."""
    model_id = obj_path.parent.name
    out_dir = Path(args.data_root) / "ShapeNet" / "SDF_v1" / f"resolution_{args.res}" / cat_id / model_id
    h5_path = out_dir / "ori_sample_grid.h5"
    if h5_path.exists() and not args.overwrite:
        return {"model_id": model_id, "obj": str(obj_path), "h5": str(h5_path), "status": "skipped"}
    if not obj_path.exists():
        return {"model_id": model_id, "obj": str(obj_path), "h5": str(h5_path), "status": "missing_obj"}

    mesh, centroid, scale = load_normalized_mesh(obj_path, surface_samples=args.surface_samples)
    if backend == "sdfgen":
        sdf, sdf_params = compute_sdfgen_sdf(
            mesh,
            out_dir,
            output_res=args.res,
            source_res=args.source_res,
            sdfgen=Path(args.sdfgen),
            expand_rate=args.expand_rate,
            index=index,
            keep_intermediate=args.keep_intermediate,
        )
    elif backend == "trimesh":
        sdf, sdf_params = compute_trimesh_sdf(mesh, resolution=args.res, extent=args.grid_extent, chunk_size=args.chunk_size)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    write_sdf_h5(h5_path, sdf, sdf_params, centroid, scale, obj_path, backend)
    return {
        "model_id": model_id,
        "obj": str(obj_path),
        "h5": str(h5_path),
        "status": "ok",
        "min": float(sdf.min()),
        "max": float(sdf.max()),
        "mean": float(sdf.mean()),
    }


def choose_backend(args: argparse.Namespace) -> str:
    """Resolve auto/trimesh/sdfgen backend selection from CLI arguments."""
    if args.backend == "auto":
        return "sdfgen" if args.sdfgen else "trimesh"
    if args.backend == "sdfgen" and not args.sdfgen:
        raise ValueError("--backend sdfgen requires --sdfgen /path/to/SDFGen")
    return args.backend


def write_filelist(records: list[dict[str, Any]], args: argparse.Namespace, cat_id: str) -> str | None:
    """Write a split file containing successfully generated model ids when requested."""
    if not args.write_filelist:
        return None
    root = Path(args.split_file_root) if args.split_file_root else Path(args.data_root) / "ShapeNet_filelists"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{cat_id}_{args.split}.lst"
    model_ids = [record["model_id"] for record in records if record.get("status") in {"ok", "skipped"}]
    path.write_text("\n".join(model_ids) + ("\n" if model_ids else ""), encoding="utf-8")
    return str(path)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for ShapeNetCore OBJ preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess ShapeNetCore model.obj files into SDFusion-style SDF h5 files.")
    parser.add_argument("--shapenet_root", required=True, help="Path to ShapeNetCore.v1, containing <cat_id>/<model_id>/model.obj.")
    parser.add_argument("--data_root", default="data", help="Output data root used by ShapeNetSDFDataset.")
    parser.add_argument("--category", default="chair", help="Category name or ShapeNet synset id.")
    parser.add_argument("--model_id", action="append", default=None, help="Process one model id. Can be passed multiple times.")
    parser.add_argument("--model_list", default=None, help="Optional text file with model ids to process.")
    parser.add_argument("--split", default="train", help="Split name used when writing a filelist.")
    parser.add_argument("--res", type=int, default=64, help="Output SDF grid resolution.")
    parser.add_argument("--max_models", type=int, default=None)
    parser.add_argument("--backend", choices=["auto", "trimesh", "sdfgen"], default="auto")
    parser.add_argument("--sdfgen", default=None, help="Path to SDFGen binary. Enables the SDFusion-style backend.")
    parser.add_argument("--source_res", type=int, default=256, help="SDFGen source resolution before downsampling.")
    parser.add_argument("--expand_rate", type=float, default=1.3)
    parser.add_argument("--grid_extent", type=float, default=1.05, help="Pure-trimesh SDF grid covers [-extent, extent]^3 after normalization.")
    parser.add_argument("--surface_samples", type=int, default=16384)
    parser.add_argument("--chunk_size", type=int, default=200000)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep_intermediate", action="store_true")
    parser.add_argument("--write_filelist", action="store_true")
    parser.add_argument("--split_file_root", default=None)
    return parser


def main() -> None:
    """Preprocess a category or model subset and write a JSON summary."""
    parser = build_parser()
    args = parser.parse_args()
    cat_id = resolve_category(args.category)
    backend = choose_backend(args)
    model_ids = read_model_ids(Path(args.model_list) if args.model_list else None)
    if args.model_id:
        model_ids = (model_ids or []) + args.model_id
    obj_paths = collect_obj_paths(Path(args.shapenet_root), cat_id, model_ids, args.max_models)
    if not obj_paths:
        raise FileNotFoundError(f"No model.obj files found under {Path(args.shapenet_root) / cat_id}")

    records: list[dict[str, Any]] = []
    if args.num_workers <= 1:
        for index, obj_path in enumerate(obj_paths):
            records.append(process_obj(obj_path, args, cat_id, backend, index))
            print(json.dumps(records[-1], sort_keys=True))
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_obj, obj_path, args, cat_id, backend, index): obj_path
                for index, obj_path in enumerate(obj_paths)
            }
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                print(json.dumps(record, sort_keys=True))

    filelist = write_filelist(records, args, cat_id)
    summary = {
        "category": args.category,
        "cat_id": cat_id,
        "backend": backend,
        "num_requested": len(obj_paths),
        "num_ok": sum(1 for record in records if record.get("status") == "ok"),
        "num_skipped": sum(1 for record in records if record.get("status") == "skipped"),
        "num_missing": sum(1 for record in records if record.get("status") == "missing_obj"),
        "filelist": filelist,
        "records": sorted(records, key=lambda row: row["model_id"]),
    }
    summary_dir = Path(args.data_root) / "ShapeNet" / "SDF_v1" / f"resolution_{args.res}" / cat_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "preprocess_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), **{k: v for k, v in summary.items() if k != "records"}}, indent=2))


if __name__ == "__main__":
    main()
