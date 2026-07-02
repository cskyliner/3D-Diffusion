from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _empty_ply() -> str:
    return "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n"


def write_ply(vertices: np.ndarray, faces: np.ndarray, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        for vertex in vertices:
            handle.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")
    return output


def sdf_to_mesh(sdf: np.ndarray, ply_path: str | Path, level: float = 0.0) -> dict[str, Any]:
    output = Path(ply_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    grid = np.asarray(sdf, dtype=np.float32).squeeze()
    stats = {"min": float(grid.min()), "max": float(grid.max()), "level": float(level)}
    try:
        from skimage import measure

        vertices, faces, normals, values = measure.marching_cubes(grid, level=level)
        write_ply(vertices, faces, output)
        return {"success": True, "vertices": int(len(vertices)), "faces": int(len(faces)), "path": str(output), "backend": "skimage", **stats}
    except ImportError as skimage_exc:
        try:
            import mcubes

            vertices, faces = mcubes.marching_cubes(grid, level)
            write_ply(vertices, faces, output)
            return {"success": True, "vertices": int(len(vertices)), "faces": int(len(faces)), "path": str(output), "backend": "mcubes", **stats}
        except Exception as mcubes_exc:
            output.write_text(_empty_ply(), encoding="utf-8")
            return {
                "success": False,
                "error": "mesh extraction dependencies are unavailable",
                "skimage_error": repr(skimage_exc),
                "mcubes_error": repr(mcubes_exc),
                "path": str(output),
                **stats,
            }
    except Exception as exc:
        output.write_text(_empty_ply(), encoding="utf-8")
        return {"success": False, "error": str(exc), "path": str(output), "backend": "skimage", **stats}
