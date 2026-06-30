from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_CATEGORY_IDS = {
    "chair": "03001627",
    "airplane": "02691156",
    "car": "02958343",
    "table": "04379243",
    "rifle": "04090263",
}


class ShapeNetSDFDataset(Dataset):
    """Load preprocessed single-category ShapeNet SDF grids for VQ-VAE and diffusion training."""

    def __init__(
        self,
        data_root: str | Path,
        category: str = "chair",
        split: str = "train",
        res: int = 64,
        max_samples: Optional[int] = None,
        filelist: Optional[str | Path | Iterable[str]] = None,
        trunc_thres: float = 0.0,
        split_file_root: Optional[str | Path] = None,
    ) -> None:
        """Resolve category paths, split/filelist choices, and available SDF sample files."""
        if category == "all":
            raise NotImplementedError("Multi-class ShapeNet loading is reserved for a later stage.")
        if category not in DEFAULT_CATEGORY_IDS:
            known = ", ".join(sorted(DEFAULT_CATEGORY_IDS))
            raise ValueError(f"Unknown ShapeNet category '{category}'. Known categories: {known}")

        self.data_root = Path(data_root)
        self.category = category
        self.cat_id = DEFAULT_CATEGORY_IDS[category]
        self.split = split
        self.res = int(res)
        self.max_samples = int(max_samples) if max_samples is not None else None
        self.trunc_thres = float(trunc_thres)
        self.split_file_root = Path(split_file_root) if split_file_root is not None else None
        self.base_dir = self.data_root / "ShapeNet" / "SDF_v1" / f"resolution_{self.res}" / self.cat_id
        self.total_candidates: int | None = None
        self.samples = self._build_samples(filelist)
        if not self.samples:
            expected = self.base_dir / "<model_id>" / "ori_sample_grid.h5"
            raise FileNotFoundError(
                "No ShapeNet SDF samples found. Expected files like "
                f"{expected}. Provide --filelist or create the preprocessed SDF tree."
            )

    def _build_samples(self, filelist: Optional[str | Path | Iterable[str]]) -> list[Path]:
        """Build concrete SDF file paths from an explicit filelist, split file, or preprocessed tree."""
        model_ids = self._read_filelist(filelist)
        if model_ids is None:
            candidates = [
                Path("dataset_info_files") / "ShapeNet_filelists" / f"{self.cat_id}_{self.split}.lst",
                Path(__file__).resolve().parents[2]
                / "SDFusion-master"
                / "dataset_info_files"
                / "ShapeNet_filelists"
                / f"{self.cat_id}_{self.split}.lst",
            ]
            if self.split_file_root is not None:
                candidates.insert(0, self.split_file_root / f"{self.cat_id}_{self.split}.lst")
            for legacy in candidates:
                if legacy.exists():
                    model_ids = self._read_filelist(legacy)
                    break
        if model_ids is None:
            if not self.base_dir.exists():
                return []
            samples: list[Path] = []
            for model_dir in self.base_dir.iterdir():
                path = model_dir / "ori_sample_grid.h5"
                if path.exists():
                    samples.append(path)
                    if self.max_samples is not None and len(samples) >= self.max_samples:
                        break
            self.total_candidates = None
            return sorted(samples)

        self.total_candidates = len(model_ids)
        if self.max_samples is not None:
            model_ids = model_ids[: self.max_samples]
        samples: list[Path] = []
        missing: list[Path] = []
        for item in model_ids:
            path = Path(item)
            if path.suffix not in {".h5", ".npy", ".npz"}:
                path = self.base_dir / item / "ori_sample_grid.h5"
            if path.exists():
                samples.append(path)
            else:
                missing.append(path)
        if not samples and missing:
            raise FileNotFoundError(
                "Filelist was read, but none of the referenced SDF files exist. "
                f"First expected path: {missing[0]}"
            )
        return samples

    def _read_filelist(self, filelist: Optional[str | Path | Iterable[str]]) -> Optional[list[str]]:
        """Read model ids or SDF paths from a text file/list, returning None when no filelist is configured."""
        if filelist is None:
            return None
        if isinstance(filelist, (str, Path)):
            filelist_path = Path(filelist)
            if not filelist_path.is_absolute() and not filelist_path.exists():
                data_relative = self.data_root / filelist_path
                if data_relative.exists():
                    filelist_path = data_relative
            if not filelist_path.exists():
                raise FileNotFoundError(f"ShapeNet filelist not found: {filelist_path}")
            with filelist_path.open("r", encoding="utf-8") as handle:
                return [line.strip() for line in handle if line.strip()]
        return [str(item).strip() for item in filelist if str(item).strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        """Load one SDF grid as a [1, res, res, res] tensor plus ShapeNet metadata."""
        path = self.samples[index]
        if path.suffix == ".h5":
            with h5py.File(path, "r") as h5_file:
                if "pc_sdf_sample" in h5_file:
                    sdf = np.asarray(h5_file["pc_sdf_sample"], dtype=np.float32)
                elif "sdf" in h5_file:
                    sdf = np.asarray(h5_file["sdf"], dtype=np.float32)
                else:
                    raise KeyError(f"Missing dataset 'pc_sdf_sample' or 'sdf' in {path}")
        elif path.suffix == ".npz":
            npz = np.load(path)
            key = "sdf" if "sdf" in npz else npz.files[0]
            sdf = np.asarray(npz[key], dtype=np.float32)
        else:
            sdf = np.asarray(np.load(path), dtype=np.float32)
        sdf = sdf.reshape(1, self.res, self.res, self.res)
        tensor = torch.from_numpy(sdf)
        if self.trunc_thres != 0.0:
            tensor = torch.clamp(tensor, min=-self.trunc_thres, max=self.trunc_thres)
        return {
            "sdf": tensor,
            "cat_id": self.cat_id,
            "cat_str": self.category,
            "model_id": path.parent.name,
            "path": str(path),
        }
