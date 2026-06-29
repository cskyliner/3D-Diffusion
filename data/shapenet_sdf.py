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
    def __init__(
        self,
        data_root: str | Path,
        category: str = "chair",
        split: str = "train",
        res: int = 64,
        max_samples: Optional[int] = None,
        filelist: Optional[str | Path | Iterable[str]] = None,
        trunc_thres: float = 0.0,
    ) -> None:
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
        self.trunc_thres = float(trunc_thres)
        self.base_dir = self.data_root / "ShapeNet" / "SDF_v1" / f"resolution_{self.res}" / self.cat_id
        self.samples = self._build_samples(filelist)
        if max_samples is not None:
            self.samples = self.samples[: int(max_samples)]
        if not self.samples:
            expected = self.base_dir / "<model_id>" / "ori_sample_grid.h5"
            raise FileNotFoundError(
                "No ShapeNet SDF samples found. Expected files like "
                f"{expected}. Provide --filelist or create the preprocessed SDF tree."
            )

    def _build_samples(self, filelist: Optional[str | Path | Iterable[str]]) -> list[Path]:
        model_ids = self._read_filelist(filelist)
        if model_ids is None:
            legacy = Path("dataset_info_files") / "ShapeNet_filelists" / f"{self.cat_id}_{self.split}.lst"
            if legacy.exists():
                model_ids = self._read_filelist(legacy)
        if model_ids is None:
            if not self.base_dir.exists():
                return []
            return sorted(self.base_dir.glob("*/ori_sample_grid.h5"))

        samples: list[Path] = []
        missing: list[Path] = []
        for item in model_ids:
            path = Path(item)
            if not path.suffix == ".h5":
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
        if filelist is None:
            return None
        if isinstance(filelist, (str, Path)):
            filelist_path = Path(filelist)
            if not filelist_path.exists():
                raise FileNotFoundError(f"ShapeNet filelist not found: {filelist_path}")
            with filelist_path.open("r", encoding="utf-8") as handle:
                return [line.strip() for line in handle if line.strip()]
        return [str(item).strip() for item in filelist if str(item).strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.samples[index]
        with h5py.File(path, "r") as h5_file:
            if "pc_sdf_sample" not in h5_file:
                raise KeyError(f"Missing dataset 'pc_sdf_sample' in {path}")
            sdf = np.asarray(h5_file["pc_sdf_sample"], dtype=np.float32).reshape(1, self.res, self.res, self.res)
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
