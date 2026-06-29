from __future__ import annotations

import torch

from .base import BaseConditioner


class PartialShapeConditioner(BaseConditioner):
    condition_type = "partial_sdf"

    def encode(self, batch: dict) -> dict:
        partial = batch.get("partial_sdf")
        if partial is None:
            sdf = batch["sdf"]
            partial = torch.zeros_like(sdf)
            partial[..., : sdf.shape[-3] // 2, :, :] = sdf[..., : sdf.shape[-3] // 2, :, :]
        return {"c_concat": [partial]}
