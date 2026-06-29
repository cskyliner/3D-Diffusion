from __future__ import annotations

from .base import BaseConditioner


class PartialShapeConditioner(BaseConditioner):
    condition_type = "partial_sdf"

    def encode(self, batch: dict) -> dict:
        raise NotImplementedError("Partial completion interface reserved for stage 2.")
