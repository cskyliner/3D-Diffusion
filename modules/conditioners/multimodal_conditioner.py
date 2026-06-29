from __future__ import annotations

from .base import BaseConditioner


class MultiModalConditioner(BaseConditioner):
    condition_type = "multimodal"

    def encode(self, batch: dict) -> dict:
        raise NotImplementedError("Multimodal interface reserved for stage 2.")
