from __future__ import annotations

from .base import BaseConditioner


class TextConditioner(BaseConditioner):
    condition_type = "text"

    def encode(self, batch: dict) -> dict:
        raise NotImplementedError("Text-to-shape interface reserved for stage 2.")
