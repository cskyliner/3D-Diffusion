from __future__ import annotations

from .base import BaseConditioner


class ImageConditioner(BaseConditioner):
    condition_type = "image"

    def encode(self, batch: dict) -> dict:
        raise NotImplementedError("Image-to-shape interface reserved for stage 2.")
