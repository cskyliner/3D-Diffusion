from __future__ import annotations

from .base import BaseConditioner


class NullConditioner(BaseConditioner):
    condition_type = "none"

    def encode(self, batch: dict) -> dict:
        return {}
