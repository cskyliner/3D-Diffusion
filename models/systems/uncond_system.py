from __future__ import annotations

from .base_system import BaseSDFusionSystem


class UncondSDFusionSystem(BaseSDFusionSystem):
    def get_condition(self, batch: dict) -> None:
        return None
