from __future__ import annotations

from modules.conditioners.partial_shape_conditioner import PartialShapeConditioner

from .base_system import BaseSDFusionSystem


class CompletionSystem(BaseSDFusionSystem):
    def __init__(self, *args, conditioner=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conditioner = conditioner or PartialShapeConditioner()

    def get_condition(self, batch: dict) -> dict:
        return self.conditioner.encode(batch)
