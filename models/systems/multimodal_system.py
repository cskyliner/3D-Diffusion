from __future__ import annotations

from sdfusion.modules.conditioners.multimodal_conditioner import MultiModalConditioner

from .base_system import BaseSDFusionSystem


class MultiModal2ShapeSystem(BaseSDFusionSystem):
    def __init__(self, *args, conditioner=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conditioner = conditioner or MultiModalConditioner()

    def get_condition(self, batch: dict) -> dict:
        return self.conditioner.encode(batch)
