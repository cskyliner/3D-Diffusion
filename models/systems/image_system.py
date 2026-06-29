from __future__ import annotations

from modules.conditioners.image_conditioner import ImageConditioner

from .base_system import BaseSDFusionSystem


class Image2ShapeSystem(BaseSDFusionSystem):
    def __init__(self, *args, conditioner=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conditioner = conditioner or ImageConditioner()

    def get_condition(self, batch: dict) -> dict:
        return self.conditioner.encode(batch)
