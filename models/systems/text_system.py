from __future__ import annotations

from modules.conditioners.text_conditioner import TextConditioner

from .base_system import BaseSDFusionSystem


class Text2ShapeSystem(BaseSDFusionSystem):
    def __init__(self, *args, conditioner=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conditioner = conditioner or TextConditioner()

    def get_condition(self, batch: dict) -> dict:
        return self.conditioner.encode(batch)
