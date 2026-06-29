from __future__ import annotations

from .image_conditioner import ImageConditioner
from .base import BaseConditioner
from .partial_shape_conditioner import PartialShapeConditioner
from .text_conditioner import TextConditioner


class MultiModalConditioner(BaseConditioner):
    condition_type = "multimodal"

    def __init__(self, context_dim: int = 128) -> None:
        super().__init__()
        self.text = TextConditioner(context_dim=context_dim)
        self.image = ImageConditioner(context_dim=context_dim)
        self.partial = PartialShapeConditioner()

    def encode(self, batch: dict) -> dict:
        output: dict[str, list] = {"c_crossattn": [], "c_concat": []}
        if any(key in batch for key in ("text", "caption", "category", "cat_str")):
            output["c_crossattn"].extend(self.text.encode(batch).get("c_crossattn", []))
        if "image" in batch:
            output["c_crossattn"].extend(self.image.encode(batch).get("c_crossattn", []))
        if "partial_sdf" in batch:
            output["c_concat"].extend(self.partial.encode(batch).get("c_concat", []))
        return {key: value for key, value in output.items() if value}
