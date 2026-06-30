from __future__ import annotations

import torch
from torch import nn

from .openai_unet3d import UNet3DModel


def _first_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class DiffusionUNet(nn.Module):
    """SDFusion/LDM-compatible wrapper around the OpenAI-style 3D UNet.

    The nested ``diffusion_net`` attribute intentionally matches the original
    SDFusion state_dict layout, e.g. ``diffusion_net.input_blocks.0.0.weight``.
    """

    def __init__(self, unet_params: dict, vq_conf=None, conditioning_key: str | None = None) -> None:
        super().__init__()
        del vq_conf
        self.diffusion_net = UNet3DModel(**unet_params)
        self.conditioning_key = conditioning_key

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: object = None,
        c_concat: list[torch.Tensor] | None = None,
        c_crossattn: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if isinstance(cond, dict):
            c_concat = cond.get("c_concat", c_concat)
            c_crossattn = cond.get("c_crossattn", c_crossattn)

        if self.conditioning_key is None:
            return self.diffusion_net(x, t)
        if self.conditioning_key == "concat":
            concat = _first_list(c_concat) or []
            return self.diffusion_net(torch.cat([x] + concat, dim=1), t)
        if self.conditioning_key == "crossattn":
            cross = _first_list(c_crossattn) or []
            context = torch.cat(cross, dim=1) if cross else None
            return self.diffusion_net(x, t, context=context)
        if self.conditioning_key == "hybrid":
            concat = _first_list(c_concat) or []
            cross = _first_list(c_crossattn) or []
            xc = torch.cat([x] + concat, dim=1)
            context = torch.cat(cross, dim=1) if cross else None
            return self.diffusion_net(xc, t, context=context)
        if self.conditioning_key == "adm":
            cross = _first_list(c_crossattn) or []
            if not cross:
                raise ValueError("ADM conditioning requires c_crossattn/y labels.")
            return self.diffusion_net(x, t, y=cross[0])
        raise NotImplementedError(f"Unknown conditioning_key: {self.conditioning_key}")
