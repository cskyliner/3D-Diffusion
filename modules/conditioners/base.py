from __future__ import annotations

from torch import nn


class BaseConditioner(nn.Module):
    condition_type: str = "base"

    def encode(self, batch: dict) -> dict:
        raise NotImplementedError
