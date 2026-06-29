from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ConditionBatch:
    text: Optional[list[str]] = None
    image: Optional[torch.Tensor] = None
    partial_sdf: Optional[torch.Tensor] = None
    category: Optional[list[str]] = None
