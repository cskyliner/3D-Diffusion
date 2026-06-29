from __future__ import annotations

import hashlib

import torch
from torch import nn

from .base import BaseConditioner


class TextConditioner(BaseConditioner):
    condition_type = "text"

    def __init__(self, vocab_size: int = 4096, context_dim: int = 128, max_tokens: int = 32) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.context_dim = int(context_dim)
        self.max_tokens = int(max_tokens)
        self.embedding = nn.Embedding(self.vocab_size, self.context_dim)
        self.proj = nn.Sequential(nn.LayerNorm(self.context_dim), nn.Linear(self.context_dim, self.context_dim))

    def _tokenize(self, texts: list[str], device: torch.device) -> torch.Tensor:
        ids = torch.zeros((len(texts), self.max_tokens), dtype=torch.long, device=device)
        for row, text in enumerate(texts):
            tokens = str(text).lower().split()[: self.max_tokens]
            for col, token in enumerate(tokens):
                digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
                ids[row, col] = int(digest[:8], 16) % self.vocab_size
        return ids

    def encode(self, batch: dict) -> dict:
        texts = batch.get("text") or batch.get("caption") or batch.get("category") or batch.get("cat_str")
        if texts is None:
            batch_size = int(batch["sdf"].shape[0])
            texts = [""] * batch_size
        if isinstance(texts, str):
            texts = [texts]
        device = next(self.parameters()).device
        token_ids = self._tokenize([str(item) for item in texts], device)
        context = self.proj(self.embedding(token_ids))
        return {"c_crossattn": [context]}
