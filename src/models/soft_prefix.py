"""Soft prefix injection (Eq. 11): p_i = W_p v_i."""
from __future__ import annotations

import torch
import torch.nn as nn


class SoftPrefix(nn.Module):
    """Project the item vector to one soft-prefix embedding of the decoder."""

    def __init__(self, item_dim: int = 128, hidden_dim: int = 2048):
        super().__init__()
        self.projection = nn.Linear(item_dim, hidden_dim)

    def forward(self, item_vectors: torch.Tensor) -> torch.Tensor:
        return self.projection(item_vectors).unsqueeze(1)
