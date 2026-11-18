"""Dual-path attention (cross + self) with per-patch gating and top-k localisation."""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPathLocalizer(nn.Module):
    """Eq. (4)-(7): cross-attention, self-attention and gated fusion."""

    def __init__(self, text_dim: int = 512, vision_dim: int = 768, attn_dim: int = 64, heads: int = 4, patch_grid: int = 14):
        super().__init__()
        if attn_dim % heads != 0:
            raise ValueError("attn_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = attn_dim // heads
        self.grid = patch_grid
        self.num_patches = patch_grid * patch_grid
        self.query = nn.Linear(text_dim, attn_dim)
        self.key = nn.Linear(vision_dim, attn_dim)
        self.gate = nn.Linear(2, 1)

    def forward(
        self, f_t: torch.Tensor, patches: torch.Tensor, fusion: str = "gate", top_k: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[List[float]]]:
        batch, num_patches, _ = patches.shape
        query = self.query(f_t).view(batch, self.heads, 1, self.head_dim)
        key = self.key(patches).view(batch, self.heads, num_patches, self.head_dim)
        scale = math.sqrt(self.head_dim)

        cross_scores = torch.matmul(query, key.transpose(-1, -2)) / scale        # B,H,1,N
        cross = torch.softmax(cross_scores, dim=-1).mean(dim=1).squeeze(1)       # B,N
        self_scores = torch.matmul(key, key.transpose(-1, -2)) / scale           # B,H,N,N
        self_attn = torch.softmax(self_scores, dim=-1).mean(dim=-2).mean(dim=1)  # B,N

        if fusion == "mean":
            fused = torch.full_like(cross, 1.0 / num_patches)
            gate_values = torch.zeros_like(cross)
        elif fusion == "fixed":
            fused = 0.5 * cross + 0.5 * self_attn
            gate_values = torch.full_like(cross, 0.5)
        else:
            gate_values = torch.sigmoid(self.gate(torch.stack([cross, self_attn], dim=-1))).squeeze(-1)
            fused = gate_values * cross + (1.0 - gate_values) * self_attn

        k = min(top_k, num_patches)
        top_indices = torch.topk(fused, k=k, dim=-1).indices
        regions: List[List[float]] = []
        for row in top_indices.tolist():
            rows = [idx // self.grid for idx in row]
            cols = [idx % self.grid for idx in row]
            regions.append(
                [
                    round(min(cols) / float(self.grid), 4),
                    round(min(rows) / float(self.grid), 4),
                    round((max(cols) + 1) / float(self.grid), 4),
                    round((max(rows) + 1) / float(self.grid), 4),
                ]
            )

        # weighted pooling over all patches (attention weights sum to one)
        f_v = torch.bmm(fused.unsqueeze(1), patches).squeeze(1)                   # B,768
        return fused, cross, self_attn, gate_values, f_v, regions


def masked_pool(patches: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gather = indices.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
    selected = torch.gather(patches, 1, gather)
    return selected.mean(dim=1)
