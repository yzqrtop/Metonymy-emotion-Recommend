"""Ranking transfer on Taobao-MM item embeddings (Table 4)."""
from __future__ import annotations

import random
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.loaders import load_taobao
from .metrics import hr_at_k, ndcg_at_k


class TransferAdapter(nn.Module):
    """Content adapter: frozen Taobao-MM embedding -> ranking factors."""

    def __init__(self, embed_dim: int, n_users: int, factors: int = 32):
        super().__init__()
        self.projection = nn.Linear(embed_dim, factors)
        self.user = nn.Embedding(n_users, factors)
        nn.init.normal_(self.user.weight, std=0.05)

    def item_vectors(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(embeddings), dim=-1)


def evaluate_transfer(cfg: Dict, seed: int = 0) -> Dict:
    bundle = load_taobao(cfg)
    embeddings = torch.from_numpy(bundle["embeddings"])
    interactions = bundle["interactions"]
    n_items = embeddings.shape[0]
    n_users = len(interactions)
    factors = int(cfg["baseline"].get("bpr_factors", 32))

    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = TransferAdapter(embeddings.shape[1], n_users, factors)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["baseline"].get("bpr_lr", 0.05)))

    pairs = [(index, item) for index, row in enumerate(interactions) for item in row["train_items"]]
    epochs = int(cfg["baseline"].get("epochs", 6))
    for _ in range(epochs):
        rng.shuffle(pairs)
        vectors = model.item_vectors(embeddings)
        for user, positive in pairs:
            negative = rng.randrange(n_items)
            score = torch.dot(model.user(torch.tensor(user)), vectors[positive] - vectors[negative])
            loss = -F.logsigmoid(score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                vectors = model.item_vectors(embeddings)

    with torch.no_grad():
        vectors = model.item_vectors(embeddings)
    limit = int(cfg["eval"].get("transfer_users", 200))
    hr5, hr10, ndcg5, ndcg10 = [], [], [], []
    for index, row in enumerate(interactions[:limit]):
        positives = row["test_items"]
        if not positives:
            continue
        negatives = rng.sample([item for item in range(n_items) if item not in set(row["train_items"])],
                               k=min(99, n_items - len(row["train_items"])))
        candidates = list(positives) + negatives
        user_vector = model.user(torch.tensor(index)).detach()
        ranked = sorted(candidates, key=lambda item: -float(torch.dot(user_vector, vectors[item])))
        relevance = [1 if item in set(positives) else 0 for item in ranked]
        hr5.append(hr_at_k(relevance, 5))
        hr10.append(hr_at_k(relevance, 10))
        ndcg5.append(ndcg_at_k(relevance, 5))
        ndcg10.append(ndcg_at_k(relevance, 10))

    return {
        "users": len(hr5),
        "HR@5": round(float(np.mean(hr5)) if hr5 else 0.0, 4),
        "HR@10": round(float(np.mean(hr10)) if hr10 else 0.0, 4),
        "NDCG@5": round(float(np.mean(ndcg5)) if ndcg5 else 0.0, 4),
        "NDCG@10": round(float(np.mean(ndcg10)) if ndcg10 else 0.0, 4),
    }
