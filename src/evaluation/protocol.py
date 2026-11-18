"""Negative sampling protocol: BPR-MF hard negatives (1 positive + 19 negatives)."""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


class BPRMF:
    """NumPy BPR-MF used to mine hard negatives and as a graph baseline."""

    def __init__(self, n_users: int, n_items: int, factors: int = 32, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.user = rng.normal(scale=0.05, size=(n_users, factors)).astype(np.float32)
        self.item = rng.normal(scale=0.05, size=(n_items, factors)).astype(np.float32)
        self.lr = 0.05
        self.reg = 0.01

    def fit(self, pairs: Sequence[Tuple[int, int]], n_items: int, epochs: int = 5, seed: int = 42) -> "BPRMF":
        rng = random.Random(seed)
        pairs = list(pairs)
        for _ in range(epochs):
            rng.shuffle(pairs)
            for user, positive in pairs:
                negative = rng.randrange(n_items)
                u = self.user[user]
                p = self.item[positive]
                n = self.item[negative]
                score = float(u @ (p - n))
                gradient = 1.0 / (1.0 + np.exp(score))
                self.user[user] += self.lr * (gradient * (p - n) - self.reg * u)
                self.item[positive] += self.lr * (gradient * u - self.reg * p)
                self.item[negative] += self.lr * (-gradient * u - self.reg * n)
        return self

    def scores(self, user: int) -> np.ndarray:
        return self.item @ self.user[user]


def sample_candidates(
    user_positives: Sequence[int],
    pool: Sequence[int],
    negatives: int = 19,
    hard_order: Sequence[int] = None,
    top_k: int = 200,
    seed: int = 0,
) -> List[int]:
    """Return indices (into ``pool``) of 1 positive + ``negatives`` negatives."""
    rng = random.Random(seed)
    positive_set = set(user_positives)
    available = [index for index, item in enumerate(pool) if item not in positive_set]
    if not available:
        return []
    ranked = available
    if hard_order is not None:
        position = {item: rank for rank, item in enumerate(hard_order)}
        ranked = sorted(available, key=lambda idx: position.get(pool[idx], len(position)))[:top_k]
    chosen = ranked[: min(len(ranked), negatives)]
    positives = [index for index, item in enumerate(pool) if item in positive_set]
    if positives:
        chosen = [rng.choice(positives)] + chosen
    return chosen
