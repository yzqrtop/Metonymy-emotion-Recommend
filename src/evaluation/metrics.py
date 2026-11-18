"""Metrics implemented with NumPy only (no scikit-learn dependency)."""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np


def _rankdata(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or sorted_values[index] != sorted_values[start]:
            average = (start + index - 1) / 2.0 + 1.0
            ranks[order[start:index]] = average
            start = index
    return ranks


def auc_score(labels: Sequence[float], scores: Sequence[float]) -> float:
    labels = np.asarray(labels, dtype=np.float32)
    positive = float(labels.sum())
    negative = float(len(labels) - positive)
    if positive == 0 or negative == 0:
        return 0.5
    ranks = _rankdata(scores)
    return float((ranks[labels == 1].sum() - positive * (positive + 1) / 2.0) / (positive * negative))


def accuracy(labels: Sequence[int], predictions: Sequence[int]) -> float:
    if not len(labels):
        return 0.0
    return float(np.mean(np.asarray(labels) == np.asarray(predictions)))


def f1_positive(labels: Sequence[int], predictions: Sequence[int]) -> float:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    true_positive = float(((labels == 1) & (predictions == 1)).sum())
    false_positive = float(((labels == 0) & (predictions == 1)).sum())
    false_negative = float(((labels == 1) & (predictions == 0)).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return float(2 * true_positive / denominator) if denominator else 0.0


def precision_at_k(ranking: Sequence[int], k: int) -> float:
    top = list(ranking)[:k]
    return float(sum(top) / max(1, len(top)))


def ndcg_at_k(ranking: Sequence[int], k: int) -> float:
    top = list(ranking)[:k]
    dcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(top))
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(k, sum(1 for r in ranking if r))))
    return float(dcg / ideal) if ideal > 0 else 0.0


def hr_at_k(ranking: Sequence[int], k: int) -> float:
    return 1.0 if any(ranking[:k]) else 0.0


def bleu4(hypothesis: str, reference: str) -> float:
    """4-gram BLEU with add-1 smoothing (sufficient for the paper's table)."""
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    if not hyp or not ref:
        return 0.0
    precisions = []
    for order in range(1, 5):
        hyp_grams = Counter(tuple(hyp[i:i + order]) for i in range(len(hyp) - order + 1))
        ref_grams = Counter(tuple(ref[i:i + order]) for i in range(len(ref) - order + 1))
        if not hyp_grams:
            precisions.append(0.0)
            continue
        matches = sum(min(count, ref_grams[gram]) for gram, count in hyp_grams.items())
        precisions.append((matches + 1.0) / (sum(hyp_grams.values()) + 1.0))
    if min(precisions) <= 0:
        return 0.0
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / max(1, len(hyp)))
    return float(bp * math.exp(sum(math.log(value) for value in precisions) / 4.0))


def rouge_l(hypothesis: str, reference: str) -> float:
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    if not hyp or not ref:
        return 0.0
    table = [[0] * (len(ref) + 1) for _ in range(len(hyp) + 1)]
    for i in range(1, len(hyp) + 1):
        for j in range(1, len(ref) + 1):
            if hyp[i - 1] == ref[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    lcs = table[-1][-1]
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def macro_f1_per_class(labels: Sequence[int], predictions: Sequence[int], classes: Sequence[int]) -> Dict[str, float]:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    scores = {}
    for label in classes:
        true_positive = float(((labels == label) & (predictions == label)).sum())
        false_positive = float(((labels != label) & (predictions == label)).sum())
        false_negative = float(((labels == label) & (predictions != label)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores[str(label)] = round(2 * true_positive / denominator, 4) if denominator else 0.0
    return scores


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {"mean": round(float(array.mean()), 4), "std": round(float(array.std(ddof=1)) if len(array) > 1 else 0.0, 4)}
