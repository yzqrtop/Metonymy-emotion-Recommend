"""Annotation quality metrics: agreement, IoU, keyword hit rate, failure rate."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..data.schema import box_iou


def cohen_kappa(a: Sequence, b: Sequence, labels: Sequence = None) -> float:
    """Cohen's kappa for two annotators."""
    if len(a) != len(b) or not len(a):
        return 0.0
    if labels is None:
        labels = sorted(set(a) | set(b))
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for x, y in zip(a, b):
        matrix[index[x], index[y]] += 1
    total = matrix.sum()
    if total == 0:
        return 0.0
    observed = np.trace(matrix) / total
    row = matrix.sum(axis=1) / total
    col = matrix.sum(axis=0) / total
    expected = float((row * col).sum())
    if abs(1.0 - expected) < 1e-12:
        return 0.0
    return float((observed - expected) / (1.0 - expected))


def fleiss_kappa(matrix: Sequence[Sequence[int]]) -> float:
    """Fleiss' kappa; ``matrix[i][j]`` = number of annotators assigning item i to category j."""
    table = np.asarray(matrix, dtype=np.float64)
    if table.size == 0:
        return 0.0
    n_items = table.shape[0]
    n_raters = table.sum(axis=1)
    if np.any(n_raters == 0):
        return 0.0
    p_j = table.sum(axis=0) / table.sum()
    p_i = ((table ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1) + 1e-12)
    p_bar = float(p_i.mean())
    p_e = float((p_j ** 2).sum())
    if abs(1.0 - p_e) < 1e-12:
        return 0.0
    return float((p_bar - p_e) / (1.0 - p_e))


def localization_report(
    predictions: Sequence[Dict], ground_truth: Sequence[Dict], iou_threshold: float = 0.3
) -> Dict:
    """IoU / keyword-hit / failure statistics for the localisation branch."""
    truth = {row["sample_id"]: row for row in ground_truth}
    ious: List[float] = []
    hits = 0
    failures = 0
    keyword_evaluated = 0
    keyword_hit_rate = []
    for pred in predictions:
        gt = truth.get(pred["sample_id"])
        if not gt:
            continue
        iou = box_iou(pred.get("region") or [], gt.get("region") or [])
        pred_keywords = {word.lower() for word in pred.get("keywords", [])}
        gt_keywords = {word.lower() for word in (gt.get("keywords") or gt.get("keywords_text") or [])}
        hit = bool(pred_keywords & gt_keywords)
        has_gt_keywords = bool(gt_keywords)
        ious.append(iou)
        if has_gt_keywords:
            keyword_evaluated += 1
            hits += int(hit)
            keyword_hit_rate.append(1.0 if hit else 0.0)
        if iou < iou_threshold or (has_gt_keywords and not hit):
            failures += 1
    n = max(1, len(ious))
    return {
        "iou_mean": round(float(np.mean(ious)) if ious else 0.0, 4),
        "iou_median": round(float(np.median(ious)) if ious else 0.0, 4),
        "keyword_hit_rate": round(hits / keyword_evaluated, 4) if keyword_evaluated else None,
        "localization_failure_rate": round(failures / n, 4),
        "evaluated": len(ious),
        "iou_threshold": iou_threshold,
    }


def confusion(predictions: Sequence[Dict], ground_truth: Sequence[Dict], key: str, labels: Sequence[str]) -> Dict:
    truth = {row["sample_id"]: row for row in ground_truth}
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for row in predictions:
        gt = truth.get(row["sample_id"])
        if not gt:
            continue
        predicted = str(row.get(key, "null"))
        actual = str(gt.get(key, "null"))
        if predicted not in matrix or actual not in matrix[actual]:
            continue
        matrix[actual][predicted] += 1
    return matrix
