"""Dataset schema shared by all three experimental datasets."""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Sequence, Tuple

EMOTION_TYPES = ["T1", "T2", "T3", "null"]
TYPE_TO_ID = {name: i for i, name in enumerate(EMOTION_TYPES)}
ID_TO_TYPE = {i: name for name, i in TYPE_TO_ID.items()}

# polarity order follows the paper's negative / neutral / positive reporting order
POLARITIES = [-1, 0, 1]
POLARITY_TO_ID = {-1: 0, 0: 1, 1: 2}
ID_TO_POLARITY = {0: -1, 1: 0, 2: 1}

EMOTION_TO_POLARITY = {
    "joy": 1,
    "love": 1,
    "trust": 1,
    "anticipation": 1,
    "neutral": 0,
    "fear": -1,
    "sadness": -1,
    "disgust": -1,
    "anger": -1,
    "surprise": -1,
}


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def read_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two ``[x0, y0, x1, y1]`` boxes in normalised coordinates."""
    if not a or not b:
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def group_split(
    records: Sequence[Dict],
    group_key: str = "business_id",
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Split *groups* (not rows) into train/val/test to prevent leakage."""
    import random

    groups = sorted({row[group_key] for row in records})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_train = int(len(groups) * ratios[0])
    n_val = int(len(groups) * ratios[1])
    return {
        "train": groups[:n_train],
        "val": groups[n_train:n_train + n_val],
        "test": groups[n_train + n_val:],
    }


def split_records(records: Sequence[Dict], splits: Dict[str, List[str]], group_key: str = "business_id") -> Dict[str, List[Dict]]:
    membership = {key: set(ids) for key, ids in splits.items()}
    out: Dict[str, List[Dict]] = {key: [] for key in splits}
    for row in records:
        group = row[group_key]
        for split, ids in membership.items():
            if group in ids:
                out[split].append(row)
                break
    return out


def dataset_statistics(records: Sequence[Dict]) -> Dict:
    stats = {"total": len(records), "type_counts": {}, "polarity_counts": {}, "positive_rate": 0.0}
    for name in EMOTION_TYPES:
        stats["type_counts"][name] = sum(1 for row in records if row.get("type") == name)
    for pol in POLARITIES:
        stats["polarity_counts"][str(pol)] = sum(1 for row in records if int(row.get("polarity", 0)) == pol)
    if records:
        stats["positive_rate"] = round(sum(1 for row in records if row.get("label", 0) == 1) / len(records), 4)
    return stats
