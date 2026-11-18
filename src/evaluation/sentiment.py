"""EmoMeta implicit-emotion / polarity evaluation (Table 3)."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.loaders import EmoMetaDataset, collate
from ..data.schema import POLARITIES
from .metrics import accuracy, macro_f1_per_class
from .recommendation import _collect
from ..annotation.quality import cohen_kappa


def evaluate_sentiment(model, cfg: Dict, limit: int = 400) -> Dict:
    dataset = EmoMetaDataset(cfg, split="test", limit=limit)
    if not len(dataset):
        return {"error": "empty EmoMeta split"}
    records = dataset.records
    collected = _collect(model, cfg, records)
    truth = [int(row.get("polarity", 0)) for row in records]
    predicted = collected["polarity_predictions"]
    type_truth = [row.get("type", "null") for row in records]

    result = {
        "samples": len(records),
        "Acc": round(accuracy(truth, predicted), 4),
        "per_class_F1": macro_f1_per_class(truth, predicted, POLARITIES),
        "Kappa": round(cohen_kappa([str(value) for value in truth], [str(value) for value in predicted],
                                   labels=["-1", "0", "1"]), 4),
    }
    from ..data.schema import TYPE_TO_ID

    if -1 not in collected["type_predictions"]:
        result["type_accuracy"] = round(
            accuracy([TYPE_TO_ID.get(name, 3) for name in type_truth], collected["type_predictions"]), 4
        )
    return result
