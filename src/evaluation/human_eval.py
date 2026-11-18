"""Human evaluation export / ingestion (coherence + helpfulness, 3 annotators)."""
from __future__ import annotations

import csv
import os
import random
from typing import Dict, List, Sequence

from ..config import ensure_dir, project_path
from ..data.loaders import RecordDataset, YelpDataset, collate
from ..data.schema import read_jsonl
from ..models.prompting import build_target, score_target, step3_segment
from .metrics import mean_std
from .recommendation import _collect
from ..annotation.quality import fleiss_kappa

ANNOTATORS = 3
RATING_COLUMNS = ["coherence", "helpfulness"]


def export_human_eval(model, cfg: Dict, path: str = None) -> Dict:
    directory = project_path(cfg, "yelp_dir")
    records = read_jsonl(os.path.join(directory, "businesses.jsonl"))
    test_records = [row for row in records if row.get("split") == "test"] or records
    limit = min(int(cfg["eval"].get("human_sample", 60)), len(test_records))
    rng = random.Random(cfg.get("seed", 42))
    sample = rng.sample(test_records, limit)

    dataset = RecordDataset(cfg, sample)
    loader = torch_loader(cfg, dataset)
    rows = []
    for batch in loader:
        texts, scores = model.generate(batch, max_new_tokens=int(cfg["decoder"]["max_new_tokens"]))
        for index, text in enumerate(texts):
            row = sample[len(rows)] if len(rows) < len(sample) else sample[-1]
            rows.append(
                {
                    "sample_id": batch["sample_id"][index],
                    "user_query": batch["queries"][index],
                    "item_text": batch["texts"][index],
                    "ground_truth_type": row.get("type", "null"),
                    "ground_truth_polarity": row.get("polarity", 0),
                    "generated_rationale": step3_segment(text),
                    "generated_score": scores[index],
                    "generated_text": text,
                    "reference_target": build_target(
                        row.get("type", "null"), int(row.get("polarity", 0)),
                        score_target(row["sample_id"], int(row.get("label", 0))),
                    ),
                }
            )
            for annotator in range(1, ANNOTATORS + 1):
                for column in RATING_COLUMNS:
                    rows[-1]["annotator_{}_{}".format(annotator, column)] = ""

    path = path or os.path.join(project_path(cfg, "output_dir"), "human_eval", "annotation_sheet.csv")
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return {"path": path, "samples": len(rows), "status": "pending_annotators"}


def torch_loader(cfg: Dict, dataset):
    from torch.utils.data import DataLoader

    return DataLoader(dataset, batch_size=int(cfg["eval"]["batch_size"]), shuffle=False, collate_fn=collate)


def ingest_human_eval(path: str) -> Dict:
    with open(path, "r", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"status": "empty"}

    result = {"status": "pending_annotators", "samples": len(rows)}
    for column in RATING_COLUMNS:
        per_annotator = []
        for annotator in range(1, ANNOTATORS + 1):
            key = "annotator_{}_{}".format(annotator, column)
            values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
            per_annotator.append(values)
        if all(len(values) == len(rows) for values in per_annotator):
            averages = [sum(values) / len(values) for values in per_annotator]
            result[column] = {
                "annotator_means": [round(value, 4) for value in averages],
                "mean": round(sum(averages) / len(averages), 4),
                "std": mean_std(averages)["std"],
            }

    complete = all(
        row.get("annotator_{}_coherence".format(annotator)) not in (None, "")
        for row in rows for annotator in range(1, ANNOTATORS + 1)
    )
    if complete:
        matrix = []
        for row in rows:
            counts = [0] * 5
            for annotator in range(1, ANNOTATORS + 1):
                value = int(float(row["annotator_{}_coherence".format(annotator)]))
                if 1 <= value <= 5:
                    counts[value - 1] += 1
            matrix.append(counts)
        result["coherence_fleiss_kappa"] = round(fleiss_kappa(matrix), 4)
        result["status"] = "scored"
    return result
