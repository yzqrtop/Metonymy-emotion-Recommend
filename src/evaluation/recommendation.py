"""Business-level, user-level and generation evaluation protocols."""
from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import project_path
from ..data.loaders import RecordDataset, YelpDataset, collate
from ..data.schema import ID_TO_TYPE, TYPE_TO_ID, read_json, read_jsonl
from ..models.prompting import build_target, score_target
from .metrics import accuracy, auc_score, bleu4, f1_positive, hr_at_k, ndcg_at_k, precision_at_k, rouge_l
from .protocol import BPRMF, sample_candidates


def _collect(model, cfg: Dict, records: Sequence[Dict]) -> Dict:
    """Score every record once and gather explanation / localisation outputs."""
    dataset = RecordDataset(cfg, records)
    loader = DataLoader(dataset, batch_size=int(cfg["eval"]["batch_size"]), shuffle=False, collate_fn=collate)
    scores: List[float] = []
    type_predictions: List[int] = []
    polarity_predictions: List[int] = []
    gates: List[float] = []
    regions: List[List[float]] = []
    keywords: List[List[str]] = []
    for batch in loader:
        if hasattr(model, "encode") and not getattr(model, "variant", {}).get("majority"):
            outputs = model.encode(batch, return_keywords=True)
            logits = model.score_head(outputs["item_vector"]).squeeze(-1)
            batch_scores = torch.sigmoid(logits).tolist()
            type_predictions.extend(outputs["type_logits"].argmax(dim=-1).tolist())
            polarity_predictions.extend((outputs["polarity_logits"].argmax(dim=-1) - 1).tolist())
            gates.extend(outputs["gate"].mean(dim=-1).tolist())
            regions.extend(outputs["region"])
            keywords.extend(outputs["keywords"])
        else:
            batch_scores = model.score(batch).tolist()
            type_predictions.extend([-1] * len(batch_scores))
            polarity_predictions.extend([0] * len(batch_scores))
            gates.extend([0.0] * len(batch_scores))
            regions.extend([[0.0, 0.0, 1.0, 1.0]] * len(batch_scores))
            keywords.extend(batch["keywords"])
        scores.extend(float(value) for value in batch_scores)
    return {
        "scores": np.asarray(scores, dtype=np.float32),
        "type_predictions": type_predictions,
        "polarity_predictions": polarity_predictions,
        "gates": gates,
        "regions": regions,
        "keywords": keywords,
    }


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / area) if area > 0 else 0.0


def _localisation(records: Sequence[Dict], collected: Dict) -> Dict:
    ious = []
    hits = 0
    evaluated = 0
    for row, region, words in zip(records, collected["regions"], collected["keywords"]):
        ious.append(_iou(region, row.get("region") or []))
        truth = {word.lower() for word in (row.get("keywords") or row.get("keywords_text") or [])}
        if truth:
            evaluated += 1
            hits += int(bool({word.lower() for word in words} & truth))
    n = max(1, len(ious))
    return {
        "iou_mean": round(float(np.mean(ious)) if ious else 0.0, 4),
        "keyword_hit_rate": round(hits / evaluated, 4) if evaluated else 0.0,
        "localization_failure_rate": round(sum(1 for value in ious if value < 0.3) / n, 4),
    }


def _grouped_ranking(records: Sequence[Dict], scores: np.ndarray, seed: int = 0) -> Dict[str, float]:
    """1 positive + 19 negatives per user; P@10 / NDCG@10."""
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(records):
        groups[str(row.get("user_id"))].append(index)
    rng = random.Random(seed)
    precision, ndcg = [], []
    for user, indices in groups.items():
        positives = [index for index in indices if row_label(records[index]) == 1]
        if not positives:
            continue
        negative_pool = [
            index for index, row in enumerate(records)
            if row_label(row) == 0 and row.get("user_id") != user
        ]
        if len(negative_pool) < 19:
            continue
        candidates = [rng.choice(positives)] + rng.sample(negative_pool, 19)
        rng.shuffle(candidates)
        ranked = sorted(candidates, key=lambda index: -float(scores[index]))
        relevance = [row_label(records[index]) for index in ranked]
        precision.append(precision_at_k(relevance, 10))
        ndcg.append(ndcg_at_k(relevance, 10))
    return {
        "P@10": round(float(np.mean(precision)) if precision else 0.0, 4),
        "NDCG@10": round(float(np.mean(ndcg)) if ndcg else 0.0, 4),
        "users": len(precision),
    }


def row_label(row: Dict) -> int:
    if "label" in row:
        return int(row["label"])
    return 1 if int(row.get("polarity", 0)) > 0 else 0


def _cold_start_auc(records: Sequence[Dict], scores: np.ndarray, all_records: Sequence[Dict]) -> float:
    counts: Dict[str, int] = defaultdict(int)
    for row in all_records:
        counts[row.get("business_id")] += 1
    indices = [index for index, row in enumerate(records) if counts[row.get("business_id")] <= 1]
    if len(indices) < 2:
        indices = list(range(len(records)))
    labels = [row_label(records[index]) for index in indices]
    return auc_score(labels, [float(scores[index]) for index in indices])


def _generation_metrics(model, cfg: Dict, records: Sequence[Dict]) -> Dict:
    if not hasattr(model, "generate"):
        return {}
    limit = min(int(cfg["eval"].get("generation_samples", 60)), len(records))
    rng = random.Random(cfg.get("seed", 42))
    sample = rng.sample(list(records), limit) if limit < len(records) else list(records)
    dataset = RecordDataset(cfg, sample)
    loader = DataLoader(dataset, batch_size=int(cfg["eval"]["batch_size"]), shuffle=False, collate_fn=collate)
    bleu, rouge, failures = [], [], []
    for batch in loader:
        texts, scores = model.generate(batch, max_new_tokens=int(cfg["decoder"]["max_new_tokens"]))
        for index, text in enumerate(texts):
            reference = build_target(
                ID_TO_TYPE.get(int(batch["type_id"][index]), "null"),
                int(batch["polarity"][index]),
                score_target(batch["sample_id"][index], int(batch["label"][index])),
            )
            bleu.append(bleu4(text, reference))
            rouge.append(rouge_l(text, reference))
            failures.append(1.0 if scores[index] is None else 0.0)
    return {
        "BLEU-4": round(float(np.mean(bleu)) if bleu else 0.0, 4),
        "ROUGE-L": round(float(np.mean(rouge)) if rouge else 0.0, 4),
        "format_failure_rate": round(float(np.mean(failures)) if failures else 0.0, 4),
        "generated": len(bleu),
    }


def _type_weight(gates: Sequence[float], records: Sequence[Dict]) -> Dict[str, float]:
    buckets: Dict[str, List[float]] = defaultdict(list)
    for value, row in zip(gates, records):
        type_id = TYPE_TO_ID.get(row.get("type", "null"), TYPE_TO_ID["null"])
        buckets[ID_TO_TYPE[type_id]].append(value)
    return {key: round(float(np.mean(values)), 4) for key, values in buckets.items() if values}


def evaluate_business_level(model, cfg: Dict, split: str = "test", mode: str = "head") -> Dict:
    dataset = YelpDataset(cfg, split=split)
    records = dataset.records
    if not records:
        return {"error": "empty split {}".format(split)}
    collected = _collect(model, cfg, records)
    labels = [row_label(row) for row in records]
    predictions = [1 if score >= 0.5 else 0 for score in collected["scores"]]
    full = YelpDataset(cfg, split="train").records + YelpDataset(cfg, split="val").records + records

    result = {
        "split": split,
        "samples": len(records),
        "Acc": round(accuracy(labels, predictions), 4),
        "F1": round(f1_positive(labels, predictions), 4),
        "AUC": round(auc_score(labels, collected["scores"].tolist()), 4),
        "Cold-Start AUC": round(_cold_start_auc(records, collected["scores"], full), 4),
        "type_accuracy": round(
            accuracy(
                [TYPE_TO_ID.get(row.get("type", "null"), TYPE_TO_ID["null"]) for row in records],
                collected["type_predictions"],
            ), 4
        ) if -1 not in collected["type_predictions"] else None,
    }
    result.update(_grouped_ranking(records, collected["scores"], seed=cfg.get("seed", 42)))
    result["localization"] = _localisation(records, collected)
    result["text_weight_by_type"] = _type_weight(collected["gates"], records)
    if mode == "generate":
        result["generation"] = _generation_metrics(model, cfg, records)
    return result


def evaluate_user_ranking(model, cfg: Dict, mode: str = "head") -> Dict:
    directory = project_path(cfg, "yelp_dir")
    users = read_jsonl(os.path.join(directory, "users.jsonl"))
    records = read_jsonl(os.path.join(directory, "businesses.jsonl"))
    splits = read_json(os.path.join(directory, "splits.json"), default={})
    test_groups = set(splits.get("test", []))
    train_groups = set(splits.get("train", []))

    by_business: Dict[str, Dict] = {}
    for row in records:
        by_business.setdefault(row["business_id"], row)
    business_ids = sorted(by_business)
    item_index = {business: index for index, business in enumerate(business_ids)}

    user_ids = sorted({user["user_id"] for user in users})
    user_index = {user: index for index, user in enumerate(user_ids)}
    train_pairs = []
    for user in users:
        for business in user.get("history", []):
            if business in train_groups and business in item_index:
                train_pairs.append((user_index[user["user_id"]], item_index[business]))
    bpr = BPRMF(len(user_ids), len(business_ids), factors=32, seed=cfg.get("seed", 42))
    bpr.fit(train_pairs, len(business_ids), epochs=int(cfg["eval"].get("bpr_epochs", 5)), seed=cfg.get("seed", 42))

    test_records = [row for row in records if row.get("split") == "test"]
    collected = _collect(model, cfg, test_records) if test_records else {"scores": np.zeros(0)}
    score_of = {row["sample_id"]: float(collected["scores"][index]) for index, row in enumerate(test_records)}

    rng = random.Random(cfg.get("seed", 42))
    limit = int(cfg["eval"].get("user_ranking_users", 80))
    hr5, hr10, ndcg5, ndcg10 = [], [], [], []
    evaluated = 0
    for user in users[:limit]:
        positives = [business for business in user.get("history", []) if business in test_groups]
        if not positives:
            continue
        pool = list(test_records)
        if len(pool) < 19:
            continue
        user_vector = user_index.get(user["user_id"], 0)
        ranking = np.argsort(-bpr.scores(user_vector))
        hard_order = [business_ids[index] for index in ranking]
        samples = sample_candidates(positives, [row["business_id"] for row in pool],
                                    negatives=19, hard_order=hard_order, seed=rng.randint(0, 10 ** 6))
        if not samples:
            continue
        candidates = [pool[index] for index in samples]
        rng.shuffle(candidates)
        positives_set = set(positives)
        if hasattr(model, "score_candidates"):
            candidate_scores = model.score_candidates(user["user_id"], candidates)
            ranked = [
                row for _, row in sorted(zip(candidate_scores, candidates), key=lambda pair: -float(pair[0]))
            ]
        else:
            ranked = sorted(candidates, key=lambda row: -score_of.get(row["sample_id"], 0.0))
        relevance = [1 if row["business_id"] in positives_set else 0 for row in ranked]
        hr5.append(hr_at_k(relevance, 5))
        hr10.append(hr_at_k(relevance, 10))
        ndcg5.append(ndcg_at_k(relevance, 5))
        ndcg10.append(ndcg_at_k(relevance, 10))
        evaluated += 1

    result = {
        "users": evaluated,
        "HR@5": round(float(np.mean(hr5)) if hr5 else 0.0, 4),
        "HR@10": round(float(np.mean(hr10)) if hr10 else 0.0, 4),
        "NDCG@5": round(float(np.mean(ndcg5)) if ndcg5 else 0.0, 4),
        "NDCG@10": round(float(np.mean(ndcg10)) if ndcg10 else 0.0, 4),
    }
    if mode == "generate":
        generated = []
        for user in users[: max(1, int(cfg["eval"].get("generation_users", 20)))]:
            for business in user.get("history", []):
                if business in test_groups and business in by_business:
                    generated.append(by_business[business])
        if generated:
            result["generation"] = _generation_metrics(model, cfg, generated)
    return result
