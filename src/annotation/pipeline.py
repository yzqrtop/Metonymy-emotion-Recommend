"""Confidence-gated annotation pipeline.

The pipeline implements the annotation protocol described in the Experiments
section:

* a strong VLM (Qwen2.5-VL-72B-Instruct, or the offline deterministic backend)
  pre-annotates every image-text pair with ``type`` / ``polarity`` /
  ``keywords`` / ``region`` / ``confidence``;
* keywords are read from both sides and stored separately: ``keywords_text``
  holds vocabulary nouns mentioned in the review, ``keywords_image`` holds the
  top-3 words of the localised region (CLIP text embeddings of the fixed
  vocabulary, or the offline region descriptor), and ``keywords`` merges them;
* regions come from the fused-attention top-k (k=20 of 196 patches) instead of
  a probability threshold, and ``region_source`` records where they came from;
* samples with ``confidence >= accept_threshold`` (0.95) are accepted directly;
* samples with ``confidence < review_threshold`` (0.70) are queued for full
  human review;
* the remaining band is sampled at ``sample_review_rate`` for spot checks;
* human decisions (``review_decisions.jsonl``) are merged back and the final
  annotation set plus a quality report (Cohen's kappa, IoU, keyword hit rate,
  localisation failure rate, confusion matrix) is written to disk.
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Sequence

from ..config import ensure_dir, project_path
from ..data.keywords import extract_keywords
from ..data.schema import (
    EMOTION_TYPES,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ..data.vocab import load_vocab
from .backends import build_backend, image_pixels
from .quality import cohen_kappa, confusion, localization_report


class AnnotationPipeline:
    """Run pre-annotation, confidence gating and quality control."""

    def __init__(self, cfg: Dict, backend=None):
        self.cfg = cfg
        self.settings = cfg["annotation"]
        vocab_path = project_path(cfg, "keyword_vocab")
        self.vocab = load_vocab(vocab_path)
        self.backend = backend or build_backend(cfg, vocab_path=vocab_path)
        self.accept_threshold = float(self.settings.get("accept_threshold", 0.95))
        self.review_threshold = float(self.settings.get("review_threshold", 0.7))
        self.sample_review_rate = float(self.settings.get("sample_review_rate", 0.2))
        self.rng = random.Random(cfg.get("seed", 42))

    # ------------------------------------------------------------------ data
    def _load_split(self, dataset: str) -> Dict:
        if dataset == "yelp":
            base = project_path(self.cfg, "yelp_dir")
            return {
                "records": read_jsonl(os.path.join(base, "businesses.jsonl")),
                "ground_truth": read_jsonl(os.path.join(base, "annotations_gt.jsonl")),
                "image_root": os.path.join(project_path(self.cfg, "image_root")),
            }
        if dataset == "emometa":
            base = project_path(self.cfg, "emometa_dir")
            return {
                "records": read_jsonl(os.path.join(base, "pairs.jsonl")),
                "ground_truth": read_jsonl(os.path.join(base, "annotations_gt.jsonl")),
                "image_root": os.path.join(project_path(self.cfg, "image_root")),
            }
        raise ValueError("unknown dataset: {}".format(dataset))

    def _output_dir(self, dataset: str) -> str:
        return ensure_dir(os.path.join(project_path(self.cfg, "output_dir"), "annotation", dataset))

    # -------------------------------------------------------------- pipeline
    def run(
        self,
        dataset: str = "yelp",
        limit: Optional[int] = None,
        decisions_path: Optional[str] = None,
        force: bool = False,
    ) -> Dict:
        """Annotate ``dataset`` and write the accepted set + quality report."""
        out_dir = self._output_dir(dataset)
        annotations_path = os.path.join(out_dir, "annotations.jsonl")
        report_path = os.path.join(out_dir, "quality_report.json")

        bundle = self._load_split(dataset)
        records = bundle["records"]
        if limit:
            records = records[: int(limit)]
        if os.path.exists(annotations_path) and not force:
            cached = read_json(report_path, default=None)
            if cached and cached.get("samples") == len(records):
                return cached

        image_root = bundle["image_root"]
        truth = {row["sample_id"]: row for row in bundle["ground_truth"]}

        decisions = {}
        candidates = [
            decisions_path,
            os.path.join(out_dir, "review_decisions.jsonl"),
            os.path.join(project_path(self.cfg, "output_dir"), "annotation", "review_decisions.jsonl"),
        ]
        for path in candidates:
            if path and os.path.exists(path):
                for row in read_jsonl(path):
                    decisions[row["sample_id"]] = row
                break

        accepted: List[Dict] = []
        review_queue: List[Dict] = []
        all_predictions: List[Dict] = []

        for row in records:
            sample_id = row["sample_id"]
            image_path = os.path.join(image_root, row["image"])
            pred = self.backend.annotate(row.get("text", ""), image_path, row.get("rating"))
            region = pred.get("region", [0.0, 0.0, 1.0, 1.0])
            keywords_text = list(pred.get("keywords_text") or [])
            keywords_image = list(pred.get("keywords_image") or [])
            keywords = list(pred.get("keywords") or [])
            keyword_source = pred.get("keyword_source")
            if not keywords_text and not keywords_image:
                pixels = image_pixels(image_path)
                extracted = extract_keywords(row.get("text", ""), pixels, region, self.vocab, topk=3)
                keywords_text = extracted["keywords_text"]
                keywords_image = extracted["keywords_image"]
                keywords = keywords or extracted["keywords"]
                keyword_source = keyword_source or extracted["keyword_source"]
            predicted = {
                "sample_id": sample_id,
                "type": pred["type"],
                "polarity": int(pred["polarity"]),
                "keywords": keywords[:3],
                "keywords_text": keywords_text[:3],
                "keywords_image": keywords_image[:3],
                "keyword_source": keyword_source or "backend",
                "region": region,
                "region_source": pred.get("region_source", getattr(self.backend, "name", "unknown")),
                "confidence": round(float(pred.get("confidence", 0.5)), 4),
                "rationale": pred.get("rationale", ""),
                "backend": getattr(self.backend, "name", "unknown"),
            }
            all_predictions.append(predicted)

            confidence = predicted["confidence"]
            if confidence >= self.accept_threshold:
                status = "accepted"
            elif confidence < self.review_threshold:
                status = "queued"
            else:
                status = "sampled" if self.rng.random() < self.sample_review_rate else "accepted_light"

            if sample_id in decisions:
                decision = decisions[sample_id]
                merged = dict(predicted)
                for key in ("type", "polarity", "keywords", "region"):
                    if key in decision:
                        merged[key] = decision[key]
                if "keywords" in decision:
                    merged["keyword_source"] = "human_reviewed"
                    merged["keywords_text"] = list(decision.get("keywords_text") or merged["keywords_text"])
                    merged["keywords_image"] = list(decision.get("keywords_image") or merged["keywords_image"])
                merged["status"] = "human_reviewed"
                merged["reviewer"] = decision.get("reviewer", "human")
                merged["confidence"] = 1.0
                accepted.append(merged)
            elif status == "queued" or status == "sampled":
                review_queue.append(dict(predicted, status=status, text=row.get("text", ""), image=row.get("image")))
                if status == "sampled":
                    # sampled items stay in the accepted set until a human reply
                    accepted.append(dict(predicted, status="accepted_sampled"))
            else:
                accepted.append(dict(predicted, status=status))

        write_jsonl(annotations_path, accepted)
        write_jsonl(os.path.join(out_dir, "review_queue.jsonl"), review_queue)
        write_jsonl(os.path.join(out_dir, "predictions_raw.jsonl"), all_predictions)

        report = self.quality_report(
            dataset=dataset,
            predictions=all_predictions,
            accepted=accepted,
            review_queue=review_queue,
            ground_truth=truth,
        )
        report["outputs"] = {
            "annotations": os.path.relpath(annotations_path, project_path(self.cfg, "data_root")),
            "review_queue": os.path.relpath(os.path.join(out_dir, "review_queue.jsonl"), project_path(self.cfg, "data_root")),
            "predictions_raw": os.path.relpath(os.path.join(out_dir, "predictions_raw.jsonl"), project_path(self.cfg, "data_root")),
        }
        write_json(report_path, report)
        return report

    # --------------------------------------------------------------- quality
    def quality_report(
        self,
        dataset: str,
        predictions: Sequence[Dict],
        accepted: Sequence[Dict],
        review_queue: Sequence[Dict],
        ground_truth: Dict[str, Dict],
    ) -> Dict:
        matched_pred = [row for row in predictions if row["sample_id"] in ground_truth]
        matched_gt = [ground_truth[row["sample_id"]] for row in matched_pred]

        type_agreement = cohen_kappa(
            [row["type"] for row in matched_pred],
            [row["type"] for row in matched_gt],
            labels=EMOTION_TYPES,
        )
        polarity_agreement = cohen_kappa(
            [str(row["polarity"]) for row in matched_pred],
            [str(int(row["polarity"])) for row in matched_gt],
            labels=["-1", "0", "1"],
        )
        localization = localization_report(
            matched_pred, matched_gt, iou_threshold=self.cfg["encoder"].get("iou_threshold", 0.3)
        )
        review_type_agreement = None
        reviewed = [row for row in accepted if row.get("status") == "human_reviewed"]
        if reviewed:
            pairs = [(row, ground_truth[row["sample_id"]]) for row in reviewed if row["sample_id"] in ground_truth]
            if pairs:
                review_type_agreement = cohen_kappa(
                    [row["type"] for row, _ in pairs],
                    [gt["type"] for _, gt in pairs],
                    labels=EMOTION_TYPES,
                )

        total = max(1, len(predictions))
        status_counts: Dict[str, int] = {}
        for row in accepted:
            status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        for row in review_queue:
            status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

        return {
            "dataset": dataset,
            "backend": getattr(self.backend, "name", "unknown"),
            "samples": len(predictions),
            "accepted": len(accepted),
            "review_queue": len(review_queue),
            "accept_rate": round(len(accepted) / total, 4),
            "queue_rate": round(len(review_queue) / total, 4),
            "status_counts": status_counts,
            "thresholds": {
                "accept": self.accept_threshold,
                "review": self.review_threshold,
                "sample_review_rate": self.sample_review_rate,
            },
            "agreement": {
                "type_cohen_kappa": round(float(type_agreement), 4),
                "polarity_cohen_kappa": round(float(polarity_agreement), 4),
                "human_reviewed_type_kappa": (
                    round(float(review_type_agreement), 4) if review_type_agreement is not None else None
                ),
                "human_reviewed": len(reviewed),
            },
            "localization": localization,
            "type_confusion": confusion(predictions, matched_gt, "type", EMOTION_TYPES),
        }
