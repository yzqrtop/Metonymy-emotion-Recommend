"""Run the confidence-gated annotation pipeline and print the quality report."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.annotation import AnnotationPipeline  # noqa: E402
from src.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cpu")
    parser.add_argument("--dataset", choices=["yelp", "emometa", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--decisions", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipeline = AnnotationPipeline(cfg)
    datasets = ["yelp", "emometa"] if args.dataset == "all" else [args.dataset]
    reports = {}
    for dataset in datasets:
        reports[dataset] = pipeline.run(dataset, limit=args.limit, decisions_path=args.decisions, force=args.force)
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
