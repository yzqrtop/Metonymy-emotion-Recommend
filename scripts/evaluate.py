"""Evaluate a trained variant (business-level, user-level, sentiment, transfer, human)."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ensure_dir, load_config, project_path  # noqa: E402
from src.evaluation import (  # noqa: E402
    evaluate_business_level,
    evaluate_user_ranking,
    evaluate_sentiment,
    evaluate_transfer,
    export_human_eval,
    ingest_human_eval,
)
from src.models.factory import build_eval_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cpu")
    parser.add_argument("--variant", default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["head", "generate"], default=None)
    parser.add_argument("--tasks", default="business,user,sentiment")
    parser.add_argument("--human-csv", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg["eval"].get("ranking_score_mode", "head")
    model = build_eval_model(cfg, args.variant, args.seed)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    results = {"variant": args.variant, "seed": args.seed, "mode": mode}
    if "business" in tasks:
        results["business"] = evaluate_business_level(model, cfg, "test", mode)
    if "user" in tasks:
        results["user"] = evaluate_user_ranking(model, cfg, mode)
    if "sentiment" in tasks:
        results["sentiment"] = evaluate_sentiment(model, cfg)
    if "transfer" in tasks:
        results["transfer"] = evaluate_transfer(cfg, args.seed)
    if "human" in tasks:
        results["human"] = (
            ingest_human_eval(args.human_csv) if args.human_csv else export_human_eval(model, cfg)
        )

    path = os.path.join(
        ensure_dir(os.path.join(project_path(cfg, "output_dir"), "results")),
        "{}_seed{}.json".format(args.variant, args.seed),
    )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("[evaluate] written to {}".format(path))


if __name__ == "__main__":
    main()
