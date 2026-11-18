"""Train one variant (stage 1, stage 2, or both)."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.training import train_stage1, train_stage2  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cpu")
    parser.add_argument("--variant", default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", choices=["1", "2", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=None, help="override stage-2 epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["stage2"]["max_epochs"] = int(args.epochs)
        cfg["stage2"]["patience"] = max(2, int(args.epochs) // 4)
    results = {}
    if args.stage in ("1", "both"):
        results["stage1"] = train_stage1(cfg, args.variant, args.seed)
    if args.stage in ("2", "both"):
        stage1_path = results.get("stage1", {}).get("checkpoint")
        results["stage2"] = train_stage2(cfg, args.variant, args.seed, stage1_checkpoint=stage1_path)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
