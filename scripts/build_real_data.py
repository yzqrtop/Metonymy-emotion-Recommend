"""Build processed datasets exclusively from downloaded official files."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import ensure_dir, load_config, project_path  # noqa: E402
from src.data.real_builder import build_emometa, build_taobao, build_yelp  # noqa: E402
from src.data.schema import read_json, write_json  # noqa: E402
from src.data.vocab import ensure_vocab  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/real_cpu.yaml")
    parser.add_argument("--dataset", choices=["all", "emometa", "taobao", "yelp"], default="all")
    parser.add_argument("--emometa-limit", type=int, default=0)
    parser.add_argument("--taobao-items", type=int, default=64)
    parser.add_argument("--taobao-users", type=int, default=32)
    parser.add_argument("--yelp-limit", type=int, default=0)
    args = parser.parse_args()
    if args.config.endswith(".yaml") and not os.path.exists(args.config):
        raise SystemExit("config not found: {} (point --config at a real-data profile)".format(args.config))
    cfg = load_config(args.config)
    ensure_vocab(project_path(cfg, "keyword_vocab"))

    path = os.path.join(ensure_dir(project_path(cfg, "data_root")), "REAL_BUILD_MANIFEST.json")
    manifest = read_json(path, default={}) or {}
    manifest["data_mode"] = "official_real_data_only"
    manifest.setdefault("results", {})
    if args.dataset in ("all", "emometa"):
        manifest["results"]["emometa"] = build_emometa(cfg, args.emometa_limit)
    if args.dataset in ("all", "taobao"):
        manifest["results"]["taobao_mm"] = build_taobao(cfg, args.taobao_items, args.taobao_users)
    if args.dataset == "yelp":
        manifest["results"]["yelp"] = build_yelp(cfg, args.yelp_limit)
        manifest["yelp"] = {"status": "ready", "meta": manifest["results"]["yelp"]}
    elif manifest.get("yelp", {}).get("status") != "ready":
        manifest["yelp"] = {
            "status": "manual_download_required",
            "official_url": "https://www.yelp.com/dataset",
            "note": "No Yelp archive was auto-downloaded or replaced. See RUNBOOK.md for manual ingestion requirements.",
        }
    write_json(path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("[build_real_data] manifest: " + path)


if __name__ == "__main__":
    main()
