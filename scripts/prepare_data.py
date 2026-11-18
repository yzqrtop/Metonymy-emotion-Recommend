"""Prepare the three experimental datasets (seed by default, real ingestion optional)."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, project_path  # noqa: E402
from src.data.schema import dataset_statistics, read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from src.data.seed_builder import build_seed  # noqa: E402
from src.data.vocab import ensure_vocab  # noqa: E402

RAW_LAYOUT = {
    "yelp": "data/raw/yelp/businesses.jsonl (sample_id, business_id, user_id, query, text, image, rating, label, type, polarity, region, keywords) + images in data/raw/images/yelp",
    "emometa": "data/raw/emometa/pairs.jsonl (sample_id, text, image, type, polarity, emotion, region, keywords) + images in data/raw/images/emometa",
    "taobao_mm": "data/raw/taobao_mm/item_embeddings.npy + interactions.jsonl (user_id, train_items, test_items)",
}


def _validate(cfg) -> dict:
    report = {}
    for name, key, filename in (
        ("yelp", "yelp_dir", "businesses.jsonl"),
        ("emometa", "emometa_dir", "pairs.jsonl"),
        ("taobao_mm", "taobao_dir", "meta.json"),
    ):
        directory = project_path(cfg, key)
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            report[name] = {"status": "missing", "path": path}
            continue
        if filename.endswith(".jsonl"):
            rows = read_jsonl(path)
            report[name] = {"status": "ok", "records": len(rows), "statistics": dataset_statistics(rows)}
        else:
            report[name] = {"status": "ok", "meta": read_json(path, default={})}
    return report


def _ingest_real(cfg, raw_dir: str) -> dict:
    """Copy schema-compatible raw JSONL/NPY into ``data/processed``.

    Download links (manual, both require accepting the original licences):
      * Yelp Open Dataset photos: https://www.yelp.com/dataset
      * EmoMeta:                 http://www.emometa.com  (request access)
      * Taobao-MM:               https://github.com/... (release page)
    """
    import shutil

    mapping = {
        "yelp": project_path(cfg, "yelp_dir"),
        "emometa": project_path(cfg, "emometa_dir"),
        "taobao_mm": project_path(cfg, "taobao_dir"),
    }
    written = {}
    for name, target in mapping.items():
        source = os.path.join(raw_dir, name)
        if not os.path.isdir(source):
            print("[prepare_data] raw folder missing: {} ({})".format(source, RAW_LAYOUT[name]))
            continue
        os.makedirs(target, exist_ok=True)
        for entry in os.listdir(source):
            shutil.copy2(os.path.join(source, entry), os.path.join(target, entry))
        written[name] = target
    image_source = os.path.join(raw_dir, "images")
    if os.path.isdir(image_source):
        shutil.copytree(image_source, project_path(cfg, "image_root"), dirs_exist_ok=True)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cpu")
    parser.add_argument("--source", choices=["seed", "real"], default="seed")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_vocab(project_path(cfg, "keyword_vocab"))
    if args.source == "seed":
        manifest = build_seed(cfg, force=args.force)
        print(json.dumps({"seed_manifest": manifest}, ensure_ascii=False, indent=2))
    else:
        written = _ingest_real(cfg, args.raw_dir)
        write_json(
            os.path.join(project_path(cfg, "data_root"), "processed", "REAL_INGEST.json"),
            {"written": written, "layout": RAW_LAYOUT},
        )
        print(json.dumps({"ingested": written, "layout": RAW_LAYOUT}, ensure_ascii=False, indent=2))
    print(json.dumps({"validation": _validate(cfg)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
