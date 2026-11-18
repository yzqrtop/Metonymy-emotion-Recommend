"""End-to-end experiment runner: data -> annotation -> training -> evaluation -> tables."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src.annotation import AnnotationPipeline  # noqa: E402
from src.baselines import GraphRanker, load_prompt_baseline, train_prompt_baseline  # noqa: E402
from src.config import ensure_dir, load_config, project_path  # noqa: E402
from src.data.loaders import YelpDataset  # noqa: E402
from src.data.seed_builder import build_seed  # noqa: E402
from src.evaluation import (  # noqa: E402
    evaluate_business_level,
    evaluate_sentiment,
    evaluate_transfer,
    evaluate_user_ranking,
    export_human_eval,
    ingest_human_eval,
    write_tables,
)
from src.evaluation.recommendation import _generation_metrics  # noqa: E402
from src.models.factory import build_eval_model  # noqa: E402
from src.training import train_stage1, train_stage2  # noqa: E402
from src.variants import TABLE1_ORDER, get_variant  # noqa: E402

ABLATION_ORDER = ["full", "wo_metonymy", "wo_prototype", "wo_gate", "wo_image", "wo_text"]
AGGREGATE_KEYS = ["Acc", "F1", "AUC", "P@10", "NDCG@10", "Cold-Start AUC"]
RANKING_KEYS = ["HR@5", "HR@10", "NDCG@5", "NDCG@10"]


def _aggregate(runs, keys):
    out, spread = {}, {}
    for key in keys:
        values = [run[key] for run in runs if isinstance(run.get(key), (int, float))]
        if not values:
            continue
        out[key] = round(float(np.mean(values)), 4)
        spread[key] = round(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 4)
    return out, spread


def _ensure_datasets(cfg) -> None:
    marker = os.path.join(project_path(cfg, "data_root"), "processed", "MANIFEST.json")
    if not os.path.exists(marker):
        print("[runner] building seed datasets")
        build_seed(cfg)


def _ensure_annotations(cfg) -> dict:
    pipeline = AnnotationPipeline(cfg)
    limits = {"yelp": int(cfg["annotation"].get("yelp_limit", 0)),
              "emometa": int(cfg["annotation"].get("emometa_limit", 0))}
    reports = {}
    for dataset in ("yelp", "emometa"):
        reports[dataset] = pipeline.run(dataset, force=False, limit=limits[dataset] or None)
        print("[runner] annotation {} -> kappa(type)={}".format(
            dataset, reports[dataset].get("agreement", {}).get("type_cohen_kappa")))
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cpu")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--variants", default=None, help="comma separated; defaults to the Table-1 variants")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = list(range(args.seeds or int(cfg["eval"].get("seeds", 1))))
    variants = [name.strip() for name in args.variants.split(",")] if args.variants else list(TABLE1_ORDER)
    mode = cfg["eval"].get("ranking_score_mode", "head")

    _ensure_datasets(cfg)
    annotation_reports = _ensure_annotations(cfg)

    training_log, business_runs, ranking_runs = {}, {}, {}
    stage2_variants = cfg["stage2"].get("variants", ["full"])
    for name in variants:
        variant = get_variant(name)
        business_runs[name], ranking_runs[name] = [], []
        for seed in seeds:
            if not args.skip_train:
                stage1 = train_stage1(cfg, name, seed)
                training_log.setdefault(name, {})[str(seed)] = {"stage1": stage1}
                wants_stage2 = "*" in stage2_variants or name in stage2_variants
                if variant.get("generative") and wants_stage2:
                    stage2 = train_stage2(cfg, name, seed, stage1_checkpoint=stage1.get("checkpoint"))
                    training_log[name][str(seed)]["stage2"] = stage2
            model = build_eval_model(cfg, name, seed)
            business = evaluate_business_level(model, cfg, "test", mode)
            ranking = evaluate_user_ranking(model, cfg, mode)
            business_runs[name].append(business)
            ranking_runs[name].append(ranking)
            print("[runner] {} seed {}: Acc={} AUC={} P@10={} HR@10={}".format(
                name, seed, business.get("Acc"), business.get("AUC"),
                business.get("P@10"), ranking.get("HR@10")))

    results = {"config": cfg.get("profile"), "seeds": seeds, "annotation": annotation_reports}
    results["table1"], table1_std = {}, {}
    for name in variants:
        values, spread = _aggregate(business_runs[name], AGGREGATE_KEYS)
        runs = business_runs[name]
        iterations = [
            training_log.get(name, {}).get(str(seed), {}).get("stage1", {}).get("iterations")
            for seed in seeds
        ]
        iterations = [value for value in iterations if value is not None]
        if iterations:
            values["Iterations"] = round(float(np.mean(iterations)), 2)
        results["table1"][name] = values
        table1_std[name] = spread

    results["table2"] = {name: results["table1"][name] for name in ABLATION_ORDER if name in results["table1"]}
    results["table1_std"] = table1_std

    full_model = build_eval_model(cfg, "full", seeds[0]) if "full" in variants else None
    if full_model is not None:
        sentiment = evaluate_sentiment(full_model, cfg, limit=int(cfg["stage1"].get("emometa_subset", 400)))
        per_class = sentiment.get("per_class_F1", {})
        results["table3"] = {
            "Full": {
                "Acc": sentiment.get("Acc"),
                "F1_neg": per_class.get("-1"),
                "F1_neu": per_class.get("0"),
                "F1_pos": per_class.get("1"),
                "Kappa": sentiment.get("Kappa"),
            }
        }
        generation = _generation_metrics(full_model, cfg, YelpDataset(cfg, split="test").records)
        human = export_human_eval(full_model, cfg)
        ingested = ingest_human_eval(human["path"])
        results["table5"] = {
            "Full": {
                "BLEU-4": generation.get("BLEU-4", 0.0),
                "ROUGE-L": generation.get("ROUGE-L", 0.0),
                "Human-Coherence": ingested.get("coherence", {}).get("mean"),
                "format_failure_rate": generation.get("format_failure_rate", 0.0),
            }
        }
        results["human_eval_file"] = human["path"]

    results["table4"] = {"Ours (transfer adapter)": evaluate_transfer(cfg, seeds[0])}

    if not args.skip_external:
        external = {}
        for name, kind in (("MMGCN", "mmgcn"), ("LATTICE", "lattice")):
            model = GraphRanker(cfg, kind).fit(cfg, seeds[0])
            external[name] = evaluate_user_ranking(model, cfg, "head")
        for name in ("SEALR", "LLaMA-Rec"):
            train_prompt_baseline(cfg, name, seeds[0])
            external[name] = evaluate_user_ranking(load_prompt_baseline(cfg, name, seeds[0]), cfg, "head")
        if "emocrec" not in variants:
            train_stage1(cfg, "emocrec", seeds[0])
        external["EmoRec"] = evaluate_user_ranking(build_eval_model(cfg, "emocrec", seeds[0]), cfg, "head")
        results["external"] = external
        results["table_external"] = {
            name: {key: values.get(key) for key in RANKING_KEYS} for name, values in external.items()
        }

    results["training"] = training_log
    output_dir = ensure_dir(project_path(cfg, "output_dir"))
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "ablation.json"), "w", encoding="utf-8") as handle:
        json.dump(results.get("table2", {}), handle, ensure_ascii=False, indent=2)
    tables = write_tables(cfg, results)
    print(json.dumps({"results": os.path.join(output_dir, "results.json"), "tables": tables}, indent=2))


if __name__ == "__main__":
    main()
