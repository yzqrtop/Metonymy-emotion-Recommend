"""Fast end-to-end smoke test on a miniature copy of the seed datasets."""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.annotation import AnnotationPipeline  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data.seed_builder import build_seed  # noqa: E402
from src.evaluation import (  # noqa: E402
    evaluate_business_level,
    evaluate_sentiment,
    evaluate_user_ranking,
    write_tables,
)
from src.models.factory import build_eval_model  # noqa: E402
from src.training import train_stage1, train_stage2  # noqa: E402

OVERRIDES = {
    "profile": "smoke",
    "paths": {
        "data_root": "data_smoke",
        "yelp_dir": "data_smoke/processed/yelp",
        "emometa_dir": "data_smoke/processed/emometa",
        "taobao_dir": "data_smoke/processed/taobao_mm",
        "image_root": "data_smoke/images",
        "output_dir": "outputs_smoke",
        "keyword_vocab": "data_smoke/processed/keyword_vocab.txt",
    },
    "seed_data": {
        "yelp_businesses": 80,
        "emometa_pairs": 240,
        "taobao_items": 256,
        "taobao_users": 40,
        "image_size": 96,
    },
    "stage1": {"batch_size": 8, "max_epochs": 1, "patience": 1, "emometa_subset": 80},
    "stage2": {"batch_size": 8, "max_epochs": 1, "patience": 1},
    "decoder": {"hidden_dim": 64, "layers": 1, "heads": 2, "max_len": 64, "max_new_tokens": 24},
    "eval": {
        "batch_size": 8,
        "seeds": 1,
        "generation_samples": 4,
        "human_sample": 4,
        "user_ranking_users": 6,
        "generation_users": 2,
        "transfer_users": 10,
        "bpr_epochs": 2,
        "ranking_score_mode": "head",
    },
    "baseline": {"epochs": 2},
}


def main() -> None:
    started = time.time()
    cfg = load_config("cpu", OVERRIDES)
    manifest = build_seed(cfg, force=True)
    print("[smoke] seed:", {key: manifest[key] for key in ("yelp", "emometa", "taobao_mm")})

    pipeline = AnnotationPipeline(cfg)
    yelp_report = pipeline.run("yelp", limit=40, force=True)
    emometa_report = pipeline.run("emometa", limit=40, force=True)
    print("[smoke] annotation:", yelp_report["agreement"], emometa_report["agreement"])

    stage1 = train_stage1(cfg, "full", 0)
    stage2 = train_stage2(cfg, "full", 0, stage1_checkpoint=stage1["checkpoint"])
    print("[smoke] stage1:", stage1["best_val_auc"], "stage2:", stage2.get("best_val_loss"))

    model = build_eval_model(cfg, "full", 0)
    business = evaluate_business_level(model, cfg, "test", "head")
    ranking = evaluate_user_ranking(model, cfg, "head")
    sentiment = evaluate_sentiment(model, cfg, limit=40)
    print("[smoke] business:", business)
    print("[smoke] user:", ranking)
    print("[smoke] sentiment:", sentiment)

    results = {
        "table1": {"full": {key: business[key] for key in ("Acc", "F1", "AUC", "P@10", "NDCG@10", "Cold-Start AUC")}},
        "table3": {"Full": {"Acc": sentiment["Acc"], "F1_neg": sentiment["per_class_F1"]["-1"],
                            "F1_neu": sentiment["per_class_F1"]["0"], "F1_pos": sentiment["per_class_F1"]["1"],
                            "Kappa": sentiment["Kappa"]}},
        "table5": {"Full": {"BLEU-4": 0.0, "ROUGE-L": 0.0, "Human-Coherence": None}},
    }
    tables = write_tables(cfg, results)
    print("[smoke] tables:", tables)
    print("[smoke] elapsed {:.1f}s".format(time.time() - started))
    print(json.dumps({"status": "ok"}, indent=2))


if __name__ == "__main__":
    main()
