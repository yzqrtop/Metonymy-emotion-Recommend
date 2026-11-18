"""Configuration loading and shared paths."""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")

DEFAULT_CONFIG = {
    "profile": "cpu",
    "seed": 42,
    "paths": {
        "data_root": "data",
        "raw_dir": "data/raw",
        "yelp_dir": "data/processed/yelp",
        "emometa_dir": "data/processed/emometa",
        "taobao_dir": "data/processed/taobao_mm",
        "image_root": "data/images",
        "output_dir": "outputs",
        "keyword_vocab": "data/processed/keyword_vocab.txt",
    },
    "seed_data": {
        "yelp_businesses": 900,
        "yelp_photos_ratio": [0.70, 0.25, 0.05],
        "positive_rate": 0.654,
        "emometa_pairs": 5000,
        "taobao_items": 4096,
        "taobao_users": 1200,
        "taobao_embed_dim": 64,
        "image_size": 160,
    },
    "encoder": {
        "text_backend": "mock",
        "vision_backend": "mock",
        "text_model": "Qwen/Qwen2.5-7B-Instruct",
        "vision_model": "vit_b_16",
        "text_dim": 512,
        "vision_dim": 768,
        "patch_grid": 14,
        "item_dim": 128,
        "attention_dim": 64,
        "num_heads": 4,
        "top_k": 20,
        "keyword_topk": 3,
        "iou_threshold": 0.3,
    },
    "prototype": {"enabled": True, "num_prototypes": 3, "max_gate": 0.01},
    "loss": {
        "lambda_type": 0.5,
        "lambda_cl": 0.25,
        "lambda_consis": 0.2,
        "lambda_sent": 0.3,
        "lambda_pol": 0.3,
        "polarity_weights": [1.91, 0.62, 1.17],
        "temperature": 0.07,
    },
    "stage1": {
        "batch_size": 16,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "max_epochs": 10,
        "patience": 2,
        "emometa_subset": 1500,
        "select_by": "val_auc",
    },
    "stage2": {
        "batch_size": 16,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "max_epochs": 8,
        "patience": 2,
        "lambda_score": 0.5,
        "warmup_steps": 0,
    },
    "decoder": {
        "backend": "tiny",
        "model_name": "Qwen/Qwen2.5-1.8B",
        "hidden_dim": 256,
        "layers": 3,
        "heads": 4,
        "ff_mult": 4,
        "dropout": 0.1,
        "max_len": 256,
        "lora_rank": 8,
        "max_new_tokens": 80,
    },
    "eval": {
        "batch_size": 32,
        "candidates": 20,
        "neg_top_k": 200,
        "bpr_epochs": 5,
        "seeds": 3,
        "topk": [5, 10],
        "human_sample": 200,
        "generation_samples": 200,
        "ranking_score_mode": "head",
    },
    "annotation": {
        "backend": "mock",
        "model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "accept_threshold": 0.95,
        "review_threshold": 0.7,
        "sample_review_rate": 0.2,
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
    "baseline": {
        "mmgcn_layers": 2,
        "lattice_neighbors": 10,
        "bpr_lr": 0.05,
        "bpr_factors": 32,
        "epochs": 6,
    },
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(profile: Optional[str] = "cpu", overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load a YAML profile merged on top of :data:`DEFAULT_CONFIG`."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if profile:
        path = profile
        if not os.path.isabs(path) and not path.endswith(".yaml"):
            path = os.path.join(CONFIG_DIR, "{}.yaml".format(profile))
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            cfg = _deep_update(cfg, loaded)
            cfg["profile"] = loaded.get("profile", os.path.splitext(os.path.basename(path))[0])
    if overrides:
        cfg = _deep_update(cfg, overrides)
    return cfg


def project_path(cfg: Dict[str, Any], key: str) -> str:
    """Resolve a path entry of ``cfg['paths']`` against the project root."""
    return os.path.join(PROJECT_ROOT, cfg["paths"][key])


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
