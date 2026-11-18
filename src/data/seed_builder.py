"""Build the annotated seed datasets used by every experiment.

The repository does not ship the proprietary Yelp photo dump, EmoMeta or the
Taobao-MM embeddings.  ``scripts/prepare_data.py`` therefore produces a fully
annotated *seed* version of the three datasets with exactly the same schema as
the real converters, so the whole pipeline (annotation -> stage-1 -> stage-2 ->
evaluation -> paper tables) can be executed end-to-end on CPU.

Seed images are procedurally rendered so that T1 (outcome/action), T2 (object)
and T3 (scene) cues occupy a known bounding box; this makes the localisation
metrics (IoU / keyword hit rate) and the gate statistics measurable.
"""
from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ..config import ensure_dir, project_path
from .schema import (
    dataset_statistics,
    group_split,
    write_json,
    write_jsonl,
)
from .text_utils import lexicon_polarity
from .vocab import ensure_vocab

SEED_VERSION = 5

YELP_TEMPLATES: Dict[Tuple[str, int], List[str]] = {
    ("T1", 1): [
        "We finished every last bite and the empty plate said it all.",
        "The cup was drained in one go, best latte I have had in months.",
        "Nothing was left on the tray, everything tasted fresh and amazing.",
        "Our plates were cleared in minutes, warm, generous and delicious.",
    ],
    ("T1", -1): [
        "The plate went back to the kitchen untouched.",
        "My cup sat full and cold the whole evening.",
        "We left the tray exactly as it arrived, stale and disappointing.",
        "Half of the plate was still there when we paid.",
    ],
    ("T1", 0): [
        "The plate was cleared after the meal.",
        "We ordered a cup of coffee with the set.",
    ],
    ("T2", 1): [
        "The package was opened before we got home, still warm and fresh.",
        "A neatly wrapped box arrived on the counter, exactly what we hoped for.",
        "The little paper bag was full, generous portions and lovely packaging.",
        "Inside the box everything was crisp and clean, worth the wait.",
    ],
    ("T2", -1): [
        "The package came damaged, and the box was crushed.",
        "A torn wrapper on the counter, the contents were stale.",
        "The takeaway bag leaked all over the table, awful.",
        "The box was empty although we paid for the full set.",
    ],
    ("T2", 0): [
        "The package was placed on the counter.",
        "A box of pastries sat on the shelf.",
    ],
    ("T3", 1): [
        "Sunlight through the window made the corner seat warm and cozy.",
        "Bright window, quiet room, we stayed for hours and loved it.",
        "The candle on the table and the soft light made it lovely.",
        "A sunny windowsill, fresh flowers and a very comfortable chair.",
    ],
    ("T3", -1): [
        "The window looked onto a noisy street, cold and dirty.",
        "Dark room, curtain closed, we wanted to leave.",
        "The corner by the window was dim and unpleasant.",
        "Bright sign outside, but the inside was greasy and chaotic.",
    ],
    ("T3", 0): [
        "The window faced the street.",
        "A table by the window was available.",
    ],
    ("null", 1): ["Good service overall.", "The place was nice and clean."],
    ("null", -1): ["The visit was forgettable.", "Service felt rushed."],
    ("null", 0): ["We visited on a weekday.", "The menu has several options."],
}

EMOMETA_TEMPLATES: Dict[Tuple[str, int], List[str]] = {
    ("T1", 1): [
        "一口不剩，盘子干净得发亮，太满足了。",
        "咖啡见底，心情却满格，回味很香。",
        "最后一口也吃完了，香得不行，下次还来。",
        "汤都喝完了，暖到心里，真的很赞。",
    ],
    ("T1", -1): [
        "盘子原封不动地退了回去，太失望了。",
        "杯子一直是满的，凉了也没人管。",
        "剩了一半就吃不下了，味道很差。",
        "汤一口没动，腥味很重。",
    ],
    ("T1", 0): ["吃完后把盘子收走了。", "点了一杯咖啡配餐。"],
    ("T2", 1): [
        "拆开包装的那一刻，香味扑面而来，期待成真。",
        "盒子还热乎，分量很足，包装也干净。",
        "礼盒一打开就很惊喜，细节满满。",
        "袋子沉甸甸的，打开全是新鲜出炉的。",
    ],
    ("T2", -1): [
        "包装压扁了，盒子里的东西也碎了。",
        "袋子漏了一桌，太糟心了。",
        "拆开一看全是碎的，标签也贴错了。",
        "盒子看着挺好，里面却又干又硬。",
    ],
    ("T2", 0): ["包装盒放在柜台上。", "桌上摆着一盒点心。"],
    ("T3", 1): [
        "阳光透过窗户，角落的位置又暖又舒服。",
        "窗边很亮，坐了一下午都不想走。",
        "窗帘半开，光线柔和，整个房间都很安静。",
        "窗台上有花，阳光正好，心情也亮了起来。",
    ],
    ("T3", -1): [
        "窗外是吵闹的街道，屋里又冷又暗。",
        "窗帘拉着，房间昏暗，坐着很难受。",
        "窗边位置很挤，光线刺眼。",
        "窗外灰蒙蒙的，屋里也冷冷清清。",
    ],
    ("T3", 0): ["窗户朝着街面。", "窗边有空位。"],
}

EMOTION_BY_POLARITY = {
    1: ["joy", "love", "trust", "anticipation"],
    0: ["neutral"],
    -1: ["fear", "sadness", "disgust", "anger", "surprise"],
}

TYPE_TO_KIND = {"T1": "outcome", "T2": "object", "T3": "scene", "null": "object"}

QUERIES = {
    "T1": [
        "I want a place where I can finish a satisfying meal.",
        "Find me somewhere with portions big enough to clear the plate.",
    ],
    "T2": [
        "I am looking for somewhere to grab something to take home.",
        "Find a shop with well packed takeaway boxes.",
    ],
    "T3": [
        "I want a bright, cozy corner to sit and relax.",
        "Find a quiet place with good window light.",
    ],
    "null": ["I want a place nearby.", "Find a well reviewed business."],
}


def _draw_scene(kind: str, rng: random.Random, size: int) -> Tuple[Image.Image, List[float]]:
    """Render a synthetic photo and return its normalised cue bounding box."""
    if kind == "outcome":
        top, bottom = (250, 240, 220), (206, 178, 146)
    elif kind == "object":
        top, bottom = (245, 245, 242), (212, 206, 196)
    else:
        top, bottom = (188, 219, 245), (242, 241, 235)

    image = Image.new("RGB", (size, size), top)
    draw = ImageDraw.Draw(image)
    for y in range(size):
        ratio = y / max(1, size - 1)
        color = tuple(int(a + (b - a) * ratio) for a, b in zip(top, bottom))
        draw.line([(0, y), (size, y)], fill=color)

    if kind == "scene":
        x0, y0, x1, y1 = 0.18 * size, 0.12 * size, 0.82 * size, 0.74 * size
        draw.rectangle([x0, y0, x1, y1], fill=(135, 206, 250), outline=(250, 250, 250), width=max(3, size // 40))
        draw.line([(0.5 * size, y0), (0.5 * size, y1)], fill=(250, 250, 250), width=max(2, size // 60))
        draw.line([(x0, 0.43 * size), (x1, 0.43 * size)], fill=(250, 250, 250), width=max(2, size // 60))
        sun = 0.07 * size
        draw.ellipse([0.66 * size - sun, 0.26 * size - sun, 0.66 * size + sun, 0.26 * size + sun], fill=(255, 240, 165))
        draw.rectangle([0.10 * size, 0.10 * size, 0.20 * size, 0.80 * size], fill=(228, 176, 188))
        draw.rectangle([0.80 * size, 0.10 * size, 0.90 * size, 0.80 * size], fill=(228, 176, 188))
        return image, [round(x0 / size, 4), round(y0 / size, 4), round(x1 / size, 4), round(y1 / size, 4)]

    # table surface
    draw.rectangle([0, 0.62 * size, size, size], fill=(150 + rng.randint(-10, 10), 108, 74))
    draw.line([(0, 0.62 * size), (size, 0.62 * size)], fill=(110, 78, 52), width=max(2, size // 60))

    if kind == "outcome":
        draw.ellipse([0.30 * size, 0.48 * size, 0.70 * size, 0.68 * size], fill=(250, 250, 250), outline=(200, 200, 195), width=2)
        draw.ellipse([0.38 * size, 0.53 * size, 0.62 * size, 0.63 * size], fill=(206, 170, 130))
        for _ in range(4):
            cx = 0.44 * size + rng.random() * 0.12 * size
            cy = 0.55 * size + rng.random() * 0.05 * size
            rad = 0.012 * size
            draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(150, 95, 60))
        draw.rectangle([0.44 * size, 0.38 * size, 0.56 * size, 0.52 * size], fill=(245, 245, 245), outline=(190, 190, 185))
        draw.arc([0.54 * size, 0.40 * size, 0.60 * size, 0.48 * size], 270, 90, fill=(190, 190, 185), width=2)
        return image, [0.30, 0.38, 0.70, 0.68]

    # object
    draw.rectangle([0.30 * size, 0.36 * size, 0.70 * size, 0.68 * size], fill=(205, 170, 120), outline=(160, 128, 88), width=2)
    draw.rectangle([0.47 * size, 0.36 * size, 0.53 * size, 0.68 * size], fill=(236, 220, 180))
    draw.rectangle([0.55 * size, 0.52 * size, 0.67 * size, 0.62 * size], fill=(250, 250, 250), outline=(150, 150, 150))
    return image, [0.30, 0.36, 0.70, 0.68]


def _write_image(path: str, kind: str, rng: random.Random, size: int) -> List[float]:
    image, box = _draw_scene(kind, rng, size)
    ensure_dir(os.path.dirname(path))
    image.save(path, format="JPEG", quality=88)
    return box


def _make_queries(rng: random.Random, types: List[str], n_users: int) -> List[Dict]:
    users = []
    for idx in range(n_users):
        preferred = types[idx % len(types)]
        users.append(
            {
                "user_id": "U{:04d}".format(idx),
                "preferred_type": preferred,
                "query": rng.choice(QUERIES[preferred]),
            }
        )
    return users


def build_yelp(cfg: Dict, force: bool = False) -> Dict:
    out_dir = ensure_dir(project_path(cfg, "yelp_dir"))
    image_dir = os.path.join(project_path(cfg, "image_root"), "yelp")
    size = int(cfg["seed_data"]["image_size"])
    n_biz = int(cfg["seed_data"]["yelp_businesses"])
    weights = cfg["seed_data"]["yelp_photos_ratio"]
    positive_rate = float(cfg["seed_data"]["positive_rate"])
    rng = random.Random(cfg.get("seed", 42))

    photo_counts = rng.choices([1, 2, 3], weights=weights, k=n_biz)
    total_photos = sum(photo_counts)
    n_t3 = int(round(0.683 * total_photos))
    n_t1 = min(203, max(20, int(0.167 * total_photos)))
    n_null = max(10, int(0.025 * total_photos))
    n_t2 = max(10, total_photos - n_t3 - n_t1 - n_null)
    type_pool = ["T1"] * n_t1 + ["T2"] * n_t2 + ["T3"] * n_t3 + ["null"] * n_null
    while len(type_pool) < total_photos:
        type_pool.append("T3")
    type_pool = type_pool[:total_photos]
    rng.shuffle(type_pool)

    records: List[Dict] = []
    business_types: Dict[str, str] = {}
    cursor = 0
    for b_idx, count in enumerate(photo_counts):
        business_id = "B{:04d}".format(b_idx)
        for p_idx in range(count):
            sample_type = type_pool[cursor]
            cursor += 1
            if p_idx == 0:
                business_types[business_id] = sample_type

            label = 1 if rng.random() < positive_rate else 0
            if sample_type == "null":
                polarity = 0
            else:
                polarity = 1 if label == 1 else -1
                if rng.random() < 0.10:
                    polarity = 0
            rating = round(rng.uniform(4.0, 5.0) if label == 1 else rng.uniform(1.0, 3.5), 1)
            template_pool = YELP_TEMPLATES.get((sample_type, polarity)) or YELP_TEMPLATES[("null", 0)]
            text = rng.choice(template_pool)

            sample_id = "{}_P{}".format(business_id, p_idx + 1)
            image_rel = "yelp/{}.jpg".format(sample_id)
            image_path = os.path.join(image_dir, sample_id + ".jpg")
            box = _write_image(image_path, TYPE_TO_KIND[sample_type], rng, size)
            records.append(
                {
                    "sample_id": sample_id,
                    "business_id": business_id,
                    "user_id": None,
                    "query": rng.choice(QUERIES[sample_type]),
                    "text": text,
                    "image": image_rel,
                    "rating": rating,
                    "label": label,
                    "type": sample_type,
                    "polarity": polarity,
                    "region": box,
                    "text_polarity": lexicon_polarity(text)[0],
                    "sentiment_score": lexicon_polarity(text)[1],
                    "split": None,
                }
            )

    rng.shuffle(records)
    splits = group_split(records, "business_id", (0.8, 0.1, 0.1), cfg.get("seed", 42))
    split_of = {}
    for split_name, ids in splits.items():
        for bid in ids:
            split_of[bid] = split_name
    for row in records:
        row["split"] = split_of[row["business_id"]]

    all_types = [row["type"] for row in records]
    users = _make_queries(rng, all_types, 600)
    history_by_user: Dict[str, List[str]] = {}
    business_ids = sorted({row["business_id"] for row in records})
    for user in users:
        size_hist = rng.randint(3, 9)
        history_by_user[user["user_id"]] = rng.sample(business_ids, min(size_hist, len(business_ids)))
    for row in records:
        candidates = [uid for uid, hist in history_by_user.items() if row["business_id"] in hist]
        user_id = rng.choice(candidates) if candidates else rng.choice(users)["user_id"]
        row["user_id"] = user_id
        row["query"] = next(u["query"] for u in users if u["user_id"] == user_id)

    write_jsonl(os.path.join(out_dir, "businesses.jsonl"), records)
    write_json(os.path.join(out_dir, "splits.json"), splits)
    write_jsonl(os.path.join(out_dir, "users.jsonl"), [{**u, "history": history_by_user[u["user_id"]]} for u in users])
    annotations = [
        {
            "sample_id": row["sample_id"],
            "type": row["type"],
            "polarity": row["polarity"],
            "region": row["region"],
            "region_source": "seed_cue_bbox",
            "confidence": 1.0,
            "source": "seed_construction_text_image",
            "reviewed": True,
        }
        for row in records
    ]
    write_jsonl(os.path.join(out_dir, "annotations_gt.jsonl"), annotations)
    return {
        "path": out_dir,
        "samples": len(records),
        "images": len(records),
        "statistics": dataset_statistics(records),
        "splits": {key: len(value) for key, value in splits.items()},
        "type_counts": {t: all_types.count(t) for t in ["T1", "T2", "T3", "null"]},
    }


def build_emometa(cfg: Dict, force: bool = False) -> Dict:
    out_dir = ensure_dir(project_path(cfg, "emometa_dir"))
    image_dir = os.path.join(project_path(cfg, "image_root"), "emometa")
    size = int(cfg["seed_data"]["image_size"])
    n_pairs = int(cfg["seed_data"]["emometa_pairs"])
    rng = random.Random(cfg.get("seed", 42) + 7)

    n_pos, n_neu, n_neg = 1425, 2703, 872
    if n_pos + n_neu + n_neg != n_pairs:
        scale = n_pairs / float(n_pos + n_neu + n_neg)
        n_pos, n_neu, n_neg = int(n_pos * scale), int(n_neu * scale), 0
        n_neg = n_pairs - n_pos - n_neu
    polarities = [1] * n_pos + [0] * n_neu + [-1] * n_neg
    rng.shuffle(polarities)

    types = ["T1"] * int(n_pairs * 0.24) + ["T2"] * int(n_pairs * 0.36)
    types += ["T3"] * (n_pairs - len(types))
    rng.shuffle(types)

    records: List[Dict] = []
    n_train = int(n_pairs * 0.8)
    n_val = int(n_pairs * 0.1)
    for idx in range(n_pairs):
        sample_type = types[idx]
        polarity = polarities[idx]
        emotion = rng.choice(EMOTION_BY_POLARITY[polarity])
        template_pool = EMOMETA_TEMPLATES.get((sample_type, polarity)) or EMOMETA_TEMPLATES[(sample_type, 0)]
        text = rng.choice(template_pool)
        sample_id = "E{:05d}".format(idx)
        image_rel = "emometa/{}.jpg".format(sample_id)
        image_path = os.path.join(image_dir, sample_id + ".jpg")
        box = _write_image(image_path, TYPE_TO_KIND[sample_type], rng, size)
        split = "train" if idx < n_train else ("val" if idx < n_train + n_val else "test")
        records.append(
            {
                "sample_id": sample_id,
                "text": text,
                "image": image_rel,
                "type": sample_type,
                "polarity": polarity,
                "emotion": emotion,
                "region": box,
                "text_polarity": lexicon_polarity(text)[0],
                "sentiment_score": lexicon_polarity(text)[1],
                "split": split,
            }
        )

    write_jsonl(os.path.join(out_dir, "pairs.jsonl"), records)
    write_jsonl(
        os.path.join(out_dir, "annotations_gt.jsonl"),
        [
            {
                "sample_id": row["sample_id"],
                "type": row["type"],
                "polarity": row["polarity"],
                "emotion": row["emotion"],
                "region": row["region"],
                "region_source": "seed_cue_bbox",
                "confidence": 1.0,
                "source": "seed_construction_text_image",
                "reviewed": True,
            }
            for row in records
        ],
    )
    stats = dataset_statistics(records)
    stats["emotion_counts"] = {}
    for row in records:
        stats["emotion_counts"][row["emotion"]] = stats["emotion_counts"].get(row["emotion"], 0) + 1
    return {"path": out_dir, "samples": len(records), "statistics": stats}


def _bpr_embedding(rng: random.Random, dim: int, n_clusters: int = 32) -> np.ndarray:
    centers = np.random.RandomState(rng.randint(0, 10 ** 6)).normal(size=(n_clusters, dim)).astype(np.float32)
    return centers


def build_taobao(cfg: Dict, force: bool = False) -> Dict:
    out_dir = ensure_dir(project_path(cfg, "taobao_dir"))
    n_items = int(cfg["seed_data"]["taobao_items"])
    n_users = int(cfg["seed_data"]["taobao_users"])
    dim = int(cfg["seed_data"]["taobao_embed_dim"])
    rng = random.Random(cfg.get("seed", 42) + 13)
    rs = np.random.RandomState(cfg.get("seed", 42) + 13)

    n_clusters = 32
    centers = _bpr_embedding(rng, dim, n_clusters)
    assignments = rs.randint(0, n_clusters, size=n_items)
    noise = rs.normal(scale=0.35, size=(n_items, dim)).astype(np.float32)
    embeddings = centers[assignments] + noise
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    users = []
    for u_idx in range(n_users):
        preferred = set(rs.choice(n_clusters, size=3, replace=False).tolist())
        candidates = np.where(np.isin(assignments, list(preferred)))[0]
        if len(candidates) < 5:
            candidates = np.arange(n_items)
        size_hist = int(rs.randint(6, 26))
        items = rs.choice(candidates, size=min(size_hist, len(candidates)), replace=False)
        items = [int(x) for x in items]
        rng.shuffle(items)
        cut = max(1, int(len(items) * 0.8))
        users.append(
            {
                "user_id": "TU{:05d}".format(u_idx),
                "train_items": items[:cut],
                "test_items": items[cut:] or items[-1:],
            }
        )

    np.save(os.path.join(out_dir, "item_embeddings.npy"), embeddings)
    write_jsonl(os.path.join(out_dir, "interactions.jsonl"), users)
    write_json(
        os.path.join(out_dir, "meta.json"),
        {"items": n_items, "users": n_users, "dim": dim, "clusters": n_clusters},
    )
    return {
        "path": out_dir,
        "items": n_items,
        "users": n_users,
        "dim": dim,
        "interactions": int(sum(len(u["train_items"]) + len(u["test_items"]) for u in users)),
    }


def build_seed(cfg: Dict, force: bool = False) -> Dict:
    processed_root = project_path(cfg, "data_root") + os.sep + "processed"
    manifest_path = os.path.join(ensure_dir(processed_root), "MANIFEST.json")
    ensure_vocab(project_path(cfg, "keyword_vocab"))
    if os.path.exists(manifest_path) and not force:
        import json

        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("seed_version") == SEED_VERSION:
            return manifest

    manifest = {
        "seed_version": SEED_VERSION,
        "yelp": build_yelp(cfg, force=force),
        "emometa": build_emometa(cfg, force=force),
        "taobao_mm": build_taobao(cfg, force=force),
        "notes": "Procedurally rendered seed datasets. Replace with prepare_data.py --source real.",
    }
    write_json(manifest_path, manifest)
    return manifest
