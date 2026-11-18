"""Converters for official EmoMeta and Taobao-MM files.

No images, labels or embeddings are synthesized. Derived fields are explicitly
named and separated from fields supplied by the dataset publishers.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import struct
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import requests
from PIL import Image

from ..config import ensure_dir, project_path
from .schema import EMOTION_TO_POLARITY, dataset_statistics, group_split, write_json, write_jsonl
from .text_utils import lexicon_polarity

EMOTIONS = {
    1: "joy", 2: "love", 3: "trust", 4: "fear", 5: "sadness",
    6: "disgust", 7: "anger", 8: "surprise", 9: "anticipation", 10: "neutral",
}
TAOBOO_ROOT = "https://hf-mirror.com/datasets/TaoBao-MM/Taobao-MM/resolve/main/"


def build_emometa(cfg: Dict, limit: int = 0) -> Dict:
    repository = os.path.join(project_path(cfg, "raw_dir"), "emometa", "EmoMeta-master")
    workbook = os.path.join(repository, "DataAll.xlsx")
    if not os.path.exists(workbook):
        raise FileNotFoundError("Run scripts/download_real_data.py --dataset emometa first: " + workbook)
    frame = pd.read_excel(workbook)
    metaphor_column = next(c for c in frame.columns if str(c).startswith("MetaphorOccurrence"))
    emotion_column = next(c for c in frame.columns if str(c).startswith("Emotion"))
    output_dir = ensure_dir(project_path(cfg, "emometa_dir"))
    image_dir = ensure_dir(os.path.join(project_path(cfg, "image_root"), "emometa"))

    records = []
    missing = []
    for _, row in frame.iterrows():
        image_name = str(row["Img_id"])
        source = os.path.join(repository, "dataF", image_name)
        if not os.path.exists(source):
            missing.append({"Img_id": image_name, "Img_url": row.get("Img_url"), "reason": "image_not_downloaded"})
            continue
        try:
            with Image.open(source) as image:
                image.verify()
        except Exception as error:
            missing.append({"Img_id": image_name, "Img_url": row.get("Img_url"), "reason": str(error)})
            continue
        destination = os.path.join(image_dir, image_name)
        if not os.path.exists(destination):
            shutil.copy2(source, destination)
        emotion_id = int(row[emotion_column])
        emotion = EMOTIONS[emotion_id]
        polarity = EMOTION_TO_POLARITY[emotion]
        text = "" if pd.isna(row["Text"]) else str(row["Text"])
        text_polarity, sentiment_score = lexicon_polarity(text)
        stem = os.path.splitext(image_name)[0]
        records.append({
            "sample_id": "emometa_" + stem,
            "group_id": stem,
            "text": text,
            "image": "emometa/" + image_name,
            "type": None,
            "type_source": "official_T1_T2_T3_unavailable",
            "polarity": polarity,
            "polarity_source": "derived_from_official_emotion",
            "label": int(polarity > 0),
            "label_source": "derived_for_model_mechanics_not_recommendation_ground_truth",
            "emotion": emotion,
            "official_emotion_id": emotion_id,
            "official_metaphor_occurrence": int(row[metaphor_column]),
            "official_target_domain": None if pd.isna(row["Target domain"]) else str(row["Target domain"]),
            "official_source_domain": None if pd.isna(row["Source domain"]) else str(row["Source domain"]),
            "official_image_url": None if pd.isna(row["Img_url"]) else str(row["Img_url"]),
            "region": None,
            "region_source": "official_bbox_unavailable",
            "keywords": [],
            "keywords_source": "official_keywords_unavailable",
            "text_polarity": text_polarity,
            "sentiment_score": sentiment_score,
        })
        if limit and len(records) >= limit:
            break
    if not records:
        raise RuntimeError("No usable official EmoMeta images were found")
    splits = group_split(records, "group_id", (0.8, 0.1, 0.1), int(cfg.get("seed", 42)))
    split_of = {group: name for name, groups in splits.items() for group in groups}
    for row in records:
        row["split"] = split_of[row["group_id"]]
    write_jsonl(os.path.join(output_dir, "pairs.jsonl"), records)
    write_jsonl(os.path.join(output_dir, "missing_images.jsonl"), missing)
    write_json(os.path.join(output_dir, "splits.json"), splits)
    stats = dataset_statistics(records)
    stats["emotion_counts"] = dict(Counter(row["emotion"] for row in records))
    manifest = {
        "source": "DUTIR-YSQ/EmoMeta official DataAll.xlsx and dataF",
        "official_rows": len(frame),
        "usable_real_image_pairs": len(records),
        "missing_or_invalid_images": len(missing),
        "statistics": stats,
        "split_groups": {name: len(groups) for name, groups in splits.items()},
        "official_fields": ["Img_id", "Img_url", "Text", "MetaphorOccurrence", "Target domain", "Source domain", "Emotion"],
        "unavailable_official_fields": ["T1/T2/T3", "bbox", "keywords", "recommendation label", "user query"],
        "warning": "Derived polarity/label fields support code testing, not official recommendation evaluation.",
    }
    write_json(os.path.join(output_dir, "meta.json"), manifest)
    return manifest


def _clicked(row: Dict) -> bool:
    label = row.get("label_0")
    return bool(label is not None and len(label) == 2 and int(label[1]) == 1 and int(label[0]) == 0)


def _remote_header(url: str) -> Dict:
    response = requests.get(url, headers={"Range": "bytes=0-4095"}, timeout=(30, 60))
    response.raise_for_status()
    data = response.content
    if data[:6] != b"\x93NUMPY":
        raise ValueError("Remote file is not NPY: " + url)
    major = data[6]
    if major == 1:
        length = struct.unpack("<H", data[8:10])[0]
        offset = 10 + length
        header = data[10:offset]
    elif major == 2:
        length = struct.unpack("<I", data[8:12])[0]
        offset = 12 + length
        header = data[12:offset]
    else:
        raise ValueError("Unsupported NPY version: {}".format(major))
    meta = ast.literal_eval(header.decode("latin1").strip())
    meta["data_offset"] = offset
    return meta


def _range_row(url: str, offset: int, length: int) -> bytes:
    response = requests.get(url, headers={"Range": "bytes={}-{}".format(offset, offset + length - 1)}, timeout=(30, 60))
    response.raise_for_status()
    if len(response.content) != length:
        raise IOError("Range request returned {} bytes, expected {}".format(len(response.content), length))
    return response.content


def _feature_map_embeddings(raw_dir: str, item_ids: Sequence[int]) -> np.ndarray:
    feature_dir = os.path.join(raw_dir, "feature_map")
    keys_path = os.path.join(feature_dir, "scl_emb_int8_p90_keys.npy")
    values_path = os.path.join(feature_dir, "scl_emb_int8_p90_values.npy")
    if not os.path.exists(keys_path):
        raise FileNotFoundError("Missing sorted SCL keys. Run scripts/download_real_data.py --dataset taobao --taobao-keys")
    keys = np.load(keys_path, mmap_mode="r")
    positions = np.searchsorted(keys, np.asarray(item_ids, dtype=np.int64))
    if any(int(position) >= len(keys) or int(keys[int(position)]) != int(item) for item, position in zip(item_ids, positions)):
        raise KeyError("At least one selected item has no official SCL embedding")
    if os.path.exists(values_path):
        values = np.load(values_path, mmap_mode="r")
        return np.asarray(values[positions], dtype=np.float32)
    url = TAOBOO_ROOT + "feature_map/scl_emb_int8_p90_values.npy"
    header = _remote_header(url)
    if list(header["shape"])[1:] != [128] or header["descr"] not in ("|i1", "i1", "int8"):
        raise ValueError("Unexpected official SCL NPY layout: {}".format(header))
    rows = []
    for position in positions:
        payload = _range_row(url, int(header["data_offset"]) + int(position) * 128, 128)
        rows.append(np.frombuffer(payload, dtype=np.int8))
    return np.stack(rows).astype(np.float32)


def _parquet_embeddings(raw_dir: str, item_ids: Sequence[int]) -> np.ndarray:
    import pyarrow.parquet as pq

    path = os.path.join(raw_dir, "raw", "scl_embedding_int8_p90.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError("No local SCL feature map or raw/scl_embedding_int8_p90.parquet")
    wanted = {int(item) for item in item_ids}
    found = {}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=65536, columns=["205", "205_c"]):
        for row in batch.to_pylist():
            item = int(row["205"])
            if item in wanted:
                found[item] = row["205_c"]
        if len(found) == len(wanted):
            break
    if len(found) != len(wanted):
        raise KeyError("SCL parquet covers {}/{} selected items".format(len(found), len(wanted)))
    return np.asarray([found[int(item)] for item in item_ids], dtype=np.float32)


def build_taobao(cfg: Dict, max_items: int = 64, max_users: int = 32) -> Dict:
    raw_dir = os.path.join(project_path(cfg, "raw_dir"), "taobao_mm")
    import pyarrow.parquet as pq

    train_path = os.path.join(raw_dir, "train_train-shard-000159.parquet")
    if not os.path.exists(train_path):
        raise FileNotFoundError("Run scripts/download_real_data.py --dataset taobao first")
    rows = pq.read_table(train_path, columns=["129_1", "205", "label_0"]).to_pylist()
    positives = [row for row in rows if _clicked(row)]
    by_user = defaultdict(list)
    for row in positives:
        by_user[int(row["129_1"])].append(int(row["205"]))
    eligible = sorted(((u, items) for u, items in by_user.items() if len(items) >= 2), key=lambda x: (-len(x[1]), str(x[0])))

    selected_users = []
    selected_items = set()
    for user, items in eligible:
        new_items = set(items) - selected_items
        if len(selected_items) + len(new_items) > max_items:
            continue
        selected_users.append(user)
        selected_items.update(items)
        if len(selected_users) >= max_users:
            break
    if not selected_users:
        raise RuntimeError("This shard has no users with at least two official clicks within the item limit")
    for row in rows:
        if len(selected_items) >= max_items:
            break
        selected_items.add(int(row["205"]))
    item_ids = sorted(int(item) for item in selected_items)
    embeddings = _feature_map_embeddings(raw_dir, item_ids) if os.path.exists(os.path.join(raw_dir, "feature_map", "scl_emb_int8_p90_keys.npy")) else _parquet_embeddings(raw_dir, item_ids)
    index = {item: position for position, item in enumerate(item_ids)}
    interactions = []
    for user in selected_users:
        items = by_user[user]
        interactions.append({
            "user_id": str(user),
            "train_items": [index[item] for item in items[:-1]],
            "test_items": [index[items[-1]]],
            "official_train_item_ids": items[:-1],
            "official_test_item_ids": [items[-1]],
        })

    output_dir = ensure_dir(project_path(cfg, "taobao_dir"))
    np.save(os.path.join(output_dir, "item_embeddings.npy"), embeddings)
    write_jsonl(os.path.join(output_dir, "interactions.jsonl"), interactions)
    meta = {
        "source": "TaoBao-MM/Taobao-MM official train shard and SCL int8 p90 feature map",
        "source_shard": "train/train-shard-000159.parquet",
        "source_rows": len(rows),
        "official_click_rows": len(positives),
        "items": len(item_ids),
        "users": len(interactions),
        "dim": int(embeddings.shape[1]),
        "official_item_ids": item_ids,
        "quantization": "official int8 values cast to float32; no synthetic rescaling",
        "label_source": "official label_0 [0,1] click rows only",
        "split_source": "derived leave-last-click-out per selected user inside the downloaded shard; not the full official split",
        "interactions": int(sum(len(row["train_items"]) + len(row["test_items"]) for row in interactions)),
        "scope_warning": "Small real-data mechanics test, not a full Taobao-MM benchmark.",
    }
    write_json(os.path.join(output_dir, "meta.json"), meta)
    return meta


def _find_official_file(root: str, names: Sequence[str]) -> str:
    for current, _, files in os.walk(root):
        for name in names:
            if name in files:
                return os.path.join(current, name)
    return ""


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_yelp(cfg: Dict, limit: int = 0) -> Dict:
    """Convert manually supplied official Yelp photos, businesses and reviews."""
    raw_dir = project_path(cfg, "raw_dir")
    yelp_dir = os.path.join(raw_dir, "yelp")
    photo_path = _find_official_file(yelp_dir, ["yelp_academic_dataset_photo.json", "photos.json", "photo.json"])
    business_path = _find_official_file(yelp_dir, ["yelp_academic_dataset_business.json", "business.json"])
    review_path = _find_official_file(yelp_dir, ["yelp_academic_dataset_review.json", "review.json"])
    if not all([photo_path, business_path, review_path]):
        raise FileNotFoundError(
            "Official Yelp photo/business/review JSON files are required under data/raw/yelp. "
            "Download and extract https://www.yelp.com/dataset manually after accepting its licence."
        )
    businesses = {row["business_id"]: row for row in _iter_jsonl(business_path)}
    images = {}
    for current, _, files in os.walk(yelp_dir):
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg")):
                images[os.path.splitext(name)[0]] = os.path.join(current, name)

    candidates = []
    for row in _iter_jsonl(photo_path):
        photo_id = str(row["photo_id"])
        business_id = str(row["business_id"])
        if photo_id in images and business_id in businesses:
            candidates.append((row, images[photo_id]))
            if limit and len(candidates) >= limit:
                break
    if not candidates:
        raise RuntimeError("No official Yelp photo could be joined to business metadata and a local image")

    required = Counter(str(row["business_id"]) for row, _ in candidates)
    reviews = defaultdict(list)
    for row in _iter_jsonl(review_path):
        business_id = str(row["business_id"])
        if business_id in required and len(reviews[business_id]) < required[business_id]:
            reviews[business_id].append(row)
        if all(len(reviews[business]) >= count for business, count in required.items()):
            break

    output_dir = ensure_dir(project_path(cfg, "yelp_dir"))
    image_dir = ensure_dir(os.path.join(project_path(cfg, "image_root"), "yelp"))
    annotations_path = os.path.join(project_path(cfg, "output_dir"), "annotation", "yelp", "annotations.jsonl")
    annotations = {}
    if cfg.get("annotation", {}).get("apply_annotations") and os.path.exists(annotations_path):
        annotations = {row["sample_id"]: row for row in _iter_jsonl(annotations_path)}
    used = defaultdict(int)
    records = []
    skipped_without_review = 0
    for photo, source in candidates:
        business_id = str(photo["business_id"])
        position = used[business_id]
        used[business_id] += 1
        if position >= len(reviews[business_id]):
            skipped_without_review += 1
            continue
        review = reviews[business_id][position]
        business = businesses[business_id]
        stars = float(review["stars"])
        polarity = 1 if stars >= 4 else (-1 if stars <= 2 else 0)
        photo_id = str(photo["photo_id"])
        image_name = photo_id + ".jpg"
        destination = os.path.join(image_dir, image_name)
        if not os.path.exists(destination):
            shutil.copy2(source, destination)
        categories = business.get("categories") or "local business"
        text_polarity, sentiment_score = lexicon_polarity(str(review.get("text") or ""))
        record = {
            "sample_id": photo_id,
            "business_id": business_id,
            "user_id": str(review["user_id"]),
            "query": "I am looking for {}.".format(categories),
            "query_source": "derived_from_official_business_categories",
            "text": str(review.get("text") or ""),
            "image": "yelp/" + image_name,
            "rating": stars,
            "label": int(stars >= 4),
            "label_source": "derived_from_official_review_stars_threshold_4",
            "type": None,
            "type_source": "official_T1_T2_T3_unavailable",
            "polarity": polarity,
            "polarity_source": "derived_from_official_review_stars",
            "region": None,
            "region_source": "official_bbox_unavailable",
            "text_polarity": text_polarity,
            "sentiment_score": sentiment_score,
            "official_photo_caption": photo.get("caption") or "",
            "official_photo_label": photo.get("label"),
            "official_business_name": business.get("name"),
            "official_review_id": review.get("review_id"),
        }
        if photo_id in annotations:
            annotation = annotations[photo_id]
            for key in ("type", "polarity", "region"):
                if key in annotation:
                    record[key] = annotation[key]
            record["region_source"] = annotation.get("region_source", "real_vlm_and_review_pipeline")
            record["annotation_source"] = "real_vlm_and_review_pipeline:" + str(annotation.get("backend", "unknown"))
        records.append(record)
    if not records:
        raise RuntimeError("No official Yelp photo-review pairs with local images were constructed")

    splits = group_split(records, "business_id", (0.8, 0.1, 0.1), int(cfg.get("seed", 42)))
    split_of = {group: name for name, groups in splits.items() for group in groups}
    histories = defaultdict(list)
    for row in records:
        row["split"] = split_of[row["business_id"]]
        if row["business_id"] not in histories[row["user_id"]]:
            histories[row["user_id"]].append(row["business_id"])
    write_jsonl(os.path.join(output_dir, "businesses.jsonl"), records)
    write_json(os.path.join(output_dir, "splits.json"), splits)
    users = [{"user_id": user, "history": history, "query": ""} for user, history in histories.items()]
    write_jsonl(os.path.join(output_dir, "users.jsonl"), users)
    meta = {
        "source": "Yelp Open Dataset official photo, business and review JSON plus local official photos",
        "photo_json": photo_path,
        "business_json": business_path,
        "review_json": review_path,
        "samples": len(records),
        "businesses": len({row["business_id"] for row in records}),
        "users": len(users),
        "skipped_without_review": skipped_without_review,
        "statistics": dataset_statistics(records),
        "split_businesses": {name: len(groups) for name, groups in splits.items()},
        "annotations_applied": len(annotations),
        "warning": "Recommendation labels are derived from real review stars; they are not an official explicit recommendation label.",
    }
    write_json(os.path.join(output_dir, "meta.json"), meta)
    return meta
