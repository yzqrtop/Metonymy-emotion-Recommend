"""Dataset classes for Yelp, EmoMeta and Taobao-MM."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..config import project_path
from .schema import (
    POLARITY_TO_ID,
    TYPE_TO_ID,
    read_json,
    read_jsonl,
)


def load_image(path: str, size: int = 224) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class _BaseDataset(Dataset):
    def __init__(self, records: Sequence[Dict], image_root: str, image_size: int = 224, with_image: bool = True):
        self.records = list(records)
        self.image_root = image_root
        self.image_size = image_size
        self.with_image = with_image

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict:
        row = self.records[index]
        item = {
            "sample_id": row["sample_id"],
            "text": row["text"],
            "type_id": TYPE_TO_ID.get(row.get("type"), -1),
            "polarity": int(row.get("polarity", 0)),
            "polarity_id": POLARITY_TO_ID.get(int(row.get("polarity", 0)), 1),
            "label": int(row.get("label", 1 if int(row.get("polarity", 0)) > 0 else 0)),
            "text_polarity": int(row.get("text_polarity", 0)),
            "region": torch.tensor(row.get("region") or [0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
            "keywords": list(row.get("keywords", [])),
            "query": row.get("query", ""),
            "user_id": row.get("user_id") or "",
            "business_id": row.get("business_id") or "",
        }
        if self.with_image:
            item["image"] = load_image(os.path.join(self.image_root, row["image"]), self.image_size)
        else:
            item["image"] = torch.zeros(3, self.image_size, self.image_size)
        return item


class YelpDataset(_BaseDataset):
    def __init__(self, cfg: Dict, split: str = "train", with_image: bool = True, limit: Optional[int] = None):
        directory = project_path(cfg, "yelp_dir")
        records = read_jsonl(os.path.join(directory, "businesses.jsonl"))
        splits = read_json(os.path.join(directory, "splits.json"), default={})
        groups = set(splits.get(split, []))
        selected = [row for row in records if row.get("split") == split or row.get("business_id") in groups]
        if limit:
            selected = selected[:limit]
        super().__init__(selected, project_path(cfg, "image_root"), 224, with_image)
        self.split = split

    @property
    def business_ids(self) -> List[str]:
        return [row["business_id"] for row in self.records]

    @property
    def user_ids(self) -> List[str]:
        return [row.get("user_id") for row in self.records]


class EmoMetaDataset(_BaseDataset):
    def __init__(self, cfg: Dict, split: str = "train", limit: Optional[int] = None, with_image: bool = True):
        directory = project_path(cfg, "emometa_dir")
        records = read_jsonl(os.path.join(directory, "pairs.jsonl"))
        split = split or "all"
        if split != "all":
            records = [row for row in records if row.get("split", "train") == split]
        if limit:
            records = records[:limit]
        super().__init__(records, project_path(cfg, "image_root"), 224, with_image)
        self.split = split


class RecordDataset(_BaseDataset):
    """Dataset over an explicit list of Yelp records (candidate sets, slices)."""

    def __init__(self, cfg: Dict, records: Sequence[Dict], with_image: bool = True):
        super().__init__(records, project_path(cfg, "image_root"), 224, with_image)
        self.split = "custom"


def collate(items: Sequence[Dict]) -> Dict:
    batch = {
        "texts": [item["text"] for item in items],
        "queries": [item["query"] for item in items],
        "keywords": [item["keywords"] for item in items],
        "images": torch.stack([item["image"] for item in items], dim=0),
        "type_id": torch.tensor([item["type_id"] for item in items], dtype=torch.long),
        "polarity": torch.tensor([item["polarity"] for item in items], dtype=torch.float32),
        "polarity_id": torch.tensor([item["polarity_id"] for item in items], dtype=torch.long),
        "label": torch.tensor([item["label"] for item in items], dtype=torch.float32),
        "text_polarity": torch.tensor([item["text_polarity"] for item in items], dtype=torch.long),
        "region": torch.stack([item["region"] for item in items], dim=0),
        "sample_id": [item["sample_id"] for item in items],
        "user_ids": [item["user_id"] for item in items],
        "business_ids": [item["business_id"] for item in items],
    }
    return batch


def load_taobao(cfg: Dict) -> Dict:
    directory = project_path(cfg, "taobao_dir")
    embeddings = np.load(os.path.join(directory, "item_embeddings.npy")).astype(np.float32)
    interactions = read_jsonl(os.path.join(directory, "interactions.jsonl"))
    meta = read_json(os.path.join(directory, "meta.json"), default={})
    return {"embeddings": embeddings, "interactions": interactions, "meta": meta}
