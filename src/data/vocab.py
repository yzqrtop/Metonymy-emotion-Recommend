"""Keyword vocabulary used for CLIP-style region keyword extraction."""
from __future__ import annotations

import os

KEYWORDS = [
    "cup", "coffee", "latte", "plate", "bowl", "tray", "table", "counter", "napkin",
    "fork", "knife", "spoon", "glass", "bottle", "wine", "beer", "menu", "receipt",
    "pizza", "burger", "taco", "noodles", "sushi", "salad", "cake", "pastry", "bread",
    "box", "package", "bag", "wrapper", "carton", "label", "tape", "shelf",
    "window", "sunlight", "curtain", "candle", "flower", "plant", "chair", "bench",
    "wall", "wood", "metal", "light", "sign", "street", "sky", "beach", "car",
    "book", "laptop", "dog", "chef", "oven", "grill", "storefront", "umbrella",
]


def ensure_vocab(path: str) -> str:
    """Materialise the keyword vocabulary next to the processed datasets."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(KEYWORDS) + "\n")
    return path


def load_vocab(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            words = [line.strip() for line in handle if line.strip()]
        if words:
            return words
    return list(KEYWORDS)
