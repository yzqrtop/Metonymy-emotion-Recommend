"""Text + image keyword extraction shared by annotation and evaluation.

The paper reads the localisation region out of the fused attention
(top-20 patches, no softmax threshold), back-projects it to the 14x14 grid, and
then extracts ``top-3`` keywords from the pooled region feature against a fixed
object/scene vocabulary.  Accordingly an annotation record keeps the two
sources separate (``keywords_text`` / ``keywords_image``) and merges them into
``keywords`` so that the evaluation can report keyword hit rate without hiding
where each word came from.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import numpy as np

# Chinese surface forms used by EmoMeta reviews -> English vocabulary entries.
CJK_CUES = {
    "咖啡": "coffee", "拿铁": "latte", "盘": "plate", "碗": "bowl", "托盘": "tray",
    "桌": "table", "柜台": "counter", "餐巾": "napkin", "叉": "fork", "刀": "knife",
    "勺": "spoon", "瓶": "bottle", "酒": "wine", "啤酒": "beer", "菜单": "menu",
    "披萨": "pizza", "汉堡": "burger", "面条": "noodles", "拉面": "noodles", "面馆": "noodles",
    "寿司": "sushi", "沙拉": "salad",
    "蛋糕": "cake", "面包": "bread", "盒": "box", "包装": "package", "袋": "bag",
    "标签": "label", "货架": "shelf", "窗": "window", "阳光": "sunlight",
    "窗帘": "curtain", "蜡烛": "candle", "鲜花": "flower", "花瓶": "flower",
    "植物": "plant", "椅": "chair", "墙": "wall", "木头": "wood", "招牌": "sign",
    "街": "street", "天空": "sky", "杯": "cup", "水杯": "cup", "玻璃": "glass",
}

FEATURES = ["white", "blue", "brown", "yellow", "pink", "gray", "green", "brightness", "saturation"]

# Region appearance prototypes.  They are used only when CLIP weights are not
# available; every record built this way is tagged ``image_backend=attribute``.
CATEGORY_PROFILES: Dict[str, Dict[str, float]] = {
    "tableware": {"white": 1.0, "brightness": 0.5, "gray": 0.25},
    "outdoor": {"blue": 1.0, "brightness": 0.25},
    "sunlight": {"yellow": 0.8, "brightness": 0.2},
    "furniture": {"brown": 1.0, "saturation": 0.25},
    "decor": {"pink": 1.0},
    "signage": {"gray": 1.0},
    "plant": {"green": 1.0},
}

CATEGORY_WORDS: Dict[str, List[str]] = {
    "tableware": ["plate", "cup", "bowl", "napkin", "tray", "glass"],
    "outdoor": ["window", "sky", "street"],
    "sunlight": ["sunlight", "light", "candle"],
    "furniture": ["table", "chair", "counter", "shelf", "wood", "box", "package", "bag", "wrapper",
                  "carton", "bread", "pastry", "cake"],
    "decor": ["curtain", "flower"],
    "signage": ["sign", "menu", "receipt", "label", "metal", "wall"],
    "plant": ["plant", "flower"],
}


def clamp_region(region: Optional[Sequence[float]]) -> List[float]:
    if not region:
        return [0.0, 0.0, 1.0, 1.0]
    x0, y0, x1, y1 = [float(value) for value in region]
    x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    if x1 - x0 < 0.05:
        x1 = min(1.0, x0 + 0.05)
    if y1 - y0 < 0.05:
        y1 = min(1.0, y0 + 0.05)
    return [x0, y0, x1, y1]


def text_keywords(text: str, vocab: Sequence[str]) -> List[str]:
    """Vocabulary nouns actually mentioned in the review (plurals included)."""
    if not text:
        return []
    lowered = text.lower()
    found: List[tuple] = []
    seen = set()
    for word in vocab:
        pattern = re.compile(r"\b" + re.escape(word) + r"(?:s|es)?\b")
        match = pattern.search(lowered)
        if match and word not in seen:
            found.append((match.start(), word))
            seen.add(word)
    for surface, word in CJK_CUES.items():
        if word in vocab and word not in seen and surface in text:
            found.append((text.find(surface), word))
            seen.add(word)
    found.sort(key=lambda item: item[0])
    return [word for _, word in found]


def _descriptor(pixels: np.ndarray) -> Dict[str, float]:
    flat = pixels.reshape(-1, 3)
    red, green, blue = flat[:, 0], flat[:, 1], flat[:, 2]
    high = flat.max(axis=1)
    low = flat.min(axis=1)
    saturation = np.where(high > 1e-6, (high - low) / np.maximum(high, 1e-6), 0.0)
    return {
        "white": float(((flat > 0.82).all(axis=1)).mean()),
        "blue": float(((blue > red + 0.06) & (blue > green + 0.02) & (blue > 0.45)).mean()),
        "brown": float(((red > green) & (green > blue) & (red > 0.25) & (red < 0.85) & (saturation > 0.15)).mean()),
        "yellow": float(((red > 0.60) & (green > 0.55) & (blue < 0.60) & (np.abs(red - green) < 0.22)).mean()),
        "pink": float(((red > 0.55) & (red > blue + 0.05) & (blue > green) & (saturation > 0.12)).mean()),
        "gray": float(((saturation < 0.12) & (high > 0.30) & (high < 0.80)).mean()),
        "green": float(((green > red + 0.04) & (green > blue + 0.04) & (green > 0.30)).mean()),
        "brightness": float(high.mean()),
        "saturation": float(saturation.mean()),
    }


def image_keywords(
    pixels: Optional[np.ndarray], region: Optional[Sequence[float]], vocab: Sequence[str], topk: int = 3
) -> List[str]:
    """Top-``k`` vocabulary words for the annotated region appearance."""
    if pixels is None or pixels.size == 0:
        return []
    x0, y0, x1, y1 = clamp_region(region)
    height, width = pixels.shape[:2]
    top, bottom = int(y0 * height), max(int(y1 * height), int(y0 * height) + 1)
    left, right = int(x0 * width), max(int(x1 * width), int(x0 * width) + 1)
    crop = pixels[top:bottom, left:right]
    if crop.size == 0:
        crop = pixels
    descriptor = _descriptor(crop)
    scored = []
    for category, profile in CATEGORY_PROFILES.items():
        score = sum(weight * descriptor.get(name, 0.0) for name, weight in profile.items())
        scored.append((score, category))
    scored.sort(key=lambda item: (-item[0], item[1]))

    words: List[str] = []
    vocab_set = set(vocab)
    for score, category in scored:
        if score <= 0.0:
            continue
        for word in CATEGORY_WORDS[category]:
            if word in vocab_set and word not in words:
                words.append(word)
                if len(words) >= topk:
                    return words
    return words


def merge_keywords(text_words: Sequence[str], image_words: Sequence[str], topk: int = 3) -> List[str]:
    """Text cues take at most ``topk-1`` slots so the image always contributes."""
    merged: List[str] = []
    for word in list(text_words)[: max(1, topk - 1)]:
        if word not in merged:
            merged.append(word)
    for word in image_words:
        if word not in merged:
            merged.append(word)
        if len(merged) >= topk:
            break
    return merged[:topk]


def extract_keywords(
    text: str,
    pixels: Optional[np.ndarray],
    region: Optional[Sequence[float]],
    vocab: Sequence[str],
    topk: int = 3,
) -> Dict[str, object]:
    """Keywords read from the review text and from the image region."""
    from_text = text_keywords(text, vocab)
    from_image = image_keywords(pixels, region, vocab, topk)
    merged = merge_keywords(from_text, from_image, topk)
    if from_text and from_image:
        source = "text+image"
    elif from_text:
        source = "text"
    elif from_image:
        source = "image"
    else:
        source = "unavailable"
    return {
        "keywords": merged,
        "keywords_text": from_text[:topk],
        "keywords_image": from_image,
        "keyword_source": source,
        "image_backend": "attribute",
    }
