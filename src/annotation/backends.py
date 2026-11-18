"""Annotation backends: offline deterministic backend and Qwen2.5-VL-72B API."""
from __future__ import annotations

import base64
import io
import json
import os
import re
import zlib
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from ..data.keywords import extract_keywords
from ..data.text_utils import lexicon_polarity
from ..data.vocab import load_vocab

TYPE_CUES = {
    "T1": {
        "english": ["plate", "cup", "tray", "fork", "finished", "cleared", "drained", "empty", "bite", "left"],
        "chinese": ["盘", "杯", "吃", "喝", "一口", "剩", "见底", "干净"],
    },
    "T2": {
        "english": ["box", "package", "bag", "wrapper", "carton", "wrapped", "packaging", "takeaway"],
        "chinese": ["盒", "包装", "袋", "拆", "礼盒", "标签"],
    },
    "T3": {
        "english": ["window", "sunlight", "curtain", "sunny", "light", "seat", "corner", "room"],
        "chinese": ["窗", "阳光", "窗帘", "光线", "角落", "窗台"],
    },
}


def _text_type_cues(text: str) -> Dict[str, int]:
    lowered = (text or "").lower()
    scores = {}
    for emo_type, cues in TYPE_CUES.items():
        hit = 0
        for word in cues["english"]:
            if word in lowered:
                hit += 1
        for char in cues["chinese"]:
            if char in text:
                hit += 1
        scores[emo_type] = hit
    return scores


def image_pixels(image_path: str, size: int = 224) -> Optional[np.ndarray]:
    if not os.path.exists(image_path):
        return None
    with Image.open(image_path) as image:
        array = np.asarray(image.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32)
    return array / 255.0


def _attention_region(image_path: str, grid: int = 14, topk: int = 20) -> List[float]:
    """Offline stand-in for the fused-attention localiser.

    Mirrors the three steps of the paper: score the 14x14 grid, take the
    ``topk`` patches (~10% of 196) without a probability threshold, and use the
    bounding box of the selected patches.  Real attention replaces the patch
    statistics once the encoder is trained.
    """
    pixels = image_pixels(image_path)
    if pixels is None:
        return [0.0, 0.0, 1.0, 1.0]
    cell = pixels.shape[0] // grid
    scores = np.zeros((grid, grid), dtype=np.float32)
    for gy in range(grid):
        for gx in range(grid):
            patch = pixels[gy * cell:(gy + 1) * cell, gx * cell:(gx + 1) * cell]
            scores[gy, gx] = float(patch.std()) + float(np.abs(patch - 0.75).mean())
    indices = np.argsort(-scores.ravel())[: min(topk, grid * grid)]
    rows, cols = indices // grid, indices % grid
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    return [round(x0 / grid, 4), round(y0 / grid, 4), round(x1 / grid, 4), round(y1 / grid, 4)]


class MockVLMBackend:
    """Deterministic offline annotator.

    It emulates the large-model pre-annotation stage: text cues propose the
    implicit-emotion type, the image is used for the localisation region, and a
    calibrated confidence decides whether the sample is accepted directly,
    sampled for review, or queued for full human review.
    """

    name = "mock"

    def __init__(self, vocab_path: Optional[str] = None, seed: int = 42):
        self.vocab = load_vocab(vocab_path) if vocab_path else []
        self.seed = seed

    def annotate(self, text: str, image_path: str, rating: Optional[float] = None) -> Dict:
        cues = _text_type_cues(text)
        best_type = max(cues, key=cues.get)
        best_hits = cues[best_type]
        total_hits = sum(cues.values())
        pol, pol_score = lexicon_polarity(text)
        if rating is not None and pol == 0:
            pol = 1 if float(rating) >= 4.0 else (-1 if float(rating) <= 2.5 else 0)

        region = _attention_region(image_path)
        keywords = extract_keywords(text, image_pixels(image_path), region, self.vocab, topk=3)

        confidence = 0.56
        if best_hits >= 2:
            confidence += 0.30
        elif best_hits == 1:
            confidence += 0.20
        else:
            confidence -= 0.06
        if abs(pol_score) >= 2:
            confidence += 0.08
        elif abs(pol_score) >= 1:
            confidence += 0.04
        if total_hits > 0 and best_hits / max(1, total_hits) >= 0.6:
            confidence += 0.04
        jitter = (zlib.crc32((text or "").encode("utf-8")) % 7) / 100.0 - 0.03
        confidence = float(min(0.99, max(0.45, confidence + jitter)))
        if best_hits == 0:
            best_type = "null"

        return {
            "type": best_type,
            "polarity": int(pol),
            "confidence": round(confidence, 4),
            "keywords": keywords["keywords"],
            "keywords_text": keywords["keywords_text"],
            "keywords_image": keywords["keywords_image"],
            "keyword_source": keywords["keyword_source"],
            "region": region,
            "region_source": "attention_top20",
            "rationale": "text cues: {}; polarity score {:.1f}".format(cues, pol_score),
        }


class QwenVLBackend:
    """Qwen2.5-VL-72B-Instruct via an OpenAI-compatible endpoint."""

    name = "qwen2.5-vl-72b"

    SYSTEM_PROMPT = (
        "You are an implicit-emotion annotator for merchant image-text pairs. "
        "Implicit emotion is expressed indirectly. Categories:\n"
        "T1: an action or outcome stands for the emotion (e.g. an empty cup means satisfaction).\n"
        "T2: an object stands for the emotion (e.g. an opened package means anticipation).\n"
        "T3: a scene stands for the emotion (e.g. a sunny windowsill means comfort).\n"
        "null: no implicit emotional cue.\n"
        "Return strict JSON with keys: type (T1|T2|T3|null), polarity (-1|0|1), "
        "confidence (0-1), keywords (list of 1-3 English nouns), "
        "region ([x0,y0,x1,y1] normalised), rationale (one sentence). "
        "Do not output markdown fences."
    )

    def __init__(self, model: str, api_base: str, api_key_env: str = "DASHSCOPE_API_KEY", timeout: int = 60):
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = timeout

    def _api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError(
                "missing API key: set environment variable {}".format(self.api_key_env)
            )
        return key

    def annotate(self, text: str, image_path: str, rating: Optional[float] = None) -> Dict:
        import urllib.request

        with open(image_path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Review text: {}\nRating: {}".format(text, rating)},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}},
                    ],
                },
            ],
        }
        request = urllib.request.Request(
            self.api_base + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self._api_key(),
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        return self._parse(content)

    @staticmethod
    def _parse(content: str) -> Dict:
        cleaned = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("annotation response is not JSON: {}".format(content[:200]))
        data = json.loads(match.group(0))
        return {
            "type": data.get("type", "null"),
            "polarity": int(data.get("polarity", 0)),
            "confidence": float(data.get("confidence", 0.5)),
            "keywords": list(data.get("keywords", []))[:3],
            "keywords_text": [],
            "keywords_image": [],
            "keyword_source": "vlm",
            "region": [float(x) for x in data.get("region", [0.0, 0.0, 1.0, 1.0])][:4],
            "region_source": "vlm",
            "rationale": str(data.get("rationale", ""))[:300],
        }


def build_backend(cfg: Dict, vocab_path: Optional[str] = None):
    settings = cfg["annotation"]
    if settings.get("backend") == "qwen2.5-vl-72b":
        return QwenVLBackend(settings["model"], settings["api_base"], settings["api_key_env"])
    return MockVLMBackend(vocab_path=vocab_path, seed=cfg.get("seed", 42))
