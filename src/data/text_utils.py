"""Tokenisation, sentiment lexicon and the small decoder vocabulary."""
from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Sequence

_ASCII_WORD = re.compile(r"[a-z0-9']+")
_CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

POS_WORDS = {
    "amazing", "best", "bright", "clean", "cozy", "comfortable", "crisp", "delicious",
    "enjoyed", "fresh", "friendly", "full", "generous", "great", "happy", "helpful",
    "inviting", "love", "loved", "lovely", "perfect", "pleasant", "quiet", "recommend",
    "satisfied", "tasty", "warm", "wonderful", "fair", "quick",
}
NEG_WORDS = {
    "awful", "bland", "broken", "burnt", "cold", "damaged", "dirty", "disappointing",
    "empty", "greasy", "leak", "noisy", "overpriced", "regret", "rude", "slow",
    "soggy", "stale", "terrible", "unhappy", "waiting", "worst", "late", "missed",
}
CJK_POS = set("好香美赞鲜暖净快足")
CJK_NEG = set("差脏慢难苦冷贵臭破旧少")


def tokenize(text: str) -> List[str]:
    """Whitespace-free tokenizer: ASCII words plus individual CJK characters."""
    text = (text or "").lower()
    tokens = _ASCII_WORD.findall(text)
    tokens.extend(_CJK_CHAR.findall(text))
    return tokens


def lexicon_polarity(text: str):
    """Return ``(polarity, score)`` with polarity in {-1, 0, 1}."""
    tokens = tokenize(text)
    pos = sum(1 for tok in tokens if tok in POS_WORDS or tok in CJK_POS)
    neg = sum(1 for tok in tokens if tok in NEG_WORDS or tok in CJK_NEG)
    score = float(pos - neg)
    if score > 0:
        return 1, score
    if score < 0:
        return -1, score
    return 0, 0.0


class SimpleTokenizer:
    """Word-level tokenizer used by the (tiny or Qwen) decoder."""

    def __init__(self, itos: Sequence[str]):
        self.itos = list(itos)
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD_TOKEN]
        self.bos_id = self.stoi[BOS_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = [self.stoi.get(tok, self.unk_id) for tok in tokenize(text)]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        words = []
        for idx in ids:
            if idx in (self.pad_id, self.bos_id, self.eos_id):
                continue
            if 0 <= idx < len(self.itos):
                words.append(self.itos[idx])
        return " ".join(words)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"itos": self.itos}, handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(json.load(handle)["itos"])

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 1) -> "SimpleTokenizer":
        freq: Dict[str, int] = {}
        for text in texts:
            for tok in tokenize(text):
                freq[tok] = freq.get(tok, 0) + 1
        words = [tok for tok, count in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])) if count >= min_freq]
        return cls(SPECIAL_TOKENS + words)
