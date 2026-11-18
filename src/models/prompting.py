"""Three-step reasoning prompts, targets and the decoder vocabulary."""
from __future__ import annotations

import json
import os
import re
import zlib
from typing import Dict, List, Optional

from ..config import ensure_dir, project_path
from ..data.schema import read_jsonl
from ..data.text_utils import SimpleTokenizer

TYPE_DESCRIPTION = {
    "T1": "an action or outcome stands for the emotion",
    "T2": "an object stands for the emotion",
    "T3": "a scene stands for the emotion",
    "null": "no implicit emotional cue",
}

POLARITY_WORD = {-1: "negative", 0: "neutral", 1: "positive"}
POLARITY_TYPE = {-1: "dislike", 0: "neutral", 1: "preference"}

PROMPT_TEMPLATE = (
    "[User Query] {query}\n"
    "[Candidate Item] {text}\n"
    "[Task] Identify the implicit emotion of this merchant photo and decide whether to recommend it."
)

TARGET_TEMPLATE = (
    "Step 1: {cue}.\n"
    "Step 2: The implicit emotion is {polarity_word} ({emo_type}: {description}).\n"
    "Step 3: The item reflects a {preference} toward {topic}.\n"
    "SCORE: {score}"
)

STEP1_BY_TYPE = {
    "T1": "The action or outcome in the photo indirectly expresses how the visit felt",
    "T2": "The object in the photo indirectly expresses how the visit felt",
    "T3": "The scene around the photo indirectly expresses how the visit felt",
    "null": "The photo carries no clear implicit emotional cue",
}

SCAFFOLD_TEXTS = [
    PROMPT_TEMPLATE.format(query="user query", text="candidate item"),
    TARGET_TEMPLATE.format(
        cue=STEP1_BY_TYPE["T1"],
        polarity_word="positive",
        emo_type="T1",
        description=TYPE_DESCRIPTION["T1"],
        preference="preference",
        topic="the place",
        score=5,
    ),
    "Step 1 Step 2 Step 3 SCORE 1 2 3 4 5",
    "positive negative neutral preference dislike",
]


def build_prompt(query: str, text: str) -> str:
    return PROMPT_TEMPLATE.format(query=query or "", text=text or "")


def score_target(sample_id: str, label: int) -> int:
    """Deterministic 4/5 (positive) or 1/2 (negative) supervision."""
    parity = zlib.crc32(sample_id.encode("utf-8")) % 2
    if int(label) == 1:
        return 5 if parity == 0 else 4
    return 2 if parity == 0 else 1


def build_target(emo_type: str, polarity: int, score: int, topic: str = "the place") -> str:
    emo_type = emo_type if emo_type in TYPE_DESCRIPTION else "null"
    polarity = int(polarity)
    return TARGET_TEMPLATE.format(
        cue=STEP1_BY_TYPE[emo_type],
        polarity_word=POLARITY_WORD.get(polarity, "neutral"),
        emo_type=emo_type,
        description=TYPE_DESCRIPTION[emo_type],
        preference=POLARITY_TYPE.get(polarity, "neutral"),
        topic=topic,
        score=int(score),
    )


_SCORE_RE = re.compile(r"SCORE\s*[:\-]?\s*([1-5])", re.IGNORECASE)


def parse_score(text: str) -> Optional[int]:
    if not text:
        return None
    match = _SCORE_RE.search(text)
    if match:
        return int(match.group(1))
    digits = re.findall(r"([1-5])", text)
    return int(digits[-1]) if digits else None


def step3_segment(text: str) -> str:
    """Return the Step-3 explanation, used for the human-evaluation table."""
    if not text:
        return ""
    lowered = text.lower()
    start = lowered.find("step 3")
    if start < 0:
        return text.strip()
    end = lowered.find("score", start)
    segment = text[start:end if end > 0 else len(text)]
    return segment.split(":", 1)[-1].strip(" .\n")


def _decoder_corpus(cfg: Dict) -> List[str]:
    corpus: List[str] = list(SCAFFOLD_TEXTS)
    for emo_type in TYPE_DESCRIPTION:
        for polarity in (-1, 0, 1):
            for score in (1, 2, 4, 5):
                corpus.append(build_target(emo_type, polarity, score))
    yelp = read_jsonl(os.path.join(project_path(cfg, "yelp_dir"), "businesses.jsonl"))
    emometa = read_jsonl(os.path.join(project_path(cfg, "emometa_dir"), "pairs.jsonl"))
    for row in yelp:
        corpus.append(row.get("text", ""))
        corpus.append(row.get("query", ""))
    for row in emometa:
        corpus.append(row.get("text", ""))
    return corpus


def build_tokenizer(cfg: Dict) -> SimpleTokenizer:
    return SimpleTokenizer.build(_decoder_corpus(cfg))


def load_or_build_tokenizer(cfg: Dict) -> SimpleTokenizer:
    path = os.path.join(project_path(cfg, "data_root"), "processed", "decoder_vocab.json")
    if os.path.exists(path):
        try:
            return SimpleTokenizer.load(path)
        except (json.JSONDecodeError, KeyError):
            pass
    tokenizer = build_tokenizer(cfg)
    ensure_dir(os.path.dirname(path))
    tokenizer.save(path)
    return tokenizer
