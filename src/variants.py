"""Ablation / baseline configuration switches.

Every switch maps directly to the descriptions in the Experiments section:

    text_only      text-only path
    image_only     visual-only path
    concat         direct concatenation of text and visual features
    gated_fusion   dual-path attention + gated fusion, no metonymy supervision
    full           complete model (encoder + MTL + prototype prior + decoder)
    wo_metonymy    without metonymy supervision
    wo_prototype   without prototype prior
    wo_gate        gating degraded to a fixed 0.5/0.5 mixture
    wo_image       visual branch removed
    wo_text        text branch removed
    majority       majority-class baseline
"""
from __future__ import annotations

from typing import Dict

VARIANTS: Dict[str, Dict[str, object]] = {
    "text_only": {
        "display": "TextOnly",
        "use_text": True,
        "use_image": False,
        "fusion": "mean",
        "metonymy": False,
        "prototype": False,
        "generative": False,
    },
    "image_only": {
        "display": "ImageOnly",
        "use_text": False,
        "use_image": True,
        "fusion": "mean",
        "metonymy": False,
        "prototype": False,
        "generative": False,
    },
    "concat": {
        "display": "Concat",
        "use_text": True,
        "use_image": True,
        "fusion": "mean",
        "metonymy": False,
        "prototype": False,
        "generative": False,
    },
    "gated_fusion": {
        "display": "GatedFusion",
        "use_text": True,
        "use_image": True,
        "fusion": "gate",
        "metonymy": False,
        "prototype": False,
        "generative": False,
    },
    "full": {
        "display": "Full",
        "use_text": True,
        "use_image": True,
        "fusion": "gate",
        "metonymy": True,
        "prototype": True,
        "generative": True,
    },
    "wo_metonymy": {
        "display": "w/o Metonymy",
        "use_text": True,
        "use_image": True,
        "fusion": "gate",
        "metonymy": False,
        "prototype": True,
        "generative": True,
    },
    "wo_prototype": {
        "display": "w/o Prototype",
        "use_text": True,
        "use_image": True,
        "fusion": "gate",
        "metonymy": True,
        "prototype": False,
        "generative": True,
    },
    "wo_gate": {
        "display": "w/o Gate",
        "use_text": True,
        "use_image": True,
        "fusion": "fixed",
        "metonymy": True,
        "prototype": True,
        "generative": True,
    },
    "wo_image": {
        "display": "w/o Image",
        "use_text": True,
        "use_image": False,
        "fusion": "mean",
        "metonymy": True,
        "prototype": True,
        "generative": True,
    },
    "wo_text": {
        "display": "w/o Text",
        "use_text": False,
        "use_image": True,
        "fusion": "mean",
        "metonymy": True,
        "prototype": True,
        "generative": True,
    },
    "majority": {
        "display": "Majority",
        "use_text": False,
        "use_image": False,
        "fusion": "mean",
        "metonymy": False,
        "prototype": False,
        "generative": False,
        "majority": True,
    },
    "emocrec": {
        "display": "EmoRec",
        "use_text": True,
        "use_image": True,
        "fusion": "gate",
        "metonymy": True,
        "prototype": True,
        "generative": False,
    },
}

TABLE1_ORDER = [
    "text_only",
    "image_only",
    "concat",
    "gated_fusion",
    "full",
    "wo_metonymy",
    "wo_prototype",
    "wo_gate",
    "wo_image",
    "wo_text",
    "majority",
]


def get_variant(name: str) -> Dict[str, object]:
    if name not in VARIANTS:
        raise KeyError("unknown variant {!r}; choose from {}".format(name, sorted(VARIANTS)))
    variant = dict(VARIANTS[name])
    variant["name"] = name
    return variant
