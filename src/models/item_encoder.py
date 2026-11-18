"""Multimodal encoder with implicit-emotion enhancement and prototype prior."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.schema import TYPE_TO_ID
from ..data.vocab import load_vocab
from .dual_path import DualPathLocalizer
from .encoders import build_text_encoder, build_vision_encoder
from .keywords import KeywordExtractor


class PrototypePrior(nn.Module):
    """Prototype bank for T1/T2/T3 with a deliberately small learnable gate.

    The Method section of the paper does not spell this module out, but the
    ablation table contains ``w/o Prototype`` and the analysis states that the
    prototype gate stays below 0.01, so a minimal implementation is provided.
    """

    def __init__(self, dim: int = 128, num_prototypes: int = 3, max_gate: float = 0.01, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.max_gate = max_gate
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, dim) * 0.02)
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, item_vectors: torch.Tensor):
        if not self.enabled:
            return item_vectors, torch.zeros((), device=item_vectors.device)
        similarity = F.normalize(item_vectors, dim=-1) @ F.normalize(self.prototypes, dim=-1).t()
        weights = torch.softmax(similarity * 10.0, dim=-1)
        lam = torch.sigmoid(self.gate) * self.max_gate
        updated = item_vectors + lam * (weights @ self.prototypes)
        return updated, lam


class MetonymyEncoder(nn.Module):
    """f_t + dual-path localisation + Eq. (8) item vector + multi-task heads."""

    def __init__(self, cfg: Dict):
        super().__init__()
        settings = cfg["encoder"]
        self.text_dim = int(settings["text_dim"])
        self.vision_dim = int(settings["vision_dim"])
        self.item_dim = int(settings["item_dim"])
        self.grid = int(settings["patch_grid"])
        self.num_patches = self.grid * self.grid
        self.top_k = int(settings["top_k"])

        self.text_encoder = build_text_encoder(settings)
        self.vision_encoder = build_vision_encoder(settings)
        self.localizer = DualPathLocalizer(
            self.text_dim, self.vision_dim, int(settings["attention_dim"]), int(settings["num_heads"]), self.grid
        )
        self.fuse_projection = nn.Linear(self.text_dim + self.vision_dim, self.item_dim)
        self.type_head = nn.Linear(self.item_dim, len(TYPE_TO_ID))
        self.polarity_head = nn.Linear(self.item_dim, 3)
        self.consistency_head = nn.Linear(self.item_dim, 1)
        self.sentiment_head = nn.Linear(self.item_dim, 1)
        self.cl_text = nn.Linear(self.text_dim, self.item_dim)
        self.cl_vision = nn.Linear(self.vision_dim, self.item_dim)
        self.prototype = PrototypePrior(
            self.item_dim,
            int(cfg["prototype"]["num_prototypes"]),
            float(cfg["prototype"]["max_gate"]),
            enabled=bool(cfg["prototype"]["enabled"]),
        )
        self.keyword_extractor = KeywordExtractor(
            load_vocab(cfg["paths"]["keyword_vocab"]), int(settings["keyword_topk"])
        )
        self._keyword_vocab = load_vocab(cfg["paths"]["keyword_vocab"])

    def forward(self, batch: Dict, variant: Dict, return_keywords: bool = False) -> Dict:
        device = batch["images"].device
        use_text = bool(variant["use_text"])
        use_image = bool(variant["use_image"])
        fusion = str(variant["fusion"])

        if use_text:
            f_t = self.text_encoder(batch["texts"])
        else:
            f_t = torch.zeros(batch["images"].shape[0], self.text_dim, device=device)
        if use_image:
            patches = self.vision_encoder(batch["images"])
        else:
            patches = torch.zeros(batch["images"].shape[0], self.num_patches, self.vision_dim, device=device)

        if use_image:
            fused, cross, self_attn, gate_values, f_v, regions = self.localizer(f_t, patches, fusion, self.top_k)
        else:
            fused = torch.zeros(batch["images"].shape[0], self.num_patches, device=device)
            cross = self_attn = gate_values = fused
            f_v = torch.zeros(batch["images"].shape[0], self.vision_dim, device=device)
            regions = [[0.0, 0.0, 1.0, 1.0] for _ in range(batch["images"].shape[0])]

        raw_item = self.fuse_projection(torch.cat([f_t, f_v], dim=-1))
        prototype_enabled = bool(variant.get("prototype")) and self.prototype.enabled
        self.prototype.enabled = prototype_enabled
        item_vector, prototype_gate = self.prototype(raw_item)

        outputs = {
            "item_vector": item_vector,
            "raw_item_vector": raw_item,
            "type_logits": self.type_head(item_vector),
            "polarity_logits": self.polarity_head(item_vector),
            "consistency_logits": self.consistency_head(item_vector).squeeze(-1),
            "sentiment": torch.tanh(self.sentiment_head(item_vector)).squeeze(-1),
            "z_text": self.cl_text(f_t),
            "z_vision": self.cl_vision(f_v),
            "attention": fused,
            "cross_attention": cross,
            "self_attention": self_attn,
            "gate": gate_values,
            "region": regions,
            "prototype_gate": prototype_gate.detach(),
            "f_t": f_t,
            "f_v": f_v,
        }
        if return_keywords and use_image:
            with torch.no_grad():
                outputs["keywords"] = self.keyword_extractor(batch["images"], regions)
        else:
            outputs["keywords"] = [[] for _ in range(batch["images"].shape[0])]
        return outputs

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class EncoderRanker(nn.Module):
    """Discriminative variant (also used for EmoRec and every ablation)."""

    def __init__(self, cfg: Dict, variant: Dict):
        super().__init__()
        self.variant = variant
        self.encoder = MetonymyEncoder(cfg)
        self.score_head = nn.Linear(self.encoder.item_dim, 1)
        self.majority_score: Optional[float] = None

    def forward(self, batch: Dict, return_keywords: bool = False) -> Dict:
        outputs = self.encoder(batch, self.variant, return_keywords=return_keywords)
        outputs["score_logits"] = self.score_head(outputs["item_vector"]).squeeze(-1)
        return outputs

    def score(self, batch: Dict) -> torch.Tensor:
        if self.variant.get("majority"):
            value = self.majority_score if self.majority_score is not None else 0.654
            return torch.full((batch["images"].shape[0],), float(value))
        with torch.no_grad():
            return torch.sigmoid(self.forward(batch)["score_logits"])

    def predict_type(self, batch: Dict) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(batch)["type_logits"].argmax(dim=-1)

    def predict_polarity(self, batch: Dict) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(batch)["polarity_logits"].argmax(dim=-1) - 1
