"""Model factory: encoder + soft prefix + decoder (full generative pipeline)."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from ..checkpoint import load_checkpoint, save_checkpoint
from ..config import ensure_dir, project_path
from ..data.schema import ID_TO_TYPE
from .decoder import build_decoder
from .item_encoder import MetonymyEncoder
from .prompting import build_prompt, build_target, load_or_build_tokenizer, parse_score, score_target
from .soft_prefix import SoftPrefix


class GenerativeRecommender(nn.Module):
    """The complete model: item encoder, soft prefix, decoder and score head.

    ``score_head`` is the auxiliary discriminative head fitted in stage 1.  It
    provides the fast ranking signal used for the CPU evaluation protocol,
    while ``generate`` produces the three-step rationale + SCORE used for the
    BLEU-4 / ROUGE-L / human-evaluation table.
    """

    def __init__(self, cfg: Dict, variant: Dict, decoder_backend: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.variant = variant
        self.encoder = MetonymyEncoder(cfg)
        self.score_head = nn.Linear(self.encoder.item_dim, 1)
        self.item_dim = self.encoder.item_dim
        self.hidden_dim = int(cfg["decoder"]["hidden_dim"])
        self.soft_prefix = SoftPrefix(self.item_dim, self.hidden_dim) if variant.get("generative") else None
        effective = dict(cfg)
        if decoder_backend:
            effective["decoder"] = dict(cfg["decoder"], backend=decoder_backend)
        self.decoder_backend = effective["decoder"].get("backend", "tiny")
        self.tokenizer = load_or_build_tokenizer(cfg)
        self.decoder = build_decoder(effective, self.tokenizer)
        self.majority_score: Optional[float] = None

    # ---------------------------------------------------------------- prompts
    def prompt_texts(self, batch: Dict) -> List[str]:
        return [
            build_prompt(batch["queries"][i], batch["texts"][i]) for i in range(len(batch["texts"]))
        ]

    def target_texts(self, batch: Dict) -> List[str]:
        targets = []
        for index in range(len(batch["texts"])):
            emo_type = ID_TO_TYPE.get(int(batch["type_id"][index]), "null")
            polarity = int(batch["polarity"][index])
            score = score_target(batch["sample_id"][index], int(batch["label"][index]))
            targets.append(build_target(emo_type, polarity, score))
        return targets

    def encode_prompt(self, text: str) -> List[int]:
        if hasattr(self.decoder, "encode_text"):
            return self.decoder.encode_text(text)
        return self.tokenizer.encode(text)

    def encode_target(self, text: str) -> List[int]:
        if hasattr(self.decoder, "encode_text"):
            return self.decoder.encode_text(text, add_eos=True)
        return self.tokenizer.encode(text, add_eos=True)

    # ----------------------------------------------------------------- encode
    def encode(self, batch: Dict, return_keywords: bool = False) -> Dict:
        return self.encoder(batch, self.variant, return_keywords=return_keywords)

    def score_logits(self, batch: Dict) -> torch.Tensor:
        return self.score_head(self.encode(batch)["item_vector"]).squeeze(-1)

    def score(self, batch: Dict) -> torch.Tensor:
        if self.variant.get("majority"):
            value = self.majority_score if self.majority_score is not None else 0.654
            return torch.full((batch["images"].shape[0],), float(value))
        with torch.no_grad():
            return torch.sigmoid(self.score_logits(batch))

    def predict_type(self, batch: Dict) -> torch.Tensor:
        with torch.no_grad():
            return self.encode(batch)["type_logits"].argmax(dim=-1)

    def predict_polarity(self, batch: Dict) -> torch.Tensor:
        with torch.no_grad():
            return self.encode(batch)["polarity_logits"].argmax(dim=-1) - 1

    # --------------------------------------------------------------- generate
    def generate(self, batch: Dict, max_new_tokens: Optional[int] = None):
        outputs = self.encode(batch)
        if self.soft_prefix is not None:
            prefix = self.soft_prefix(outputs["item_vector"])
        else:
            prefix = torch.zeros(batch["images"].shape[0], 1, self.hidden_dim, device=batch["images"].device)
        prompts = self.prompt_texts(batch)
        prompt_ids = [self.encode_prompt(prompt) for prompt in prompts]
        texts = self.decoder.generate(prefix, prompt_ids, max_new_tokens=max_new_tokens)
        return texts, [parse_score(text) for text in texts]

    def generate_scores(self, batch: Dict, max_new_tokens: Optional[int] = None):
        texts, scores = self.generate(batch, max_new_tokens)
        values = torch.tensor([0.0 if s is None else s / 5.0 for s in scores], dtype=torch.float32)
        failures = [1.0 if s is None else 0.0 for s in scores]
        return values, texts, failures


def build_model(cfg: Dict, variant_name: str, decoder_backend: Optional[str] = None) -> GenerativeRecommender:
    from ..variants import get_variant

    return GenerativeRecommender(cfg, get_variant(variant_name), decoder_backend=decoder_backend)


def checkpoint_path(cfg: Dict, variant_name: str, seed: int, stage: str) -> str:
    return os.path.join(
        project_path(cfg, "output_dir"), "checkpoints", "{}_{}_seed{}_{}.pt".format(
            variant_name, "gen" if stage == "stage2" else "enc", seed, stage
        )
    )


def save_model(model: GenerativeRecommender, path: str) -> str:
    ensure_dir(os.path.dirname(path))
    save_checkpoint(
        {
            "state_dict": model.state_dict(),
            "variant": model.variant,
            "itos": model.tokenizer.itos,
        },
        path,
    )
    return path


def load_model(cfg: Dict, variant_name: str, path: str, decoder_backend: Optional[str] = None) -> GenerativeRecommender:
    from ..variants import get_variant

    model = GenerativeRecommender(cfg, get_variant(variant_name), decoder_backend=decoder_backend)
    checkpoint = load_checkpoint(path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def load_stage1_into(model: GenerativeRecommender, path: Optional[str]) -> bool:
    """Initialise the encoder / score head from a stage-1 checkpoint."""
    if not path or not os.path.exists(path):
        return False
    checkpoint = load_checkpoint(path)
    state = checkpoint.get("state_dict", checkpoint)
    own = model.state_dict()
    matched = {key: value for key, value in state.items() if key in own and own[key].shape == value.shape}
    model.load_state_dict(matched, strict=False)
    return bool(matched)


def build_eval_model(cfg: Dict, variant_name: str, seed: int) -> GenerativeRecommender:
    """Load the best available checkpoint (stage 2 if trained, else stage 1)."""
    from ..variants import get_variant

    model = GenerativeRecommender(cfg, get_variant(variant_name))
    stage2_path = checkpoint_path(cfg, variant_name, seed, "stage2")
    stage1_path = checkpoint_path(cfg, variant_name, seed, "stage1")
    if os.path.exists(stage2_path):
        checkpoint = load_checkpoint(stage2_path)
        model.load_state_dict(checkpoint["state_dict"])
        model.generative_trained = True
    elif os.path.exists(stage1_path):
        checkpoint = load_checkpoint(stage1_path)
        load_stage1_into(model, stage1_path)
        if checkpoint.get("majority") is not None:
            model.majority_score = float(checkpoint["majority"])
        model.generative_trained = False
    else:
        raise FileNotFoundError("no checkpoint for variant {} seed {}".format(variant_name, seed))
    model.eval()
    return model
