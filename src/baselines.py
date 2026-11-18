"""External baselines: MMGCN, LATTICE, SEALR, EmoRec and LLaMA-Rec.

MMGCN / LATTICE are reproduced as lightweight graph rankers over the Yelp
user-business graph.  SEALR / LLaMA-Rec reuse the tiny generative decoder
without soft-prefix injection (SEALR additionally injects sentiment labels).
EmoRec is the discriminative variant of our encoder (``variant="emocrec"``).
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ensure_dir, project_path
from .data.loaders import YelpDataset, load_image
from .data.schema import read_json, read_jsonl
from .data.text_utils import lexicon_polarity
from .models.decoder import TinyDecoder
from .models.encoders import MockTextEncoder, MockVisionEncoder
from .models.prompting import build_target, load_or_build_tokenizer, score_target, step3_segment

SENTIMENT_WORD = {-1: "negative", 0: "neutral", 1: "positive"}


class GraphRanker(nn.Module):
    """MMGCN / LATTICE-style content graph ranker with BPR training."""

    def __init__(self, cfg: Dict, kind: str = "mmgcn", factors: int = 64):
        super().__init__()
        self.kind = kind
        self.text_encoder = MockTextEncoder(512)
        self.vision_encoder = MockVisionEncoder(768)
        self.project = nn.Linear(512 + 768, factors)
        self.user_embedding = None
        self.item_embedding = nn.Parameter(torch.zeros(1, factors))
        self.neighbors = int(cfg["baseline"]["lattice_neighbors"])
        self.factors = factors
        self._item_vectors = None
        self._item_index: Dict[str, int] = {}

    def _features(self, records: Sequence[Dict], cfg: Dict) -> torch.Tensor:
        texts = [row["text"] for row in records]
        images = torch.stack(
            [load_image(os.path.join(project_path(cfg, "image_root"), row["image"]), 224) for row in records]
        )
        with torch.no_grad():
            text_features = self.text_encoder(texts)
            visual = self.vision_encoder(images).mean(dim=1)
        return F.normalize(torch.cat([text_features, visual], dim=-1), dim=-1)

    def fit(self, cfg: Dict, seed: int = 0) -> "GraphRanker":
        torch.manual_seed(seed)
        rng = random.Random(seed)
        records = read_jsonl(os.path.join(project_path(cfg, "yelp_dir"), "businesses.jsonl"))
        users = read_jsonl(os.path.join(project_path(cfg, "yelp_dir"), "users.jsonl"))
        splits = read_json(os.path.join(project_path(cfg, "yelp_dir"), "splits.json"), default={})
        train_groups = set(splits.get("train", []))
        business_records: Dict[str, Dict] = {}
        for row in records:
            business_records.setdefault(row["business_id"], row)
        business_ids = sorted(business_records)
        self._item_index = {bid: idx for idx, bid in enumerate(business_ids)}
        features = self._features([business_records[bid] for bid in business_ids], cfg)

        user_ids = sorted({user["user_id"] for user in users})
        user_index = {uid: idx for idx, uid in enumerate(user_ids)}
        self._user_index = user_index
        history = {
            user_index[user["user_id"]]: [
                self._item_index[bid] for bid in user.get("history", [])
                if bid in self._item_index and (not train_groups or bid in train_groups)
            ]
            for user in users
        }
        self.user_embedding = nn.Parameter(torch.randn(len(user_ids), self.factors) * 0.05)
        item_bias = nn.Parameter(torch.zeros(len(business_ids), self.factors))
        optimizer = torch.optim.Adam([*self.parameters(), item_bias], lr=float(cfg["baseline"]["bpr_lr"]))
        pairs = [(user, item) for user, items in history.items() for item in items]
        epochs = int(cfg["baseline"]["epochs"])

        for _ in range(epochs):
            rng.shuffle(pairs)
            item_vectors = self._item_representations(features) + item_bias
            for user, positive in pairs:
                negative = rng.randrange(len(business_ids))
                while negative in history[user]:
                    negative = rng.randrange(len(business_ids))
                user_vec = self.user_embedding[user]
                score = torch.dot(user_vec, item_vectors[positive] - item_vectors[negative])
                loss = -torch.log(torch.sigmoid(score) + 1e-8)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                item_vectors = self._item_representations(features) + item_bias

        with torch.no_grad():
            self._item_vectors = (self._item_representations(features) + item_bias).detach()
        self.eval()
        return self

    def _item_representations(self, features: torch.Tensor) -> torch.Tensor:
        projected = F.normalize(self.project(features), dim=-1)
        if self.kind == "lattice":
            similarity = projected @ projected.t()
            top = torch.topk(similarity, k=min(self.neighbors + 1, projected.shape[0]), dim=-1).indices
            neighbor_mean = projected[top].mean(dim=1)
            return F.normalize(0.5 * projected + 0.5 * neighbor_mean, dim=-1)
        # MMGCN: mean aggregation over the user-item graph (one propagation step)
        aggregated = 0.5 * projected + 0.5 * projected.mean(dim=0, keepdim=True)
        return F.normalize(aggregated, dim=-1)

    def score(self, batch: Dict) -> torch.Tensor:
        if self._item_vectors is None:
            return torch.zeros(len(batch["business_ids"]))
        values = []
        with torch.no_grad():
            for business, user in zip(batch["business_ids"], batch["user_ids"]):
                index = self._item_index.get(business, -1)
                if index < 0 or user not in self._user_index:
                    values.append(0.0)
                    continue
                user_vector = self.user_embedding[self._user_index[user]]
                values.append(float(torch.dot(self._item_vectors[index], user_vector)))
        return torch.sigmoid(torch.tensor(values, dtype=torch.float32))

    def score_candidates(self, user_id: str, records: Sequence[Dict]) -> np.ndarray:
        if self._item_vectors is None or user_id not in self._user_index:
            return np.zeros(len(records), dtype=np.float32)
        indices = [self._item_index[row["business_id"]] for row in records if row["business_id"] in self._item_index]
        if len(indices) != len(records):
            return np.zeros(len(records), dtype=np.float32)
        user_vector = self.user_embedding[self._user_index[user_id]].detach()
        return (self._item_vectors[indices] @ user_vector).detach().numpy()


class PromptOnlyGenerative(nn.Module):
    """SEALR / LLaMA-Rec: generation without soft-prefix injection."""

    def __init__(self, cfg: Dict, name: str):
        super().__init__()
        self.name = name
        self.cfg = cfg
        self.tokenizer = load_or_build_tokenizer(cfg)
        self.decoder = TinyDecoder(cfg, self.tokenizer)
        self.hidden_dim = int(cfg["decoder"]["hidden_dim"])
        self.variant = {"name": name, "display": name}

    def prompt_text(self, batch: Dict, index: int) -> str:
        query = batch["queries"][index]
        text = batch["texts"][index]
        if self.name == "SEALR":
            polarity = lexicon_polarity(text)[0]
            return "[User Query] {}\n[Sentiment Label] {}\n[Candidate Item] {}".format(
                query, SENTIMENT_WORD[polarity], text
            )
        return "[User Query] {}\n[Candidate Item] {}".format(query, text)

    def prompt_texts(self, batch: Dict) -> List[str]:
        return [self.prompt_text(batch, i) for i in range(len(batch["texts"]))]

    def generate(self, batch: Dict, max_new_tokens: int = None):
        prompts = self.prompt_texts(batch)
        prompt_ids = [self.tokenizer.encode(prompt) for prompt in prompts]
        zero_prefix = torch.zeros(len(prompts), 1, self.hidden_dim)
        texts = self.decoder.generate(zero_prefix, prompt_ids, max_new_tokens=max_new_tokens)
        from .models.prompting import parse_score

        return texts, [parse_score(text) for text in texts]

    def generate_scores(self, batch: Dict, max_new_tokens: int = None):
        texts, scores = self.generate(batch, max_new_tokens)
        return (
            torch.tensor([0.0 if s is None else s / 5.0 for s in scores]),
            texts,
            [1.0 if s is None else 0.0 for s in scores],
        )

    def score(self, batch: Dict) -> torch.Tensor:
        prompts = self.prompt_texts(batch)
        prompt_ids = [self.tokenizer.encode(prompt) for prompt in prompts]
        zero_prefix = torch.zeros(len(prompts), 1, self.hidden_dim)
        return self.decoder.score_expectation(zero_prefix, prompt_ids)


def train_prompt_baseline(cfg: Dict, name: str, seed: int = 0) -> Dict:
    torch.manual_seed(seed)
    dataset = YelpDataset(cfg, split="train")
    model = PromptOnlyGenerative(cfg, name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["stage2"]["lr"]))
    batch_size = int(cfg["stage2"]["batch_size"])
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    epochs = int(cfg["baseline"].get("epochs", 6))
    for epoch in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            items = [dataset[i] for i in indices[start:start + batch_size]]
            from .data.loaders import collate

            batch = collate(items)
            prompts = model.prompt_texts(batch)
            targets = [
                build_target(row["type"], int(row["polarity"]), score_target(row["sample_id"], int(row["label"])))
                for row in (dataset.records[i] for i in indices[start:start + batch_size])
            ]
            prompt_ids = [model.tokenizer.encode(prompt) for prompt in prompts]
            target_ids = [model.tokenizer.encode(target, add_eos=True) for target in targets]
            zero_prefix = torch.zeros(len(prompts), 1, model.hidden_dim)
            text_loss, score_loss, _ = model.decoder.loss_batch(zero_prefix, prompt_ids, target_ids)
            loss = text_loss + float(cfg["stage2"]["lambda_score"]) * score_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    output_dir = ensure_dir(os.path.join(project_path(cfg, "output_dir"), "checkpoints"))
    path = os.path.join(output_dir, "{}_seed{}_stage2.pt".format(name, seed))
    save_checkpoint({"state_dict": model.state_dict(), "variant": model.variant}, path)
    model.eval()
    return {"variant": name, "seed": seed, "iterations": epochs, "checkpoint": path}


def load_prompt_baseline(cfg: Dict, name: str, seed: int = 0):
    path = os.path.join(project_path(cfg, "output_dir"), "checkpoints", "{}_seed{}_stage2.pt".format(name, seed))
    model = PromptOnlyGenerative(cfg, name)
    if os.path.exists(path):
        checkpoint = load_checkpoint(path)
        model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
