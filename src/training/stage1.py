"""Stage 1: multi-task training of the multimodal item encoder (Eq. 9)."""
from __future__ import annotations

import os
import random
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from ..config import project_path
from ..checkpoint import save_checkpoint
from ..data.loaders import EmoMetaDataset, YelpDataset, collate
from ..models.factory import checkpoint_path
from ..models.item_encoder import EncoderRanker
from ..models.losses import mtl_loss, ranking_loss
from ..variants import get_variant


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def auc_score(labels, scores) -> float:
    labels = np.asarray(labels, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    positives = labels.sum()
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float32)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float32)
    # average ranks for ties
    sorted_scores = scores[order]
    start = 0
    for index in range(1, len(sorted_scores) + 1):
        if index == len(sorted_scores) or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def _evaluate_auc(model: EncoderRanker, loader: DataLoader) -> float:
    model.eval()
    labels, scores = [], []
    with torch.no_grad():
        for batch in loader:
            labels.extend(batch["label"].tolist())
            scores.extend(model.score(batch).tolist())
    return auc_score(labels, scores)


def train_stage1(cfg: Dict, variant_name: str, seed: int = 0) -> Dict:
    set_seed(seed)
    variant = get_variant(variant_name)
    settings = cfg["stage1"]
    batch_size = int(settings["batch_size"])

    train_yelp = YelpDataset(cfg, split="train")
    emometa = EmoMetaDataset(cfg, split="train", limit=int(settings.get("emometa_subset", 0)) or None)
    train_data = ConcatDataset([train_yelp, emometa])
    val_data = YelpDataset(cfg, split="val")

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=generator
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = EncoderRanker(cfg, variant)
    if variant.get("majority"):
        positive = sum(row.get("label", 0) for row in train_yelp.records) / max(1, len(train_yelp.records))
        model.majority_score = float(positive)
        path = checkpoint_path(cfg, variant_name, seed, "stage1")
        save_checkpoint({"state_dict": model.state_dict(), "variant": variant, "majority": model.majority_score}, path)
        return {"variant": variant_name, "seed": seed, "iterations": 0, "best_val_auc": 0.5,
                "history": [], "checkpoint": path}

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["lr"]), weight_decay=float(settings["weight_decay"])
    )
    max_epochs = int(settings["max_epochs"])
    patience = int(settings["patience"])
    select_by = settings.get("select_by", "val_auc")
    use_metonymy = bool(variant.get("metonymy"))

    best_auc = 0.0
    best_state = None
    best_epoch = 0
    history = []
    stale = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in train_loader:
            outputs = model(batch)
            loss_parts = mtl_loss(outputs, batch, cfg, use_metonymy=use_metonymy)
            loss = loss_parts["total"] + 0.5 * ranking_loss(outputs["score_logits"], batch["label"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss)
            seen += 1
        val_auc = _evaluate_auc(model, val_loader)
        history.append({"epoch": epoch + 1, "loss": round(total_loss / max(1, seen), 4), "val_auc": round(val_auc, 4)})
        print("[stage1][{}][seed {}] epoch {}/{} loss={:.4f} val_auc={:.4f}".format(
            variant_name, seed, epoch + 1, max_epochs, total_loss / max(1, seen), val_auc))
        if select_by == "last":
            best_auc = max(best_auc, val_auc)
            best_epoch = epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        elif val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    path = checkpoint_path(cfg, variant_name, seed, "stage1")
    save_checkpoint({"state_dict": model.state_dict(), "variant": variant, "best_val_auc": best_auc}, path)
    return {
        "variant": variant_name,
        "seed": seed,
        "iterations": len(history),
        "best_epoch": best_epoch,
        "best_val_auc": round(best_auc, 4),
        "history": history,
        "checkpoint": path,
    }
