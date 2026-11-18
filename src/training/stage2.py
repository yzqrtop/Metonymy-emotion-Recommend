"""Stage 2: soft-prefix + LoRA decoder training with the frozen encoder."""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..config import ensure_dir, project_path
from ..data.loaders import YelpDataset, collate
from ..models.factory import GenerativeRecommender, checkpoint_path, load_stage1_into, save_model
from ..variants import get_variant
from .stage1 import set_seed


def _freeze(module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def _val_loss(model: GenerativeRecommender, loader: DataLoader) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model.encode(batch)
            if model.soft_prefix is not None:
                prefix = model.soft_prefix(outputs["item_vector"])
            else:
                prefix = torch.zeros(batch["images"].shape[0], 1, model.hidden_dim)
            prompts = [model.encode_prompt(text) for text in model.prompt_texts(batch)]
            targets = [model.encode_target(text) for text in model.target_texts(batch)]
            text_loss, _, _ = model.decoder.loss_batch(prefix, prompts, targets)
            total += float(text_loss)
            count += 1
    return total / max(1, count)


def train_stage2(cfg: Dict, variant_name: str, seed: int = 0, stage1_checkpoint: Optional[str] = None) -> Dict:
    set_seed(seed)
    variant = get_variant(variant_name)
    if not variant.get("generative"):
        return {"variant": variant_name, "seed": seed, "skipped": True, "reason": "non-generative variant"}

    model = GenerativeRecommender(cfg, variant)
    stage1_path = stage1_checkpoint or checkpoint_path(cfg, variant_name, seed, "stage1")
    load_stage1_into(model, stage1_path)
    _freeze(model.encoder)
    _freeze(model.score_head)

    settings = cfg["stage2"]
    dataset = YelpDataset(cfg, split="train")
    val_data = YelpDataset(cfg, split="val", limit=240)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=int(settings["batch_size"]), shuffle=True,
                        collate_fn=collate, generator=generator)
    val_loader = DataLoader(val_data, batch_size=int(settings["batch_size"]), shuffle=False, collate_fn=collate)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        return {"variant": variant_name, "seed": seed, "skipped": True, "reason": "no trainable parameters"}
    optimizer = torch.optim.AdamW(trainable, lr=float(settings["lr"]), weight_decay=float(settings["weight_decay"]))

    max_epochs = int(settings["max_epochs"])
    patience = int(settings["patience"])
    lambda_score = float(settings["lambda_score"])
    best_loss = float("inf")
    best_state = None
    history = []
    stale = 0

    for epoch in range(max_epochs):
        model.train()
        model.encoder.eval()
        total = 0.0
        count = 0
        for batch in loader:
            with torch.no_grad():
                outputs = model.encode(batch)
            if model.soft_prefix is not None:
                prefix = model.soft_prefix(outputs["item_vector"])
            else:
                prefix = torch.zeros(batch["images"].shape[0], 1, model.hidden_dim)
            prompts = [model.encode_prompt(text) for text in model.prompt_texts(batch)]
            targets = [model.encode_target(text) for text in model.target_texts(batch)]
            text_loss, score_loss, _ = model.decoder.loss_batch(prefix, prompts, targets)
            loss = text_loss + lambda_score * score_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            total += float(loss)
            count += 1

        value = _val_loss(model, val_loader)
        history.append({"epoch": epoch + 1, "loss": round(total / max(1, count), 4), "val_loss": round(value, 4)})
        print("[stage2][{}][seed {}] epoch {}/{} loss={:.4f} val_loss={:.4f}".format(
            variant_name, seed, epoch + 1, max_epochs, total / max(1, count), value))
        if value < best_loss:
            best_loss = value
            best_state = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    path = checkpoint_path(cfg, variant_name, seed, "stage2")
    save_model(model, path)
    return {
        "variant": variant_name,
        "seed": seed,
        "iterations": len(history),
        "best_val_loss": round(best_loss, 4),
        "history": history,
        "checkpoint": path,
        "trainable_parameters": sum(p.numel() for p in trainable),
    }
