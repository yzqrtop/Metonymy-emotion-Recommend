"""Losses of the multi-task stage (Eq. 9 in the Method section)."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def info_nce(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE between text and visual item embeddings."""
    if z_a.shape[0] < 2:
        return torch.zeros((), device=z_a.device)
    a = F.normalize(z_a, dim=-1)
    b = F.normalize(z_b, dim=-1)
    logits = a @ b.t() / temperature
    labels = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def mtl_loss(outputs: Dict, batch: Dict, cfg: Dict, use_metonymy: bool = True) -> Dict[str, torch.Tensor]:
    """L_type + L_CL + L_consis + L_sent with the paper weights."""
    weights = cfg["loss"]
    device = outputs["type_logits"].device

    valid_types = batch["type_id"] >= 0
    if valid_types.any():
        type_loss = F.cross_entropy(outputs["type_logits"][valid_types], batch["type_id"][valid_types])
    else:
        type_loss = outputs["type_logits"].sum() * 0.0
    cl_loss = info_nce(outputs["z_text"], outputs["z_vision"], float(weights.get("temperature", 0.07)))

    polarity = batch["polarity"]
    class_weights = weights.get("polarity_weights")
    if class_weights:
        polarity_loss = F.cross_entropy(
            outputs["polarity_logits"], batch["polarity_id"],
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        )
    else:
        polarity_loss = F.cross_entropy(outputs["polarity_logits"], batch["polarity_id"])
    agree_target = ((batch["text_polarity"] == polarity.long()) & (polarity != 0)).float()
    consis_loss = F.binary_cross_entropy_with_logits(outputs["consistency_logits"], agree_target)
    sent_loss = F.mse_loss(outputs["sentiment"], polarity / 1.0)

    if use_metonymy:
        total = (
            float(weights["lambda_type"]) * type_loss
            + float(weights["lambda_cl"]) * cl_loss
            + float(weights["lambda_consis"]) * consis_loss
            + float(weights["lambda_sent"]) * sent_loss
            + float(weights.get("lambda_pol", 0.3)) * polarity_loss
        )
    else:
        # ablations without metonymy supervision keep only the cross-modal term
        total = float(weights["lambda_cl"]) * cl_loss

    return {
        "total": total,
        "type": type_loss.detach(),
        "polarity": polarity_loss.detach(),
        "cl": cl_loss.detach(),
        "consis": consis_loss.detach(),
        "sent": sent_loss.detach(),
        "zero": torch.zeros((), device=device),
    }


def ranking_loss(score_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(score_logits, labels)
