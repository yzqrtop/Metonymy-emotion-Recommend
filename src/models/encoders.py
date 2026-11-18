"""Text and vision encoders with mock (CPU) and paper (GPU) backends."""
from __future__ import annotations

import zlib
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.text_utils import tokenize


class MockTextEncoder(nn.Module):
    """Deterministic hashed bag-of-words encoder (512-d by default).

    Stands in for the frozen Qwen2.5-7B-Instruct + linear projection when no
    GPU / weights are available.  It is cached because texts repeat a lot.
    """

    def __init__(self, out_dim: int = 512, buckets: int = 8192, seed: int = 1234):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        table = torch.randn(buckets, out_dim, generator=generator) * 0.06
        self.register_buffer("table", table)
        self.out_dim = out_dim
        self.buckets = buckets
        self._cache: Dict[str, torch.Tensor] = {}

    def _encode(self, text: str) -> torch.Tensor:
        tokens = tokenize(text) or ["<empty>"]
        indices = [zlib.crc32(tok.encode("utf-8")) % self.buckets for tok in tokens]
        vector = self.table[torch.tensor(indices)].mean(dim=0)
        return F.normalize(vector, dim=-1)

    def forward(self, texts: List[str]) -> torch.Tensor:
        device = self.table.device
        rows = []
        for text in texts:
            if text not in self._cache:
                self._cache[text] = self._encode(text).detach()
            rows.append(self._cache[text])
        return torch.stack(rows, dim=0).to(device)


class MockVisionEncoder(nn.Module):
    """Patch statistics (colour / contrast / gradients) -> 196 visual tokens."""

    def __init__(self, out_dim: int = 768, patch_grid: int = 14, seed: int = 4321):
        super().__init__()
        self.out_dim = out_dim
        self.grid = patch_grid
        self.num_patches = patch_grid * patch_grid
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("projection", torch.randn(10, out_dim, generator=generator) * 0.12)
        self.position = nn.Parameter(torch.zeros(1, self.num_patches, out_dim))
        nn.init.normal_(self.position, std=0.01)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = images.shape
        ph, pw = height // self.grid, width // self.grid
        patches = images.unfold(2, ph, ph).unfold(3, pw, pw)  # B, 3, G, G, ph, pw
        mean = patches.mean(dim=(-1, -2))                      # B, 3, G, G
        std = patches.std(dim=(-1, -2))
        bright = patches.mean(dim=1).mean(dim=(-1, -2))        # B, G, G
        saturation = (patches.max(dim=1).values - patches.min(dim=1).values).mean(dim=(-1, -2))
        grad_x = patches.diff(dim=-1).abs().mean(dim=(-1, -2)).mean(dim=1)
        grad_y = patches.diff(dim=-2).abs().mean(dim=(-1, -2)).mean(dim=1)
        features = torch.stack(
            [
                mean[:, 0], mean[:, 1], mean[:, 2],
                std[:, 0], std[:, 1], std[:, 2],
                bright, saturation, grad_x, grad_y,
            ],
            dim=-1,
        )                                                      # B, G, G, 10
        features = features.reshape(batch, self.num_patches, 10)
        tokens = torch.matmul(features, self.projection) + self.position
        return tokens


class QwenTextEncoder(nn.Module):
    """Frozen Qwen2.5-7B-Instruct mean-pooled states projected to ``out_dim``."""

    def __init__(self, model_name: str, out_dim: int = 512, device: str = "cuda", dtype: str = "float16", max_length: int = 256):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer  # lazy import

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        torch_dtype = getattr(torch, dtype) if dtype else torch.float32
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.projection = nn.Linear(self.model.config.hidden_size, out_dim)
        self.device_name = device
        self.max_length = int(max_length)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, texts: List[str]) -> torch.Tensor:
        device = next(self.projection.parameters()).device
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1].float()
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.projection(pooled.to(device))


class ViTB16Encoder(nn.Module):
    """ViT-B/16 patch tokens (196 x 768) from transformers."""

    def __init__(self, model_name: str = "google/vit-base-patch16-224-in21k"):
        super().__init__()
        from transformers import ViTModel  # lazy import

        self.model = ViTModel.from_pretrained(model_name)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-1] != 224:
            images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.5, 0.5, 0.5], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=images.device).view(1, 3, 1, 1)
        with torch.no_grad():
            outputs = self.model(pixel_values=(images - mean) / std)
        return outputs.last_hidden_state[:, 1:, :]


def build_text_encoder(settings: Dict, device: str = "cpu") -> nn.Module:
    backend = settings.get("text_backend")
    if backend in ("qwen2.5", "qwen2.5-7b"):
        dtype = settings.get("dtype") or ("float32" if device == "cpu" else "float16")
        return QwenTextEncoder(
            settings["text_model"], settings["text_dim"], device=device, dtype=dtype,
            max_length=int(settings.get("text_max_length", 256)),
        )
    if backend != "mock":
        raise ValueError("Unknown text backend: {}".format(backend))
    return MockTextEncoder(settings["text_dim"])


def build_vision_encoder(settings: Dict) -> nn.Module:
    backend = settings.get("vision_backend")
    if backend == "vit_b16":
        return ViTB16Encoder(settings.get("vision_model", "google/vit-base-patch16-224-in21k"))
    if backend != "mock":
        raise ValueError("Unknown vision backend: {}".format(backend))
    return MockVisionEncoder(settings["vision_dim"], settings["patch_grid"])
