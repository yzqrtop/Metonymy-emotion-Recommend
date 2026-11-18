"""Keyword extraction from the localised region (CLIP-style in concept).

The paper matches the pooled region vector of the top-``k`` patches against the
CLIP text embeddings of a fixed vocabulary.  When CLIP weights are available we
follow that path exactly; otherwise a deterministic region-appearance
descriptor is used and reported as the ``attribute`` backend.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from ..data.keywords import image_keywords, text_keywords


class KeywordExtractor:
    def __init__(self, vocab: Sequence[str], topk: int = 3, backend: str = "attribute"):
        self.vocab = list(vocab)
        self.topk = topk
        self.backend = backend
        self._clip = None

    def _clip_model(self):
        if self._clip is None:
            from transformers import CLIPModel, CLIPProcessor

            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model.eval()
            self._clip = (model, processor)
        return self._clip

    def __call__(self, images: torch.Tensor, regions: Sequence[Sequence[float]], backend: Optional[str] = None) -> List[List[str]]:
        backend = backend or self.backend
        if backend == "clip":
            return self._clip_keywords(images, regions)
        return self._attribute_keywords(images, regions)

    def text_keywords(self, text: str) -> List[str]:
        """Vocabulary nouns mentioned in the review text."""
        return text_keywords(text, self.vocab)

    def _crops(self, images: torch.Tensor, regions: Sequence[Sequence[float]]):
        crops = []
        height, width = images.shape[-2:]
        for image, region in zip(images, regions):
            x0, y0, x1, y1 = [float(v) for v in region]
            x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
            y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
            if x1 - x0 < 0.05:
                x1 = min(1.0, x0 + 0.05)
            if y1 - y0 < 0.05:
                y1 = min(1.0, y0 + 0.05)
            crops.append(image[:, int(y0 * height):max(int(y1 * height), int(y0 * height) + 1),
                                 int(x0 * width):max(int(x1 * width), int(x0 * width) + 1)])
        return crops

    def _attribute_keywords(self, images: torch.Tensor, regions: Sequence[Sequence[float]]) -> List[List[str]]:
        results = []
        for image, region in zip(images, regions):
            pixels = image.detach().cpu().permute(1, 2, 0).numpy()
            results.append(image_keywords(pixels, region, self.vocab, self.topk))
        return results

    def _clip_keywords(self, images: torch.Tensor, regions: Sequence[Sequence[float]]) -> List[List[str]]:
        model, processor = self._clip_model()
        crops = self._crops(images, regions)
        results = []
        text_inputs = processor(text=self.vocab, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        for crop in crops:
            array = (crop.permute(1, 2, 0).numpy() * 255).astype("uint8")
            inputs = processor(images=array, return_tensors="pt")
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).squeeze(0)
            top = torch.topk(similarity, k=min(self.topk, len(self.vocab))).indices.tolist()
            results.append([self.vocab[i] for i in top])
        return results


def region_from_indices(indices: torch.Tensor, grid: int) -> List[List[float]]:
    """Convert patch indices to one bounding box per sample."""
    boxes = []
    for row in indices.tolist():
        rows = [idx // grid for idx in row]
        cols = [idx % grid for idx in row]
        boxes.append(
            [
                round(min(cols) / float(grid), 4),
                round(min(rows) / float(grid), 4),
                round((max(cols) + 1) / float(grid), 4),
                round((max(rows) + 1) / float(grid), 4),
            ]
        )
    return boxes
