"""Small end-to-end mechanics test using only real data and pretrained weights."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import ensure_dir, load_config, project_path  # noqa: E402
from src.data.loaders import EmoMetaDataset, collate, load_taobao  # noqa: E402
from src.evaluation.transfer import evaluate_transfer  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.models.losses import mtl_loss, ranking_loss  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/real_cpu.yaml")
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.set_num_threads(args.threads)
    output_dir = ensure_dir(os.path.join(project_path(cfg, "output_dir"), "logs"))
    report_path = os.path.join(output_dir, "real_smoke_test.json")

    try:
        dataset = EmoMetaDataset(cfg, split="train", limit=1)
        if not dataset.records:
            raise RuntimeError("No processed real EmoMeta record is available")
        batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)))
        record = dataset.records[0]
        image_path = os.path.join(project_path(cfg, "image_root"), record["image"])
        with Image.open(image_path) as image:
            image_size = list(image.size)
            image_format = image.format

        model = build_model(cfg, "full")
        encoder_trainable = [p for p in list(model.encoder.parameters()) + list(model.score_head.parameters()) if p.requires_grad]
        decoder_trainable = [p for p in list(model.soft_prefix.parameters()) + list(model.decoder.parameters()) if p.requires_grad]
        if not encoder_trainable or not decoder_trainable:
            raise RuntimeError("Expected trainable encoder and LoRA/soft-prefix parameters")

        model.train()
        outputs = model.encode(batch)
        score_logits = model.score_head(outputs["item_vector"]).squeeze(-1)
        loss_parts = mtl_loss(outputs, batch, cfg, use_metonymy=True)
        encoder_loss = loss_parts["total"] + 0.5 * ranking_loss(score_logits, batch["label"])
        encoder_optimizer = torch.optim.AdamW(encoder_trainable, lr=float(cfg["stage1"]["lr"]))
        encoder_optimizer.zero_grad()
        encoder_loss.backward()
        encoder_gradient = any(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in encoder_trainable)
        if not encoder_gradient:
            raise RuntimeError("No finite encoder gradient was produced")
        encoder_optimizer.step()

        with torch.no_grad():
            item_vector = model.encode(batch)["item_vector"]
        prefix = model.soft_prefix(item_vector)
        prompts = [model.encode_prompt(text) for text in model.prompt_texts(batch)]
        targets = [model.encode_target(text) for text in model.target_texts(batch)]
        if not any(int(token) in model.decoder.digit_index for token in targets[0]):
            raise RuntimeError("Real Qwen tokenization did not expose the target SCORE digit")
        text_loss, score_loss, decoder_total = model.decoder.loss_batch(prefix, prompts, targets)
        decoder_optimizer = torch.optim.AdamW(decoder_trainable, lr=float(cfg["stage2"]["lr"]))
        decoder_optimizer.zero_grad()
        decoder_total.backward()
        decoder_gradient = any(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in decoder_trainable)
        if not decoder_gradient:
            raise RuntimeError("No finite decoder/soft-prefix gradient was produced")
        decoder_optimizer.step()

        model.eval()
        generated, scores = model.generate(batch, max_new_tokens=args.new_tokens)
        taobao = load_taobao(cfg)
        embeddings = taobao["embeddings"]
        if embeddings.ndim != 2 or embeddings.shape[1] != 128 or not np.isfinite(embeddings).all():
            raise RuntimeError("Invalid official Taobao-MM SCL embedding subset")
        transfer = evaluate_transfer(cfg, seed=0)

        report = {
            "status": "passed",
            "scope": "Real-data/model mechanics test; not a paper benchmark or real Yelp recommendation result",
            "data": {
                "emometa_sample_id": record["sample_id"],
                "emometa_image": image_path,
                "image_format": image_format,
                "image_size": image_size,
                "official_emotion": record["emotion"],
                "official_metaphor_occurrence": record["official_metaphor_occurrence"],
                "taobao_embedding_shape": list(embeddings.shape),
                "taobao_users": len(taobao["interactions"]),
                "taobao_source": taobao["meta"]["source"],
            },
            "models": {
                "text_encoder": type(model.encoder.text_encoder).__name__,
                "text_model": cfg["encoder"]["text_model"],
                "vision_encoder": type(model.encoder.vision_encoder).__name__,
                "vision_model": cfg["encoder"]["vision_model"],
                "decoder": type(model.decoder).__name__,
                "decoder_model": cfg["decoder"]["model_name"],
                "prefix_width": int(prefix.shape[-1]),
                "encoder_trainable_parameters": sum(p.numel() for p in encoder_trainable),
                "decoder_lora_and_prefix_trainable_parameters": sum(p.numel() for p in decoder_trainable),
            },
            "training": {
                "encoder_loss": float(encoder_loss.detach()),
                "encoder_backward_and_step": True,
                "decoder_text_loss": float(text_loss.detach()),
                "decoder_score_loss": float(score_loss.detach()),
                "decoder_total_loss": float(decoder_total.detach()),
                "decoder_backward_and_step": True,
                "generated_text": generated[0],
                "parsed_score": scores[0],
                "generated_tokens_limit": args.new_tokens,
            },
            "transfer": transfer,
        }
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("[real_smoke_test] report: " + report_path)
    except Exception as error:
        failure = {"status": "failed", "error": str(error), "traceback": traceback.format_exc()}
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(failure, handle, ensure_ascii=False, indent=2)
        raise


if __name__ == "__main__":
    main()
