"""Generative decoder: tiny causal LM for CPU, Qwen2.5-1.8B + LoRA for paper runs."""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ensure_dir, project_path
from ..data.text_utils import SimpleTokenizer
from .prompting import parse_score


class TinyCausalLM(nn.Module):
    """Small GPT-style decoder with an explicit causal mask."""

    def __init__(self, vocab_size: int, hidden_dim: int = 256, layers: int = 3, heads: int = 4,
                 ff_mult: int = 4, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position = nn.Embedding(max_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(hidden_dim))
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.max_len = max_len

    def forward(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        length = embeddings.shape[1]
        positions = torch.arange(length, device=embeddings.device).unsqueeze(0)
        hidden = embeddings + self.position(positions)
        causal = torch.triu(torch.ones(length, length, device=embeddings.device, dtype=torch.bool), diagonal=1)
        if attention_mask is None:
            padding = torch.zeros(embeddings.shape[0], length, dtype=torch.bool, device=embeddings.device)
        else:
            padding = attention_mask == 0
        hidden = self.encoder(hidden, mask=causal, src_key_padding_mask=padding)
        return hidden


class TinyDecoder(nn.Module):
    """Decoder wrapper with teacher-forced loss and greedy generation."""

    def __init__(self, cfg: Dict, tokenizer: SimpleTokenizer):
        super().__init__()
        settings = cfg["decoder"]
        self.tokenizer = tokenizer
        self.hidden_dim = int(settings["hidden_dim"])
        self.max_len = int(settings["max_len"])
        self.model = TinyCausalLM(
            len(tokenizer),
            self.hidden_dim,
            int(settings["layers"]),
            int(settings["heads"]),
            int(settings["ff_mult"]),
            float(settings["dropout"]),
            self.max_len,
        )
        self.digit_ids = [tokenizer.stoi[str(d)] for d in range(1, 6)]
        self.digit_index = {token_id: idx for idx, token_id in enumerate(self.digit_ids)}

    # ------------------------------------------------------------------ loss
    def _pad(self, sequences: Sequence[Sequence[int]], pad_id: int) -> torch.Tensor:
        width = max(1, max(len(seq) for seq in sequences))
        out = torch.full((len(sequences), width), pad_id, dtype=torch.long)
        for row, seq in enumerate(sequences):
            out[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return out

    def _left_pad(self, sequences: Sequence[Sequence[int]], pad_id: int) -> torch.Tensor:
        width = max(1, max(len(seq) for seq in sequences))
        out = torch.full((len(sequences), width), pad_id, dtype=torch.long)
        for row, seq in enumerate(sequences):
            out[row, width - len(seq):] = torch.tensor(seq, dtype=torch.long)
        return out

    def loss_batch(self, prefix: torch.Tensor, prompt_ids: Sequence[Sequence[int]], target_ids: Sequence[Sequence[int]]):
        pad_id = self.tokenizer.pad_id
        max_target = max(1, min(max(len(target) for target in target_ids), self.max_len - 2))
        target_ids = [list(target)[:max_target] for target in target_ids]
        budget = max(1, self.max_len - prefix.shape[1] - max_target)
        prompt_ids = [list(prompt)[-budget:] for prompt in prompt_ids]
        prompts = self._left_pad(prompt_ids, pad_id)
        targets = self._pad(target_ids, pad_id)
        teacher = torch.cat([targets[:, :-1], torch.full((targets.shape[0], 1), pad_id, dtype=torch.long)], dim=1)

        prompt_embeds = self.model.token_embedding(prompts)
        target_embeds = self.model.token_embedding(teacher)
        embeddings = torch.cat([prefix, prompt_embeds, target_embeds], dim=1)[:, : self.max_len]
        attention = torch.cat(
            [
                torch.ones(prefix.shape[0], prefix.shape[1], dtype=torch.bool),
                prompts != pad_id,
                teacher != pad_id,
            ],
            dim=1,
        )[:, : self.max_len]
        hidden = self.model(embeddings, attention)
        start = prefix.shape[1] + prompts.shape[1] - 1
        target_hidden = hidden[:, start: start + targets.shape[1]]
        logits = self.model.lm_head(target_hidden)

        pad_mask = targets == pad_id
        text_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=pad_id
        )

        score_positions = []
        score_targets = []
        score_indices = []
        for row, target in enumerate(target_ids):
            tokens = list(target)
            if tokens and tokens[-1] == self.tokenizer.eos_id:
                tokens = tokens[:-1]
            if not tokens or tokens[-1] not in self.digit_index:
                continue
            score_positions.append(row)
            score_targets.append(self.digit_index[tokens[-1]])
            score_indices.append(len(tokens) - 1)
        if score_positions:
            index = torch.tensor(score_positions, dtype=torch.long)
            positions = torch.tensor(score_indices, dtype=torch.long)
            digit_logits = logits[index, positions][:, self.digit_ids]
            score_loss = F.cross_entropy(digit_logits, torch.tensor(score_targets, dtype=torch.long))
        else:
            score_loss = torch.zeros((), device=prefix.device)
        return text_loss, score_loss, text_loss + score_loss

    # -------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(self, prefix: torch.Tensor, prompt_ids: Sequence[Sequence[int]], max_new_tokens: Optional[int] = None) -> List[str]:
        limit = int(max_new_tokens or self.max_len // 3)
        outputs: List[str] = []
        for row in range(prefix.shape[0]):
            prompt = list(prompt_ids[row])
            generated: List[int] = []
            for _ in range(limit):
                sequence = prompt + generated
                sequence = sequence[-self.max_len + 1:]
                embeds = self.model.token_embedding(torch.tensor([sequence], dtype=torch.long)) # 1,L,H
                embeds = torch.cat([prefix[row: row + 1], embeds], dim=1)[:, -self.max_len:]
                hidden = self.model(embeds)
                logits = self.model.lm_head(hidden[:, -1])
                token = int(torch.argmax(logits, dim=-1))
                if token == self.tokenizer.eos_id:
                    break
                generated.append(token)
            outputs.append(self.tokenizer.decode(generated))
        return outputs

    @torch.no_grad()
    def score_expectation(self, prefix: torch.Tensor, prompt_ids: Sequence[Sequence[int]]) -> torch.Tensor:
        """Probability-weighted SCORE in [0, 1] from one teacher-forced forward pass."""
        scaffold = [self.tokenizer.stoi.get(tok, self.tokenizer.unk_id) for tok in ("score", ":")]
        values = []
        for row in range(prefix.shape[0]):
            sequence = (list(prompt_ids[row]) + scaffold)[-self.max_len + 1:]
            embeds = self.model.token_embedding(torch.tensor([sequence], dtype=torch.long))
            embeds = torch.cat([prefix[row: row + 1], embeds], dim=1)[:, -self.max_len:]
            hidden = self.model(embeds)
            logits = self.model.lm_head(hidden[:, -1])[0, self.digit_ids]
            probabilities = torch.softmax(logits, dim=-1)
            expectation = float((probabilities * torch.arange(1, 6, dtype=torch.float32)).sum())
            values.append(expectation / 5.0)
        return torch.tensor(values, dtype=torch.float32)

    def save(self, path: str) -> None:
        from ..checkpoint import save_checkpoint

        save_checkpoint({"model": self.model.state_dict(), "itos": self.tokenizer.itos}, path)


class LoRALinear(nn.Module):
    """Standard low-rank residual adaptation around a frozen pretrained linear."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = self.lora_B(self.lora_A(self.dropout(inputs))) * self.scaling
        return self.base(inputs) + update.to(self.base.weight.dtype)


def inject_lora(model: nn.Module, rank: int, alpha: int, dropout: float) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    targets = {"q_proj", "k_proj", "v_proj", "o_proj"}
    for module in model.modules():
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank, alpha, dropout))


class QwenLoRADecoder(nn.Module):
    """Real Qwen2.5 causal LM with LoRA and multimodal soft-prefix inputs."""

    def __init__(self, cfg: Dict, tokenizer: Optional[SimpleTokenizer] = None):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        settings = cfg["decoder"]
        default_dtype = "float16" if torch.cuda.is_available() else "float32"
        dtype = getattr(torch, settings.get("dtype", default_dtype))
        self.tokenizer = AutoTokenizer.from_pretrained(settings["model_name"], trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            settings["model_name"], torch_dtype=dtype, trust_remote_code=True
        )
        rank = int(settings["lora_rank"])
        inject_lora(model, rank, rank * 2, float(settings["dropout"]))
        self.model = model
        self.hidden_dim = int(self.model.config.hidden_size)
        if int(settings["hidden_dim"]) != self.hidden_dim:
            raise ValueError("decoder.hidden_dim must equal the real Qwen hidden size {}".format(self.hidden_dim))
        self.max_len = int(settings["max_len"])
        self.max_new_tokens = int(settings.get("max_new_tokens", 96))
        self.pad_id = int(self.tokenizer.pad_token_id)
        self.eos_id = int(self.tokenizer.eos_token_id)
        self.digit_ids = []
        self.digit_index = {}
        for digit in range(1, 6):
            canonical = self.tokenizer.encode(" " + str(digit), add_special_tokens=False)
            alternatives = [canonical, self.tokenizer.encode(str(digit), add_special_tokens=False)]
            token_id = int(canonical[0] if len(canonical) == 1 else alternatives[1][0])
            self.digit_ids.append(token_id)
            for ids in alternatives:
                if len(ids) == 1:
                    self.digit_index[int(ids[0])] = digit - 1

    def encode_text(self, text: str, add_eos: bool = False) -> List[int]:
        ids = list(self.tokenizer.encode(text, add_special_tokens=True))
        if add_eos and (not ids or ids[-1] != self.eos_id):
            ids.append(self.eos_id)
        return ids

    def loss_batch(self, prefix: torch.Tensor, prompt_ids: Sequence[Sequence[int]], target_ids: Sequence[Sequence[int]]):
        batch_size, prefix_len = prefix.shape[:2]
        if prefix_len + 3 > self.max_len:
            raise ValueError("decoder.max_len is too small for the multimodal prefix")
        max_target = max(1, min(max(len(target) for target in target_ids), self.max_len - prefix_len - 2))
        rows = []
        for prompt, target in zip(prompt_ids, target_ids):
            target = list(target)[:max_target] or [self.eos_id]
            budget = max(1, self.max_len - prefix_len - len(target) + 1)
            prompt = list(prompt)[-budget:]
            sequence = prompt + target[:-1]
            rows.append((sequence, prompt, target))

        width = max(prefix_len + len(sequence) for sequence, _, _ in rows)
        input_ids = torch.full((batch_size, width), self.pad_id, dtype=torch.long, device=prefix.device)
        attention = torch.zeros((batch_size, width), dtype=torch.long, device=prefix.device)
        labels = torch.full((batch_size, width), -100, dtype=torch.long, device=prefix.device)
        starts = []
        for index, (sequence, prompt, target) in enumerate(rows):
            length = len(sequence)
            input_ids[index, prefix_len:prefix_len + length] = torch.tensor(sequence, dtype=torch.long)
            attention[index, :prefix_len + length] = 1
            start = prefix_len + len(prompt) - 1
            labels[index, start:start + len(target)] = torch.tensor(target, dtype=torch.long)
            starts.append(start)

        token_embeds = self.model.get_input_embeddings()(input_ids)
        embeds = torch.cat([prefix.to(token_embeds.dtype), token_embeds[:, prefix_len:]], dim=1)
        logits = self.model(
            inputs_embeds=embeds, attention_mask=attention, use_cache=False, return_dict=True
        ).logits
        text_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)

        score_logits = []
        score_targets = []
        for row, (_, _, target) in enumerate(rows):
            positions = [index for index, token in enumerate(target) if int(token) in self.digit_index]
            if not positions:
                continue
            position = positions[-1]
            score_logits.append(logits[row, starts[row] + position, self.digit_ids])
            score_targets.append(self.digit_index[int(target[position])])
        if score_logits:
            score_loss = F.cross_entropy(
                torch.stack(score_logits), torch.tensor(score_targets, dtype=torch.long, device=prefix.device)
            )
        else:
            score_loss = text_loss * 0.0
        return text_loss, score_loss, text_loss + score_loss

    @torch.no_grad()
    def generate(self, prefix: torch.Tensor, prompt_ids: Sequence[Sequence[int]], max_new_tokens: Optional[int] = None) -> List[str]:
        outputs = []
        limit = int(max_new_tokens or self.max_new_tokens)
        for row in range(prefix.shape[0]):
            prompt = list(prompt_ids[row])[-max(1, self.max_len - prefix.shape[1]):]
            inputs = torch.tensor([prompt], dtype=torch.long, device=prefix.device)
            prompt_embeds = self.model.get_input_embeddings()(inputs)
            embeds = torch.cat([prefix[row:row + 1].to(prompt_embeds.dtype), prompt_embeds], dim=1)
            attention = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
            generated = self.model.generate(
                inputs_embeds=embeds,
                attention_mask=attention,
                max_new_tokens=limit,
                do_sample=False,
                eos_token_id=self.eos_id,
                pad_token_id=self.pad_id,
            )
            outputs.append(self.tokenizer.decode(generated[0], skip_special_tokens=True))
        return outputs


def build_decoder(cfg: Dict, tokenizer: SimpleTokenizer):
    backend = cfg["decoder"].get("backend", "tiny")
    if backend in ("qwen2.5", "qwen2.5-0.5b", "qwen2.5-1.8b"):
        return QwenLoRADecoder(cfg, tokenizer)
    if backend != "tiny":
        raise ValueError("Unknown decoder backend: {}".format(backend))
    return TinyDecoder(cfg, tokenizer)
