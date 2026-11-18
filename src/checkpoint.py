"""Checkpoint I/O that tolerates non-ASCII (Chinese) paths on Windows.

``torch.save`` / ``torch.load`` go through a C++ file writer that fails on
Windows paths containing Chinese characters.  Serialising through an in-memory
buffer and using Python's ``open`` sidesteps the issue.
"""
from __future__ import annotations

import io
import os
from typing import Any

import torch


def save_checkpoint(payload: Any, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    with open(path, "wb") as handle:
        handle.write(buffer.getvalue())
    return path


def load_checkpoint(path: str, map_location: str = "cpu") -> Any:
    with open(path, "rb") as handle:
        return torch.load(io.BytesIO(handle.read()), map_location=map_location)
