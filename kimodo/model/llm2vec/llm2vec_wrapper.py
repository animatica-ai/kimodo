# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM2Vec encoder wrapper for Kimodo text conditioning."""

import os
from typing import Optional, Union

import numpy as np
import torch
from torch import nn


def _text_encoder_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Pick a device for dummy / lightweight encoders; map unavailable CUDA to CPU."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.device(device)
    if d.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return d

from .llm2vec import LLM2Vec


class LLM2VecEncoder:
    """LLM2Vec text embeddings."""

    def __init__(
        self,
        base_model_name_or_path: str,
        peft_model_name_or_path: str,
        dtype: str,
        llm_dim: int,
    ) -> None:
        torch_dtype = getattr(torch, dtype)
        self.llm_dim = llm_dim

        cache_dir = os.environ.get("HUGGINGFACE_CACHE_DIR")

        if "TEXT_ENCODERS_DIR" in os.environ:
            base_model_name_or_path = os.path.join(os.environ["TEXT_ENCODERS_DIR"], base_model_name_or_path)
            peft_model_name_or_path = os.path.join(os.environ["TEXT_ENCODERS_DIR"], peft_model_name_or_path)

        self.model = LLM2Vec.from_pretrained(
            base_model_name_or_path=base_model_name_or_path,
            peft_model_name_or_path=peft_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def to(self, device: torch.device):
        self.model = self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def get_device(self):
        return self.model.model.device

    def __call__(self, text: list[str] | str):
        is_string = False
        if isinstance(text, str):
            text = [text]
            is_string = True

        with torch.no_grad():
            encoded_text = self.model.encode(text, batch_size=len(text), show_progress_bar=False)

        assert len(encoded_text.shape)
        assert self.llm_dim == encoded_text.shape[-1]

        encoded_text = encoded_text[:, None]
        lengths = np.ones(len(encoded_text), dtype=int).tolist()

        if is_string:
            encoded_text = encoded_text[0]
            lengths = lengths[0]

        encoded_text = torch.tensor(encoded_text).to(self.get_device())
        return encoded_text, lengths


class DummyTextEncoder(nn.Module):
    """Zero-vector text encoder for constraint-only generation without LLM weights.

    Activated by setting TEXT_ENCODER_MODE=dummy. Returns zero embeddings
    of the correct shape (llm_dim=4096), which the model treats as
    unconditional (same as empty-text in classifier-free guidance training).

    This allows running Kimodo on GPUs with <17GB VRAM and without
    Llama-3 access, using only kinematic constraints for motion control.

    Subclasses ``nn.Module`` so ``Kimodo.to(device)`` moves outputs to the same
    device as the denoiser; defaults to CPU when PyTorch has no CUDA build.
    """

    def __init__(self, llm_dim: int = 4096, device: Optional[Union[str, torch.device]] = None) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        dev = _text_encoder_device(device)
        self.register_buffer("_device_anchor", torch.zeros((), device=dev), persistent=False)
        print(f"[Kimodo] Using DummyTextEncoder (zero embeddings, dim={llm_dim})")
        print("[Kimodo] Text prompts will be ignored. Use constraints for motion control.")

    def get_device(self):
        return self._device_anchor.device

    def forward(self, text: list[str] | str):
        is_string = False
        if isinstance(text, str):
            text = [text]
            is_string = True

        encoded_text = torch.zeros(len(text), 1, self.llm_dim, device=self.get_device())
        lengths = np.ones(len(text), dtype=int).tolist()

        if is_string:
            encoded_text = encoded_text[0]
            lengths = lengths[0]

        return encoded_text, lengths