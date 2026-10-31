# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Gemma-only FP32 computation without converting frozen weight storage.

Install after adapter injection and before FSDP wrapping. Parameters and their
names are preserved; no global PyTorch functions or dispatch tables are changed.
"""

from functools import wraps
from types import MethodType

import torch
from torch.nn import functional as F
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel, Gemma3TextScaledWordEmbedding


class FP32Linear(torch.nn.Linear):
    compute_dtype = torch.float32

    def forward(self, inputs):
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return F.linear(inputs.float(), self.weight.float(), None if self.bias is None else self.bias.float())


class FP32GemmaEmbedding(Gemma3TextScaledWordEmbedding):
    def forward(self, input_ids):
        values = F.embedding(
            input_ids,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # Use the original scalar, not the buffer that model.to(bfloat16) rounds.
        return values.float() * self.scalar_embed_scale


@wraps(Gemma3TextModel.forward)
def _fp32_gemma_forward(self, *args, **kwargs):
    with torch.autocast(device_type=self.embed_tokens.weight.device.type, enabled=False):
        return Gemma3TextModel.forward(self, *args, **kwargs)


def install_fp32_gemma_compute(gemma):
    """Keep BF16 base weights while protecting activations from autocast.

    Reapply after adding an adapter to include the new LoRA linear layers.
    FSDP must preserve parameter dtypes and must not recast layer inputs to BF16.
    """
    language_model = gemma.model.language_model
    if not isinstance(language_model, Gemma3TextModel):
        raise TypeError("FP32 Gemma computation requires Gemma3TextModel")
    # Transformers registers output capture by the class used at construction.
    # Preserve that class so output_hidden_states=True still captures every layer
    # for LTX video conditioning. Wrap only this instance's forward computation.
    language_model.forward = MethodType(_fp32_gemma_forward, language_model)
    embedding = language_model.embed_tokens
    if not isinstance(embedding, Gemma3TextScaledWordEmbedding):
        raise TypeError("FP32 Gemma computation requires scaled Gemma embeddings")
    embedding.__class__ = FP32GemmaEmbedding
    for name, parameter in language_model.named_parameters():
        if ".lora_" in name and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
    for module in language_model.modules():
        if type(module) is torch.nn.Linear:
            module.__class__ = FP32Linear
    if type(gemma.lm_head) in (torch.nn.Linear, FP32Linear):
        gemma.lm_head.__class__ = FP32Linear
    else:
        # ChunkedLogprobHead owns the same parameter but bypasses nn.Linear.
        from ..training.logprob_head import ChunkedLogprobHead

        if not isinstance(gemma.lm_head, ChunkedLogprobHead):
            raise TypeError("Unsupported Gemma LM head")
        gemma.lm_head.compute_dtype = torch.float32
