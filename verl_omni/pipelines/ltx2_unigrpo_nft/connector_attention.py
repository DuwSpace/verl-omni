# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Differentiable connector rotation matching native LTX-2.3 materialization."""

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn


def native_connector_rope(value, frequencies, rope_type, head_dim):
    """Keep multiply/addcmul rounding in projection dtype, as native LTX does.

    Functional updates preserve autograd without modifying views in place.
    In particular, casting only the final FP32 rotation result is not equivalent.
    """
    cos, sin = (item.to(value.dtype) for item in frequencies)
    if rope_type == "interleaved":
        real, imag = value.unflatten(2, (-1, 2)).unbind(-1)
        rotated = torch.stack((-imag, real), dim=-1).flatten(2)
        return value * cos + rotated * sin
    if rope_type != "split":
        raise ValueError(f"Unsupported connector RoPE type: {rope_type}")
    batch, length, _ = value.shape
    heads = value.shape[-1] // head_dim
    split = value.reshape(batch, length, heads, 2, head_dim // 2).transpose(1, 2)
    first, second = split.unbind(-2)
    first_out = torch.addcmul(first * cos, -sin, second)
    second_out = torch.addcmul(second * cos, sin, first)
    return torch.stack((first_out, second_out), dim=-2).flatten(-2).transpose(1, 2).reshape(value.shape)


class NativeConnectorAttnProcessor:
    """Native connector arithmetic with training-capable Diffusers attention."""

    def __init__(self, has_learned_registers):
        self.has_learned_registers = has_learned_registers

    def project_qkv(self, attn, hidden_states, encoder_hidden_states):
        return attn.to_q(hidden_states), attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)

    def project_output(self, attn, hidden_states):
        return attn.to_out[0](hidden_states)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        query_rotary_emb=None,
        key_rotary_emb=None,
    ):
        encoder_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        gate = attn.to_gate_logits(hidden_states) if attn.to_gate_logits is not None else None
        query, key, value = self.project_qkv(attn, hidden_states, encoder_hidden_states)
        query, key = attn.norm_q(query).to(value.dtype), attn.norm_k(key).to(value.dtype)
        if query_rotary_emb is not None:
            query = native_connector_rope(query, query_rotary_emb, attn.rope_type, attn.head_dim)
            key = native_connector_rope(
                key, query_rotary_emb if key_rotary_emb is None else key_rotary_emb, attn.rope_type, attn.head_dim
            )
        if self.has_learned_registers:
            # LTX-2.3 replaces padding by learned registers before attention.
            attention_mask = None
        elif attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, key.shape[1], query.shape[0])
            attention_mask = attention_mask.view(query.shape[0], attn.heads, -1, attention_mask.shape[-1])
        query, key, value = (item.unflatten(-1, (-1, attn.head_dim)) for item in (query, key, value))
        output = dispatch_attention_fn(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        if gate is not None:
            output = output * (2.0 * gate.sigmoid()).unsqueeze(-1)
        return attn.to_out[1](self.project_output(attn, output.flatten(2).to(query.dtype)))


def install_native_connector_attention(connectors):
    for name in ("video_connector", "audio_connector"):
        connector = getattr(connectors, name, None)
        for block in getattr(connector, "transformer_blocks", ()):
            block.attn1.set_processor(NativeConnectorAttnProcessor(connector.learnable_registers is not None))
