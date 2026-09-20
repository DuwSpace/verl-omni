# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compatibility helpers shared by Qwen2-VL native reward models."""

import torch

__all__ = ["ensure_omninft_qwen2vl_layout", "omninft_qwen2vl_reward_forward"]


class _TensorVisual(torch.nn.Module):
    """Unwrap vision outputs exposing last_hidden_state; pass tensor outputs through."""

    def __init__(self, visual):
        super().__init__()
        self._visual = visual

    def get_dtype(self):
        if hasattr(self._visual, "get_dtype"):
            return self._visual.get_dtype()
        return next(self._visual.parameters()).dtype

    @property
    def dtype(self):
        return self.get_dtype()

    def forward(self, pixel_values, grid_thw=None, **kwargs):
        outputs = self._visual(pixel_values, grid_thw=grid_thw, **kwargs)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs


def _unwrap_peft(model):
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def ensure_omninft_qwen2vl_layout(model):
    """Expose the reward forward's visual/token interfaces on this model in place.

    Unwrap PEFT if present, wrap the visual module to extract last_hidden_state,
    and alias nested language embeddings when needed. Return the original model.
    Repeated calls do not rewrap the visual module; changes persist on the instance
    with no automatic restoration or process-wide Transformers monkey patch.
    """
    root = _unwrap_peft(model)
    inner = getattr(root, "model", None)
    if inner is None:
        return model
    visual = getattr(root, "visual", None) or getattr(inner, "visual", None)
    if visual is not None and not isinstance(visual, _TensorVisual):
        visual = _TensorVisual(visual)
        if hasattr(inner, "visual"):
            inner.visual = visual
        root.visual = visual
    if not hasattr(inner, "embed_tokens"):
        language = getattr(inner, "language_model", None)
        if language is not None and hasattr(language, "embed_tokens"):
            inner.embed_tokens = language.embed_tokens
    return model


def omninft_qwen2vl_reward_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    rope_deltas=None,
    **kwargs,
):
    """Embed visual tokens and pool reward-head logits in sequence-token order.

    Args:
        input_ids: ``[B, L]`` token IDs, required for reward-token selection even
            when ``inputs_embeds`` is supplied. Rows must contain the configured
            reward tokens in the order/count expected by the scorer; this is not
            explicitly validated here.
        attention_mask: Optional sequence mask, moved to embedding device only
            when embeddings are built here.
        inputs_embeds: Optional ``[B, L, D]`` embeddings. If supplied, visual
            embedding/scatter is skipped; otherwise image/video pixels and their
            grid metadata fill the matching placeholder token positions.
        labels: Ignored; this forward computes no supervised loss.
        rope_deltas: Ignored. Other explicit model-forward options are passed to
            the backbone; extra ``kwargs`` are not forwarded.

    Returns:
        A dict with ``logits[B, N * O]`` for N selected reward tokens and O head
        outputs. Hidden states are cast to FP32 for the head; logits stay on its
        device without detachment. ``return_dict`` controls the backbone only;
        the outer result is always a dict. Gradient mode belongs to the caller.

    Raises:
        ValueError: No pad token for B>1, or reward_token is not ``special``.
    """
    del labels, rope_deltas
    kwargs.pop("mm_token_type_ids", None)
    output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
    output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
    return_dict = getattr(self.config, "use_return_dict", True) if return_dict is None else return_dict
    visual = getattr(self, "visual", None) or getattr(self.model, "visual", None)
    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(visual.get_dtype() if hasattr(visual, "get_dtype") else visual.dtype)
            image_embeds = visual(pixel_values, grid_thw=image_grid_thw)
            image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask, image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            )
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(
                visual.get_dtype() if hasattr(visual, "get_dtype") else visual.dtype
            )
            video_embeds = visual(pixel_values_videos, grid_thw=video_grid_thw)
            video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask, video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            )
        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)
    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_states = outputs[0]
    logits = self.rm_head(hidden_states.float())
    batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
    if self.config.pad_token_id is None and batch_size != 1:
        raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
    if self.reward_token != "special":
        raise ValueError("OmniNFT Qwen2-VL rewards must pool special tokens.")
    special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in self.special_token_ids:
        special_token_mask |= input_ids == token_id
    pooled_logits = logits[special_token_mask, ...].view(batch_size, -1)
    return {"logits": pooled_logits}
