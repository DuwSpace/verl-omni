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

"""Request-local prompt preparation used only by LTX-2.3 OmniNFT rollout."""

from typing import Any

import torch
import torch.nn.functional as F

from verl_omni.pipelines.request_batch import collate_prompt_rows


def shared_ltx_int(value: torch.Tensor, name: str) -> int:
    """Read a scalar LTX layout value shared by an actor micro-batch."""
    values = value.reshape(-1)
    if values.numel() == 0 or not torch.all(values == values[0]):
        raise ValueError(f"LTX-2.3 requires one shared {name} per micro-batch, got {values.tolist()}.")
    return int(values[0].item())


def _left_pad_prompt_rows(
    prompts: list[Any],
    aliases: tuple[str, ...],
    *,
    device: torch.device,
    field_name: str,
    pad_value: int,
    target_len: int,
) -> tuple[torch.Tensor | None, list[int] | None]:
    """Apply LTX's fixed-length left padding to a request batch."""
    rows, lengths = collate_prompt_rows(
        prompts,
        aliases,
        None,
        device=device,
        field_name=field_name,
        pad_value=pad_value,
    )
    if rows is None or lengths is None:
        return None, None
    padded = torch.full((rows.shape[0], target_len), pad_value, dtype=rows.dtype, device=rows.device)
    clipped_lengths = []
    for index, row_len in enumerate(lengths):
        row = rows[index, :row_len][:target_len]
        clipped_len = int(row.shape[0])
        if clipped_len:
            padded[index, target_len - clipped_len :] = row
        clipped_lengths.append(clipped_len)
    return padded, clipped_lengths


def _left_pad_prompt_mask(
    prompts: list[Any],
    aliases: tuple[str, ...],
    *,
    device: torch.device,
    field_name: str,
    token_lengths: list[int] | None,
    target_len: int,
) -> torch.Tensor | None:
    mask, _ = _left_pad_prompt_rows(
        prompts,
        aliases,
        device=device,
        field_name=field_name,
        pad_value=0,
        target_len=target_len,
    )
    if mask is not None:
        return mask != 0
    if token_lengths is None:
        return None
    mask = torch.zeros((len(token_lengths), target_len), dtype=torch.bool, device=device)
    for index, row_len in enumerate(token_lengths):
        valid = min(int(row_len), target_len)
        if valid:
            mask[index, target_len - valid :] = True
    return mask


def _prompt_pad_id(pipeline: Any) -> int:
    pad_id = pipeline.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = pipeline.tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("LTX tokenizer must define a pad_token_id or eos_token_id.")
    return int(pad_id)


def _prepare_token_ids(
    pipeline: Any,
    token_ids: torch.Tensor | list[int],
    attention_mask: torch.Tensor | None,
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(token_ids, list):
        token_ids = torch.tensor(token_ids, device=pipeline.device, dtype=torch.long)
    else:
        token_ids = token_ids.to(device=pipeline.device, dtype=torch.long)
    if token_ids.ndim == 1:
        token_ids = token_ids.unsqueeze(0)

    if attention_mask is None:
        attention_mask = torch.ones_like(token_ids)
    else:
        attention_mask = attention_mask.to(device=pipeline.device)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)

    token_ids = token_ids[:, :max_sequence_length]
    attention_mask = attention_mask[:, :max_sequence_length]
    pad_length = max_sequence_length - token_ids.shape[1]
    if pad_length > 0:
        token_ids = F.pad(token_ids, (pad_length, 0), value=_prompt_pad_id(pipeline))
        attention_mask = F.pad(attention_mask, (pad_length, 0), value=0)
    return token_ids, attention_mask


def _encode_prepared_rows(
    pipeline: Any,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_states = pipeline.text_encoder(
        input_ids=token_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    ).hidden_states
    prompt_embeds = torch.stack(hidden_states, dim=-1).flatten(2, 3).to(dtype=pipeline.text_encoder.dtype)
    return prompt_embeds, attention_mask


def encode_ltx_token_ids(
    pipeline: Any,
    token_ids: torch.Tensor | list[int],
    attention_mask: torch.Tensor | None,
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode pre-tokenized prompts through LTX's full Gemma-3 hidden-state stack."""
    token_ids, attention_mask = _prepare_token_ids(pipeline, token_ids, attention_mask, max_sequence_length)
    return _encode_prepared_rows(pipeline, token_ids, attention_mask)


class LTXTokenIdPromptMixin:
    """Convert verl token-ID request fields into native LTX prompt embeddings."""

    def _prompt_pad_id(self) -> int:
        return _prompt_pad_id(self)

    def _encode_token_ids(
        self,
        token_ids: torch.Tensor | list[int],
        attention_mask: torch.Tensor | None,
        max_sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return encode_ltx_token_ids(self, token_ids, attention_mask, max_sequence_length)

    def _encode_unique_prompt_rows(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode unique packed rows and return the inverse row mapping."""
        if token_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("LTX prompt rows must be two-dimensional tensors.")
        if token_ids.shape != attention_mask.shape:
            raise ValueError(
                "LTX prompt token ids and attention mask must have matching shapes, "
                f"got {tuple(token_ids.shape)} and {tuple(attention_mask.shape)}."
            )

        # The mask is part of the key because equal token IDs with different
        # padding are different text-encoder inputs.
        unique_keys, inverse = torch.unique(
            torch.cat((token_ids, attention_mask.to(dtype=token_ids.dtype)), dim=1),
            dim=0,
            return_inverse=True,
        )
        unique_len = token_ids.shape[1]
        unique_token_ids = unique_keys[:, :unique_len]
        unique_attention_mask = unique_keys[:, unique_len:].to(dtype=attention_mask.dtype)
        embeds, masks = _encode_prepared_rows(self, unique_token_ids, unique_attention_mask)
        return embeds, masks, inverse

    def _inject_batch_prompt_embeds(self, req: Any) -> None:
        """Batch-encode request prompts, deduplicating identical encoder rows."""
        prompts = [request.prompt for request in req.requests]
        if any(not isinstance(prompt, dict) for prompt in prompts):
            raise TypeError("LTX-2.3 OmniNFT rollout expects dict prompts containing `prompt_token_ids`.")

        max_sequence_length = max(
            int(request.sampling_params.max_sequence_length or self.tokenizer_max_length) for request in req.requests
        )
        prompt_ids, prompt_lengths = _left_pad_prompt_rows(
            prompts,
            ("prompt_token_ids", "prompt_ids"),
            device=self.device,
            field_name="prompt_token_ids",
            pad_value=self._prompt_pad_id(),
            target_len=max_sequence_length,
        )
        prompt_mask = _left_pad_prompt_mask(
            prompts,
            ("prompt_mask", "attention_mask"),
            device=self.device,
            field_name="prompt_mask",
            token_lengths=prompt_lengths,
            target_len=max_sequence_length,
        )
        negative_ids, negative_lengths = _left_pad_prompt_rows(
            prompts,
            ("negative_prompt_ids", "negative_prompt_token_ids"),
            device=self.device,
            field_name="negative_prompt_ids",
            pad_value=self._prompt_pad_id(),
            target_len=max_sequence_length,
        )
        negative_mask = _left_pad_prompt_mask(
            prompts,
            ("negative_prompt_mask", "negative_attention_mask"),
            device=self.device,
            field_name="negative_prompt_mask",
            token_lengths=negative_lengths,
            target_len=max_sequence_length,
        )

        if prompt_ids is None and negative_ids is None:
            return
        if prompt_ids is not None and prompt_mask is None:
            prompt_mask = torch.ones_like(prompt_ids)

        positive_count = 0 if prompt_ids is None else prompt_ids.shape[0]
        if negative_ids is not None:
            if negative_mask is None:
                negative_mask = torch.ones_like(negative_ids)
            if prompt_ids is None:
                encoder_ids = negative_ids
                encoder_mask = negative_mask
            else:
                encoder_ids = torch.cat((prompt_ids, negative_ids), dim=0)
                encoder_mask = torch.cat((prompt_mask, negative_mask), dim=0)
        else:
            assert prompt_ids is not None and prompt_mask is not None
            encoder_ids = prompt_ids
            encoder_mask = prompt_mask

        unique_embeds, unique_masks, inverse = self._encode_unique_prompt_rows(encoder_ids, encoder_mask)
        prompt_embeds = prompt_masks = None
        if prompt_ids is not None:
            prompt_embeds = unique_embeds.index_select(0, inverse[:positive_count])
            prompt_masks = unique_masks.index_select(0, inverse[:positive_count])
        negative_embeds = negative_masks = None
        if negative_ids is not None:
            negative_start = positive_count
            negative_inverse = inverse[negative_start:] if prompt_ids is not None else inverse
            negative_embeds = unique_embeds.index_select(0, negative_inverse)
            negative_masks = unique_masks.index_select(0, negative_inverse)

        for index, request in enumerate(req.requests):
            payload = dict(request.prompt)
            if prompt_embeds is not None and prompt_masks is not None:
                payload["prompt_embeds"] = prompt_embeds[index]
                payload["prompt_attention_mask"] = prompt_masks[index]
            if negative_embeds is not None and negative_masks is not None:
                payload["negative_prompt_embeds"] = negative_embeds[index]
                payload["negative_prompt_attention_mask"] = negative_masks[index]
            request.prompt = payload


__all__ = ["LTXTokenIdPromptMixin", "encode_ltx_token_ids", "shared_ltx_int"]
