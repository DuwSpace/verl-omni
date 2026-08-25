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
"""Encode verl token-ID prompts into LTX text-encoder embeddings."""

from typing import Any

import torch


def normalize_ltx_output_type(output_type: str | None) -> str | None:
    """Map the generic image default to an LTX decoded-tensor output type."""
    return "pt" if output_type == "image" else output_type


class LTXTokenIdPromptMixin:
    """Convert verl token-ID request fields into LTX prompt embeddings."""

    def _encode_token_ids(
        self,
        token_ids: torch.Tensor | list[int],
        attention_mask: torch.Tensor | None,
        max_sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long)
        else:
            token_ids = token_ids.to(device=self.device, dtype=torch.long)
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)

        if attention_mask is None:
            attention_mask = torch.ones_like(token_ids)
        else:
            attention_mask = attention_mask.to(device=self.device)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)

        token_ids = token_ids[:, :max_sequence_length]
        attention_mask = attention_mask[:, :max_sequence_length]
        pad_length = max_sequence_length - token_ids.shape[1]
        if pad_length > 0:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id
            token_ids = torch.nn.functional.pad(token_ids, (pad_length, 0), value=pad_id)
            attention_mask = torch.nn.functional.pad(attention_mask, (pad_length, 0), value=0)

        encoded = self.text_encoder(
            input_ids=token_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        prompt_embeds = torch.stack(encoded.hidden_states, dim=-1).flatten(2, 3)
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype)
        return prompt_embeds, attention_mask

    def _inject_precomputed_prompt_embeds(self, req: Any) -> None:
        if not isinstance(req.prompt, dict):
            raise TypeError("LTX-2.3 rollout expects a dict prompt containing `prompt_token_ids`.")
        payload = dict(req.prompt)
        prompt_ids = payload.get("prompt_token_ids")
        if prompt_ids is None:
            return

        max_sequence_length = req.sampling_params.max_sequence_length or self.tokenizer_max_length
        prompt_embeds, prompt_mask = self._encode_token_ids(
            prompt_ids,
            payload.get("prompt_mask"),
            max_sequence_length,
        )
        payload["prompt_embeds"] = prompt_embeds[0]
        payload["prompt_attention_mask"] = prompt_mask[0]

        negative_ids = payload.get("negative_prompt_ids")
        if negative_ids is not None:
            negative_embeds, negative_mask = self._encode_token_ids(
                negative_ids,
                payload.get("negative_prompt_mask"),
                max_sequence_length,
            )
            payload["negative_prompt_embeds"] = negative_embeds[0]
            payload["negative_prompt_attention_mask"] = negative_mask[0]
        req.prompt = payload
