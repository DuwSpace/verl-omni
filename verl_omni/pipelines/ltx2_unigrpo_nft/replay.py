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

"""Exact recaption replay metadata paired with the actor batch's AV clean latents."""

import math
from dataclasses import dataclass

import torch


@dataclass
class LTXJointReplay:
    """CPU token record; the enclosing batch carries media and clean latent tensors.

    Condition tokens/mask are the exact padded encoder inputs captured during
    rollout, not a later decode/re-tokenization. Generated response tokens include
    a sampled terminal EOS when present; prompt tokens never receive GRPO credit.
    """

    original_prompt: str
    recaption: str
    prompt_token_ids: torch.Tensor
    response_token_ids: torch.Tensor
    rollout_logprobs: torch.Tensor
    condition_token_ids: torch.Tensor
    condition_attention_mask: torch.Tensor
    policy_version: int
    seed: int
    temperature: float
    eos_token_ids: tuple[int, ...]
    generation_params: dict
    logprob_kind: str = "temperature_scaled"

    def validate(self):
        if not self.original_prompt.strip() or not self.recaption.strip():
            raise ValueError("Joint replay requires original task and nonempty recaption text")
        for tokens in (self.prompt_token_ids, self.response_token_ids, self.condition_token_ids):
            if tokens.ndim != 1 or tokens.dtype != torch.int64 or tokens.numel() == 0 or bool((tokens < 0).any()):
                raise ValueError("Joint replay token IDs must be nonempty nonnegative int64 vectors")
        tensors = (
            self.prompt_token_ids,
            self.response_token_ids,
            self.rollout_logprobs,
            self.condition_token_ids,
            self.condition_attention_mask,
        )
        if any(t.device.type != "cpu" or t.requires_grad for t in tensors):
            raise ValueError("Joint replay tensors must be detached on CPU")
        if self.rollout_logprobs.shape != self.response_token_ids.shape or self.rollout_logprobs.dtype != torch.float32:
            raise ValueError("Every sampled recaption token requires one FP32 rollout logprob")
        if not bool(torch.isfinite(self.rollout_logprobs).all()):
            raise ValueError("Recaption rollout logprobs must be finite")
        mask = self.condition_attention_mask
        if mask.shape != self.condition_token_ids.shape or mask.dtype != torch.bool or not bool(mask.any()):
            raise ValueError("Condition mask must be boolean and match the nonempty encoder token sequence")
        if bool((mask[:-1] & ~mask[1:]).any()):
            raise ValueError("LTX condition replay must use left padding")
        if type(self.policy_version) is not int or self.policy_version < 0 or type(self.seed) is not int:
            raise ValueError("Joint replay needs integer seed and nonnegative integer policy version")
        if not math.isfinite(self.temperature) or self.temperature <= 0 or self.logprob_kind != "temperature_scaled":
            raise ValueError("Replay supports finite positive temperature and temperature-scaled logprobs only")
        if not self.eos_token_ids or any(type(token) is not int or token < 0 for token in self.eos_token_ids):
            raise ValueError("Replay requires tokenizer EOS IDs")
        if any(int(token) in self.eos_token_ids for token in self.response_token_ids[:-1]):
            raise ValueError("Recaption contains tokens after a sampled EOS")
        for key in ("height", "width", "num_frames", "frame_rate", "num_inference_steps"):
            value = self.generation_params.get(key)
            if not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Missing/invalid generation parameter: {key}")
            if key != "frame_rate" and type(value) is not int:
                raise ValueError(f"Generation parameter must be an integer: {key}")
        if (
            self.generation_params["height"] % 32
            or self.generation_params["width"] % 32
            or (self.generation_params["num_frames"] - 1) % 8
        ):
            raise ValueError("LTX replay requires dimensions divisible by 32 and 8k+1 video frames")

    def ar_inputs(self, device):
        ids = torch.cat((self.prompt_token_ids, self.response_token_ids)).to(device).unsqueeze(0)
        response_mask = torch.zeros_like(ids, dtype=torch.bool)
        response_mask[:, self.prompt_token_ids.numel() :] = True
        return dict(
            input_ids=ids,
            attention_mask=torch.ones_like(ids, dtype=torch.bool),
            response_mask=response_mask,
            temperature=self.temperature,
        )
