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

"""Exact token/latent replay boundary for BAGEL AR-GRPO + image OmniNFT."""

from dataclasses import dataclass

import torch


@dataclass
class BagelJointReplay:
    """CPU rollout record; no decoded/re-encoded text, images, or KV tensors.

    Native BAGEL returns cached thinking tokens INCLUDING its fixed BOS, but
    excludes the final next-token draw. Only sampled tokens receive AR credit.
    A terminating EOS is scored when observed, but never added to image context.
    Latents are patchified ``[L, D]`` FP32; positions and IDs are ``[L]`` int64.
    """

    prompt_token_ids: torch.Tensor
    cached_thinking_token_ids: torch.Tensor
    response_token_ids: torch.Tensor
    rollout_log_probs: torch.Tensor
    latents_clean: torch.Tensor
    latent_pos_ids: torch.Tensor
    image_position_ids: torch.Tensor
    image_boundary_token_ids: torch.Tensor
    temperature: float
    policy_version: int
    eos_token_id: int

    @property
    def ar_prompt_token_ids(self) -> torch.Tensor:
        """The prefill plus fixed thinking BOS, before the first sampled token."""
        return torch.cat((self.prompt_token_ids, self.cached_thinking_token_ids[:1]))

    @property
    def condition_token_ids(self) -> torch.Tensor:
        """Exactly the token sequence in the KV cache used for image generation."""
        return torch.cat((self.prompt_token_ids, self.cached_thinking_token_ids))

    def validate(self) -> None:
        """Reject ambiguous, detached-policy, nonfinite, or unsupported replay data."""
        token_fields = (
            self.prompt_token_ids,
            self.cached_thinking_token_ids,
            self.response_token_ids,
            self.latent_pos_ids,
            self.image_position_ids,
            self.image_boundary_token_ids,
        )
        if any(t.ndim != 1 or t.dtype != torch.int64 or t.numel() == 0 for t in token_fields):
            raise ValueError("Replay token/position fields must be nonempty int64 vectors")
        tensors = (*token_fields, self.rollout_log_probs, self.latents_clean)
        if any(t.device.type != "cpu" or t.requires_grad for t in tensors):
            raise ValueError("Replay data must be detached and on CPU")
        if self.latents_clean.ndim != 2 or self.latents_clean.dtype != torch.float32:
            raise ValueError("Clean latents must be patchified FP32 [L, D]")
        if self.latent_pos_ids.numel() != self.latents_clean.shape[0]:
            raise ValueError("Latent position count differs from clean latent length")
        if self.image_position_ids.numel() != self.latents_clean.shape[0] + 2:
            raise ValueError("Image RoPE positions must include SOI and EOI")
        if self.image_boundary_token_ids.numel() != 2:
            raise ValueError("Image boundary IDs must contain SOI and EOI")
        if not bool((self.image_position_ids == self.condition_token_ids.numel()).all()):
            raise ValueError("Image RoPE positions differ from the actual text cache length")
        if self.rollout_log_probs.shape != self.response_token_ids.shape:
            raise ValueError("One old-policy log probability is required per sampled response token")
        if self.rollout_log_probs.dtype != torch.float32:
            raise ValueError("Old log probabilities must be FP32")
        if not bool(torch.isfinite(self.latents_clean).all() and torch.isfinite(self.rollout_log_probs).all()):
            raise ValueError("Nonfinite rollout latents/log probabilities")
        if self.temperature <= 0 or self.policy_version < 0:
            raise ValueError("Positive sampling temperature and a nonnegative policy version are required")
        cached_response = self.cached_thinking_token_ids[1:]
        if not torch.equal(self.response_token_ids[: cached_response.numel()], cached_response):
            raise ValueError("Sampled response prefix differs from the tokens cached for image generation")
        if self.response_token_ids.numel() not in (cached_response.numel(), cached_response.numel() + 1):
            raise ValueError("Only an uncached terminal EOS may follow the cached sampled tokens")
        if (
            self.response_token_ids.numel() > cached_response.numel()
            and int(self.response_token_ids[-1]) != self.eos_token_id
        ):
            raise ValueError("The uncached terminal response token must be EOS")


def replay_from_metadata(metadata: dict) -> BagelJointReplay:
    """Validate the versioned wire record returned through rollout RL metadata."""
    if metadata.get("bagel_joint_contract_version") != 1:
        raise ValueError("Missing/unsupported BAGEL joint rollout contract")
    replay = metadata.get("bagel_joint_replay")
    if not isinstance(replay, BagelJointReplay):
        raise TypeError("BAGEL rollout must return its exact BagelJointReplay record")
    replay.validate()
    return replay
