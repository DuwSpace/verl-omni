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

"""Configuration for Gemma recaption GRPO plus LTX OmniNFT."""

import math
from dataclasses import dataclass

from verl_omni.pipelines.ltx2_omni_nft.config import LTXDiffusionPipelineConfig
from verl_omni.workers.config.diffusion.actor import DiffusionLossConfig


@dataclass
class LTXJointPipelineConfig(LTXDiffusionPipelineConfig):
    # Frozen weights remain BF16; float32 preserves Gemma activations for AR consistency.
    gemma_compute_dtype: str = "bfloat16"
    # Prompt template used for audiovisual recaptioning.
    recaption_prompt: str = "ltx_t2av"
    # Temperature-only sampling; no top-k or top-p truncation.
    recaption_temperature: float = 1.0
    # Responses must reach EOS before this limit.
    recaption_max_new_tokens: int = 4096
    # Bound the AR KV-cache; CPU conditions are retained for the entire round.
    recaption_batch_size: int = 2
    # Deadline for request collection and text preparation, independently.
    stage_timeout_s: int = 3600
    # Maximum token rows in each local vocabulary projection during actor logprob computation.
    recaption_logprob_chunk_size: int = 128
    # Select which adapters receive gradients; rollout remains complete.
    update_mode: str = "joint"
    # Copy Gemma current into old by default; AV retains the configured EMA.
    text_ema_enabled: bool = False

    def __post_init__(self):
        if self.gemma_compute_dtype not in {"bfloat16", "float32"}:
            raise ValueError("gemma_compute_dtype must be bfloat16 or float32")
        if self.recaption_prompt != "ltx_t2av":
            raise ValueError("recaption_prompt must be ltx_t2av")
        if self.update_mode not in {"joint", "ar_only", "av_only"}:
            raise ValueError("update_mode must be joint, ar_only or av_only")
        if type(self.text_ema_enabled) is not bool:
            raise ValueError("text_ema_enabled must be boolean")
        for name in (
            "recaption_max_new_tokens",
            "recaption_batch_size",
            "stage_timeout_s",
            "recaption_logprob_chunk_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.recaption_temperature) or self.recaption_temperature <= 0:
            raise ValueError("recaption_temperature must be finite and positive")


@dataclass
class LTXJointLossConfig(DiffusionLossConfig):
    # Registered joint objective; AV arithmetic comes from main's OmniNFTLoss.
    loss_mode: str = "ltx2_unigrpo_nft"
    # Fixed initial text credit definition, without alternate reward paths.
    text_advantage_source: str = "component_advantage_mean"
    # Token-level PPO clipping for the generated caption.
    ar_clip_ratio: float = 0.01
    # Relative branch coefficients; ablations use update_mode instead.
    text_loss_weight: float = 1.0
    av_loss_weight: float = 1.0
    # Sampled reference KL against the frozen base Gemma.
    text_kl_coef: float = 0.001
    # Full-batch population std; centering remains within each prompt group.
    text_global_std: bool = True
    text_advantage_epsilon: float = 1e-6

    def __post_init__(self):
        super().__post_init__()
        if self.loss_mode != "ltx2_unigrpo_nft" or self.text_advantage_source != "component_advantage_mean":
            raise ValueError("LTX text GRPO requires component_advantage_mean")
        if not 0 < self.ar_clip_ratio < 1:
            raise ValueError("ar_clip_ratio must be in (0, 1)")
        for name in ("text_loss_weight", "av_loss_weight", "text_advantage_epsilon"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.text_kl_coef) or self.text_kl_coef < 0:
            raise ValueError("text_kl_coef must be finite and nonnegative")
