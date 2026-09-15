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

"""Configuration used only by the LTX OmniNFT pipeline."""

from dataclasses import dataclass
from typing import Optional

from verl_omni.workers.config.diffusion.actor import DiffusionLossConfig
from verl_omni.workers.config.diffusion.rollout import DiffusionPipelineConfig


@dataclass
class OmniNFTLossConfig(DiffusionLossConfig):
    """Loss weights for the audio and video branches of OmniNFT."""

    loss_mode: str = "omni_nft"
    video_weight: float = 1.0
    audio_weight: float = 1.0
    video_ref_kl_coef: float = 0.0
    audio_ref_kl_coef: float = 0.0

    def __post_init__(self):
        if self.loss_mode != "omni_nft":
            raise ValueError(f"OmniNFT loss_mode must be 'omni_nft', got {self.loss_mode!r}.")
        if self.adv_clip_max <= 0:
            raise ValueError(f"Diffusion adv_clip_max must be positive, got {self.adv_clip_max}.")
        if self.mix_beta <= 0:
            raise ValueError(f"mix_beta must be positive, got {self.mix_beta}.")
        if self.adaptive_weight_min <= 0:
            raise ValueError(f"adaptive_weight_min must be positive, got {self.adaptive_weight_min}.")
        if self.kl_mask_threshold <= 0:
            raise ValueError(f"kl_mask_threshold must be positive, got {self.kl_mask_threshold}.")
        for name in ("video_weight", "audio_weight", "video_ref_kl_coef", "audio_ref_kl_coef"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")


@dataclass
class LTXDiffusionPipelineConfig(DiffusionPipelineConfig):
    """LTX guidance controls forwarded to vLLM-Omni sampling parameters."""

    video_cfg_scale: Optional[float] = None
    audio_cfg_scale: Optional[float] = None
    video_modality_scale: Optional[float] = None
    audio_modality_scale: Optional[float] = None
    video_rescale_scale: Optional[float] = None
    audio_rescale_scale: Optional[float] = None
