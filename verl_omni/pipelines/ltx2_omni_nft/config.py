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

"""OmniNFT objective configuration."""

import math
from dataclasses import dataclass

from verl_omni.workers.config.diffusion.actor import DiffusionLossConfig


@dataclass
class OmniNFTLossConfig(DiffusionLossConfig):
    """Combine modality losses independently of reward-to-modality routing."""

    loss_mode: str = "omni_nft"
    video_weight: float = 1.0
    audio_weight: float = 1.0
    video_ref_kl_coef: float = 0.0
    audio_ref_kl_coef: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.loss_mode != "omni_nft":
            raise ValueError(f"OmniNFT loss_mode must be 'omni_nft', got {self.loss_mode!r}.")
        for name in ("video_weight", "audio_weight", "video_ref_kl_coef", "audio_ref_kl_coef"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}.")
        if self.video_weight == 0 and self.audio_weight == 0:
            raise ValueError("At least one of video_weight or audio_weight must be positive.")


__all__ = ["OmniNFTLossConfig"]
