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

"""Placeholder LTX OmniNFT training adapter.

`main_diffusion` looks up `(LTX2Pipeline, omni_nft)` before constructing a trainer.
This class occupies that registry key so the entrypoint can load a processor. Actor
forward, scheduler, and optimizer paths are not implemented.
"""

from typing import Optional

import torch
from diffusers import ModelMixin, SchedulerMixin
from tensordict import TensorDict

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config import DiffusionModelConfig

_NOT_IMPLEMENTED = "LTX OmniNFT training adapter methods are not implemented."


@DiffusionModelBase.register("LTX2Pipeline", algorithm="omni_nft")
class LTX23OmniNFT(DiffusionModelBase):
    """Registry placeholder for OmniNFT training; does not run Actor updates."""

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig) -> SchedulerMixin:
        """Reject scheduler construction until the real OmniNFT actor lands."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    @classmethod
    def set_timesteps(cls, scheduler: SchedulerMixin, model_config: DiffusionModelConfig, device: str):
        """Reject timestep setup until the real OmniNFT actor lands."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    @classmethod
    def prepare_model_inputs(
        cls,
        module: ModelMixin,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_embeds_mask: Optional[torch.Tensor],
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, Optional[dict]]:
        """Reject actor input construction until the real OmniNFT actor lands."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module: ModelMixin,
        scheduler: SchedulerMixin,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        """Reject reverse-process sampling until the real OmniNFT actor lands."""
        raise NotImplementedError(_NOT_IMPLEMENTED)
