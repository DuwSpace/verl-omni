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

"""LTX-2.3 joint audio-video training adapter for OmniNFT."""

from typing import Optional

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, ModelMixin
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config import DiffusionModelConfig

from .common import apply_x0_cfg, calculate_shift

__all__ = ["LTX23OmniNFT"]


def _single_int(value: torch.Tensor, name: str) -> int:
    values = value.reshape(-1)
    if values.numel() == 0 or not torch.all(values == values[0]):
        raise ValueError(f"LTX-2.3 requires one shared {name} per micro-batch, got {values.tolist()}.")
    return int(values[0].item())


@DiffusionModelBase.register("LTX2Pipeline", algorithm="omni_nft")
class LTX23OmniNFT(DiffusionModelBase):
    """Return separate video/audio velocities from one joint LTX forward."""

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig) -> FlowMatchEulerDiscreteScheduler:
        """Load LTX's native deterministic flow-matching scheduler."""
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_config.local_path,
            subfolder="scheduler",
        )
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(
        cls,
        scheduler: FlowMatchEulerDiscreteScheduler,
        model_config: DiffusionModelConfig,
        device: str,
    ) -> None:
        """Configure the native LTX sigma grid without FlowGRPO's SDE transition."""
        num_steps = model_config.pipeline.num_inference_steps
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        latent_frames = (model_config.pipeline.num_frames - 1) // 8 + 1
        latent_height = model_config.pipeline.height // 32
        latent_width = model_config.pipeline.width // 32
        video_seq_len = latent_frames * latent_height * latent_width
        mu = calculate_shift(
            video_seq_len,
            scheduler.config.get("base_image_seq_len", 1024),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.95),
            scheduler.config.get("max_shift", 2.05),
        )
        scheduler.set_timesteps(num_steps, device=device, sigmas=sigmas, mu=mu)

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
        """Build one joint forward-process input from already-noised AV latents."""
        del step
        required = ["audio_prompt_embeds", "video_seq_len"]
        missing = [key for key in required if key not in micro_batch]
        if missing:
            raise KeyError(f"LTX-2.3 OmniNFT rollout is missing required fields: {missing}.")

        video_seq_len = _single_int(micro_batch["video_seq_len"], "video_seq_len")
        video_latents = latents[:, :video_seq_len]
        audio_latents = latents[:, video_seq_len:]
        latent_frames = (model_config.pipeline.num_frames - 1) // 8 + 1
        latent_height = model_config.pipeline.height // 32
        latent_width = model_config.pipeline.width // 32

        common = {
            "hidden_states": video_latents,
            "audio_hidden_states": audio_latents,
            "timestep": timesteps,
            "sigma": timesteps,
            "num_frames": latent_frames,
            "height": latent_height,
            "width": latent_width,
            "fps": model_config.pipeline.frame_rate,
            "audio_num_frames": audio_latents.shape[1],
            "return_dict": False,
        }
        model_inputs = {
            **common,
            "encoder_hidden_states": prompt_embeds,
            "audio_encoder_hidden_states": micro_batch["audio_prompt_embeds"],
            "encoder_attention_mask": prompt_embeds_mask,
            "audio_encoder_attention_mask": prompt_embeds_mask,
        }

        guidance_scale = model_config.pipeline.guidance_scale or 1.0
        if guidance_scale <= 1.0:
            return model_inputs, None
        if negative_prompt_embeds is None or negative_prompt_embeds_mask is None:
            raise ValueError("LTX-2.3 OmniNFT CFG requires negative prompt embeddings and attention masks.")
        if "negative_audio_prompt_embeds" not in micro_batch:
            raise KeyError("LTX-2.3 OmniNFT CFG requires `negative_audio_prompt_embeds` from rollout.")
        negative_model_inputs = {
            **common,
            "encoder_hidden_states": negative_prompt_embeds,
            "audio_encoder_hidden_states": micro_batch["negative_audio_prompt_embeds"],
            "encoder_attention_mask": negative_prompt_embeds_mask,
            "audio_encoder_attention_mask": negative_prompt_embeds_mask,
        }
        return model_inputs, negative_model_inputs

    @staticmethod
    def _predict(module: ModelMixin, model_inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the joint LTX transformer and return float32 AV velocities."""
        video_prediction, audio_prediction = module(**model_inputs)
        return video_prediction.float(), audio_prediction.float()

    @classmethod
    def forward(
        cls,
        module: ModelMixin,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_prediction, audio_prediction = cls._predict(module, model_inputs)

        guidance_scale = model_config.pipeline.guidance_scale or 1.0
        if guidance_scale > 1.0:
            if negative_model_inputs is None:
                raise ValueError("LTX-2.3 OmniNFT CFG requires negative model inputs.")
            negative_video_prediction, negative_audio_prediction = cls._predict(module, negative_model_inputs)
            sigma = (model_inputs["timestep"].float() / 1000.0).view(-1, 1, 1)
            video_prediction = apply_x0_cfg(
                model_inputs["hidden_states"].float(),
                video_prediction,
                negative_video_prediction,
                sigma,
                guidance_scale,
            )
            audio_prediction = apply_x0_cfg(
                model_inputs["audio_hidden_states"].float(),
                audio_prediction,
                negative_audio_prediction,
                sigma,
                guidance_scale,
            )

        return video_prediction, audio_prediction

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module: ModelMixin,
        scheduler: FlowMatchEulerDiscreteScheduler,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        """Reject the reverse-transition API, which is not part of OmniNFT."""
        del module, scheduler, model_config, model_inputs, negative_model_inputs, scheduler_inputs, step
        raise NotImplementedError("LTX-2.3 OmniNFT does not sample reverse transitions or compute their log-probs.")
