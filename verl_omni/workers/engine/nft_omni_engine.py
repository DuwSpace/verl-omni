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

"""FSDP2 engine for joint OmniNFT audio-video updates."""

import torch
from tensordict import TensorDict
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig
from verl.workers.engine.base import EngineRegistry

from verl_omni.pipelines.utils import forward, prepare_model_inputs
from verl_omni.workers.config import DiffusionModelConfig
from verl_omni.workers.engine.fsdp.diffusers_impl import NFTDiffusersFSDPEngine, device_name


def _validate_omni_nft_fsdp2_config(engine_config: FSDPEngineConfig) -> None:
    """Keep the first OmniNFT implementation on the verified FSDP2-only path."""
    if engine_config.strategy != "fsdp2":
        raise NotImplementedError(
            f"OmniNFT currently supports only actor.strategy=fsdp2, got {engine_config.strategy!r}."
        )
    if engine_config.ulysses_sequence_parallel_size != 1:
        raise NotImplementedError(
            "OmniNFT FSDP2 does not implement Ulysses/context parallelism yet; "
            "set actor.fsdp_config.ulysses_sequence_parallel_size=1."
        )


@EngineRegistry.register(model_type="omni_nft_model", backend=["fsdp2"], device=[device_name])
class OmniNFTDiffusersFSDPEngine(NFTDiffusersFSDPEngine):
    """FSDP2 actor engine for joint LTX video/audio DiffusionNFT updates."""

    def __init__(
        self,
        model_config: DiffusionModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        _validate_omni_nft_fsdp2_config(engine_config)
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    @staticmethod
    def _select_forward_noise(micro_batch: TensorDict, key: str, x0: torch.Tensor, step: int) -> torch.Tensor:
        noise = micro_batch.get(key, None)
        if noise is None:
            return torch.randn_like(x0.float())
        return noise[:, step] if noise.ndim == x0.ndim + 1 else noise

    def prepare_model_inputs(self, micro_batch: TensorDict, step: int):
        video_x0 = micro_batch["video_latents_clean"]
        audio_x0 = micro_batch["audio_latents_clean"]
        timestep = micro_batch["train_timesteps"][:, step]
        t = timestep.float() / 1000.0
        video_t = t.view(-1, *([1] * (video_x0.ndim - 1)))
        audio_t = t.view(-1, *([1] * (audio_x0.ndim - 1)))

        video_noise = self._select_forward_noise(micro_batch, "video_forward_noise", video_x0, step)
        audio_noise = self._select_forward_noise(micro_batch, "audio_forward_noise", audio_x0, step)
        video_xt = (1.0 - video_t) * video_x0 + video_t * video_noise
        audio_xt = (1.0 - audio_t) * audio_x0 + audio_t * audio_noise

        prompt_embeds = micro_batch["prompt_embeds"]
        prompt_embeds_mask = micro_batch["prompt_embeds_mask"]
        negative_prompt_embeds = micro_batch.get("negative_prompt_embeds", None)
        negative_prompt_embeds_mask = micro_batch.get("negative_prompt_embeds_mask", None)
        if prompt_embeds.is_nested:
            prompt_embeds, prompt_embeds_mask = self._unpad_nested_embeds(prompt_embeds, prompt_embeds_mask)
        if isinstance(negative_prompt_embeds, torch.Tensor) and negative_prompt_embeds.is_nested:
            negative_prompt_embeds, negative_prompt_embeds_mask = self._unpad_nested_embeds(
                negative_prompt_embeds, negative_prompt_embeds_mask
            )

        model_inputs, negative_model_inputs = prepare_model_inputs(
            module=self.module,
            model_config=self.model_config,
            latents=torch.cat((video_xt, audio_xt), dim=1),
            timesteps=timestep,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            micro_batch=micro_batch,
            step=step,
        )
        return (
            model_inputs,
            negative_model_inputs,
            video_x0,
            audio_x0,
            video_xt,
            audio_xt,
            video_t,
            audio_t,
        )

    def prepare_model_outputs(self, output, micro_batch: TensorDict):
        del micro_batch
        (
            video_old,
            audio_old,
            video_forward,
            audio_forward,
            video_ref,
            audio_ref,
            video_x0,
            audio_x0,
            video_xt,
            audio_xt,
            video_t,
            audio_t,
        ) = output
        return {
            "video_old_prediction": video_old,
            "audio_old_prediction": audio_old,
            "video_forward_prediction": video_forward,
            "audio_forward_prediction": audio_forward,
            "video_ref_forward_prediction": video_ref,
            "audio_ref_forward_prediction": audio_ref,
            "video_x0": video_x0,
            "audio_x0": audio_x0,
            "video_xt": video_xt,
            "audio_xt": audio_xt,
            "video_t_expanded": video_t,
            "audio_t_expanded": audio_t,
        }

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only, step):
        (
            model_inputs,
            negative_model_inputs,
            video_x0,
            audio_x0,
            video_xt,
            audio_xt,
            video_t,
            audio_t,
        ) = self.prepare_model_inputs(micro_batch=micro_batch, step=step)

        with self.use_adapter("old"), torch.no_grad():
            video_old, audio_old = forward(
                module=self.module,
                model_config=self.model_config,
                model_inputs=model_inputs,
                negative_model_inputs=negative_model_inputs,
            )
            video_old, audio_old = video_old.detach(), audio_old.detach()

        video_forward, audio_forward = forward(
            module=self.module,
            model_config=self.model_config,
            model_inputs=model_inputs,
            negative_model_inputs=negative_model_inputs,
        )

        with torch.no_grad(), self.disable_adapter():
            video_ref, audio_ref = forward(
                module=self.module,
                model_config=self.model_config,
                model_inputs=model_inputs,
                negative_model_inputs=negative_model_inputs,
            )
            video_ref, audio_ref = video_ref.detach(), audio_ref.detach()
        self._set_adapter("default")

        model_output = self.prepare_model_outputs(
            output=(
                video_old,
                audio_old,
                video_forward,
                audio_forward,
                video_ref,
                audio_ref,
                video_x0,
                audio_x0,
                video_xt,
                audio_xt,
                video_t,
                audio_t,
            ),
            micro_batch=micro_batch,
        )
        if loss_function is not None:
            data = tu.get_tensordict(
                {
                    "video_reward_prob": micro_batch["video_reward_prob"][:, step],
                    "audio_reward_prob": micro_batch["audio_reward_prob"][:, step],
                }
            )
            tu.assign_non_tensor(
                data,
                gradient_accumulation_steps=tu.get_non_tensor_data(
                    micro_batch, "gradient_accumulation_steps", default=None
                ),
                sp_size=1,
            )
            loss, metrics = loss_function(
                model_output=model_output,
                data=data,
                dp_group=self.get_data_parallel_group(),
            )
        else:
            assert forward_only, "forward_only must be True when loss_function is None"
            loss = torch.tensor(1.0, device=video_x0.device)
            metrics = {}

        return loss, {
            "model_output": model_output,
            "loss": loss.detach().item(),
            "metrics": metrics,
        }
