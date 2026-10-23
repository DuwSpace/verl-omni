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

"""Register the jointly sharded Gemma/DiT actor with the existing FSDP engine."""

from verl_omni.pipelines.ltx2_flow_grpo.diffusers_training_adapter import LTX23FlowGRPO
from verl_omni.pipelines.model_base import DiffusionModelBase


@DiffusionModelBase.register("LTX2Pipeline", algorithm="ltx2_unigrpo_nft")
class LTXJointTrainingAdapter(LTX23FlowGRPO):
    """Reuse LTX tokenizer/schedule contracts and replace trajectory training with joint hooks."""

    @classmethod
    def validate_lora_config(cls, model_config):
        if model_config.lora_rank <= 0:
            raise ValueError("LTX joint training requires Gemma and DiT LoRA")
        if tuple(model_config.policy_state_adapters) != ("default",):
            raise ValueError("LTX joint hooks own old/ref snapshots; configure only the default PEFT adapter")

    @classmethod
    def build_module(cls, model_config, torch_dtype):
        import torch
        from verl.utils.device import get_device_name

        from .model import LTXJointModel

        cls.validate_lora_config(model_config)
        if get_device_name() == "npu":
            torch.npu.config.allow_internal_format = False
            print("LTX_NPU_FORMAT side=actor allow_internal_format=False", flush=True)
        return LTXJointModel.from_pretrained(model_config.local_path, torch_dtype=torch_dtype)

    @classmethod
    def fsdp2_sharding_units(cls, module):
        # The Gemma embedding and tied LM head stay in the root unit: independently
        # sharding them would split ownership of the same parameter. All calls
        # enter root.forward, including the recaption branch.
        return [
            *module.gemma.model.language_model.layers,
            *module.transformer.transformer_blocks,
            module.connectors,
            module.transformer,
            module,
        ]

    @classmethod
    def build_engine_hooks(cls, module, model_config, optimizer_config):
        from .hooks import LTXJointHooks

        return LTXJointHooks(module, model_config, optimizer_config)

    @classmethod
    def build_scheduler(cls, model_config):
        from diffusers import FlowMatchEulerDiscreteScheduler
        from verl.utils.device import get_device_name

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_config.local_path, subfolder="scheduler")
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def prepare_model_inputs(cls, *args, **kwargs):
        raise NotImplementedError("LTX joint hooks prepare exact recaption-conditioned forward-process inputs")

    @classmethod
    def forward_and_sample_previous_step(cls, *args, **kwargs):
        raise NotImplementedError("LTX joint OmniNFT does not use reverse-transition logprobs")


__all__ = ["LTXJointTrainingAdapter"]
