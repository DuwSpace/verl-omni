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

"""Register composite Gemma/LTX training through main's NFT engine."""

from verl_omni.pipelines.ltx2_omni_nft.diffusers_training_adapter import LTX23OmniNFT
from verl_omni.pipelines.model_base import DiffusionModelBase

from .model import LTXJointModel
from .update import JointUpdate


@DiffusionModelBase.register("LTX2Pipeline", algorithm="ltx2_unigrpo_nft")
class LTXJointTrainingAdapter(LTX23OmniNFT):
    @classmethod
    def build_module(cls, model_config, torch_dtype):
        if model_config.lora_rank <= 0 or tuple(model_config.policy_state_adapters) != ("default", "old"):
            raise ValueError("Joint LTX requires LoRA with default and old policy adapters")
        module = LTXJointModel.from_pretrained(model_config.local_path, torch_dtype=torch_dtype)
        module.gemma_compute_dtype = model_config.pipeline.gemma_compute_dtype
        if module.gemma_compute_dtype == "float32":
            from ..text.precision import install_fp32_gemma_compute

            install_fp32_gemma_compute(module.gemma)
        module.gemma.lm_head.chunk_size = model_config.pipeline.recaption_logprob_chunk_size
        return module

    @classmethod
    def forward_backward_batch(cls, module, model_config, data, loss_function, forward_only, *, use_adapter):
        return JointUpdate(module, model_config, use_adapter).run(data, loss_function, forward_only)

    @classmethod
    def adapter_ema_decay(cls, model_config, parameter_name, decay):
        settings = model_config.pipeline
        if parameter_name.startswith("gemma.") and not settings.text_ema_enabled:
            return 0.0
        return decay
