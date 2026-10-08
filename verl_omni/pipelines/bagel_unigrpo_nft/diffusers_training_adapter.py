# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full-weight BAGEL adapter for hybrid AR-GRPO/image-OmniNFT training."""

from verl_omni.pipelines.bagel_unigrpo.diffusers_training_adapter import BagelUniGRPO
from verl_omni.pipelines.model_base import DiffusionModelBase


@DiffusionModelBase.register("OmniBagelForConditionalGeneration", algorithm="unigrpo_nft")
class BagelUniGRPONFT(BagelUniGRPO):
    """Reuse MoT loading, expert selection and FSDP2 units; replace the algorithm hooks."""

    @classmethod
    def build_engine_hooks(cls, module, model_config, optimizer_config):
        """Attach the joint objective and durable full-weight policy state."""
        from .hooks import BagelUniGRPONFTHooks

        return BagelUniGRPONFTHooks(module, model_config, optimizer_config)
