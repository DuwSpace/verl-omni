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

"""Native vllm-omni rollout for BAGEL AR-GRPO + image OmniNFT."""

from itertools import chain

import torch
from vllm_omni.diffusion.layers.custom_op import CustomOp
from vllm_omni.diffusion.layers.mot.mot_layernorm import MoTRMSNorm
from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelPipeline

from verl_omni.pipelines.diffusion_rollout_output import with_rollout_data
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec

from .capture import BagelReplayCapture
from .weight_sync import native_actor_weight_name


@VllmOmniPipelineBase.register("OmniBagelForConditionalGeneration", algorithm="unigrpo_nft")
class BagelUniGRPONFTPipeline(BagelPipeline):
    """Run native thinking and image generation, then export exact training inputs."""

    supports_request_batch = False
    diffusion_io_spec = DiffusionIOSpec(primary=MediaSpec("image"))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.device.type == "npu":
            # The pinned NPU image leaves MoT normalization's NPU dispatch
            # unimplemented. Use its own FP32-accumulating PyTorch kernel,
            # scoped to this pipeline; never patch installed/global classes.
            for module in self.modules():
                if isinstance(module, MoTRMSNorm) and type(module).forward_npu is CustomOp.forward_npu:
                    module._forward_method = module.forward_native

    def load_weights(self, weights):
        """Sync both expert paths AND image projections through the native stacked loader."""
        iterator = iter(weights)
        first = next(iterator, None)
        if first is None:
            return set()
        if not first[0].startswith("transformer."):
            return super().load_weights(chain((first,), iterator))
        loaded = set()
        for name, tensor in chain((first,), iterator):
            if not name.startswith("transformer."):
                raise ValueError("Actor policy-sync buckets must not mix checkpoint and actor keys")
            part = super().load_weights(((native_actor_weight_name(name), tensor),))
            if not part:
                raise ValueError(f"Native BAGEL did not load actor tensor: {name}")
            loaded.update(part)
            if getattr(self, "_joint_sync_in_progress", False):
                self._joint_loaded_actor_keys.add(name)
        return loaded

    def forward(self, req):
        """Single text-to-image request; reject hidden fallbacks and unsupported KV inputs."""
        if len(req.prompts) != 1:
            raise ValueError("BAGEL joint rollout requires one sequence per native request")
        prompt = req.prompts[0]
        if not isinstance(prompt, dict) or not isinstance(prompt.get("prompt"), str):
            raise ValueError("BAGEL joint rollout requires original prompt text, not decoded token IDs")
        if prompt.get("multi_modal_data") or "text" in prompt.get("modalities", []):
            raise ValueError("This BAGEL joint recipe requires text-to-image output")
        if getattr(req.sampling_params, "past_key_values", None) is not None:
            raise ValueError("Injected KV caches cannot provide an exact joint actor replay")
        extra = req.sampling_params.extra_args
        if not extra.get("think") or not extra.get("do_sample"):
            raise ValueError("Joint training requires think=true and do_sample=true")
        if "policy_version" not in extra:
            raise ValueError("Rollout must declare the synchronized old-policy version")
        if getattr(self, "_joint_sync_in_progress", False):
            raise RuntimeError("BAGEL policy sync has not committed")
        if int(extra["policy_version"]) != getattr(self, "_joint_policy_version", 0):
            raise ValueError("Requested rollout policy version has not been loaded by this worker")
        capture = BagelReplayCapture(self.bagel)
        # The pinned native pipeline seeds only AFTER thinking. Seed both policies.
        if req.sampling_params.seed is not None:
            torch.manual_seed(int(req.sampling_params.seed))
        with capture.installed():
            output = super().forward(req)
        replay = capture.result(policy_version=int(extra["policy_version"]))
        return with_rollout_data(
            output,
            rl={
                "bagel_joint_contract_version": 1,
                "bagel_joint_replay": replay,
                "latents_clean": replay.latents_clean.unsqueeze(0),
            },
            to_cpu=False,
        )
