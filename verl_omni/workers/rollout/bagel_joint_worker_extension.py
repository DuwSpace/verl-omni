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

"""Worker RPC boundary for atomic full-weight BAGEL rollout-policy sync."""

import os

import torch


class BagelJointWorkerExtension:
    """Mixin for the existing vllm-omni diffusion worker; no private dependency patches."""

    def re_init_pipeline(self, custom_pipeline_args):
        """Reuse the native custom-pipeline initialization contract."""
        from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension

        return CustomPipelineWorkerExtension.re_init_pipeline(self, custom_pipeline_args)

    def _joint_pipeline(self):
        pipeline = getattr(self.model_runner, "pipeline", None)
        if pipeline is None:
            pipeline = self.model_runner.model
        if not hasattr(pipeline, "bagel"):
            raise TypeError("Joint policy worker requires the native BAGEL pipeline")
        return pipeline

    def joint_model_info(self):
        """Report authoritative worker model/loader identity and small parameter probes."""
        pipeline = self._joint_pipeline()
        probe_names = (
            "language_model.model.layers.0.self_attn.qkv_proj.weight",
            "language_model.model.layers.0.self_attn.qkv_proj.gen_exp.weight",
            "bagel.vae2llm.weight",
            "bagel.llm2vae.weight",
            "bagel.time_embedder.mlp.0.weight",
        )
        state = pipeline.state_dict()
        probes = {
            name: {
                "shape": list(state[name].shape),
                "dtype": str(state[name].dtype),
                "values": state[name].flatten()[:8].float().cpu().tolist(),
            }
            for name in probe_names
            if name in state
        }
        return {
            "pipeline": f"{type(pipeline).__module__}.{type(pipeline).__name__}",
            "device": str(self.device),
            "pid": os.getpid(),
            "policy_version": getattr(pipeline, "_joint_policy_version", 0),
            "parameter_elements": sum(p.numel() for p in pipeline.parameters()),
            "probes": probes,
        }

    def begin_joint_policy_sync(self, version: int, expected_keys: list[str]):
        """Invalidate rollout until all named actor tensors load and the version commits."""
        pipeline = self._joint_pipeline()
        if getattr(pipeline, "_joint_sync_in_progress", False):
            raise RuntimeError("A joint policy sync is already in progress")
        if version < getattr(pipeline, "_joint_policy_version", 0):
            raise ValueError("Cannot sync an older rollout policy")
        if not expected_keys or len(set(expected_keys)) != len(expected_keys):
            raise ValueError("Policy sync must declare a nonempty unique actor tensor manifest")
        pipeline._joint_expected_actor_keys = set(expected_keys)
        pipeline._joint_loaded_actor_keys = set()
        pipeline._joint_pending_policy_version = int(version)
        pipeline._joint_sync_in_progress = True
        return {"version": version, "expected_tensors": len(expected_keys)}

    def load_joint_weight_file(self, path: str):
        """Load a trusted local CPU bucket, used by the file-based smoke/training runner."""
        pipeline = self._joint_pipeline()
        if not getattr(pipeline, "_joint_sync_in_progress", False):
            raise RuntimeError("Begin joint policy sync before loading weight buckets")
        weights = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(weights, dict) or not weights:
            raise ValueError("Actor weight bucket must be a nonempty named tensor dictionary")
        unexpected = set(weights) - pipeline._joint_expected_actor_keys
        if unexpected:
            raise ValueError(f"Undeclared actor tensors in policy sync: {sorted(unexpected)}")
        pipeline.load_weights(
            (name, tensor.to(device=self.device, dtype=torch.bfloat16)) for name, tensor in weights.items()
        )
        return {"loaded_tensors": len(weights)}

    def commit_joint_policy_sync(self):
        """Admit rollout only after complete tensor-manifest coverage."""
        pipeline = self._joint_pipeline()
        if not getattr(pipeline, "_joint_sync_in_progress", False):
            raise RuntimeError("No joint policy sync to commit")
        if pipeline._joint_loaded_actor_keys != pipeline._joint_expected_actor_keys:
            missing = sorted(pipeline._joint_expected_actor_keys - pipeline._joint_loaded_actor_keys)
            raise RuntimeError(f"Incomplete BAGEL rollout-policy sync; missing tensors: {missing}")
        pipeline._joint_policy_version = pipeline._joint_pending_policy_version
        pipeline._joint_sync_in_progress = False
        return {"version": pipeline._joint_policy_version, "loaded_tensors": len(pipeline._joint_loaded_actor_keys)}
