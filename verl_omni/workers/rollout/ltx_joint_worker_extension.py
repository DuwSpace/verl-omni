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

"""Commit Gemma PEFT and native DiT LoRA together inside a vLLM-Omni worker."""

from pathlib import Path

from verl_omni.pipelines.ltx2_unigrpo_nft.weight_sync import validate_joint_lora_bundle


class LTXJointWorkerExtension:
    def snapshot_joint_frozen_components(self):
        """Opt-in before/after audit of native-only decoders and frozen connectors."""
        import hashlib

        import torch

        pipeline = self.model_runner.pipeline
        report = {}
        for name in ("connectors", "vae", "audio_vae", "vocoder"):
            component = getattr(pipeline, name, None)
            if not isinstance(component, torch.nn.Module):
                raise RuntimeError(f"Missing native frozen component: {name}")
            digest = hashlib.sha256()
            elements = 0
            for key, parameter in component.named_parameters():
                if parameter.requires_grad or parameter.grad is not None:
                    raise RuntimeError(f"Native frozen parameter has gradients enabled: {name}.{key}")
                value = parameter.detach().cpu().contiguous()
                digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}\n".encode())
                digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
                elements += value.numel()
            if not elements:
                raise RuntimeError(f"Native frozen component has no parameters: {name}")
            report[name] = {"sha256": digest.hexdigest(), "parameter_elements": elements}
        return report

    def release_joint_device_memory(self):
        """Release managed model residency before colocated reward/training.

        The joint controller also releases connectors, codecs and unregistered
        diffusion LoRA sidecars. Retain the legacy path for older pipelines.
        """
        import torch

        pipeline = self.model_runner.pipeline
        device = torch.device(pipeline.device)
        device_api = torch.get_device_module(device) if device.type != "cpu" else None
        if device_api is not None:
            device_api.synchronize()
        controller = getattr(pipeline, "_joint_offload", None)
        if controller is not None:
            controller.release()
        elif getattr(pipeline.od_config, "enable_cpu_offload", False):
            pipeline.text_encoder.to("cpu")
            pipeline.transformer.to("cpu")
        if device_api is not None:
            device_api.empty_cache()
            return {"allocated_bytes": device_api.memory_allocated(), "reserved_bytes": device_api.memory_reserved()}
        return {"allocated_bytes": 0, "reserved_bytes": 0}

    def re_init_pipeline(self, custom_pipeline_args):
        from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension

        return CustomPipelineWorkerExtension.re_init_pipeline(self, custom_pipeline_args)

    def snapshot_joint_text_failure(self, reason, force=False):
        """Save live native parameters and nonpersistent buffers without reloading."""
        pipeline = self.model_runner.pipeline
        diagnostic = getattr(pipeline, "_joint_text_diagnostics", None)
        if diagnostic is None or not diagnostic.enabled:
            raise RuntimeError("Native text diagnostics were not enabled")
        return diagnostic.snapshot(pipeline.text_encoder, reason, force=force)

    def load_joint_lora_bundle(self, directory, text_diagnostics=None, source_rank=None):
        """Validate → park → load Gemma/DiT → commit one paired policy version.

        An exception deliberately leaves synchronization in progress: a partially
        loaded policy must never become eligible for rollout.
        """
        pipeline = self.model_runner.pipeline
        manifest = validate_joint_lora_bundle(directory)
        base_model = manifest.get("base_model")
        if not isinstance(base_model, str) or Path(base_model).resolve() != Path(pipeline.od_config.model).resolve():
            raise ValueError("Joint LoRA base model differs from the live native model")
        version = manifest["policy_version"]
        if version <= getattr(pipeline, "_joint_policy_version", -1):
            raise ValueError("Joint rollout policy versions must increase")
        pipeline._joint_sync_in_progress = True
        controller = getattr(pipeline, "_joint_offload", None)
        if controller is not None:
            controller.begin_policy_update(version)
        directory = Path(directory)
        weights = self._load_gemma_adapter(directory)
        self._record_text_policy(weights, version, text_diagnostics, source_rank)
        if controller is not None:
            self._verify_gemma_adapter(weights, version)
        request = self._load_dit_adapter(directory, version, controller)
        if controller is not None:
            controller.commit_policy_update(version)
        pipeline._joint_lora_request = request
        pipeline._joint_policy_version = version
        pipeline._joint_sync_in_progress = False
        return {"policy_version": version, "components": manifest["components"]}

    def _load_gemma_adapter(self, directory):
        """Load the PEFT adapter on the parked model and return the transport tensors."""
        from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
        from safetensors.torch import load_file

        pipeline = self.model_runner.pipeline
        gemma_config = LoraConfig.from_pretrained(directory / "gemma")
        # PEFT may compress target_modules in place during injection. Compare
        # future bundles with the original transport config, not that rewrite.
        transport_config = gemma_config.to_dict()
        if not hasattr(pipeline, "_joint_gemma_config"):
            inject_adapter_in_model(gemma_config, pipeline.text_encoder, adapter_name="joint")
            pipeline._joint_gemma_config = transport_config
        elif transport_config != pipeline._joint_gemma_config:
            raise ValueError("Cannot change Gemma LoRA structure while syncing policy weights")
        weights = load_file(directory / "gemma" / "adapter_model.safetensors")
        result = set_peft_model_state_dict(pipeline.text_encoder, weights, adapter_name="joint")
        if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
            raise ValueError("Gemma LoRA did not load exactly")
        pipeline.text_encoder.requires_grad_(False)
        return weights

    def _record_text_policy(self, weights, version, text_diagnostics, source_rank):
        """Keep optional text evidence separate from mandatory policy verification."""
        pipeline = self.model_runner.pipeline
        if text_diagnostics is not None:
            from verl_omni.pipelines.ltx2_unigrpo_nft.text_diagnostics import TextDiagnostics

            if not hasattr(pipeline, "_joint_text_diagnostics"):
                pipeline._joint_text_diagnostics = TextDiagnostics(
                    text_diagnostics, side="rollout", source_rank=source_rank
                )
            pipeline._joint_text_diagnostics.policy(pipeline.text_encoder, version, expected=weights)

    def _verify_gemma_adapter(self, weights, version):
        """Compare actual loaded Gemma LoRA against the bundle before publication."""
        pipeline = self.model_runner.pipeline
        from verl_omni.utils.weight_integrity import first_different_byte_block

        # This is part of policy publication, independent of optional text
        # diagnostics. Compare the loaded CPU adapter against the bundle.
        parameters = dict(pipeline.text_encoder.named_parameters())
        for name, expected in weights.items():
            key = name.replace(".lora_A.weight", ".lora_A.joint.weight").replace(
                ".lora_B.weight", ".lora_B.joint.weight"
            )
            actual = parameters.get(key)
            if actual is None or first_different_byte_block(actual, expected) is not None:
                raise RuntimeError(f"Actual native Gemma LoRA differs from policy {version}: {name}")

    def _load_dit_adapter(self, directory, version, controller):
        """Load native DiT slots and leave them on CPU when component offload owns residency."""
        from vllm.lora.request import LoRARequest
        from vllm_omni.diffusion.lora.layers.base_linear import DiffusionBaseLinearLayerWithLoRA

        pipeline = self.model_runner.pipeline
        request = LoRARequest(
            lora_name=f"ltx-joint-{version}", lora_int_id=version + 1, lora_path=str(directory / "dit")
        )
        self.lora_manager.set_active_adapter(request, 1.0)
        # Native LoRA sidecars are tuples rather than registered buffers. The
        # joint controller owns their residency too, including rank resizing.
        sidecar_device = "cpu" if controller is not None else pipeline.device
        for module in pipeline.transformer.modules():
            if isinstance(module, DiffusionBaseLinearLayerWithLoRA):
                module._set_diffusion_lora_buffer_device(sidecar_device)
        return request
