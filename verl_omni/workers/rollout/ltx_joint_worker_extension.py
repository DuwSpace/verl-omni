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

    def collect_joint_av_probe(self, record_data, inputs, identity):
        """Replay an actor forward-process sample through the committed native pair."""
        import torch
        from vllm_omni.diffusion.forward_context import set_forward_context

        from verl_omni.pipelines.ltx2_unigrpo_nft.consistency_probe import capture_forward, probe_selected
        from verl_omni.pipelines.ltx2_unigrpo_nft.replay import LTXJointReplay

        record = LTXJointReplay(**record_data)
        record.validate()
        pipeline = self.model_runner.pipeline
        if not probe_selected(record.policy_version) or pipeline._joint_policy_version != record.policy_version:
            raise RuntimeError("Native probe window or paired policy does not match actor old")
        models = torch.nn.ModuleDict(
            {"gemma": pipeline.text_encoder, "connectors": pipeline.connectors, "transformer": pipeline.transformer}
        )
        device = pipeline.device
        try:
            with (
                torch.no_grad(),
                set_forward_context(
                    vllm_config=self.model_runner.vllm_config, omni_diffusion_config=pipeline.od_config
                ),
                capture_forward(models, side="generate", identity=identity),
            ):
                embeddings, ids, mask = pipeline._condition(record.recaption, record.condition_token_ids.numel())
                if not torch.equal(ids.cpu()[0], record.condition_token_ids) or not torch.equal(
                    mask.cpu()[0], record.condition_attention_mask.bool()
                ):
                    raise RuntimeError("Native diagnostic condition tokens differ from actor replay")
                video, audio, attention = pipeline.connectors(embeddings, mask, padding_side="left")
                kwargs = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in inputs.items()
                    if key not in ("input_ids", "attention_mask")
                }
                predictions = pipeline.predict_noise(
                    **kwargs,
                    encoder_hidden_states=video,
                    audio_encoder_hidden_states=audio,
                    encoder_attention_mask=attention,
                    audio_encoder_attention_mask=attention,
                    return_dict=False,
                )
                if not all(torch.isfinite(value).all() for value in predictions):
                    raise RuntimeError("Nonfinite native diagnostic prediction")
        finally:
            self.release_joint_device_memory()
        return {"policy_version": record.policy_version, "sample_id": identity["sample_id"]}

    def release_joint_device_memory(self):
        """Release managed model residency before colocated reward/training.

        Sequential offload swaps Gemma and DiT on their next forward; it does
        not promise to offload the last-used model at the end of a request.
        Their installed pre-forward hooks restore residency on the next batch.
        VAEs and connectors retain their existing backend lifecycle.
        """
        import torch

        pipeline = self.model_runner.pipeline
        device = torch.device(pipeline.device)
        device_api = torch.get_device_module(device) if device.type != "cpu" else None
        if device_api is not None:
            device_api.synchronize()
        if getattr(pipeline.od_config, "enable_cpu_offload", False):
            pipeline.text_encoder.to("cpu")
            pipeline.transformer.to("cpu")
        if device_api is not None:
            device_api.empty_cache()
            return {"allocated_bytes": device_api.memory_allocated(), "reserved_bytes": device_api.memory_reserved()}
        return {"allocated_bytes": 0, "reserved_bytes": 0}

    def re_init_pipeline(self, custom_pipeline_args):
        from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension

        return CustomPipelineWorkerExtension.re_init_pipeline(self, custom_pipeline_args)

    def load_joint_lora_bundle(self, directory):
        from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
        from safetensors.torch import load_file
        from vllm.lora.request import LoRARequest
        from vllm_omni.diffusion.lora.layers.base_linear import DiffusionBaseLinearLayerWithLoRA

        pipeline = self.model_runner.pipeline
        manifest = validate_joint_lora_bundle(directory)
        base_model = manifest.get("base_model")
        if not isinstance(base_model, str) or Path(base_model).resolve() != Path(pipeline.od_config.model).resolve():
            raise ValueError("Joint LoRA base model differs from the live native model")
        version = manifest["policy_version"]
        if version <= getattr(pipeline, "_joint_policy_version", -1):
            raise ValueError("Joint rollout policy versions must increase")
        pipeline._joint_sync_in_progress = True
        directory = Path(directory)
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
        request = LoRARequest(
            lora_name=f"ltx-joint-{version}", lora_int_id=version + 1, lora_path=str(directory / "dit")
        )
        self.lora_manager.set_active_adapter(request, 1.0)
        # Model-level offload moves parameters/buffers, but native LoRA stores
        # its small sidecars in tensor tuples. Keep them on the compute device
        # using the backend's residency hook (also retained on rank resizing).
        for module in pipeline.transformer.modules():
            if isinstance(module, DiffusionBaseLinearLayerWithLoRA):
                module._set_diffusion_lora_buffer_device(pipeline.device)
        pipeline._joint_lora_request = request
        pipeline._joint_policy_version = version
        pipeline._joint_sync_in_progress = False
        return {"policy_version": version, "components": manifest["components"]}
