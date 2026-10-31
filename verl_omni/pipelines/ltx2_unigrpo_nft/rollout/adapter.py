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

"""Batched Gemma recaptioning followed by main's native audiovisual rollout."""

import re
from copy import copy
from dataclasses import asdict

import torch
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from transformers import Gemma3ForConditionalGeneration
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.ltx2_omni_nft.vllm_omni_rollout_adapter import LTX23OmniNFTPipeline
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.rollout_request import prompt_ids_from_payload

from ..replay import LTXJointReplay
from ..text.policy import base_encoder
from ..text.prompt import direct_recaption
from ..text.sampling import generate_recaptions
from .cpu_weights import FrozenCpuWeights
from .offload import AdapterCpuWeights, JointRolloutOffload


def split_lora(tensors, peft_config):
    """Route one transported joint adapter into exact Gemma and DiT targets."""
    gemma, dit = {}, {}
    for name, value in tensors.items():
        key = (
            name.removeprefix("base_model.model.")
            .replace(".default.weight", ".weight")
            .replace(".old.weight", ".weight")
        )
        # The diffusion engine wraps exported composite-model names in transformer.
        if key.startswith(("transformer.gemma.", "transformer.transformer.")):
            key = key.removeprefix("transformer.")
        if key.startswith("gemma.model.language_model."):
            gemma[key.removeprefix("gemma.")] = value
        elif key.startswith("transformer."):
            dit["base_model.model." + key] = value
        else:
            raise ValueError(f"Unexpected joint LoRA tensor: {name}")
    if not gemma or not dit:
        raise ValueError("Joint publication requires both Gemma and DiT adapters")

    def config(weights, prefix):
        targets = sorted({key.removeprefix(prefix).split(".lora_")[0] for key in weights})
        return {**peft_config, "target_modules": "^(?:" + "|".join(re.escape(key) for key in targets) + ")$"}

    return gemma, config(gemma, ""), dit, config(dit, "base_model.model.")


@VllmOmniPipelineBase.register("LTX2Pipeline", algorithm="ltx2_unigrpo_nft")
class LTXJointPipeline(LTX23OmniNFTPipeline):
    supports_request_batch = True
    support_image_input = False

    def __init__(self, *, od_config, prefix=""):
        super().__init__(od_config=od_config, prefix=prefix)
        # Load the causal LM used for recaptioning in the same BF16 dtype as the actor.
        del self.text_encoder
        self.text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            od_config.model, subfolder="text_encoder", torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        self._gemma_compute_dtype = (od_config.additional_config or {}).get("gemma_compute_dtype", "bfloat16")
        if self._gemma_compute_dtype not in {"bfloat16", "float32"}:
            raise ValueError("gemma_compute_dtype must be bfloat16 or float32")
        if self._gemma_compute_dtype == "float32":
            from ..text.precision import install_fp32_gemma_compute

            install_fp32_gemma_compute(self.text_encoder)
        for name in JointRolloutOffload.component_names:
            getattr(self, name).requires_grad_(False)
        self._joint_offload = None
        self._joint_policy_version = -1
        self._joint_lora_id = None
        self._joint_sync_in_progress = False
        self._gemma_config = None
        self._pending_dit = False
        self._prepared_text = {}

    def enable_omni_model_cpu_offload(self, *, device, pin_memory, use_hsdp):
        if use_hsdp:
            raise ValueError("Joint component offload currently requires an unsharded rollout worker")
        if self._joint_offload is None:
            for name in JointRolloutOffload.component_names:
                component = getattr(self, name)
                component.requires_grad_(False)
                component._joint_cpu_weights = FrozenCpuWeights(component)
            self._joint_offload = JointRolloutOffload(self, device)

    def disable_omni_model_cpu_offload(self):
        if self._joint_offload is not None:
            self._joint_offload.disable()
            self._joint_offload = None

    def map_lora_update_to_engine(self, tensors, peft_config):
        if self._prepared_text:
            raise RuntimeError("Cannot publish weights while a staged rollout is pending")
        gemma, config, dit, dit_config = split_lora(tensors, peft_config)
        self._joint_sync_in_progress = True
        if self._joint_offload is not None:
            self._joint_offload.release()
        if self._gemma_config is None:
            inject_adapter_in_model(LoraConfig(**config), self.text_encoder, adapter_name="joint")
            if self._gemma_compute_dtype == "float32":
                from ..text.precision import install_fp32_gemma_compute

                install_fp32_gemma_compute(self.text_encoder)
            self._gemma_config = config
        elif config != self._gemma_config:
            raise ValueError("Cannot change joint adapter layout during a run")
        result = set_peft_model_state_dict(self.text_encoder, gemma, adapter_name="joint")
        if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
            raise ValueError("Gemma LoRA update did not load exactly")
        self.text_encoder.requires_grad_(False)
        # Native manager publishes DiT after this mapper returns; forward is admitted only afterwards.
        self._pending_dit = True
        return dit, dit_config

    def encode_prompt(self, *args, **kwargs):
        kwargs["dtype"] = next(self.connectors.parameters()).dtype
        with base_encoder(self.text_encoder):
            return super().encode_prompt(*args, **kwargs)

    def _prepare_text(self, requests):
        first = requests[0].sampling_params
        extra = first.extra_args or {}
        settings = (extra["recaption_temperature"], extra["recaption_max_new_tokens"], extra["recaption_prompt"])
        prompts = []
        for request in requests:
            local = request.sampling_params.extra_args or {}
            if local.get("gemma_compute_dtype", "bfloat16") != getattr(self, "_gemma_compute_dtype", "bfloat16"):
                raise ValueError("Request and rollout engine Gemma compute precision must match")
            if (
                tuple(local[key] for key in ("recaption_temperature", "recaption_max_new_tokens", "recaption_prompt"))
                != settings
            ):
                raise ValueError("AR batches require uniform text sampling settings")
            ids = prompt_ids_from_payload(request.prompt)
            text = (
                self.tokenizer.decode(ids, skip_special_tokens=True)
                if ids is not None
                else request.prompt.get("prompt")
            )
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Joint rollout requires a nonempty original prompt")
            prompts.append(text)
        if self._joint_offload is not None:
            self._joint_offload.activate("text_encoder")
        generated = generate_recaptions(
            self.text_encoder,
            self.tokenizer,
            prompts,
            [request.sampling_params.seed for request in requests],
            temperature=settings[0],
            max_new_tokens=settings[1],
            preset=settings[2],
            duration_seconds=(first.num_frames - 1) / first.frame_rate,
        )
        prepared, negatives = [], {}
        for request, original, sample in zip(requests, prompts, generated, strict=True):
            text, prompt_ids, response_ids, logprobs, eos = sample
            caption = direct_recaption(text, ended_with_eos=int(response_ids[-1]) in eos)
            maximum = request.sampling_params.max_sequence_length or self.tokenizer_max_length
            self.tokenizer.padding_side = "left"
            tokens = self.tokenizer(
                caption,
                padding="max_length",
                max_length=maximum,
                truncation=False,
                add_special_tokens=True,
                return_tensors="pt",
            )
            if tokens.input_ids.shape[1] > maximum:
                raise ValueError("Caption exceeds condition encoder context")
            ids, mask = tokens.input_ids.to(self.device), tokens.attention_mask.to(self.device).bool()
            with base_encoder(self.text_encoder):
                hidden = self.text_encoder(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
            embeddings = torch.stack(hidden, dim=-1).flatten(2, 3).to(next(self.connectors.parameters()).dtype).cpu()
            del hidden
            negative = None
            local = request.sampling_params.extra_args or {}
            if (local.get("video_cfg_scale") or 1) > 1 or (local.get("audio_cfg_scale") or 1) > 1:
                if maximum not in negatives:
                    with base_encoder(self.text_encoder):
                        negatives[maximum] = tuple(
                            value.cpu()
                            for value in self._get_gemma_prompt_embeds(
                                prompt="", max_sequence_length=maximum, dtype=next(self.connectors.parameters()).dtype
                            )
                        )
                negative = negatives[maximum]
            record = LTXJointReplay(
                original_prompt=original,
                recaption=caption,
                prompt_token_ids=prompt_ids,
                response_token_ids=response_ids,
                rollout_logprobs=logprobs,
                condition_token_ids=ids[0].cpu(),
                condition_attention_mask=mask[0].cpu(),
                policy_version=self._joint_policy_version,
                seed=int(request.sampling_params.seed),
                temperature=settings[0],
                eos_token_ids=eos,
                response_text=text,
                response_format="caption",
                generation_params={
                    key: getattr(request.sampling_params, key)
                    for key in ("height", "width", "num_frames", "frame_rate", "num_inference_steps")
                },
            )
            record.validate()
            prepared.append((request, record, embeddings, mask.cpu(), negative))
        return prepared

    def _admit_policy(self, requests):
        versions = {request.sampling_params.lora_request.lora_int_id for request in requests}
        if len(versions) != 1:
            raise ValueError("Joint rollout batches must use one adapter version")
        version = versions.pop()
        if self._pending_dit:
            for name in ("text_encoder", "transformer"):
                component = getattr(self, name)
                # Native sidecars may be on device after activation; park the freshly loaded slots once.
                if name == "transformer":
                    from .offload import move_native_lora_sidecars

                    move_native_lora_sidecars(component, torch.device("cpu"))
                component._joint_adapter_cpu_weights = AdapterCpuWeights(component, self._joint_policy_version + 1)
            self._joint_policy_version += 1
            self._joint_lora_id = version
            self._pending_dit = self._joint_sync_in_progress = False
        if self._joint_sync_in_progress or version != self._joint_lora_id:
            raise RuntimeError("No matching paired Gemma/DiT rollout policy")

    @torch.no_grad()
    def prepare_rollout_requests(self, requests):
        """Encode a replica's entire round; keep only CPU conditions between stages."""
        if self._prepared_text:
            raise RuntimeError("Previous rollout conditions have not been consumed")
        self._admit_policy(requests)
        keys = [request.sampling_params.extra_args["prepared_request_id"] for request in requests]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate staged request IDs")
        try:
            batch_size = requests[0].sampling_params.extra_args["recaption_batch_size"]
            for start in range(0, len(requests), batch_size):
                for prepared in self._prepare_text(requests[start : start + batch_size]):
                    key = prepared[0].sampling_params.extra_args["prepared_request_id"]
                    self._prepared_text[key] = prepared
        except BaseException:
            self._prepared_text.clear()
            raise
        finally:
            if self._joint_offload is not None:
                self._joint_offload.release()
        return len(self._prepared_text)

    def clear_prepared_rollout(self):
        """Drop unconsumed CPU conditions after completion or failure."""
        self._prepared_text.clear()

    @torch.no_grad()
    def forward(self, req):
        if any(OmniDiffusionRequest.is_dummy_run_request_id(request.request_id) for request in req.requests):
            return super().forward(req)
        self._admit_policy(req.requests)
        results = []
        try:
            for incoming in req.requests:
                key = (incoming.sampling_params.extra_args or {}).get("prepared_request_id")
                if key not in self._prepared_text:
                    raise RuntimeError("Stage 2 requires prepared text on this worker; recaption fallback is disabled")
                original, record, embeddings, mask, negative = self._prepared_text.pop(key)
                if (
                    record.policy_version != self._joint_policy_version
                    or original.sampling_params.seed != incoming.sampling_params.seed
                ):
                    raise RuntimeError("Staged condition does not match the rollout policy/seed")
                request = copy(incoming)
                request.prompt = {
                    "prompt": record.recaption,
                    "prompt_embeds": embeddings[0].to(self.device),
                    "prompt_attention_mask": mask[0].to(self.device),
                }
                if negative is not None:
                    request.prompt.update(
                        negative_prompt_embeds=negative[0][0].to(self.device),
                        negative_prompt_attention_mask=negative[1][0].to(self.device),
                    )
                output = super().forward(DiffusionRequestBatch(requests=[request]))
                output.output["metadata"]["rl"]["ltx_joint_replay"] = asdict(record)
                results.append(output)
                if self._joint_offload is not None:
                    self._joint_offload.release()
            return results[0] if len(results) == 1 else results
        finally:
            if self._joint_offload is not None:
                self._joint_offload.release()
