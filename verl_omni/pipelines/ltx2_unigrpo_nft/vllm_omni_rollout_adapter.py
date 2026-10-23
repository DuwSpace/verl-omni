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

"""Two-stage recaption and AV generation in one native vLLM-Omni LTX worker."""

import os
from contextlib import nullcontext
from copy import copy
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import torch
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.diffusion_rollout_output import with_rollout_data
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec

from .config import GUIDANCE_DEFAULTS
from .model import base_encoder
from .replay import LTXJointReplay, direct_recaption, planning_instruction


@VllmOmniPipelineBase.register("LTX2Pipeline", algorithm="ltx2_unigrpo_nft")
class LTXJointPipeline(LTX2Pipeline):
    """The worker's existing Gemma performs both recaption sampling and encoding.

    This is native diffusion-worker execution, using Gemma's Transformers
    generation API within vLLM-Omni. It does not instantiate a second actor-side
    inference model or claim to use a separate paged-KV autoregressive engine.
    """

    supports_request_batch = False
    support_image_input = False
    diffusion_io_spec = DiffusionIOSpec(primary=MediaSpec("video"), auxiliary=(MediaSpec("audio", sample_rate=24000),))

    def __init__(self, *, od_config, prefix=""):
        from vllm_omni.platforms import current_omni_platform

        from verl_omni.utils.determinism import configure_worker_determinism

        # Omni's set_device() enables private formats. Override it before the
        # base constructor allocates or loads model tensors, matching the actor.
        if current_omni_platform.device_type == "npu":
            torch.npu.config.allow_internal_format = False
            print("LTX_NPU_FORMAT side=rollout allow_internal_format=False", flush=True)
        configure_worker_determinism()
        self._shared_cpu_weights_dir = (od_config.extras or {}).get("ltx_shared_cpu_weights_dir")
        super().__init__(od_config=od_config, prefix=prefix)
        if os.environ.get("LTX_DISABLE_PROGRESS_BAR", "0") == "1":
            self.set_progress_bar_config(disable=True)
        from transformers import Gemma3ForConditionalGeneration

        # The native loader currently applies one dtype to every component.
        # Reload the original Gemma weights: upcasting the already-BF16 model
        # cannot recover checkpoint values rounded during that initial load.
        # This happens before worker offload hooks and paired LoRA injection.
        del self.text_encoder
        self.text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            od_config.model,
            subfolder="text_encoder",
            torch_dtype=torch.float32,
            revision=getattr(od_config, "revision", None),
            attn_implementation="sdpa",
        )
        # Native workers only execute rollout; none of their base/codec
        # parameters participate in the actor optimizer or require gradients.
        for name in ("text_encoder", "transformer", "connectors", "vae", "audio_vae", "vocoder"):
            component = getattr(self, name, None)
            if isinstance(component, torch.nn.Module):
                component.requires_grad_(False)
        from verl_omni.utils.frozen_cpu_weights import FrozenCpuWeights

        # Gemma has already loaded its checkpoint; the DiT loader runs later.
        self.text_encoder._joint_cpu_weights = FrozenCpuWeights(
            self.text_encoder, shared_path=self._shared_weight_path("text_encoder")
        )

    def _shared_weight_path(self, name):
        directory = getattr(self, "_shared_cpu_weights_dir", None)
        return Path(directory) / f"{name}.pt" if directory else None

    def enable_omni_model_cpu_offload(self, *, device, pin_memory, use_hsdp):
        from verl_omni.utils.frozen_cpu_weights import FrozenCpuWeights

        from .offload import JointRolloutOffload

        if use_hsdp:
            raise ValueError("Joint rollout component offload requires an unsharded worker")
        if getattr(self, "_joint_offload", None) is None:
            # The native runner calls this AFTER checkpoint loading and BEFORE
            # creating the LoRA manager. Never capture an uninitialized DiT or
            # include later-injected mutable policy adapters in its base master.
            for name in JointRolloutOffload.component_names:
                component = getattr(self, name)
                if getattr(component, "_joint_cpu_weights", None) is None:
                    component.requires_grad_(False)
                    component._joint_cpu_weights = FrozenCpuWeights(
                        component, shared_path=self._shared_weight_path(name)
                    )
            self._joint_offload = JointRolloutOffload(self, device)

    def disable_omni_model_cpu_offload(self):
        controller = getattr(self, "_joint_offload", None)
        if controller is not None:
            controller.disable()
            self._joint_offload = None

    def encode_prompt(self, *args, **kwargs):
        # Native negative encoding otherwise inherits FP32 from Gemma, while
        # the positive embedding already uses the connector's BF16 dtype.
        kwargs["dtype"] = next(self.connectors.parameters()).dtype
        with base_encoder(self.text_encoder):
            return super().encode_prompt(*args, **kwargs)

    def _unpack_and_denormalize_stage(self, forward_ctx, latents, audio_latents):
        self._joint_clean = (latents, audio_latents[:, : forward_ctx.original_audio_num_frames])
        self._joint_forward_context = forward_ctx
        return super()._unpack_and_denormalize_stage(forward_ctx, latents, audio_latents)

    def _recaption(self, prompt, *, seed, temperature, max_new_tokens, duration_seconds=None):
        from transformers import GenerationConfig

        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("Recaption max_new_tokens must be a positive integer")
        instruction = planning_instruction(prompt, duration_seconds)
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            return_tensors="pt",
        ).to(self.device)
        eos = self.text_encoder.generation_config.eos_token_id
        eos = (eos,) if isinstance(eos, int) else tuple(eos)
        sampling = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            top_k=0,
            top_p=1.0,
            typical_p=1.0,
            repetition_penalty=1.0,
            max_new_tokens=max_new_tokens,
            eos_token_id=list(eos),
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
        device = torch.device(self.device)
        rng_kwargs = {} if device.type == "cpu" else {"device_type": device.type}
        devices = [] if device.type == "cpu" else [device.index or 0]
        residency = nullcontext()
        if getattr(self, "_joint_offload", None) is None and getattr(
            getattr(self, "od_config", None), "enable_cpu_offload", False
        ):
            from vllm_omni.diffusion.offloader.sequential_backend import sequential_offload_component

            # generate() consults model.device before invoking forward hooks.
            # After an AV request, Gemma is offloaded; activate it before any
            # generation preprocessing, not just before its first forward.
            residency = sequential_offload_component(self.text_encoder)
        with residency, torch.random.fork_rng(devices=devices, **rng_kwargs):
            torch.manual_seed(seed)
            if device.type != "cpu":
                torch.get_device_module(device).manual_seed(seed)
            output = self.text_encoder.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=sampling
            )
        response = output.sequences[0, ids.shape[1] :]
        if response.numel() != len(output.scores) or response.numel() == 0:
            raise RuntimeError("Recaption generation did not return one processed score per sampled token")
        logprobs = torch.stack(
            [score[0].float().log_softmax(-1)[token] for score, token in zip(output.scores, response, strict=True)]
        )
        text = self.tokenizer.decode(response, skip_special_tokens=True).strip()
        if not text:
            raise RuntimeError("Gemma generated an empty recaption")
        return text, ids[0].cpu(), response.cpu(), logprobs.cpu(), eos

    def _condition(self, text, max_length):
        self.tokenizer.padding_side = "left"
        encoded = self.tokenizer(
            text.strip(),
            padding="max_length",
            max_length=max_length,
            truncation=False,
            add_special_tokens=True,
            return_tensors="pt",
        )
        if encoded.input_ids.shape[1] > max_length:
            raise ValueError("Recaption exceeds encoder context; refusing silent truncation")
        ids = encoded.input_ids.to(self.device)
        mask = encoded.attention_mask.to(self.device).bool()
        with torch.no_grad(), base_encoder(self.text_encoder):
            hidden = self.text_encoder(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        embeddings = torch.stack(hidden, dim=-1).flatten(2, 3).to(next(self.connectors.parameters()).dtype)
        return embeddings, ids, mask

    @torch.no_grad()
    def forward(self, req):
        """Run one committed policy request and release residency on every exit."""
        from verl_omni.utils.determinism import configure_worker_determinism

        configure_worker_determinism()
        if len(req.requests) != 1:
            raise ValueError("Initial LTX joint rollout admits exactly one request per worker")
        incoming = req.requests[0]
        dummy = OmniDiffusionRequest.is_dummy_run_request_id(incoming.request_id)
        self._joint_clean = self._joint_forward_context = None
        try:
            if dummy:
                return super().forward(req)
            original, version, extra = self._validate_joint_request(incoming)
            device = torch.device(self.device)
            device_api = torch.get_device_module(device) if device.type != "cpu" else None
            if device_api is not None:
                device_api.reset_peak_memory_stats()
            text_record, recaption_time = self._recaption_request(incoming, original, version, extra)
            return self._generate_av_request(incoming, text_record, recaption_time, device_api)
        finally:
            self._joint_clean = self._joint_forward_context = None
            controller = getattr(self, "_joint_offload", None)
            if controller is not None:
                controller.release()

    def _validate_joint_request(self, incoming):
        """Require the committed Gemma/DiT pair and an original user prompt."""
        extra = incoming.sampling_params.extra_args or {}
        version = extra.get("policy_version")
        if getattr(self, "_joint_sync_in_progress", True) or version != getattr(self, "_joint_policy_version", None):
            raise RuntimeError("Joint policy pair is not committed for this rollout version")
        request_lora = incoming.sampling_params.lora_request
        if request_lora is None or request_lora.lora_int_id != self._joint_lora_request.lora_int_id:
            raise RuntimeError("Request DiT LoRA does not match the committed Gemma/DiT pair")
        original = incoming.prompt.get("prompt")
        if not isinstance(original, str) or not original.strip():
            raise ValueError("Joint rollout requires the original user prompt text")
        return original, version, extra

    def _recaption_request(self, incoming, original, version, extra):
        """Sample text, capture R/R-prime immediately, then parse the AV caption."""
        seed = int(incoming.sampling_params.seed)
        temperature = float(extra["recaption_temperature"])
        started = perf_counter()
        response_text, prompt_ids, response_ids, logprobs, eos = self._recaption(
            original,
            seed=seed,
            temperature=temperature,
            max_new_tokens=extra["recaption_max_new_tokens"],
            duration_seconds=(incoming.sampling_params.num_frames - 1) / incoming.sampling_params.frame_rate,
        )
        text_diagnostic = None
        diagnostic = getattr(self, "_joint_text_diagnostics", None)
        if diagnostic is not None and diagnostic.enabled:
            # Preserve R/R-prime and buffers before condition encoding, AV, or offload.
            text_diagnostic = diagnostic.rollout(
                self.text_encoder,
                prompt_ids,
                response_ids,
                logprobs,
                version=version,
                seed=seed,
                temperature=temperature,
                request_id=incoming.request_id,
            )
        ended_with_eos = int(response_ids[-1]) in eos
        try:
            text = direct_recaption(response_text, ended_with_eos=ended_with_eos)
        except ValueError as error:
            # Keep the sampled text in the RPC error: failed requests never
            # reach periodic sample saving, making the old error unactionable.
            raise ValueError(
                f"{error}; policy_version={version}, seed={seed}, "
                f"response_tokens={response_ids.numel()}, "
                f"max_new_tokens={extra['recaption_max_new_tokens']}, "
                f"ended_with_eos={ended_with_eos}, response_text={response_text!r}"
            ) from error
        recaption_time = perf_counter() - started
        return dict(
            original_prompt=original,
            recaption=text,
            planning=None,
            response_format="caption",
            response_text=response_text,
            prompt_token_ids=prompt_ids,
            response_token_ids=response_ids,
            rollout_logprobs=logprobs,
            policy_version=version,
            text_diagnostic=text_diagnostic,
            seed=seed,
            temperature=temperature,
            eos_token_ids=eos,
        ), recaption_time

    def _generate_av_request(self, incoming, text_record, recaption_time, device_api):
        """Encode the caption, run native AV generation, and attach exact replay geometry."""
        text = text_record["recaption"]
        extra = incoming.sampling_params.extra_args or {}
        generation_start = perf_counter()
        maximum = incoming.sampling_params.max_sequence_length or self.tokenizer_max_length
        embeddings, condition_ids, condition_mask = self._condition(text, maximum)
        request = copy(incoming)
        request.prompt = {
            "prompt": text,
            "prompt_embeds": embeddings[0],
            "prompt_attention_mask": condition_mask[0],
        }
        request.sampling_params = copy(incoming.sampling_params)
        request.sampling_params.output_type = "pt"
        output = super().forward(DiffusionRequestBatch(requests=[request]))
        if self._joint_clean is None or self._joint_forward_context is None:
            raise RuntimeError("Native LTX did not expose final clean latents")
        context = self._joint_forward_context
        geometry = context.request_inputs
        record = LTXJointReplay(
            **text_record,
            condition_token_ids=condition_ids[0].cpu(),
            condition_attention_mask=condition_mask[0].cpu(),
            generation_params={
                "height": geometry.height,
                "width": geometry.width,
                "num_frames": geometry.num_frames,
                "frame_rate": geometry.frame_rate,
                "num_inference_steps": geometry.num_inference_steps,
                "gemma_dtype": str(self.text_encoder.dtype),
                "max_sequence_length": maximum,
                **{key: extra.get(key, default) for key, default in GUIDANCE_DEFAULTS.items()},
            },
        )
        record.validate()
        return self._pack_av_result(output, record, context, recaption_time, generation_start, device_api)

    def _pack_av_result(self, output, record, context, recaption_time, generation_start, device_api):
        """Materialize media and clean latents before recording generation wall time."""
        video_clean, audio_clean = self._joint_clean
        geometry = context.request_inputs
        video, audio = output.output
        if video.ndim == 5:
            video = video[0]
        # Materialize outputs before ending the stage's wall-clock measurement.
        video_clean, audio_clean = video_clean.float().cpu(), audio_clean.float().cpu()
        return with_rollout_data(
            replace(output, output=(video, audio)),
            media_key="video",
            to_cpu=True,
            rl={
                "ltx_joint_replay": asdict(record),
                "video_latents_clean": video_clean,
                "audio_latents_clean": audio_clean,
                "audio": audio,
                "train_timesteps": context.timesteps.float().cpu().unsqueeze(0),
                "fps": torch.tensor([geometry.frame_rate]),
                "audio_sample_rate": torch.tensor([self.vocoder.config.output_sampling_rate]),
                "recaption_time": recaption_time,
                "generation_time": perf_counter() - generation_start,
                "memory_metrics": {
                    "rollout/max_memory_allocated_gib": device_api.max_memory_allocated() / 1024**3,
                    "rollout/max_memory_reserved_gib": device_api.max_memory_reserved() / 1024**3,
                }
                if device_api is not None
                else {},
            },
        )


__all__ = ["LTXJointPipeline"]
