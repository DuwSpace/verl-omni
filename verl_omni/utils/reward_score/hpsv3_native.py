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

"""Batch-native HPSv3 reward adapted from zghhui/OmniNFT."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from verl.utils.device import get_device_name, get_torch_device

from .hpsv3_reward import (
    _BASE_MODEL,
    _INSTRUCTION,
    _PROMPT_WITH_SPECIAL_TOKEN,
    _Qwen2VLRewardModelBT,
    _process_vision_info,
    _remap_state_dict,
)

_DEFAULT_MODEL_REVISION = "MizzenAI/HPSv3@4f81e3e09edd82fe3c5f636444c721b592a735ca"
_DEFAULT_BASE_MODEL_REVISION = "Qwen/Qwen2-VL-7B-Instruct@eed13092ef92e448dd6875b2a00151bd3f7db0ac"
_DEFINITION_VERSION = "omninft-hpsv3-top30-v4"
_FRAME_COUNT = 5
_TOP_FRACTION = 0.3
_REWARD_CAP = 15.0


@dataclass
class _HPSv3NativeState:
    model: Any
    processor: Any
    model_revision: str
    base_model_revision: str
    device: torch.device | None = None


class _TensorVisual(torch.nn.Module):
    """Return a tensor from Transformers-5 vision towers that wrap last_hidden_state."""

    def __init__(self, visual):
        super().__init__()
        self._visual = visual

    def get_dtype(self):
        if hasattr(self._visual, "get_dtype"):
            return self._visual.get_dtype()
        return next(self._visual.parameters()).dtype

    @property
    def dtype(self):
        return self.get_dtype()

    def forward(self, pixel_values, grid_thw=None, **kwargs):
        outputs = self._visual(pixel_values, grid_thw=grid_thw, **kwargs)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs


def _unwrap_peft(model):
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def ensure_omninft_qwen2vl_layout(model):
    """Expose OmniNFT's pre-v5 `visual` / `embed_tokens` attributes on a v5 Qwen2-VL model."""
    root = _unwrap_peft(model)
    inner = getattr(root, "model", None)
    if inner is None:
        return model
    visual = getattr(root, "visual", None) or getattr(inner, "visual", None)
    if visual is not None and not isinstance(visual, _TensorVisual):
        visual = _TensorVisual(visual)
        if hasattr(inner, "visual"):
            inner.visual = visual
        root.visual = visual
    if not hasattr(inner, "embed_tokens"):
        language = getattr(inner, "language_model", None)
        if language is not None and hasattr(language, "embed_tokens"):
            inner.embed_tokens = language.embed_tokens
    return model


def omninft_qwen2vl_reward_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    rope_deltas=None,
    **kwargs,
):
    """OmniNFT Qwen2VLRewardModelBT.forward, with TF5 kwargs ignored."""
    del labels, rope_deltas
    kwargs.pop("mm_token_type_ids", None)
    output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
    output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
    return_dict = getattr(self.config, "use_return_dict", True) if return_dict is None else return_dict
    visual = getattr(self, "visual", None) or getattr(self.model, "visual", None)
    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(visual.get_dtype() if hasattr(visual, "get_dtype") else visual.dtype)
            image_embeds = visual(pixel_values, grid_thw=image_grid_thw)
            image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask, image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            )
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(
                visual.get_dtype() if hasattr(visual, "get_dtype") else visual.dtype
            )
            video_embeds = visual(pixel_values_videos, grid_thw=video_grid_thw)
            video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask, video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            )
        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)
    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_states = outputs[0]
    logits = self.rm_head(hidden_states.float())
    batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
    if self.config.pad_token_id is None and batch_size != 1:
        raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
    if self.reward_token != "special":
        raise ValueError("OmniNFT Qwen2-VL rewards must pool special tokens.")
    special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in self.special_token_ids:
        special_token_mask |= input_ids == token_id
    pooled_logits = logits[special_token_mask, ...].view(batch_size, -1)
    return {"logits": pooled_logits}


class _HPSv3NativeModel(_Qwen2VLRewardModelBT):
    """HPSv3 model using the OmniNFT RewardModelBT forward."""

    forward = omninft_qwen2vl_reward_forward


def _load_components(model_path: str, base_model_path: str) -> tuple[Any, Any]:
    from transformers import AutoConfig, AutoProcessor

    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(base_model_path, padding_side="right")
    special_tokens = ["<|Reward|>"]
    processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)
    model = _HPSv3NativeModel(
        config,
        output_dim=2,
        reward_token="special",
        special_token_ids=special_token_ids,
        rm_head_type="ranknet",
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.to(torch.bfloat16)
    model.rm_head.to(torch.float32)
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    if model_path.endswith(".safetensors"):
        import safetensors.torch

        state_dict = safetensors.torch.load_file(model_path, device="cpu")
    else:
        state_dict = torch.load(model_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(_remap_state_dict(state_dict, model.state_dict().keys()), strict=True)
    ensure_omninft_qwen2vl_layout(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, processor


def initialize(
    model_path: str,
    base_model_path: str = _BASE_MODEL,
    model_revision: str = _DEFAULT_MODEL_REVISION,
    base_model_revision: str = _DEFAULT_BASE_MODEL_REVISION,
) -> _HPSv3NativeState:
    """Load HPSv3 and its Qwen2-VL processor into CPU state."""
    for name, value in (
        ("model_path", model_path),
        ("base_model_path", base_model_path),
        ("model_revision", model_revision),
        ("base_model_revision", base_model_revision),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"HPSv3 Native Reward requires a non-empty {name}.")
    model, processor = _load_components(model_path, base_model_path)
    return _HPSv3NativeState(model, processor, model_revision, base_model_revision)


def _resolve_device(device: int | str | torch.device) -> torch.device:
    if isinstance(device, int):
        return torch.device(f"{get_device_name()}:{device}")
    return torch.device(device)


def activate(state: _HPSv3NativeState, device: int | str | torch.device) -> None:
    """Move HPSv3 to the runtime-selected accelerator."""
    if state.model is None:
        raise RuntimeError("HPSv3 Native Reward has already been finalized.")
    if state.device is not None:
        raise RuntimeError("HPSv3 Native Reward is already active.")
    state.device = _resolve_device(device)
    state.model.to(state.device).eval()


def _to_pil(frame: torch.Tensor) -> Image.Image:
    frame = frame.detach().cpu()
    if frame.ndim != 3:
        raise ValueError(f"HPSv3 video frame must have shape [C,H,W], got {tuple(frame.shape)}.")
    if frame.shape[0] not in (1, 3):
        raise ValueError(f"HPSv3 video must have 1 or 3 channels, got {frame.shape[0]}.")
    if frame.dtype.is_floating_point:
        if not torch.isfinite(frame).all():
            raise ValueError("HPSv3 video must contain only finite values.")
        frame = frame.clamp(0, 1).mul(255).round().to(torch.uint8)
    elif frame.dtype != torch.uint8:
        raise ValueError("HPSv3 video must be floating-point or uint8.")
    if frame.shape[0] == 1:
        frame = frame.expand(3, -1, -1)
    return Image.fromarray(frame.permute(1, 2, 0).numpy(), mode="RGB")


def _extract_inputs(batch) -> tuple[list[list[Image.Image]], list[str]]:
    batch_size = len(batch)
    if batch_size <= 0:
        raise ValueError("HPSv3 Native Reward requires a non-empty local batch.")
    videos = batch.batch.get("responses")
    if not isinstance(videos, torch.Tensor) or videos.ndim != 5 or videos.shape[0] != batch_size:
        shape = None if not isinstance(videos, torch.Tensor) else tuple(videos.shape)
        raise ValueError(f"HPSv3 responses must have shape [B,T,C,H,W] with B={batch_size}, got {shape}.")
    if videos.shape[1] <= 0 or videos.shape[2] not in (1, 3) or videos.shape[3] <= 0 or videos.shape[4] <= 0:
        raise ValueError("HPSv3 video must have non-empty temporal and spatial dimensions.")
    if not videos.dtype.is_floating_point and videos.dtype != torch.uint8:
        raise ValueError("HPSv3 video must be floating-point or uint8.")
    if videos.dtype.is_floating_point and not torch.isfinite(videos).all():
        raise ValueError("HPSv3 video must contain only finite values.")

    reward_inputs = batch.non_tensor_batch.get("reward_inputs")
    if np.asarray(reward_inputs, dtype=object).shape != (batch_size,):
        raise ValueError(f"HPSv3 reward_inputs must have shape ({batch_size},).")
    frame_indices = torch.linspace(0, videos.shape[1] - 1, _FRAME_COUNT).round().to(torch.long).tolist()
    frames = [[_to_pil(videos[index, frame]) for frame in frame_indices] for index in range(batch_size)]
    prompts = []
    for index, reward_input in enumerate(reward_inputs):
        try:
            prompt = reward_input["text"]["video"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"HPSv3 reward_inputs[{index}] must contain text.video.") from exc
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"HPSv3 reward_inputs[{index}].text.video must be a non-empty string.")
        prompts.append(prompt)
    return frames, prompts


def _prepare_batch(state: _HPSv3NativeState, images: list[Image.Image], prompts: list[str]) -> dict[str, Any]:
    max_pixels = 256 * 28 * 28
    messages = []
    for image, prompt in zip(images, prompts, strict=True):
        messages.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image, "min_pixels": max_pixels, "max_pixels": max_pixels},
                        {
                            "type": "text",
                            "text": _INSTRUCTION.format(text_prompt=prompt) + _PROMPT_WITH_SPECIAL_TOKEN,
                        },
                    ],
                }
            ]
        )
    image_inputs = _process_vision_info(messages)
    inputs = state.processor(
        text=state.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
        images=image_inputs,
        padding=True,
        return_tensors="pt",
        videos_kwargs={"do_rescale": True},
    )
    return {
        key: value.to(state.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()
    }


def score_batch(state: _HPSv3NativeState, batch, micro_batch_size: int, **kwargs) -> dict[str, Any]:
    """Score one complete local shard using batched five-frame HPSv3 inference."""
    del kwargs
    if state.device is None:
        raise RuntimeError("HPSv3 Native Reward must be active before scoring.")
    if isinstance(micro_batch_size, bool) or not isinstance(micro_batch_size, int) or micro_batch_size <= 0:
        raise ValueError("HPSv3 micro_batch_size must be a positive integer.")
    frame_groups, prompts = _extract_inputs(batch)
    score_chunks = []
    forward_calls = 0
    with torch.inference_mode():
        for start in range(0, len(frame_groups), micro_batch_size):
            stop = min(start + micro_batch_size, len(frame_groups))
            images = [frame for group in frame_groups[start:stop] for frame in group]
            repeated_prompts = [prompt for prompt in prompts[start:stop] for _ in range(_FRAME_COUNT)]
            inputs = _prepare_batch(state, images, repeated_prompts)
            output = state.model(return_dict=True, **inputs)
            forward_calls += 1
            logits = output["logits"] if isinstance(output, dict) else output.logits
            if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape != (len(images), 2):
                raise ValueError(f"HPSv3 model logits must have shape ({len(images)}, 2).")
            if not torch.isfinite(logits).all():
                raise ValueError("HPSv3 model logits must contain only finite values.")
            frame_scores = torch.minimum(
                logits[:, 0].float(),
                torch.tensor(_REWARD_CAP, device=logits.device),
            ).reshape(stop - start, _FRAME_COUNT)
            top_count = max(1, int(np.ceil(_FRAME_COUNT * _TOP_FRACTION)))
            score_chunks.append(frame_scores.topk(top_count, dim=1).values.mean(dim=1).cpu())
    scores = torch.cat(score_chunks).to(dtype=torch.float32)
    if scores.shape != (len(batch),) or not torch.isfinite(scores).all():
        raise ValueError("HPSv3 scores must be finite and sample-aligned.")
    return {
        "scores": scores,
        "valid_mask": torch.ones(len(batch), dtype=torch.bool),
        "metrics": {
            "batch_size": len(batch),
            "micro_batch_size": micro_batch_size,
            "forward_calls": forward_calls,
            "frames_per_sample": _FRAME_COUNT,
            "top_frame_count": top_count,
            "reward_cap": _REWARD_CAP,
        },
        "model_revision": state.model_revision,
        "definition_version": _DEFINITION_VERSION,
    }


def _release_accelerator_memory() -> None:
    accelerator = get_torch_device()
    accelerator.synchronize()
    accelerator.empty_cache()
    accelerator.synchronize()


def deactivate(state: _HPSv3NativeState) -> None:
    """Move HPSv3 back to CPU and release accelerator cache."""
    if state.device is None:
        return
    device = state.device
    state.model.to("cpu")
    state.device = None
    if device.type != "cpu":
        _release_accelerator_memory()


def finalize(state: _HPSv3NativeState) -> None:
    """Release all HPSv3 state owned by this Reward Manager."""
    deactivate(state)
    state.model = None
    state.processor = None
