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

from .hpsv3_reward import (
    _BASE_MODEL,
    _INSTRUCTION,
    _PROMPT_WITH_SPECIAL_TOKEN,
    _process_vision_info,
    _Qwen2VLRewardModelBT,
    _remap_state_dict,
)
from .qwen2vl_reward_compat import ensure_omninft_qwen2vl_layout, omninft_qwen2vl_reward_forward

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


class _HPSv3NativeModel(_Qwen2VLRewardModelBT):
    """HPSv3 model using the OmniNFT RewardModelBT forward."""

    forward = omninft_qwen2vl_reward_forward


def _load_components(model_path: str, base_model_path: str) -> tuple[Any, Any]:
    """Build Qwen2-VL reward-token inputs and strictly load remapped HPSv3 weights.

    Return a frozen eval model and processor. The backbone uses BF16 and the
    two-output reward head uses FP32; the instance receives the layout adapter.
    """
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


def _load_state(
    model_path: str,
    base_model_path: str = _BASE_MODEL,
    model_revision: str = _DEFAULT_MODEL_REVISION,
    base_model_revision: str = _DEFAULT_BASE_MODEL_REVISION,
) -> _HPSv3NativeState:
    """Load HPSv3 and its Qwen2-VL processor."""
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
    """Return five uniformly indexed RGB PIL frames and text.video per sample.

    Rounded linspace indices may repeat on short clips. Floating pixels are
    clamped to [0, 1] before uint8 conversion; grayscale is expanded to RGB.
    """
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
    """Format aligned frame/prompt pairs with the reward token and move inputs.

    Vision preprocessing requests a 256 * 28 * 28 pixel budget per image; the
    processor pads text and tensors are moved to the active model device.
    """
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
    return {key: value.to(state.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}


async def compute_score_batch(batch, reward_model, *, micro_batch_size: int, **kwargs) -> dict[str, Any]:
    """Score video preference using the best two of five sampled frame scores.

    Args:
        batch: Nonempty ``responses[B, T, C, H, W]`` videos (C=1 or 3), uint8
            or finite floating pixels, and per-row ``reward_inputs.text.video``.
            Floating pixels are clamped to [0, 1]. Five rounded, uniformly spaced
            frame indices span each clip; indices may repeat. Audio is not used.
        reward_model: Active executor returning two logits per frame/prompt pair.
        micro_batch_size: Positive number of original videos per call, expanded
            to five frame/prompt pairs each for model inference.
        **kwargs: Ignored scorer options.

    Returns:
        CPU FP32 ``scores[B]`` in input order and all-true CPU bool
        ``valid_mask[B]``. Only logit 0 is used, capped above at 15; average the largest
        ``ceil(5 * 0.3) = 2`` scores per video. Higher is preferred; no lower
        bound or additional normalization is applied.

    Raises:
        ValueError: Invalid inputs, micro-batch size, logits, or final scores.
    """
    del kwargs
    if isinstance(micro_batch_size, bool) or not isinstance(micro_batch_size, int) or micro_batch_size <= 0:
        raise ValueError("HPSv3 micro_batch_size must be a positive integer.")
    frame_groups, prompts = _extract_inputs(batch)
    score_chunks = []
    for start in range(0, len(frame_groups), micro_batch_size):
        stop = min(start + micro_batch_size, len(frame_groups))
        images = [frame for group in frame_groups[start:stop] for frame in group]
        repeated_prompts = [prompt for prompt in prompts[start:stop] for _ in range(_FRAME_COUNT)]
        logits = await reward_model.infer(images, repeated_prompts)
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
    }


class HPSv3NativeModel:
    """Raw HPSv3 inference adapter owned by a native reward executor."""

    def __init__(self, model_path: str, device, **kwargs: Any) -> None:
        self._state = _load_state(model_path=model_path, **kwargs)
        self._state.device = torch.device(device)
        self._state.model.to(self._state.device).eval()

    def close(self) -> None:
        """Drop model/processor references; reuse requires constructing a new adapter."""
        self._state.model = None
        self._state.processor = None
        self._state.device = None

    @torch.inference_mode()
    def infer(self, images: list[Image.Image], prompts: list[str]) -> torch.Tensor:
        """Infer aligned PIL frame/prompt pairs without gradients.

        Format and preprocess pairs on the active device, then return detached
        CPU ``[N, 2]`` reward logits from the FP32 head. ``N`` counts frames,
        not videos; frame selection and per-video aggregation belong to the scorer.
        """
        inputs = _prepare_batch(self._state, images, prompts)
        output = self._state.model(return_dict=True, **inputs)
        logits = output["logits"] if isinstance(output, dict) else output.logits
        return logits.detach().cpu()
