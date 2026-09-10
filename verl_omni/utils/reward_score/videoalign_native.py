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

"""Batch-native VideoAlign reward adapted from zghhui/OmniNFT."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from verl.utils.device import get_device_name, get_torch_device

from .hpsv3_native import ensure_omninft_qwen2vl_layout, omninft_qwen2vl_reward_forward
from .hpsv3_reward import _Qwen2VLRewardModelBT, _smart_resize
from .reward_utils import load_torch_state_dict

_DEFAULT_MODEL_REVISION = "KlingTeam/VideoReward@4f26600130683e6f1de9f5d463887f28e8ef995c"
_DEFAULT_BASE_MODEL_REVISION = "Qwen/Qwen2-VL-2B-Instruct@895c3a49bc3fa70a340399125c650a463535e71c"
_DEFINITION_VERSION = "omninft-videoalign-vq-ta-v3"
_SPECIAL_TOKENS = ("<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>")
_TARGET_FPS = 24.0
_FRAME_FACTOR = 2
_MIN_FRAMES = 4
_MAX_FRAMES = 768
_IMAGE_FACTOR = 28
_MIN_FRAME_PIXELS = 128 * 28 * 28
_MAX_FRAME_PIXELS = 200_704
_VQ_MEAN = 3.6757
_VQ_STD = 2.2476
_TA_MEAN = 2.8105
_TA_STD = 2.5121

_PROMPT_TEMPLATE = (
    "\n"
    "You are tasked with evaluating a generated video based on three distinct criteria: Visual Quality, Motion "
    "Quality, and Text Alignment. Please provide a rating from 0 to 10 for each of the three categories, with 0 "
    "being the worst and 10 being the best. Each evaluation should be independent of the others.\n"
    "\n"
    "**Visual Quality:**  \n"
    "Evaluate the overall visual quality of the video, with a focus on static factors. The following "
    "sub-dimensions should be considered:\n"
    "- **Reasonableness:** The video should not contain any significant biological or logical errors, such as "
    "abnormal body structures or nonsensical environmental setups.\n"
    "- **Clarity:** Evaluate the sharpness and visibility of the video. The image should be clear and easy to "
    "interpret, with no blurring or indistinct areas.\n"
    "- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual "
    "elements (e.g., hair, clothing, shadows).\n"
    "- **Aesthetic and Creativity:** Assess the artistic aspects of the video, including the color scheme, "
    "composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense "
    "of harmony and balance.\n"
    "- **Safety:** The video should not contain harmful or inappropriate content, such as political, violent, or "
    "adult material. If such content is present, the image quality and satisfaction score should be the lowest "
    "possible. \n"
    "\n"
    "Please provide the ratings of Visual Quality: <|VQ_reward|>\n"
    "END\n"
    "\n"
    "**Motion Quality:**  \n"
    "Assess the dynamic aspects of the video, with a focus on dynamic factors. Consider the following "
    "sub-dimensions:\n"
    "- **Stability:** Evaluate the continuity and stability between frames. There should be no sudden, unnatural "
    "jumps, and the video should maintain stable attributes (e.g., no fluctuating colors, textures, or missing "
    "body parts).\n"
    "- **Naturalness:** The movement should align with physical laws and be realistic. For example, clothing "
    "should flow naturally with motion, and facial expressions should change appropriately (e.g., blinking, "
    "mouth movements).\n"
    "- **Aesthetic Quality:** The movement should be smooth and fluid. The transitions between different motions "
    "or camera angles should be seamless, and the overall dynamic feel should be visually pleasing.\n"
    "- **Fusion:** Ensure that elements in motion (e.g., edges of the subject, hair, clothing) blend naturally "
    "with the background, without obvious artifacts or the feeling of cut-and-paste effects.\n"
    "- **Clarity of Motion:** The video should be clear and smooth in motion. Pay attention to any areas where the "
    "video might have blurry or unsteady sections that hinder visual continuity.\n"
    "- **Amplitude:** If the video is largely static or has little movement, assign a low score for motion quality.\n"
    "\n"
    "Please provide the ratings of Motion Quality: <|MQ_reward|>\n"
    "END\n"
    "\n"
    "**Text Alignment:**  \n"
    "Assess how well the video matches the textual prompt across the following sub-dimensions:\n"
    "- **Subject Relevance** Evaluate how accurately the subject(s) in the video (e.g., person, animal, object) "
    "align with the textual description. The subject should match the description in terms of number, "
    "appearance, and behavior.\n"
    "- **Motion Relevance:** Evaluate if the dynamic actions (e.g., gestures, posture, facial expressions like "
    "talking or blinking) align with the described prompt. The motion should match the prompt in terms of type, "
    "scale, and direction.\n"
    "- **Environment Relevance:** Assess whether the background and scene fit the prompt. This includes checking "
    "if real-world locations or scenes are accurately represented, though some stylistic adaptation is "
    "acceptable.  \n"
    "- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well "
    "the video adheres to this style.\n"
    "- **Camera Movement Relevance:** Check if the camera movements (e.g., following the subject, focus shifts) "
    "are consistent with the expected behavior from the prompt.\n"
    "\n"
    "Textual prompt - {text_prompt}\n"
    "Please provide the ratings of Text Alignment: <|TA_reward|>\n"
    "END\n"
)


@dataclass
class _VideoAlignNativeState:
    model: Any
    processor: Any
    model_revision: str
    base_model_revision: str
    device: torch.device | None = None


class _VideoAlignNativeModel(_Qwen2VLRewardModelBT):
    """VideoReward model using the OmniNFT RewardModelBT forward."""

    forward = omninft_qwen2vl_reward_forward


def _find_target_linear_names(model: Any) -> list[str]:
    excluded = ("lm_head", "rm_head", "embed_tokens", "visual")
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)) and not any(part in name for part in excluded)
    ]


def _remap_checkpoint_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("base_model.model.visual."):
            key = key.replace("base_model.model.visual.", "base_model.model.model.visual.", 1)
        elif key.startswith("base_model.model.model."):
            key = key.replace("base_model.model.model.", "base_model.model.model.language_model.", 1)
        remapped[key] = value
    return remapped


def _load_components(model_path: str, base_model_path: str) -> tuple[Any, Any]:
    from accelerate import init_empty_weights
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoProcessor

    config = AutoConfig.from_pretrained(base_model_path)
    processor = AutoProcessor.from_pretrained(base_model_path, padding_side="right")
    processor.tokenizer.add_special_tokens({"additional_special_tokens": list(_SPECIAL_TOKENS)})
    special_token_ids = processor.tokenizer.convert_tokens_to_ids(list(_SPECIAL_TOKENS))

    with init_empty_weights():
        model = _VideoAlignNativeModel(
            config,
            output_dim=1,
            reward_token="special",
            special_token_ids=special_token_ids,
            rm_head_type="linear",
        )
        model.resize_token_embeddings(len(processor.tokenizer))
        model = get_peft_model(
            model,
            LoraConfig(
                target_modules=_find_target_linear_names(model),
                r=64,
                lora_alpha=128,
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                use_rslora=False,
                bias="none",
            ),
        )

    state_dict = load_torch_state_dict(model_path)
    if not isinstance(state_dict, dict) or not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise ValueError("VideoAlign checkpoint must be a tensor state dict.")
    state_dict = _remap_checkpoint_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True, assign=True)
    ensure_omninft_qwen2vl_layout(model)
    model.rm_head.to(torch.float32)
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, processor


def initialize(
    model_path: str,
    base_model_path: str,
    model_revision: str = _DEFAULT_MODEL_REVISION,
    base_model_revision: str = _DEFAULT_BASE_MODEL_REVISION,
) -> _VideoAlignNativeState:
    """Load VideoReward and its Qwen2-VL processor into CPU state."""
    for name, value in (
        ("model_path", model_path),
        ("base_model_path", base_model_path),
        ("model_revision", model_revision),
        ("base_model_revision", base_model_revision),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"VideoAlign Native Reward requires a non-empty {name}.")
    model, processor = _load_components(model_path, base_model_path)
    return _VideoAlignNativeState(model, processor, model_revision, base_model_revision)


def _resolve_device(device: int | str | torch.device) -> torch.device:
    if isinstance(device, int):
        return torch.device(f"{get_device_name()}:{device}")
    return torch.device(device)


def activate(state: _VideoAlignNativeState, device: int | str | torch.device) -> None:
    """Move VideoReward to the runtime-selected accelerator."""
    if state.model is None:
        raise RuntimeError("VideoAlign Native Reward has already been finalized.")
    if state.device is not None:
        raise RuntimeError("VideoAlign Native Reward is already active.")
    state.device = _resolve_device(device)
    state.model.to(state.device).eval()


def _sample_video(video: torch.Tensor, source_fps: float) -> tuple[torch.Tensor, list[int]]:
    requested_frames = round(video.shape[0] * _TARGET_FPS / source_fps / _FRAME_FACTOR) * _FRAME_FACTOR
    max_frames = min(_MAX_FRAMES, video.shape[0]) // _FRAME_FACTOR * _FRAME_FACTOR
    frame_count = min(max(requested_frames, _MIN_FRAMES), max_frames)
    frame_count = min(frame_count, video.shape[0])
    indices = torch.linspace(0, video.shape[0] - 1, frame_count).round().to(torch.long).tolist()
    sampled = video[indices].detach().cpu()
    if sampled.dtype.is_floating_point:
        if not torch.isfinite(sampled).all() or sampled.min() < 0 or sampled.max() > 1:
            raise ValueError("VideoAlign floating-point video values must be finite and in [0, 1].")
        sampled = sampled.mul(255).round().to(torch.uint8)
    elif sampled.dtype != torch.uint8:
        raise ValueError("VideoAlign video must be floating-point or uint8.")
    if sampled.shape[1] == 1:
        sampled = sampled.expand(-1, 3, -1, -1)
    return sampled, indices


def _resize_video(video: torch.Tensor) -> torch.Tensor:
    height, width = video.shape[-2:]
    resized_height, resized_width = _smart_resize(
        height,
        width,
        factor=_IMAGE_FACTOR,
        min_pixels=_MIN_FRAME_PIXELS,
        max_pixels=_MAX_FRAME_PIXELS,
    )
    resized = F.interpolate(
        video.float(),
        size=(resized_height, resized_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return resized.clamp(0, 255).round()


def _extract_inputs(batch) -> tuple[list[torch.Tensor], list[str], list[list[int]], list[float]]:
    batch_size = len(batch)
    if batch_size <= 0:
        raise ValueError("VideoAlign Native Reward requires a non-empty local batch.")
    videos = batch.batch.get("responses")
    if not isinstance(videos, torch.Tensor) or videos.ndim != 5 or videos.shape[0] != batch_size:
        shape = None if not isinstance(videos, torch.Tensor) else tuple(videos.shape)
        raise ValueError(f"VideoAlign responses must have shape [B,T,C,H,W] with B={batch_size}, got {shape}.")
    if videos.shape[1] < _FRAME_FACTOR or videos.shape[2] not in (1, 3) or min(videos.shape[3:]) <= 0:
        raise ValueError("VideoAlign video must have at least two frames and non-empty RGB or grayscale pixels.")

    fps = batch.batch.get("fps")
    if not isinstance(fps, torch.Tensor) or fps.shape != (batch_size,) or not fps.dtype.is_floating_point:
        raise ValueError(f"VideoAlign fps must be a floating-point tensor with shape ({batch_size},).")
    source_fps = [float(value) for value in fps.detach().cpu().tolist()]
    if any(not np.isfinite(value) or value <= 0 for value in source_fps):
        raise ValueError("VideoAlign fps values must be finite and positive.")

    reward_inputs = batch.non_tensor_batch.get("reward_inputs")
    if reward_inputs is None or np.asarray(reward_inputs, dtype=object).shape != (batch_size,):
        raise ValueError(f"VideoAlign reward_inputs must have shape ({batch_size},).")

    sampled_videos = []
    prompts = []
    frame_indices = []
    for index, (video, sample_fps) in enumerate(zip(videos, source_fps, strict=True)):
        try:
            prompt = reward_inputs[index]["text"]["video"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"VideoAlign reward_inputs[{index}] must contain text.video.") from exc
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"VideoAlign reward_inputs[{index}].text.video must be a non-empty string.")
        sampled, indices = _sample_video(video, sample_fps)
        sampled_videos.append(sampled)
        prompts.append(prompt)
        frame_indices.append(indices)
    return sampled_videos, prompts, frame_indices, source_fps


def _prepare_batch(state: _VideoAlignNativeState, videos: list[torch.Tensor], prompts: list[str]) -> dict[str, Any]:
    resized_videos = [_resize_video(video) for video in videos]
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video, "max_pixels": _MAX_FRAME_PIXELS},
                    {"type": "text", "text": _PROMPT_TEMPLATE.format(text_prompt=prompt)},
                ],
            }
        ]
        for video, prompt in zip(resized_videos, prompts, strict=True)
    ]
    inputs = state.processor(
        text=state.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
        images=None,
        videos=resized_videos,
        padding=True,
        return_tensors="pt",
        videos_kwargs={"do_rescale": True},
    )
    return {
        key: value.to(state.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()
    }


def score_batch(state: _VideoAlignNativeState, batch, micro_batch_size: int, **kwargs) -> dict[str, Any]:
    """Score one complete local shard using batched VideoReward inference."""
    del kwargs
    if state.device is None:
        raise RuntimeError("VideoAlign Native Reward must be active before scoring.")
    if isinstance(micro_batch_size, bool) or not isinstance(micro_batch_size, int) or micro_batch_size <= 0:
        raise ValueError("VideoAlign micro_batch_size must be a positive integer.")

    videos, prompts, frame_indices, source_fps = _extract_inputs(batch)
    score_chunks = []
    forward_calls = 0
    with torch.inference_mode():
        for start in range(0, len(batch), micro_batch_size):
            stop = min(start + micro_batch_size, len(batch))
            inputs = _prepare_batch(state, videos[start:stop], prompts[start:stop])
            output = state.model(return_dict=True, **inputs)
            forward_calls += 1
            logits = output["logits"] if isinstance(output, dict) else output.logits
            if not isinstance(logits, torch.Tensor) or logits.shape != (stop - start, 3):
                raise ValueError(f"VideoAlign model logits must have shape ({stop - start}, 3).")
            if not torch.isfinite(logits).all():
                raise ValueError("VideoAlign model logits must contain only finite values.")
            logits = logits.float()
            vq = (logits[:, 0] - _VQ_MEAN) / _VQ_STD
            ta = (logits[:, 2] - _TA_MEAN) / _TA_STD
            score_chunks.append(((vq + ta) / 2).cpu())

    scores = torch.cat(score_chunks).to(dtype=torch.float32)
    if scores.shape != (len(batch),) or not torch.isfinite(scores).all():
        raise ValueError("VideoAlign scores must be finite and sample-aligned.")
    return {
        "scores": scores,
        "valid_mask": torch.ones(len(batch), dtype=torch.bool),
        "metrics": {
            "batch_size": len(batch),
            "micro_batch_size": micro_batch_size,
            "forward_calls": forward_calls,
            "source_fps": sorted(set(source_fps)),
            "target_fps": _TARGET_FPS,
            "frame_indices": frame_indices,
            "frames_per_sample": [len(indices) for indices in frame_indices],
            "base_model_revision": state.base_model_revision,
        },
        "model_revision": state.model_revision,
        "definition_version": _DEFINITION_VERSION,
    }


def _release_accelerator_memory() -> None:
    accelerator = get_torch_device()
    accelerator.synchronize()
    accelerator.empty_cache()
    accelerator.synchronize()


def deactivate(state: _VideoAlignNativeState) -> None:
    """Move VideoReward back to CPU and release accelerator cache."""
    if state.device is None:
        return
    device = state.device
    state.model.to("cpu")
    state.device = None
    if device.type != "cpu":
        _release_accelerator_memory()


def finalize(state: _VideoAlignNativeState) -> None:
    """Release all VideoAlign state owned by this Reward Manager."""
    deactivate(state)
    state.model = None
    state.processor = None
