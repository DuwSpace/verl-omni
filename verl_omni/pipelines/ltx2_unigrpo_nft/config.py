# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Request-scoped LTX sampling settings shared by train and evaluation."""

import math
from dataclasses import dataclass

from verl_omni.workers.config.diffusion import DiffusionPipelineConfig

GUIDANCE_DEFAULTS = {
    "video_cfg_scale": 1.0,
    "audio_cfg_scale": 1.0,
    "video_modality_scale": 1.0,
    "audio_modality_scale": 1.0,
    "video_rescale_scale": 0.0,
    "audio_rescale_scale": 0.0,
}
GEOMETRY_KEYS = ("height", "width", "num_frames", "frame_rate", "num_inference_steps", "max_sequence_length")


@dataclass
class LTXJointPipelineConfig(DiffusionPipelineConfig):
    video_cfg_scale: float = 1.0
    audio_cfg_scale: float = 1.0
    video_modality_scale: float = 1.0
    audio_modality_scale: float = 1.0
    video_rescale_scale: float = 0.0
    audio_rescale_scale: float = 0.0


def sampling_params(config):
    """Convert a pipeline config to native kwargs; no cached engine mutation."""
    get = config.get if hasattr(config, "get") else lambda key, default=None: getattr(config, key, default)
    result = {key: get(key) for key in GEOMETRY_KEYS}
    for key in GEOMETRY_KEYS:
        value = result[key]
        if key == "frame_rate":
            if not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
                raise ValueError("LTX frame_rate must be finite and positive")
        elif type(value) is not int or value <= 0:
            raise ValueError(f"LTX {key} must be a positive integer")
    if result["height"] % 32 or result["width"] % 32 or (result["num_frames"] - 1) % 8:
        raise ValueError("LTX dimensions must be multiples of 32 and num_frames must be 8k+1")
    guidance = {key: get(key, default) for key, default in GUIDANCE_DEFAULTS.items()}
    for key, value in guidance.items():
        if not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"LTX {key} must be finite and nonnegative")
    result["extra_args"] = guidance
    return result
