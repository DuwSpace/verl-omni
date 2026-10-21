# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare an immutable HPS probe from trusted raw joint sample .pt files, never MP4s."""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers.initialization import no_init_weights

from verl_omni.utils.reward_score.hpsv3_health import build_reference
from verl_omni.utils.reward_score.hpsv3_reward import HPSv3Model, _select_reward_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", required=True, help="Trusted joint_sample_save_tensors=true .pt files")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--device", required=True, help="For example npu:1 or cuda:0")
    parser.add_argument("--output", required=True, help="New reference file; existing files are never overwritten")
    parser.add_argument("--num-frames", type=int, default=5, help="Fixed frame batch size; match the scorer")
    parser.add_argument("--sequential-position-ids", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Reference already exists: {output}; use a new path for explicit recalibration")
    if args.num_frames <= 0:
        parser.error("--num-frames must be positive")
    samples = []
    for path in args.samples:
        # Joint replay includes dataclass metadata; only load locally trusted artifacts.
        saved = torch.load(path, map_location="cpu", weights_only=False)
        frames = _select_reward_frames(saved["video"], {}, args.num_frames)
        samples.append(
            {
                "frames": torch.from_numpy(np.stack([np.asarray(frame).copy() for frame in frames])).permute(
                    0, 3, 1, 2
                ),
                "prompts": [saved["reward_inputs"]["text"]["video"]] * args.num_frames,
            }
        )
    device = torch.device(args.device)
    if device.type != "cpu":
        torch.get_device_module(device).set_device(device)
    # The HPS loader strictly loads the complete checkpoint. Avoid randomizing
    # billions of parameters that are immediately overwritten by those weights.
    with no_init_weights():
        model = HPSv3Model(args.model_path, device, args.base_model_path, args.sequential_position_ids)
    identity = {
        "model_path": args.model_path,
        "base_model_path": args.base_model_path,
        "use_sequential_position_ids": args.sequential_position_ids,
    }
    reference = build_reference(model._inferencer, samples, identity)
    reference["source_samples"] = [str(Path(path).resolve()) for path in args.samples]
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against another calibration process.
    with output.open("xb") as stream:
        torch.save(reference, stream)
    print(f"Saved {len(samples)} HPS probes to {output}")
    for index, sample in enumerate(reference["samples"]):
        print(f"probe={index} frame_logits={sample['logits'][:, 0].tolist()}")


if __name__ == "__main__":
    main()
