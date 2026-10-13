# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Actual media serialization retains task, replay and five-component evidence."""

import json
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import soundfile as sf
import torch
from verl import DataProto

from verl_omni.pipelines.ltx2_unigrpo_nft.artifacts import save_joint_samples
from verl_omni.pipelines.ltx2_unigrpo_nft.replay import LTXJointReplay


def test_joint_review_artifacts_preserve_exact_replay_and_decodable_media(tmp_path):
    record = LTXJointReplay(
        original_prompt="A bell rings. <script>alert(1)</script>",
        recaption="One bell sounds in a quiet room.",
        prompt_token_ids=torch.tensor([2, 3]),
        response_token_ids=torch.tensor([4, 1]),
        rollout_logprobs=torch.tensor([-0.4, -0.2]),
        condition_token_ids=torch.tensor([2, 4, 1]),
        condition_attention_mask=torch.ones(3, dtype=torch.bool),
        policy_version=7,
        seed=42,
        temperature=1.0,
        eos_token_ids=(1,),
        generation_params={"height": 32, "width": 32, "num_frames": 9, "frame_rate": 24.0, "num_inference_steps": 2},
    )
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.full((1, 9, 3, 32, 32), 128, dtype=torch.uint8),
            "audio": torch.zeros(1, 2, 16000),
            "audio_sample_rate": torch.tensor([48000]),
            "video_latents_clean": torch.randn(1, 2, 128),
            "audio_latents_clean": torch.randn(1, 8, 128),
            "video_forward_noise": torch.randn(1, 1, 2, 128),
            "audio_forward_noise": torch.randn(1, 1, 8, 128),
            "train_timesteps": torch.tensor([[100.0]]),
            "rm_scores": torch.arange(5.0)[None],
            "reward_prob": torch.tensor([[[0.2, 0.6]]]),
            "text_reward_prob": torch.tensor([0.4]),
            "text_advantages": torch.tensor([-1.0]),
        },
        non_tensors={"uid": np.array(["original-group"]), "ltx_joint_replay": np.array([record], dtype=object)},
    )
    names = ["audiobox", "clap", "desync", "hpsv3", "video_align"]
    save_joint_samples(batch, tmp_path, names)
    index = json.loads((tmp_path / "samples.json").read_text())[0]
    assert index["original_prompt"] == record.original_prompt and index["recaption"] == record.recaption
    assert index["policy_version"] == 7 and index["ended_with_eos"]
    assert index["reward_scores"] == dict(zip(names, range(5), strict=True))
    saved = torch.load(tmp_path / index["sample_file"], weights_only=False)
    torch.testing.assert_close(saved["replay"]["rollout_logprobs"], record.rollout_logprobs)
    assert saved["ar_inputs"]["response_mask"].tolist() == [[False, False, True, True]]
    assert saved["ar_inputs"]["attention_mask"].all()
    torch.testing.assert_close(saved["video_latents_clean"], batch.batch["video_latents_clean"])
    for modality in ("video", "audio"):
        torch.testing.assert_close(saved[f"{modality}_forward_noise"], batch.batch[f"{modality}_forward_noise"])
    audio, rate = sf.read(tmp_path / index["audio_file"])
    assert rate == 48000 and audio.shape == (16000, 2)
    with imageio.get_reader(tmp_path / index["video_file"]) as reader:
        assert reader.count_frames() == 9 and reader.get_data(0).shape == (32, 32, 3)
    with imageio.get_reader(tmp_path / index["review_video_file"]) as reader:
        assert reader.count_frames() == 9
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(tmp_path / index["review_video_file"]),
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    page = (tmp_path / "index.html").read_text()
    assert "&lt;script&gt;" in page and "<script>" not in page
    assert record.recaption in page and index["review_video_file"] in page
