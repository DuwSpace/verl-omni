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
import pytest
import soundfile as sf
import torch
from verl import DataProto

from verl_omni.pipelines.ltx2_unigrpo_nft.artifacts import save_joint_samples
from verl_omni.pipelines.ltx2_unigrpo_nft.replay import LTXJointReplay


@pytest.mark.parametrize("source", ["modality_probability", "combined_reward"])
def test_joint_review_artifacts_preserve_exact_replay_and_decodable_media(tmp_path, source):
    record = LTXJointReplay(
        original_prompt="A bell rings. <script>alert(1)</script>",
        recaption="One bell sounds in a quiet room.",
        planning="Match the strike to the chime. <script>planning</script>",
        response_text=(
            "<plan>Match the strike to the chime. <script>planning</script></plan>"
            "<recaption>One bell sounds in a quiet room.</recaption>"
        ),
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
        non_tensors={
            "uid": np.array(["original-group"]),
            "ltx_joint_replay": np.array([record], dtype=object),
            "reward/video_align/ta_raw": np.array([1.5]),
        },
    )
    names = ["audiobox", "clap", "desync", "hpsv3", "video_align"]
    if source == "combined_reward":
        batch.meta_info["text_reward_source"] = source
        batch.batch["text_reward"] = batch.batch["rm_scores"].sum(-1)
    save_joint_samples(batch, tmp_path, names, save_tensors=True)
    index = json.loads((tmp_path / "samples.json").read_text())[0]
    assert index["original_prompt"] == record.original_prompt and index["recaption"] == record.recaption
    assert index["planning"] == record.planning and index["response_text"] == record.response_text
    assert index["policy_version"] == 7 and index["ended_with_eos"]
    assert index["reward_scores"] == dict(zip(names, range(5), strict=True))
    assert index["text_reward_source"] == source
    assert index["text_reward"] == pytest.approx(10.0 if source == "combined_reward" else 0.4)
    assert index["text_reward_probability"] == pytest.approx(0.4)
    assert index["video_align_details"] == {"ta_raw": 1.5}
    assert index["planning_words"] == len(record.planning.split())
    saved = torch.load(tmp_path / index["sample_file"], weights_only=False)
    torch.testing.assert_close(saved["replay"]["rollout_logprobs"], record.rollout_logprobs)
    assert saved["replay"]["planning"] == record.planning
    assert saved["video_align_details"] == {"ta_raw": 1.5}
    assert saved["text_reward_source"] == source
    assert saved["text_reward"] == index["text_reward"]
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
    assert "<h3>Planning</h3>" in page and "Match the strike to the chime." in page
    assert record.recaption in page and index["review_video_file"] in page
    assert source in page and "Text scalar reward:" in page

    # The default must retain review evidence without serializing replay tensors.
    media_dir = tmp_path / "media_only"
    save_joint_samples(batch, media_dir, names)
    media_index = json.loads((media_dir / "samples.json").read_text())[0]
    assert not list(media_dir.glob("*.pt"))
    assert media_index["sample_file"] is None
    assert media_index["sample_id"] == "sample-0000"
    for key in ("original_prompt", "recaption", "reward_scores", "seed", "policy_version"):
        assert media_index[key] == index[key]
    for key in ("video_file", "audio_file", "review_video_file"):
        assert (media_dir / media_index[key]).stat().st_size > 0
    assert "<h2>sample-0000</h2>" in (media_dir / "index.html").read_text()
