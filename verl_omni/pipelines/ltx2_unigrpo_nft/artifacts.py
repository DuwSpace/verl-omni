# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Persist complete joint samples and an original-task quality-review index."""

import json
import subprocess
from dataclasses import asdict
from html import escape
from pathlib import Path

import torch


def write_joint_review_gallery(directory, index):
    """Make synchronized review copies; retain the separate audio and tensors."""
    import imageio_ffmpeg

    directory = Path(directory)
    cards = []
    for row in index:
        combined = Path(row["video_file"]).stem + "-av.mp4"
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(directory / row["video_file"]),
                "-i",
                str(directory / row["audio_file"]),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(directory / combined),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        row["review_video_file"] = combined
        scores = "".join(
            f"<tr><th>{escape(name)}</th><td>{score:.6f}</td></tr>" for name, score in row["reward_scores"].items()
        )
        cards.append(
            f"<article><h2>{escape(row['sample_file'])}</h2>"
            f"<p>Policy {row['policy_version']} · Seed {row['seed']} · "
            f"Tokens {row['recaption_tokens']} · EOS {row['ended_with_eos']}</p>"
            f"<h3>Original prompt</h3><p>{escape(row['original_prompt'])}</p>"
            f"<h3>Recaption</h3><p>{escape(row['recaption'])}</p>"
            f'<video controls preload="none" src="{escape(combined, quote=True)}"></video>'
            f"<table>{scores}</table><p>Video/audio probability: "
            f"{escape(str(row['reward_probability_video_audio']))}<br>"
            f"Text probability: {row['text_reward_probability']:.6f}<br>"
            f"Text advantage: {row['text_advantage']:.6f}</p></article>"
        )
    (directory / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>LTX joint sample review</title><style>"
        "body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;background:#f5f5f5}"
        "article{background:white;padding:1.5rem;margin:1rem 0;border-radius:8px}"
        "video{width:100%;max-width:512px}p{white-space:pre-wrap}th,td{text-align:left;padding:.25rem 1rem .25rem 0}"
        "</style><h1>LTX joint sample review</h1>"
        "<p>Review intent retention, event timing, dialogue and audio/video alignment. "
        "Scores alone do not establish quality. Review videos contain AAC audio; "
        "original WAV and tensor files remain available.</p>" + "".join(cards) + "</html>"
    )
    (directory / "samples.json").write_text(json.dumps(index, indent=2))


def save_joint_samples(batch, directory, reward_names):
    import imageio.v2 as imageio
    import soundfile as sf

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = batch.non_tensor_batch["ltx_joint_replay"]
    if batch.batch["rm_scores"].shape != (len(records), len(reward_names)):
        raise ValueError("Joint sample scores must retain the named reward columns")
    index = []
    for i, record in enumerate(records):
        record.validate()
        stem = f"sample-{i:04d}"
        video = batch.batch["responses"][i].detach().cpu()
        if video.dtype != torch.uint8 or video.ndim != 4 or video.shape[1] != 3:
            raise ValueError("Joint review artifacts require uint8 TCHW decoded video")
        audio = batch.batch["audio"][i].detach().float().cpu()
        rate = int(batch.batch["audio_sample_rate"][i])
        fps = float(record.generation_params["frame_rate"])
        sample = {
            "uid": str(batch.non_tensor_batch["uid"][i]),
            "replay": asdict(record),
            "ar_inputs": record.ar_inputs(torch.device("cpu")),
            "video": video,
            "audio": audio,
            "audio_sample_rate": rate,
            "fps": fps,
            "video_latents_clean": batch.batch["video_latents_clean"][i : i + 1].detach().cpu(),
            "audio_latents_clean": batch.batch["audio_latents_clean"][i : i + 1].detach().cpu(),
            "train_timesteps": batch.batch["train_timesteps"][i : i + 1].detach().cpu(),
        }
        for modality in ("video", "audio"):
            key = f"{modality}_forward_noise"
            if key in batch.batch:
                sample[key] = batch.batch[key][i : i + 1].detach().cpu()
        timings = batch.non_tensor_batch.get("ltx_rollout_metrics")
        if timings is not None:
            sample["rollout_metrics"] = dict(timings[i])
        scores = dict(zip(reward_names, batch.batch["rm_scores"][i].detach().cpu().tolist(), strict=True))
        sample["reward_scores"] = scores
        sample["reward_probability_video_audio"] = batch.batch["reward_prob"][i, 0].detach().cpu()
        sample["text_reward_probability"] = float(batch.batch["text_reward_prob"][i])
        sample["text_advantage"] = float(batch.batch["text_advantages"][i])
        torch.save(sample, directory / f"{stem}.pt")
        imageio.mimwrite(directory / f"{stem}.mp4", video.permute(0, 2, 3, 1).numpy(), fps=fps, codec="libx264")
        sf.write(directory / f"{stem}.wav", audio.T.numpy(), rate)
        index.append(
            {
                "sample_file": f"{stem}.pt",
                "video_file": f"{stem}.mp4",
                "audio_file": f"{stem}.wav",
                "uid": sample["uid"],
                "original_prompt": record.original_prompt,
                "recaption": record.recaption,
                "policy_version": record.policy_version,
                "seed": record.seed,
                "recaption_tokens": record.response_token_ids.numel(),
                "ended_with_eos": int(record.response_token_ids[-1]) in record.eos_token_ids,
                "reward_scores": scores,
                "reward_probability_video_audio": sample["reward_probability_video_audio"].tolist(),
                "text_reward_probability": sample["text_reward_probability"],
                "text_advantage": sample["text_advantage"],
                "fps": fps,
                "audio_sample_rate": rate,
                "rollout_metrics": sample.get("rollout_metrics", {}),
            }
        )
    write_joint_review_gallery(directory, index)
    return index
