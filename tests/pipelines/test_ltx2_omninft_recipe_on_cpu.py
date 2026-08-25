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
"""CPU contract tests for the staged LTX-2.3 OmniNFT recipe."""

import asyncio
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics
from verl.protocol import DataProto


REPO_ROOT = Path(__file__).parents[2]
RECIPE = REPO_ROOT / "examples/omninft_trainer/ltx2/run_ltx2_3_omninft_lora_npu.sh"


def test_recipe_has_valid_shell_syntax():
    subprocess.run(["bash", "-n", str(RECIPE)], check=True)


def test_recipe_freezes_omninft_direct_preference_contract():
    recipe = RECIPE.read_text(encoding="utf-8")
    required = (
        "python3 -m verl_omni.trainer.main_diffusion",
        "data.custom_cls.name=OmniNFTPromptDataset",
        "data.custom_cls.collate_fn=collate_omni_nft_prompt_groups",
        "algorithm.trainer_type=direct_preference",
        "algorithm.sample_source=online",
        "algorithm.paired_preference=false",
        "actor_rollout_ref.model.algorithm=omni_nft",
        "actor_rollout_ref.model.model_type=omni_nft_model",
        "actor_rollout_ref.actor.diffusion_loss.loss_mode=omni_nft",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.rollout.calculate_log_probs=False",
        "actor_rollout_ref.rollout.rollout_adapter=old",
        "actor_rollout_ref.rollout.agent.default_agent_loop=ltx2_omni_nft_single_turn_agent",
    )
    forbidden = ("calculate_log_probs=True", "algorithm=diffusion_nft", "rollout.algo.sde_")

    assert all(setting in recipe for setting in required)
    assert all(setting not in recipe for setting in forbidden)


def test_omninft_worker_reuses_shared_run_and_postprocess():
    from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopWorker
    from verl_omni.pipelines.ltx2_omni_nft.agent_loop import LTX2OmniNFTAgentLoopWorker

    assert LTX2OmniNFTAgentLoopWorker._run_agent_loop is DiffusionAgentLoopWorker._run_agent_loop
    assert LTX2OmniNFTAgentLoopWorker._agent_loop_postprocess is DiffusionAgentLoopWorker._agent_loop_postprocess


def test_omninft_worker_assigns_unique_sample_uid_after_prompt_group_expansion(monkeypatch):
    from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopWorker
    from verl_omni.pipelines.ltx2_omni_nft.agent_loop import LTX2OmniNFTAgentLoopWorker

    async def fake_generate(self, batch):
        return batch

    monkeypatch.setattr(DiffusionAgentLoopWorker, "generate_sequences", fake_generate)
    batch = DataProto.from_dict(
        tensors={"placeholder": torch.zeros(8)},
        non_tensors={"uid": np.array(["prompt-0"] * 8, dtype=object)},
    )
    worker = object.__new__(LTX2OmniNFTAgentLoopWorker)
    result = asyncio.run(worker.generate_sequences(batch))

    sample_uids = result.non_tensor_batch["sample_uid"]
    assert sample_uids.shape == (8,)
    assert len(set(sample_uids.tolist())) == 8
    assert set(result.non_tensor_batch["uid"].tolist()) == {"prompt-0"}


def test_omninft_worker_rejects_duplicate_sample_uid(monkeypatch):
    from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopWorker
    from verl_omni.pipelines.ltx2_omni_nft.agent_loop import LTX2OmniNFTAgentLoopWorker

    async def fake_generate(self, batch):
        return batch

    monkeypatch.setattr(DiffusionAgentLoopWorker, "generate_sequences", fake_generate)
    batch = DataProto.from_dict(
        tensors={"placeholder": torch.zeros(2)},
        non_tensors={"sample_uid": np.array(["duplicate", "duplicate"], dtype=object)},
    )
    worker = object.__new__(LTX2OmniNFTAgentLoopWorker)

    with pytest.raises(ValueError, match="must be unique"):
        asyncio.run(worker.generate_sequences(batch))


def test_g8_agent_loop_output_round_trips_without_identity_or_tensor_drift(tmp_path):
    from verl_omni.agent_loop.diffusion_agent_loop import (
        DiffusionAgentLoopWorker,
        _InternalDiffusionAgentLoopOutput,
    )

    tensor_specs = {
        "audio": ((1, 2, 8), torch.float32),
        "video_latents_clean": ((1, 5, 4), torch.float32),
        "audio_latents_clean": ((1, 7, 4), torch.float32),
        "prompt_embeds": ((1, 3, 4), torch.float32),
        "audio_prompt_embeds": ((1, 3, 4), torch.float32),
        "prompt_embeds_mask": ((1, 3), torch.long),
        "negative_prompt_embeds": ((1, 3, 4), torch.float32),
        "negative_audio_prompt_embeds": ((1, 3, 4), torch.float32),
        "negative_prompt_embeds_mask": ((1, 3), torch.long),
        "video_latent_shape": ((1, 2), torch.long),
        "audio_latent_shape": ((1, 2), torch.long),
        "video_seq_len": ((1,), torch.long),
        "audio_seq_len": ((1,), torch.long),
        "fps": ((1,), torch.float32),
        "audio_sample_rate": ((1,), torch.long),
        "train_timesteps": ((1, 2), torch.float32),
    }
    outputs = []
    for index in range(8):
        extra_fields = {
            key: torch.full(shape, index + field_index, dtype=dtype)
            for field_index, (key, (shape, dtype)) in enumerate(tensor_specs.items())
        }
        extra_fields["raw_prompt"] = [{"role": "user", "content": "a joint prompt"}]
        outputs.append(
            _InternalDiffusionAgentLoopOutput(
                prompt_ids=torch.tensor([[index, index + 1]], dtype=torch.long),
                response_diffusion_output=torch.full((1, 9, 3, 2, 2), index, dtype=torch.uint8),
                metrics=AgentLoopMetrics(),
                extra_fields=extra_fields,
            )
        )

    sample_uids = np.array([f"sample-{index}" for index in range(8)], dtype=object)
    worker = object.__new__(DiffusionAgentLoopWorker)
    replay = worker._postprocess(
        outputs,
        input_non_tensor_batch={
            "uid": np.array(["prompt-0"] * 8, dtype=object),
            "sample_uid": sample_uids,
        },
    )
    artifact_path = tmp_path / "replay.pkl"
    replay.save_to_disk(artifact_path)
    restored = DataProto.load_from_disk(artifact_path)

    assert "rollout_log_probs" not in replay.batch
    assert set(restored.batch.keys()) == set(replay.batch.keys())
    for key in replay.batch.keys():
        assert replay.batch[key].device.type == "cpu"
        assert restored.batch[key].device.type == "cpu"
        assert restored.batch[key].dtype == replay.batch[key].dtype
        torch.testing.assert_close(restored.batch[key], replay.batch[key])
    assert restored.batch["responses"][:, 0, 0, 0, 0].tolist() == list(range(8))
    assert restored.batch["responses"].dtype == torch.uint8
    assert restored.batch["prompt_embeds"].dtype == torch.float32
    assert restored.batch["prompt_embeds_mask"].dtype == torch.long
    assert restored.batch["audio_sample_rate"].dtype == torch.long
    assert restored.non_tensor_batch["uid"].tolist() == ["prompt-0"] * 8
    assert restored.non_tensor_batch["sample_uid"].tolist() == sample_uids.tolist()
    assert len(set(restored.non_tensor_batch["sample_uid"].tolist())) == 8
    assert restored.non_tensor_batch["raw_prompt"].tolist() == [
        [{"role": "user", "content": "a joint prompt"}]
    ] * 8
    assert restored.meta_info == replay.meta_info


def test_recipe_full_training_fails_before_runtime_initialization():
    result = subprocess.run([str(RECIPE)], cwd=REPO_ROOT, capture_output=True, text=True)

    assert result.returncode == 2
    assert "OmniNFT training prerequisites missing:" in result.stderr
    assert "rollout adapter (stage 2)" not in result.stderr
    assert "training adapter (stage 3)" not in result.stderr
    assert "omni_nft loss (stage 5)" in result.stderr
