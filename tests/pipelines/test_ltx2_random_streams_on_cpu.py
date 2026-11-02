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

"""Legacy joint random streams for data order, text sampling and training noise."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.trainer.ppo.utils import create_rl_sampler

from verl_omni.pipelines.ltx2_unigrpo_nft.text.sampling import SeededSampler
from verl_omni.pipelines.ltx2_unigrpo_nft.training.loss import LTXJointLoss


def noise_batch(seeds):
    return SimpleNamespace(
        batch={
            "train_timesteps": torch.zeros(len(seeds), 2, dtype=torch.int64),
            "video_latents_clean": torch.zeros(len(seeds), 3),
            "audio_latents_clean": torch.zeros(len(seeds), 2),
        }
    )


@pytest.mark.parametrize("default_dtype", [torch.float32, torch.bfloat16])
def test_forward_noise_matches_legacy_cpu_fixture_and_preserves_rng(default_dtype):
    # Fixed outputs of the legacy e32c5bf helper, seed 42, two selected timesteps.
    expected = {
        "video": [[-0.6524442434, 0.1380146593, 0.5821108818], [-0.7915729880, -1.1998467445, 0.8588591218]],
        "audio": [[1.5068883896, 0.6472190022], [0.1514510661, 1.1075125933]],
    }
    batch = noise_batch([42])
    state = torch.random.get_rng_state().clone()
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        LTXJointLoss._prepare_forward_noise(batch, [SimpleNamespace(seed=42)])
    finally:
        torch.set_default_dtype(previous)
    for modality, values in expected.items():
        actual = batch.batch[f"{modality}_forward_noise"]
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual[0], torch.tensor(values), rtol=0, atol=1e-7)
    assert torch.equal(state, torch.random.get_rng_state())


def test_forward_noise_is_stable_under_reordering_chunking_and_seed_wrap():
    seeds = [42, 298, 2**63 - 1, -1]
    full = noise_batch(seeds)
    LTXJointLoss._prepare_forward_noise(full, [SimpleNamespace(seed=s) for s in seeds])
    for index in (3, 1, 0, 2):
        single = noise_batch([seeds[index]])
        LTXJointLoss._prepare_forward_noise(single, [SimpleNamespace(seed=seeds[index])])
        for modality in ("video", "audio"):
            key = f"{modality}_forward_noise"
            torch.testing.assert_close(single.batch[key][0], full.batch[key][index], rtol=0, atol=0)
    for modality in ("video", "audio"):
        # Both seeds map to the same nonnegative stream modulo 2**63.
        value = full.batch[f"{modality}_forward_noise"]
        torch.testing.assert_close(value[2], value[3], rtol=0, atol=0)


@pytest.mark.parametrize("temperature", [0.7, 1.0])
def test_ar_private_stream_matches_legacy_single_row_multinomial(temperature):
    scores = torch.arange(31, dtype=torch.float32).sin()
    probabilities = (scores / temperature).softmax(-1)
    seeds = [42, 298, 810]
    expected = []
    for seed in seeds:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            expected.append(torch.stack([torch.multinomial(probabilities, 1)[0] for _ in range(12)]))
    before = torch.random.get_rng_state().clone()
    sampler = SeededSampler(seeds, temperature, "cpu")
    actual = torch.stack([sampler(None, scores.expand(3, -1)).argmax(-1) for _ in range(12)], dim=1)
    torch.testing.assert_close(actual, torch.stack(expected))
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("ablation", ["joint", "joint_tree", "ar_only", "av_only"])
def test_recipe_preserves_legacy_data_order_and_separate_actor_seeds(monkeypatch, ablation):
    for name in ("MODEL_PATH", "REWARD_ROOT", "OUTPUT_DIR"):
        monkeypatch.setenv("LTX_" + name, "/tmp/test")
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "examples/unigrpo_nft_trainer/ltx2"), version_base=None):
        cfg = compose(config_name="joint_train", overrides=[f"ablation={ablation}"])
    assert cfg.data.seed is None
    assert cfg.data.shuffle and not cfg.data.validation_shuffle
    assert cfg.actor_rollout_ref.actor.data_loader_seed == 42
    assert cfg.actor_rollout_ref.actor.fsdp_config.seed == 42
    assert cfg.actor_rollout_ref.rollout.seed == 42
    legacy = create_rl_sampler(OmegaConf.create({"shuffle": True, "seed": None}), list(range(64)))
    actual = create_rl_sampler(cfg.data, list(range(64)))
    first = list(actual)
    assert first == list(legacy)
    second = list(actual)
    assert second == list(legacy) and second != first
