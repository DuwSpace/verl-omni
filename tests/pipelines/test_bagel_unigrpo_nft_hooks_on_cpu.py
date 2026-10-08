# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hybrid hook ownership and both-branch gradient tests with a tiny CPU model."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.bagel_unigrpo_nft.hooks import BagelUniGRPONFTHooks
from verl_omni.pipelines.bagel_unigrpo_nft.replay import BagelJointReplay
from verl_omni.workers.config.diffusion.actor import DiffusionLossConfig


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.und = torch.nn.Parameter(torch.tensor(0.2))
        self.moe_gen = torch.nn.Parameter(torch.tensor(0.7))
        self.config = SimpleNamespace(start_of_image_id=10, end_of_image_id=11)


def _record():
    return BagelJointReplay(
        prompt_token_ids=torch.tensor([1, 3, 2]),
        cached_thinking_token_ids=torch.tensor([1, 4]),
        response_token_ids=torch.tensor([4, 2]),
        rollout_log_probs=torch.tensor([-0.3, -0.4]),
        latents_clean=torch.ones(2, 3),
        latent_pos_ids=torch.tensor([0, 1]),
        image_position_ids=torch.full((4,), 5),
        image_boundary_token_ids=torch.tensor([10, 11]),
        temperature=1,
        policy_version=0,
        eos_token_id=2,
    )


def _fixture(monkeypatch):
    module = _ToyModel()
    hooks = BagelUniGRPONFTHooks(module, None, None)
    monkeypatch.setattr(hooks, "_ar", lambda record: hooks.module.und * torch.tensor([0.2, 0.3]))

    def velocity(record, noise, sigma):
        clean = record.latents_clean.unsqueeze(0)
        time = sigma.reshape(1, 1, 1)
        noisy = (1 - time) * clean + time * noise.unsqueeze(0)
        prediction = (hooks.module.moe_gen + 0.1 * hooks.module.und).expand_as(clean)
        return prediction, clean, noisy, time

    monkeypatch.setattr(hooks, "_velocity", velocity)
    data = tu.get_tensordict(
        {
            "bagel_joint_replay": [_record(), _record()],
            "advantages": torch.tensor([0.7, -0.2]),
            "reward_prob": torch.tensor([0.8, 0.4]),
            "train_timesteps": torch.full((2, 2), 0.4),
            "nft_noise": torch.zeros(2, 2, 2, 3),
        }
    )
    cfg = SimpleNamespace(diffusion_loss=DiffusionLossConfig(loss_mode="unigrpo_nft", ref_kl_coef=1.5e-5))
    loss_function = partial(lambda **kwargs: None, config=cfg)
    return module, hooks, data, loss_function


def test_hooks_accumulate_both_expert_gradients_without_optimizer_step(monkeypatch):
    module, hooks, data, loss_function = _fixture(monkeypatch)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-6)
    optimizer.step = Mock(wraps=optimizer.step)
    before = {name: parameter.detach().clone() for name, parameter in module.named_parameters()}
    result = hooks.forward_backward_batch(data, loss_function)
    assert optimizer.step.call_count == 0
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad)) and float(parameter.grad.abs()) > 0
        torch.testing.assert_close(parameter, before[name])
    assert result["metrics"]["ar/ratio_mean"][0] == pytest.approx(1)
    assert result["metrics"]["image/ref_kl_loss"][0] == pytest.approx(0)
    assert result["metrics"]["image/old_deviate"][0] == pytest.approx(0)
    optimizer.step()
    assert optimizer.step.call_count == 1
    for name, parameter in module.named_parameters():
        assert float((parameter.detach() - before[name]).abs()) > 0


def test_second_update_keeps_old_and_reference_policies_frozen(monkeypatch):
    module, hooks, data, loss_function = _fixture(monkeypatch)
    module.und.data.add_(0.2)
    module.moe_gen.data.add_(0.3)
    result = hooks.forward_backward_batch(data, loss_function)
    assert result["metrics"]["image/old_deviate"][0] > 0
    assert result["metrics"]["image/ref_kl_loss"][0] > 0
    assert result["metrics"]["ar/clipfrac"][0] > 0
    torch.testing.assert_close(hooks.policies.old["und"], torch.tensor(0.2))
    torch.testing.assert_close(hooks.policies.reference["moe_gen"], torch.tensor(0.7))


def test_rollout_version_mismatch_is_not_silently_reanchored(monkeypatch):
    _, hooks, data, loss_function = _fixture(monkeypatch)
    hooks.begin_rollout(version=1)
    with pytest.raises(ValueError, match="version"):
        hooks.forward_backward_batch(data, loss_function)


def test_actor_generation_does_not_fallback_to_native_unigrpo_sampler(monkeypatch):
    _, hooks, data, _ = _fixture(monkeypatch)
    with pytest.raises(ValueError, match="original prompts"):
        hooks.generate(data)


def test_generation_freezes_old_and_calls_real_backend_interface(monkeypatch):
    _, hooks, _, _ = _fixture(monkeypatch)
    backend = Mock()
    backend.generate.return_value = (torch.zeros(2, 3, 4, 4, dtype=torch.uint8), [_record(), _record()])
    hooks._rollout_backend = backend
    data = tu.get_tensordict(
        {
            "raw_prompt": [[{"role": "user", "content": "red circle"}]] * 2,
            "_bagel_policy_version": [0, 0],
            "_bagel_sample_seed": [42, 43],
        }
    )
    output = hooks.generate(data)
    backend.load_actor_policy.assert_called_once_with(hooks.module, version=0)
    assert backend.generate.call_args.args[0] == ["red circle", "red circle"]
    assert output["responses"].shape == (2, 3, 4, 4)
