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
"""CPU coverage for single- and multi-latent training through the same NFT engine."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines import nft_hooks
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.engine.fsdp import diffusers_impl


class _Policies(torch.nn.Module):
    def __init__(self, multimodal):
        super().__init__()
        self.weights = torch.nn.ParameterDict(
            {
                name: torch.nn.Parameter(torch.tensor(value))
                for name, value in (("old", 1.0), ("current", 2.0), ("reference", 3.0))
            }
        )
        self.active = "current"
        self.multimodal = multimodal
        self.calls = []

    @contextmanager
    def select(self, name):
        previous = self.active
        self.active = name
        try:
            yield
        finally:
            self.active = previous

    def forward(self, latents):
        self.calls.append((self.active, torch.is_grad_enabled()))
        prediction = latents * self.weights[self.active]
        if self.multimodal:
            return prediction[:, :3], prediction[:, 3:]
        return prediction


def _register_adapter(monkeypatch, multimodal):
    class Adapter(DiffusionModelBase):
        @classmethod
        def prepare_model_inputs(cls, module, model_config, latents, *args, **kwargs):
            return {"latents": latents}, None

        @classmethod
        def forward(cls, module, model_config, model_inputs, negative_model_inputs=None):
            return module(**model_inputs)

    algorithm = "omni_nft" if multimodal else "diffusion_nft"
    monkeypatch.setitem(DiffusionModelBase._registry, ("_NFTHooksTest", algorithm), Adapter)
    return SimpleNamespace(
        architecture="_NFTHooksTest",
        algorithm=algorithm,
        external_lib=None,
        enable_gradient_checkpointing=True,
        lora_rank=0,
    )


def _engine(monkeypatch, multimodal, num_train_timesteps=1000, engine_class=diffusers_impl.NFTDiffusersFSDPEngine):
    model_config = _register_adapter(monkeypatch, multimodal)
    engine_config = SimpleNamespace(
        strategy="fsdp2",
        ulysses_sequence_parallel_size=1,
        mixed_precision=None,
        full_determinism=False,
        param_offload=False,
        optimizer_offload=False,
        forward_only=False,
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(diffusers_impl.DiffusersFSDPEngine, "_init_device_mesh", lambda self: None)
    engine = engine_class(model_config, engine_config, None, None)
    engine.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=num_train_timesteps))
    engine.use_ulysses_sp = False
    engine.ulysses_sequence_parallel_size = 1
    engine.module = _Policies(multimodal)
    engine.use_adapter = lambda name: engine.module.select("old" if name == "old" else "current")
    engine.disable_adapter = lambda: engine.module.select("reference")
    engine._set_adapter = Mock()
    engine.get_data_parallel_group = lambda: None
    return engine


def _batch(multimodal, per_step_noise=False, scale=1000):
    batch = TensorDict(
        {
            "train_timesteps": torch.tensor([[0.5, 0.25], [0.5, 0.75]]) * scale,
            "prompt_embeds": torch.zeros(2, 3, 4),
            "prompt_embeds_mask": torch.ones(2, 3, dtype=torch.bool),
            "reward_prob": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            "advantages": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "group_index": torch.zeros(2, dtype=torch.long),
            "group_size": torch.full((2,), 2, dtype=torch.long),
            "_omnift_reward_metric::video_mean": torch.tensor([5.0, 6.0]),
        },
        batch_size=2,
    )
    branches = (("video_", 3, 2.0, -1.0), ("audio_", 2, 3.0, -2.0)) if multimodal else (("", 3, 2.0, -1.0),)
    for prefix, length, clean_value, noise_value in branches:
        clean = torch.full((2, length, 4), clean_value)
        noise = torch.full_like(clean, noise_value)
        batch[f"{prefix}latents_clean"] = clean
        batch[f"{prefix}forward_noise"] = (
            torch.stack((torch.full_like(noise, 99.0), noise), dim=1) if per_step_noise else noise
        )
    tu.assign_non_tensor(batch, gradient_accumulation_steps=3, sp_size=1)
    return batch


@pytest.mark.parametrize(
    "strategy,sp_size,message",
    [("fsdp", 1, "only actor.strategy=fsdp2"), ("fsdp2", 2, "does not implement Ulysses")],
)
def test_omni_nft_rejects_unsupported_engine_config_before_init(monkeypatch, strategy, sp_size, message):
    model_config = _register_adapter(monkeypatch, True)
    get_rank = Mock()
    monkeypatch.setattr(torch.distributed, "get_rank", get_rank)
    config = SimpleNamespace(strategy=strategy, ulysses_sequence_parallel_size=sp_size)
    with pytest.raises(NotImplementedError, match=message):
        diffusers_impl.NFTDiffusersFSDPEngine(model_config, config, None, None)
    get_rank.assert_not_called()


def test_existing_algorithms_keep_single_latent_hooks():
    for algorithm in ("diffusion_nft", "dgpo", "external_nft_adapter"):
        assert nft_hooks.get_nft_training_hooks(algorithm) is nft_hooks.NFTTrainingHooks
    assert nft_hooks.get_nft_training_hooks("omni_nft") is nft_hooks.OmniNFTTrainingHooks


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("scale", [1000, 2000])
@pytest.mark.parametrize("per_step_noise", [False, True])
def test_shared_engine_noises_each_branch_with_scheduler_time(monkeypatch, multimodal, scale, per_step_noise):
    engine = _engine(monkeypatch, multimodal, scale)
    batch = _batch(multimodal, per_step_noise, scale)
    positive, negative, context = engine.prepare_model_inputs(batch, step=1)
    if multimodal:
        clean, noised, times = (
            tuple(context[modality][name] for modality in ("video", "audio")) for name in ("x0", "xt", "t_expanded")
        )
    else:
        clean, noised, times = (context[name] for name in ("x0", "xt", "t_expanded"))
    time = torch.tensor([0.25, 0.75]).view(2, 1, 1)
    expected_video = (1 - time) * 2.0 + time * -1.0
    torch.testing.assert_close(noised[0] if multimodal else noised, expected_video.expand(2, 3, 4))
    torch.testing.assert_close(times[0] if multimodal else times, time)
    assert tu.get_non_tensor_data(batch, "num_train_timesteps", default=None) == scale
    assert negative is None
    if multimodal:
        expected_audio = ((1 - time) * 3.0 + time * -2.0).expand(2, 2, 4)
        torch.testing.assert_close(noised[1], expected_audio)
        torch.testing.assert_close(positive["latents"], torch.cat(noised, dim=1))
        assert clean[0] is batch["video_latents_clean"] and clean[1] is batch["audio_latents_clean"]
    else:
        torch.testing.assert_close(positive["latents"], noised)
        assert clean is batch["latents_clean"]


@pytest.mark.parametrize("multimodal", [False, True])
def test_shared_forward_keeps_policy_gradients_and_loss_fields(monkeypatch, multimodal):
    engine = _engine(monkeypatch, multimodal)
    batch = _batch(multimodal)
    prefixes = ("video_", "audio_") if multimodal else ("",)

    def loss_function(*, model_output, data, dp_group):
        torch.testing.assert_close(data["reward_prob"], batch["reward_prob"][:, 1])
        assert tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=None) == 3
        assert tu.get_non_tensor_data(data, "sp_size", default=None) == 1
        if multimodal:
            torch.testing.assert_close(
                data["_omnift_reward_metric::video_mean"], batch["_omnift_reward_metric::video_mean"]
            )
        else:
            torch.testing.assert_close(data["advantages"], batch["advantages"][:, 1])
            for key in ("group_index", "group_size"):
                torch.testing.assert_close(data[key], batch[key])
        for prefix in prefixes:
            assert not model_output[f"{prefix}old_prediction"].requires_grad
            assert not model_output[f"{prefix}ref_forward_prediction"].requires_grad
            assert model_output[f"{prefix}forward_prediction"].requires_grad
            torch.testing.assert_close(model_output[f"{prefix}old_prediction"], model_output[f"{prefix}xt"])
            torch.testing.assert_close(model_output[f"{prefix}ref_forward_prediction"], model_output[f"{prefix}xt"] * 3)
        loss = sum(model_output[f"{prefix}forward_prediction"].square().mean() for prefix in prefixes)
        return loss, {"checked": 1.0}

    loss, output = engine.forward_step(batch, loss_function, forward_only=False, step=1)
    loss.backward()
    assert engine.module.calls == [("old", False), ("current", True), ("reference", False)]
    assert engine.module.weights["old"].grad is None
    assert engine.module.weights["reference"].grad is None
    assert engine.module.weights["current"].grad.item() > 0
    assert output["loss"] == pytest.approx(loss.item())
    assert output["metrics"] == {"checked": 1.0}
    engine._set_adapter.assert_called_once_with("default")


@pytest.mark.parametrize("multimodal", [False, True])
def test_forward_only_accepts_tensor_and_tuple_context(monkeypatch, multimodal):
    engine = _engine(monkeypatch, multimodal)
    loss, output = engine.forward_step(_batch(multimodal), None, forward_only=True, step=1)
    assert loss.item() == 1.0 and output["metrics"] == {}


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize(
    "engine_class",
    [
        diffusers_impl.NFTDiffusersFSDPEngine,
        diffusers_impl.PPODiffusersFSDPEngine,
        diffusers_impl.DPODiffusersFSDPEngine,
    ],
)
def test_checkpoint_override_is_scoped_and_precedes_wrapping(monkeypatch, multimodal, enabled, use_lora, engine_class):
    engine = _engine(monkeypatch, multimodal, engine_class=engine_class)
    engine.model_config.enable_gradient_checkpointing = enabled
    engine._is_lora = use_lora
    module = engine.module
    events = []
    module.enable_gradient_checkpointing = Mock(side_effect=lambda **kwargs: events.append("checkpoint"))
    monkeypatch.setattr(diffusers_impl.DiffusersFSDPEngine, "_build_module", lambda self: module)
    engine._build_lora_module = lambda loaded: events.append("lora") or loaded
    monkeypatch.setattr(diffusers_impl, "_maybe_compile_repeated_blocks", lambda *args: events.append("compile"))
    engine._build_scheduler = lambda: engine.scheduler
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(diffusers_impl, "log_gpu_memory_usage", lambda *args, **kwargs: None)
    engine._build_fsdp_module = lambda loaded: events.append("fsdp") or loaded
    engine._build_optimizer = lambda loaded: None
    engine._build_lr_scheduler = lambda optimizer: None

    engine._build_model_optimizer()
    uses_override = multimodal and enabled and engine_class is diffusers_impl.NFTDiffusersFSDPEngine
    expected = (["checkpoint"] if uses_override else []) + (["lora"] if use_lora else []) + ["compile", "fsdp"]
    assert events == expected
    assert engine.module is module
    if uses_override:
        assert callable(module.enable_gradient_checkpointing.call_args.kwargs["gradient_checkpointing_func"])
    else:
        module.enable_gradient_checkpointing.assert_not_called()


def test_checkpoint_wrapper_casts_inputs_and_preserves_gradients():
    module = torch.nn.Linear(4, 2, bias=False).to(torch.bfloat16)
    with torch.no_grad():
        module.weight.fill_(1.0)
    inputs = torch.ones(2, 4, requires_grad=True)
    checkpoint = nft_hooks._gradient_checkpointing_with_cast(torch.bfloat16)
    result = checkpoint(module, inputs)
    result.float().sum().backward()
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result, torch.full((2, 2), 4.0, dtype=torch.bfloat16))
    torch.testing.assert_close(inputs.grad, torch.full_like(inputs, 2.0))
    torch.testing.assert_close(module.weight.grad, torch.full_like(module.weight, 2.0))


@pytest.mark.parametrize("prefix", ["_omnift_reward_metric::", "renamed_reward_metric::"])
def test_omni_nft_hooks_use_loss_metric_prefix(monkeypatch, prefix):
    from verl_omni.trainer.diffusion.diffusion_algos import OmniNFTLoss

    monkeypatch.setattr(OmniNFTLoss, "_REWARD_METRIC_PREFIX", prefix)
    batch = _batch(True)
    metric_key = f"{prefix}video/quality"
    batch[metric_key] = torch.ones(2)
    inputs = nft_hooks.OmniNFTTrainingHooks.prepare_loss_inputs(batch, step=1)
    assert inputs[metric_key] is batch[metric_key]
    assert set(inputs) == {"reward_prob"} | {key for key in batch.keys() if key.startswith(prefix)}
