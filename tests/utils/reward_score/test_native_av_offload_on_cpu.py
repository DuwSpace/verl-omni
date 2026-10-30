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

"""Native AV model retention through executor sleep/wake cycles."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import transformers

from verl_omni.reward_loop import reward_model_executor as executor_module
from verl_omni.reward_loop.reward_model_executor import NativeRewardExecutor
from verl_omni.utils.reward_score import audiobox, clap, desync, hpsv3_reward, videoalign
from verl_omni.workers.config.reward import RewardModelSpec


class Weights(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1, 2], dtype=torch.bfloat16))
        self.register_buffer("calibration", torch.tensor([0.25], dtype=torch.float32))
        self.kind = kind
        self.moves = []
        self.fail_transfer = False
        self.target_transform = {key: {"mean": 0.0, "std": 1.0} for key in audiobox._AXES}

    def to(self, device):
        self.moves.append(torch.device(device))
        if self.fail_transfer:
            raise RuntimeError("transfer failed")
        # Emulate separate accelerator storage without accelerator hardware.
        if torch.device(device).type != "cpu":
            self.weight.data = self.weight.detach().clone()
            self.calibration = self.calibration.clone()
            return self
        return super().to(device)

    def forward(self, *args, **kwargs):
        value = (self.weight.float().sum() + self.calibration).reshape(1, 1)
        if self.kind == "audiobox":
            return {key: value for key in audiobox._AXES}
        if self.kind == "clap":
            return SimpleNamespace(audio_embeds=value, text_embeds=value)
        return {"logits": value}


@pytest.fixture(params=["videoalign", "hpsv3", "audiobox", "clap", "desync"])
def reward_case(request, monkeypatch):
    kind = request.param
    weights = Weights(kind)
    mel = Weights("mel")
    processor = Mock(return_value={})
    inferencer = SimpleNamespace(model=weights, processor=processor, device=torch.device("cpu"))
    inferencer.reward = lambda *args: weights()["logits"]
    kwargs = {"model_path": "unused"}
    if kind == "videoalign":
        loader = Mock(return_value=(weights, processor))
        monkeypatch.setattr(videoalign, "_load_components", loader)
        cls = videoalign.VideoAlignModel
        kwargs["base_model_path"] = "unused"
        monkeypatch.setattr(cls, "_prepare_batch", lambda *args: {})
        args = ([], [])
    elif kind == "hpsv3":
        loader = Mock(return_value=inferencer)
        monkeypatch.setattr(hpsv3_reward, "_HPSv3Inferencer", loader)
        cls = hpsv3_reward.HPSv3Model
        args = ([], [])
    elif kind == "audiobox":
        loader = Mock(return_value=weights)
        monkeypatch.setattr(audiobox, "_load_model", loader)
        cls = audiobox.AudioBoxModel
        args = (torch.ones(1, 1, 8), torch.ones(1, 8, dtype=torch.bool))
    elif kind == "clap":
        loader = Mock(return_value=weights)
        monkeypatch.setattr(transformers.ClapModel, "from_pretrained", loader)
        monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", Mock(return_value=processor))
        cls = clap.CLAPModel
        args = ([], [])
    else:
        loader = Mock(return_value=(weights, mel))
        monkeypatch.setattr(desync, "_load_components", loader)
        cls = desync.DeSyncModel
        kwargs["source_root"] = "unused"
        monkeypatch.setattr(cls, "_infer_micro_batch", lambda self, *args: self.model()["logits"])
        args = (None, None)
    return SimpleNamespace(
        cls=cls,
        kwargs=kwargs,
        args=args,
        loader=loader,
        weights=weights,
        modules=[weights, mel] if kind == "desync" else [weights],
        processor=processor,
        inferencer=inferencer,
        kind=kind,
    )


def make_executor(case, monkeypatch):
    monkeypatch.setattr(executor_module, "_load_native_model", lambda _: case.cls)
    return NativeRewardExecutor(
        RewardModelSpec(
            name=case.kind,
            backend="native",
            device_type="cpu",
            executor_config={"model": "unused:Model", "kwargs": case.kwargs},
        )
    )


@pytest.mark.asyncio
async def test_models_retain_identity_outputs_and_dtypes_across_cycles(reward_case, monkeypatch):
    case = reward_case
    executor = make_executor(case, monkeypatch)
    await executor.wake_up()
    model = executor._model
    modules = tuple(model._offload_modules())
    expected = await executor.infer(*case.args)
    for _ in range(3):
        await executor.sleep()
        await executor.sleep()
        assert executor._model is model
        assert tuple(model._offload_modules()) == modules
        assert model.device == torch.device("cpu")
        with pytest.raises(RuntimeError, match="not awake"):
            await executor.infer(*case.args)
        await executor.wake_up()
        await executor.wake_up()
        actual = await executor.infer(*case.args)
        torch.testing.assert_close(actual, expected)
        for module in modules:
            assert module.weight.dtype == torch.bfloat16
            assert module.calibration.dtype == torch.float32
    case.loader.assert_called_once()
    if case.kind == "hpsv3":
        assert model._inferencer is case.inferencer
        assert model._inferencer.processor is case.processor
    elif case.kind in ("videoalign", "clap"):
        assert model.processor is case.processor
    await executor.sleep()
    await executor.close()
    await executor.close()
    assert executor._model is None
    assert model._offload_closed
    assert not model._cpu_weights
    assert (model._inferencer if case.kind == "hpsv3" else model.model) is None
    with pytest.raises(RuntimeError, match="closed"):
        model.wake_up()


def test_all_components_restore_original_accelerator(reward_case):
    case = reward_case
    target = torch.device("cuda:3")
    model = case.cls(device=target, **case.kwargs)
    for module in case.modules:
        module.moves.clear()
    model.sleep()
    model.sleep()
    assert model.device == torch.device("cpu")
    if case.kind == "hpsv3":
        assert case.inferencer.device == torch.device("cpu")
    model.wake_up()
    model.wake_up()
    assert model.device == target
    if case.kind == "hpsv3":
        assert case.inferencer.device == target
    for module in case.modules:
        assert module.moves == [torch.device("cpu"), target]
    model.close()


@pytest.mark.asyncio
async def test_transfer_failure_releases_retained_model(reward_case, monkeypatch):
    case = reward_case
    executor = make_executor(case, monkeypatch)
    await executor.wake_up()
    model = executor._model
    case.modules[-1].fail_transfer = True
    with pytest.raises(RuntimeError, match="transfer failed"):
        await executor.sleep()
    assert executor._model is None and executor._closed
    assert model._offload_closed
    assert not model._cpu_weights
    assert (model._inferencer if case.kind == "hpsv3" else model.model) is None


def test_sleep_reuses_cpu_storage_and_preserves_runtime_buffers(reward_case):
    case = reward_case
    model = case.cls(device="cuda:3", **case.kwargs)
    masters = list(model._cpu_weights)
    for _ in range(3):
        for parameter, host in masters:
            assert not parameter.requires_grad
            assert parameter.data_ptr() != host.data_ptr()
        expected_buffers = []
        for module in case.modules:
            module.calibration.add_(1)
            expected_buffers.append(module.calibration.clone())
        model.sleep()
        for parameter, host in masters:
            assert parameter.data_ptr() == host.data_ptr()
            torch.testing.assert_close(parameter, torch.tensor([1, 2], dtype=torch.bfloat16))
        for module, expected in zip(case.modules, expected_buffers, strict=True):
            torch.testing.assert_close(module.calibration, expected)
        model.wake_up()
    model.close()
    assert not model._cpu_weights


def test_desync_chunks_features_without_changing_sample_order():
    model = object.__new__(desync.DeSyncModel)
    model.feature_batch_size = 8
    chunks = []

    def features(video, audio):
        chunks.append(len(video))
        return video[:, None, None, :].expand(-1, 24, 1, -1), audio[:, None, None, :].expand(-1, 24, 1, -1)

    model._extract_features = features
    model.model = SimpleNamespace(
        compare_v_a=lambda visual, auditory: (visual[:, 0, 0] + auditory[:, 0, 0]).expand(-1, 21)
    )
    video = torch.arange(32, dtype=torch.float32).reshape(-1, 1)
    audio = video + 10
    logits = model._infer_micro_batch(video, audio)
    assert chunks == [8, 8, 8, 8]
    expected = (video + audio).expand(-1, 21).unsqueeze(0).expand(2, -1, -1)
    torch.testing.assert_close(logits, expected)


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_desync_rejects_invalid_feature_batch_size(size):
    with pytest.raises(ValueError, match="feature_batch_size"):
        desync.DeSyncModel("unused", "cpu", "unused", feature_batch_size=size)
