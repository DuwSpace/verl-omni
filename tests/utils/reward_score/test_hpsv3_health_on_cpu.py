# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from verl_omni.utils.reward_score import hpsv3_reward
from verl_omni.utils.reward_score.hpsv3_health import (
    HPSHealthCheckError,
    HPSHealthProbe,
    build_reference,
)


class Decoder(torch.nn.Module):
    def forward(self, pixel_values, position_ids):
        return pixel_values


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.register_buffer("offset", torch.tensor(0.0), persistent=False)
        self.model = Decoder()
        self.fail = False

    def forward(self, pixel_values, input_ids, attention_mask, return_dict):
        values = self.model(pixel_values=pixel_values, position_ids=torch.arange(input_ids.shape[1]))
        if self.fail:
            raise RuntimeError("simulated device error")
        values = values.mean(dim=(1, 2, 3)) * self.weight + self.offset
        return {"logits": torch.stack([values, -values], dim=1)}


class Inferencer:
    def __init__(self):
        self.model = Model().eval()
        self.device = "cpu"
        self.token_shift = 0

    def prepare_batch(self, images, prompts):
        return {
            "pixel_values": torch.from_numpy(np.stack([np.asarray(image).copy() for image in images])).float(),
            "input_ids": torch.ones(len(prompts), 2, dtype=torch.long) + self.token_shift,
            "attention_mask": torch.ones(len(prompts), 2, dtype=torch.long),
        }

    def reward(self, images, prompts):
        return self.model(return_dict=True, **self.prepare_batch(images, prompts))["logits"]


@pytest.fixture
def setup_probe(tmp_path):
    inferencer = Inferencer()
    sample = {"frames": torch.arange(24, dtype=torch.uint8).reshape(2, 3, 2, 2), "prompts": ["original"] * 2}
    identity = {"model_path": "fake", "base_model_path": "fake-base", "use_sequential_position_ids": True}
    reference = build_reference(inferencer, [sample], identity)
    baseline = tmp_path / "reference.pt"
    torch.save(reference, baseline)
    config = {"enabled": True, "baseline_path": str(baseline), "output_dir": str(tmp_path / "checks")}
    return inferencer, config, identity


def test_fixed_reference_survives_restart_and_records_context(setup_probe):
    inferencer, config, identity = setup_probe
    baseline = Path(config["baseline_path"]).read_bytes()
    context = {"step": 14, "phase": "validation"}
    for stage in ["after_load", "after_wake", "before_sleep"]:
        probe = HPSHealthProbe(config, identity)
        probe.check(inferencer, stage=stage, context=context)
    records = [
        json.loads(line)
        for file in Path(config["output_dir"]).glob("checks-*.jsonl")
        for line in file.read_text().splitlines()
    ]
    assert len(records) == 3 and all(r["ok"] and r["context"] == context for r in records)
    assert all(r["samples"][0]["max_logit_diff"] == 0 for r in records)
    assert "effective_position_ids" in records[0]["samples"][0]["actual_input_hashes"]
    assert Path(config["baseline_path"]).read_bytes() == baseline


@pytest.mark.parametrize("mutation,changed", [("weight", "parameter:weight"), ("offset", "buffer:offset")])
def test_weight_and_nonpersistent_buffer_drift_dump_evidence_and_latch(setup_probe, mutation, changed):
    inferencer, config, identity = setup_probe
    config["dump_model_state"] = True
    probe = HPSHealthProbe(config, identity)
    probe.check(inferencer, stage="after_load", context={})
    with torch.no_grad():
        getattr(inferencer.model, mutation).add_(1)
    with pytest.raises(HPSHealthCheckError, match="evidence"):
        probe.check(inferencer, stage="after_wake", context={"step": 14})
    failure = next(Path(config["output_dir"]).glob("failure-*"))
    report = json.loads((failure / "failure.json").read_text())
    assert changed in report["changed_model_tensors"]
    assert (failure / "model_state.pt").is_file() and (failure / "buffers.pt").is_file()
    evidence = torch.load(failure / "probes.pt", weights_only=True)
    assert evidence["observed"][0]["logits"].shape == (2, 2)
    assert not inferencer.model.training
    # Fixing the model does not silently clear the failed probe or replace its baseline.
    with torch.no_grad():
        getattr(inferencer.model, mutation).sub_(1)
    with pytest.raises(HPSHealthCheckError):
        probe.check(inferencer, stage="after_wake", context={})
    assert len(list(Path(config["output_dir"]).glob("failure-*"))) == 1
    with pytest.raises(HPSHealthCheckError):
        HPSHealthProbe(config, identity).check(inferencer_with_bad_weights(), stage="after_load", context={})


def inferencer_with_bad_weights():
    inferencer = Inferencer()
    with torch.no_grad():
        inferencer.model.weight.add_(1)
    return inferencer


def test_preprocessing_drift_detected_even_when_scores_are_unchanged(setup_probe):
    inferencer, config, identity = setup_probe
    probe = HPSHealthProbe(config, identity)
    inferencer.token_shift = 1  # model deliberately ignores tokens in this fixture
    with pytest.raises(HPSHealthCheckError):
        probe.check(inferencer, stage="before_sleep", context={})
    report = json.loads(next(Path(config["output_dir"]).glob("failure-*/failure.json")).read_text())
    assert report["samples"][0]["logits_match"]
    assert report["changed_model_tensors"] == []


@pytest.mark.parametrize("fault", ["nan", "training", "forward_error"])
def test_nonfinite_training_mode_and_forward_errors_fail_closed(setup_probe, fault):
    inferencer, config, identity = setup_probe
    probe = HPSHealthProbe(config, identity)
    if fault == "nan":
        inferencer.model.offset.fill_(float("nan"))
    elif fault == "training":
        inferencer.model.model.train()
    else:
        inferencer.model.fail = True
    with pytest.raises(HPSHealthCheckError):
        probe.check(inferencer, stage="before_sleep", context={})
    evidence = torch.load(next(Path(config["output_dir"]).glob("failure-*/probes.pt")), weights_only=True)
    if fault == "forward_error":
        assert "pixel_values" in evidence["observed"][0]["inputs"]


def test_invalid_reference_and_configuration_rejected(setup_probe):
    inferencer, config, identity = setup_probe
    with pytest.raises(ValueError, match="identity mismatch"):
        HPSHealthProbe(config, {**identity, "model_path": "different"})
    for value in [-1, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="tolerances"):
            HPSHealthProbe({**config, "atol": value}, identity)
    with pytest.raises(ValueError, match="Unknown"):
        HPSHealthProbe({**config, "typo": 1}, identity)


def test_model_disabled_probe_does_no_io_and_enabled_probe_does_not_reenter_infer(setup_probe, monkeypatch):
    inferencer, config, identity = setup_probe
    monkeypatch.setattr(hpsv3_reward, "_HPSv3Inferencer", lambda **kwargs: inferencer)
    disabled = hpsv3_reward.HPSv3Model("fake", "cpu", health_check={"enabled": False, "baseline_path": "/missing"})
    disabled.check_health(stage="after_load", context={})
    enabled = hpsv3_reward.HPSv3Model("fake", "cpu", "fake-base", True, health_check=config)
    monkeypatch.setattr(enabled, "infer", lambda *args: pytest.fail("probe must not recursively acquire infer lock"))
    enabled.check_health(stage="after_load", context={"step": 20})
    with pytest.raises(ValueError, match="boolean"):
        hpsv3_reward.HPSv3Model("fake", "cpu", health_check={"enabled": "false"})


@pytest.mark.asyncio
async def test_actual_requests_preserve_scores_ids_inputs_and_dump_on_health_failure(setup_probe, monkeypatch):
    from verl import DataProto

    from verl_omni.reward_loop import reward_model_executor
    from verl_omni.workers.config.reward import RewardModelSpec

    inferencer, config, _ = setup_probe
    monkeypatch.setattr(hpsv3_reward, "_HPSv3Inferencer", lambda **kwargs: inferencer)
    monkeypatch.setattr(reward_model_executor, "_load_native_model", lambda _: hpsv3_reward.HPSv3Model)
    monkeypatch.setattr(reward_model_executor, "_empty_accelerator_cache", lambda: None)
    executor = reward_model_executor.NativeRewardExecutor(
        RewardModelSpec(
            name="hps",
            backend="native",
            model_path="fake",
            cpu_offload=True,
            executor_config={
                "model": "fake:Model",
                "kwargs": {
                    "device": "cpu",
                    "base_model_path": "fake-base",
                    "use_sequential_position_ids": True,
                    "health_check": config,
                },
            },
        )
    )
    await executor.wake_up(context={"step": 14, "phase": "train"})

    async def score(value):
        batch = DataProto(
            non_tensor_batch={
                "uid": np.array([f"uid-{value}"]),
                "_ltx_sample_seed": np.array([value]),
                "reward_inputs": np.array([{"text": {"video": f"actual prompt {value}"}}], dtype=object),
            }
        )
        return await hpsv3_reward.compute_score_hpsv3(
            "test",
            torch.full((5, 3, 2, 2), value, dtype=torch.uint8),
            "unused prompt",
            {},
            reward_model=executor,
            batch=batch,
            prompt_key="video",
            num_frames=5,
            max_batch_size=2,
        )

    results = await asyncio.gather(score(10), score(20))
    assert [result["score"] for result in results] == pytest.approx([1.0, 2.0])
    await executor.sleep()
    root = Path(config["output_dir"])
    assert not list(root.glob("input-anomaly-*"))
    records = [json.loads(line) for path in root.glob("requests-*.jsonl") for line in path.read_text().splitlines()]
    assert len(records) == 2
    for record in records:
        value = int(record["uid"].split("-")[-1])
        assert record["sample_seed"] == str(value)
        assert record["context"] == {"step": 14, "phase": "train"}
        assert record["prompt_sha256"] == hashlib.sha256(f"actual prompt {value}".encode()).hexdigest()
        assert record["raw_video"]["min"] == record["raw_video"]["max"] == value
        assert [call["frame_offset"] for call in record["calls"]] == [0, 2, 4]
        assert [row[0] for call in record["calls"] for row in call["logits"]] == pytest.approx([value] * 5)
        assert "effective_position_ids" in record["calls"][0]["inputs"]
        assert record["calls"][0]["model_device"] == "cpu"
        assert record["calls"][0]["training_modules"] == []
    await executor.wake_up(context={"step": 14, "phase": "validation"})
    with torch.no_grad():
        inferencer.model.weight.add_(1)
    with pytest.raises(HPSHealthCheckError):
        await executor.sleep()
    evidence = torch.load(next(root.glob("failure-*/recent_requests.pt")), weights_only=True)
    assert len(evidence["requests"]) == 2
    for sample in evidence["requests"]:
        value = int(sample["record"]["uid"].split("-")[-1])
        torch.testing.assert_close(sample["video"], torch.full((5, 3, 2, 2), value, dtype=torch.uint8))
        assert len(sample["calls"]) == 3
    await executor.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["nonfinite_input", "nonfinite_logits", "forward_error"])
async def test_actual_request_failures_capture_exact_inputs(setup_probe, monkeypatch, fault):
    inferencer, config, _ = setup_probe
    monkeypatch.setattr(hpsv3_reward, "_HPSv3Inferencer", lambda **kwargs: inferencer)
    model = hpsv3_reward.HPSv3Model("fake", "cpu", "fake-base", True, health_check=config)
    model.check_health(stage="after_load", context={"step": 14, "phase": "validation"})
    video = torch.ones(1, 3, 2, 2)
    if fault == "nonfinite_input":
        video[0, 0, 0, 0] = float("nan")
    elif fault == "nonfinite_logits":
        inferencer.model.offset.fill_(float("nan"))
    else:
        inferencer.model.fail = True

    class AsyncModel:
        reward_diagnostics = model.reward_diagnostics

        async def infer(self, *args, **kwargs):
            return model.infer(*args, **kwargs)

    with pytest.raises((ValueError, RuntimeError)):
        await hpsv3_reward.compute_score_hpsv3("test", video, "prompt", {"uid": "bad"}, reward_model=AsyncModel())
    root = Path(config["output_dir"])
    record = json.loads(next(root.glob("requests-*.jsonl")).read_text())
    assert not record["ok"] and record["uid"] == "bad"
    evidence = torch.load(next(root.glob("input-anomaly-*/recent_requests.pt")), weights_only=True)
    torch.testing.assert_close(evidence["requests"][0]["video"], video, equal_nan=True)
    if fault == "forward_error":
        assert "pixel_values" in evidence["requests"][0]["calls"][0]["inputs"]
