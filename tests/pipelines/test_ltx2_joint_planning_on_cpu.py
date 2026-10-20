# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Planning reaches GRPO replay, while only the final caption reaches AV."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from verl_omni.pipelines.ltx2_unigrpo_nft.planning import parse_planned_recaption, planning_instruction
from verl_omni.pipelines.ltx2_unigrpo_nft.replay import LTXJointReplay

PLAN = "Show the bell before the strike. Synchronize its motion with one chime."
CAPTION = "A brass bell swings once with a clear chime, filmed in a steady close-up."
RESPONSE = f"<plan>{PLAN}</plan>\n<recaption>{CAPTION}</recaption>"


def test_planning_prompt_preserves_task_and_duration():
    text = planning_instruction("A bell rings.", 1.333)
    assert text.endswith("USER_REQUEST:\nA bell rings.")
    assert "words" not in text and "four" not in text and "seconds" not in text
    assert "<plan>" not in text and "<recaption>" not in text
    assert "There is no prescribed length." in text
    assert "Output only the final audiovisual generation prompt" in text
    assert "Do not invent dialogue, narration, or music unless requested." in text
    assert parse_planned_recaption("\n" + RESPONSE + "\n") == (PLAN, CAPTION)


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_planning_rejects_invalid_duration(duration):
    with pytest.raises(ValueError, match="duration"):
        planning_instruction("A bell rings.", duration)


@pytest.mark.parametrize(
    "text",
    [
        CAPTION,
        "<plan>Plan only",
        "<plan>Plan</plan><recaption>Truncated",
        "<plan> </plan><recaption>Scene</recaption>",
        "<plan>Plan</plan><recaption> </recaption>",
        "<recaption>Scene</recaption><plan>Plan</plan>",
        "```\n" + RESPONSE + "\n```",
        RESPONSE + "<recaption>Another scene</recaption>",
        "<plan><plan>Nested</plan></plan><recaption>Scene</recaption>",
        RESPONSE + " trailing explanation",
    ],
)
def test_malformed_or_truncated_planning_never_becomes_an_av_caption(text):
    with pytest.raises(ValueError, match="planning"):
        parse_planned_recaption(text)


@pytest.fixture
def rollout(monkeypatch):
    from vllm_omni.diffusion.data import DiffusionOutput

    from verl_omni.pipelines.ltx2_unigrpo_nft.vllm_omni_rollout_adapter import LTX2Pipeline, LTXJointPipeline

    pipeline = object.__new__(LTXJointPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.text_encoder = SimpleNamespace(dtype=torch.float32)
    pipeline.vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=24000))
    pipeline._joint_policy_version = 3
    pipeline._joint_sync_in_progress = False
    pipeline._joint_lora_request = SimpleNamespace(lora_int_id=4)
    response_ids = torch.tensor([10, 11, 12, 13, 14, 1])
    logprobs = torch.arange(6, dtype=torch.float32).neg()
    calls = []

    def recaption(*args, **kwargs):
        return CAPTION, torch.tensor([2, 3]), response_ids, logprobs, (1,)

    def condition(text, maximum):
        assert text == CAPTION and maximum == 16
        calls.append("condition")
        return torch.ones(1, 3, 8), torch.tensor([[2, 15, 1]]), torch.ones(1, 3, dtype=torch.bool)

    geometry = dict(height=32, width=32, num_frames=9, frame_rate=24.0, num_inference_steps=2)

    def native_forward(self, request):
        assert request.requests[0].prompt["prompt"] == CAPTION
        calls.append("av")
        self._joint_clean = (torch.ones(1, 2, 4), torch.ones(1, 4, 4))
        self._joint_forward_context = SimpleNamespace(
            request_inputs=SimpleNamespace(**geometry), timesteps=torch.tensor([900.0, 500.0])
        )
        return DiffusionOutput(output=(torch.zeros(9, 3, 32, 32), torch.zeros(2, 2400)))

    monkeypatch.setattr(pipeline, "_recaption", recaption)
    monkeypatch.setattr(pipeline, "_condition", condition)
    monkeypatch.setattr(LTX2Pipeline, "forward", native_forward)
    request = SimpleNamespace(
        request_id="planning-test",
        prompt={"prompt": "A bell rings."},
        sampling_params=SimpleNamespace(
            **geometry,
            seed=42,
            max_sequence_length=16,
            lora_request=pipeline._joint_lora_request,
            extra_args={"policy_version": 3, "recaption_temperature": 1.0, "recaption_max_new_tokens": 512},
        ),
    )
    return pipeline, SimpleNamespace(requests=[request]), calls, response_ids, logprobs


def test_rollout_conditions_only_on_caption_and_replays_entire_response(rollout):
    from verl_omni.trainer.diffusion.recaption_grpo import clipped_recaption_loss

    pipeline, request, calls, response_ids, logprobs = rollout
    output = pipeline.forward(request)
    record = LTXJointReplay(**output.output["metadata"]["rl"]["ltx_joint_replay"])
    record.validate()
    assert calls == ["condition", "av"]
    assert (record.planning, record.recaption, record.response_text) == (None, CAPTION, CAPTION)
    torch.testing.assert_close(record.response_token_ids, response_ids)
    torch.testing.assert_close(record.rollout_logprobs, logprobs)
    assert record.condition_token_ids.tolist() == [2, 15, 1]
    ar = record.ar_inputs("cpu")
    assert ar["response_mask"].tolist() == [[False, False, True, True, True, True, True, True]]
    # All sampled response positions receive credit.
    current = torch.zeros(1, 7, requires_grad=True)
    loss, _ = clipped_recaption_loss(
        current, current.detach(), torch.ones(1), ar["response_mask"][:, 1:], clip_ratio=0.01
    )
    loss.backward()
    assert current.grad[0, 0] == 0
    assert (current.grad[0, 1:] != 0).all()
    assert pipeline._joint_clean is None and pipeline._joint_forward_context is None
    for changed in (replace(record, planning="different"), replace(record, response_text=None)):
        with pytest.raises(ValueError, match="Caption replay"):
            changed.validate()
    legacy = asdict(record)
    legacy.pop("planning")
    legacy.pop("response_text")
    legacy.pop("response_format")
    LTXJointReplay(**legacy).validate()


def test_rollout_rejects_token_budget_cutoff_before_av(rollout, monkeypatch):
    pipeline, request, calls, response_ids, logprobs = rollout
    monkeypatch.setattr(
        pipeline,
        "_recaption",
        lambda *args, **kwargs: ("An unfinished scene", torch.tensor([2, 3]), response_ids, logprobs, (99,)),
    )
    with pytest.raises(ValueError, match="truncated") as failure:
        pipeline.forward(request)
    assert "response_tokens=6" in str(failure.value)
    assert "ended_with_eos=False" in str(failure.value)
    assert "response_text='An unfinished scene'" in str(failure.value)
    assert "policy_version=3" in str(failure.value)
    assert calls == []
    assert pipeline._joint_clean is None and pipeline._joint_forward_context is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("width", 33),
        ("num_frames", 120),
        ("num_inference_steps", 0),
        ("frame_rate", float("nan")),
        ("audio_cfg_scale", -1.0),
    ],
)
def test_sampling_rejects_invalid_geometry_and_guidance(key, value):
    from dataclasses import asdict

    from verl_omni.pipelines.ltx2_unigrpo_nft.config import LTXJointPipelineConfig, sampling_params

    config = asdict(LTXJointPipelineConfig(height=256, width=384, num_frames=121))
    config[key] = value
    with pytest.raises(ValueError, match="LTX"):
        sampling_params(config)


def test_backend_forwards_guidance_without_mutating_cached_geometry():
    from copy import deepcopy

    from vllm_omni.diffusion.models.ltx2.ltx2_guidance import LTXModalityGuidance

    from verl_omni.pipelines.ltx2_unigrpo_nft.config import LTXJointPipelineConfig, sampling_params
    from verl_omni.workers.rollout.ltx_joint_backend import LTXJointVllmBackend

    backend = object.__new__(LTXJointVllmBackend)
    backend.geometry = sampling_params(
        LTXJointPipelineConfig(height=256, width=384, num_frames=121, num_inference_steps=20, audio_cfg_scale=3.0)
    )
    original = deepcopy(backend.geometry)
    validation = deepcopy(original)
    validation["num_inference_steps"] = 40
    validation["extra_args"].update(audio_cfg_scale=7.0, video_modality_scale=3.0, video_rescale_scale=0.7)
    backend.policy_version, backend.temperature, backend.max_new_tokens = 3, 1.0, 512
    backend.lora_request = None
    seen = []

    def receive(request):
        seen.append(request.sampling_params)
        raise RuntimeError("request captured")

    backend.engine = SimpleNamespace(add_req_and_wait_for_response=receive)
    for settings in (validation, None):
        with pytest.raises(RuntimeError, match="request captured"):
            backend.generate(["A bell rings."], [42], version=3, sampling_params=settings)
    assert [item.num_inference_steps for item in seen] == [40, 20]
    assert [item.extra_args["audio_cfg_scale"] for item in seen] == [7.0, 3.0]
    assert seen[0].extra_args["recaption_max_new_tokens"] == 512
    guidance = LTXModalityGuidance(cfg_scale=7.0, modality_scale=3.0, rescale_scale=0.7)
    assert guidance.do_cfg and guidance.do_modality_guidance
    assert backend.geometry == original
    assert "policy_version" not in validation["extra_args"]


def test_only_sampled_eos_can_terminate_unclosed_final_caption(rollout, monkeypatch):
    pipeline, request, calls, response_ids, logprobs = rollout
    unclosed = RESPONSE.removesuffix("</recaption>")
    assert parse_planned_recaption(unclosed, ended_with_eos=True) == (PLAN, CAPTION)
    with pytest.raises(ValueError, match="truncated"):
        parse_planned_recaption(unclosed)
    with pytest.raises(ValueError):
        parse_planned_recaption("<plan>unfinished", ended_with_eos=True)
    output = pipeline.forward(request)
    record = LTXJointReplay(**output.output["metadata"]["rl"]["ltx_joint_replay"])
    record = replace(record, response_format="planned", planning=PLAN, response_text=unclosed)
    record.validate()
    assert record.response_text == unclosed
    torch.testing.assert_close(record.response_token_ids, response_ids)
    torch.testing.assert_close(record.rollout_logprobs, logprobs)
    with pytest.raises(ValueError, match="truncated"):
        replace(record, response_token_ids=torch.tensor([10, 11, 12, 13, 14, 15])).validate()


@pytest.mark.parametrize("closing_tag", [True, False])
def test_plan_may_refer_to_recaption_tag_without_creating_another_section(rollout, monkeypatch, closing_tag):
    pipeline, request, calls, response_ids, logprobs = rollout
    plan = "Generate Spanish audio. The script will be provided below in the <recaption> section."
    response = f"<plan>{plan}</plan><recaption>{CAPTION}" + ("</recaption>" if closing_tag else "")
    assert parse_planned_recaption(response, ended_with_eos=True) == (plan, CAPTION)
    output = pipeline.forward(request)
    record = LTXJointReplay(**output.output["metadata"]["rl"]["ltx_joint_replay"])
    record = replace(record, response_format="planned", planning=plan, response_text=response)
    record.validate()
    assert record.planning == plan and record.response_text == response
    torch.testing.assert_close(record.response_token_ids, response_ids)
    torch.testing.assert_close(record.rollout_logprobs, logprobs)
    assert calls == ["condition", "av"]
    if not closing_tag:
        with pytest.raises(ValueError, match="truncated"):
            parse_planned_recaption(response, ended_with_eos=False)
