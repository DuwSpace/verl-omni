# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU checks for stage residency and otherwise unregistered LoRA tensors."""

from types import SimpleNamespace

import pytest
import torch

from verl_omni.pipelines.ltx2_unigrpo_nft.offload import JointRolloutOffload


class Component(torch.nn.Linear):
    def generate(self, value):
        return self(value)

    def decode(self, value):
        return self(value)


def test_stages_are_mutually_exclusive_and_nonforward_methods_activate_first(monkeypatch):
    pipeline = SimpleNamespace(**{name: Component(2, 2) for name in JointRolloutOffload.component_names})
    residence = {}
    moves = []

    def move(module, device):
        name = next(name for name, value in vars(pipeline).items() if value is module)
        residence[name] = device.type
        moves.append((name, device.type))
        assert sum(value == "cuda" for value in residence.values()) <= 1

    monkeypatch.setattr(JointRolloutOffload, "_move", staticmethod(move))
    monkeypatch.setattr(JointRolloutOffload, "_flush", lambda self: None)
    # These methods check residency BEFORE invoking __call__/forward hooks.
    original_generate = pipeline.text_encoder.generate
    original_decode = pipeline.vae.decode

    def generate(value):
        assert residence["text_encoder"] == "cuda"
        return original_generate(value)

    def decode(value):
        assert residence["vae"] == "cuda"
        return original_decode(value)

    pipeline.text_encoder.generate = generate
    pipeline.vae.decode = decode
    controller = JointRolloutOffload(pipeline, "cuda:0")
    value = torch.ones(1, 2)
    for _ in range(2):
        pipeline.text_encoder.generate(value)
        pipeline.text_encoder(value)
        pipeline.connectors(value)
        pipeline.transformer(value)
        count = len(moves)
        pipeline.transformer(value)
        assert len(moves) == count  # no transfers per denoising step
        pipeline.vae.decode(value)
        pipeline.audio_vae.decode(value)
        pipeline.vocoder(value)
        controller.release()
        assert set(residence.values()) == {"cpu"}
    controller.disable()
    assert pipeline.text_encoder.generate is generate
    assert pipeline.vae.decode is decode
    assert "decode" not in pipeline.audio_vae.__dict__
    assert all(not module._forward_pre_hooks for module in vars(pipeline).values())


def test_lora_sidecar_target_moves_before_module_apply():
    calls = []

    class NativeLora(Component):
        def _set_diffusion_lora_buffer_device(self, device):
            calls.append(("sidecar", device))

        def _apply(self, fn, recurse=True):
            assert calls and calls[-1][0] == "sidecar"
            calls.append(("apply", None))
            return super()._apply(fn, recurse=recurse)

    JointRolloutOffload._move(NativeLora(2, 2), torch.device("cpu"))
    assert calls == [("sidecar", torch.device("cpu")), ("apply", None)]


def test_negative_encoding_uses_connector_dtype(monkeypatch):
    from verl_omni.pipelines.ltx2_unigrpo_nft.vllm_omni_rollout_adapter import LTX2Pipeline, LTXJointPipeline

    pipeline = object.__new__(LTXJointPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.connectors = Component(2, 2).to(torch.bfloat16)
    pipeline.text_encoder = Component(2, 2).float()
    captured = {}

    def encode(self, **kwargs):
        captured.update(kwargs)
        return "encoded"

    monkeypatch.setattr(LTX2Pipeline, "encode_prompt", encode)
    assert pipeline.encode_prompt(negative_prompt="", do_classifier_free_guidance=True) == "encoded"
    assert captured["dtype"] == torch.bfloat16
    assert next(pipeline.text_encoder.parameters()).dtype == torch.float32


@pytest.mark.parametrize("raises", [False, True])
def test_request_always_releases_controller(monkeypatch, raises):
    from verl_omni.pipelines.ltx2_unigrpo_nft.vllm_omni_rollout_adapter import LTX2Pipeline, LTXJointPipeline

    pipeline = object.__new__(LTXJointPipeline)
    torch.nn.Module.__init__(pipeline)
    released = []
    pipeline._joint_offload = SimpleNamespace(release=lambda: released.append(True))

    def forward(self, req):
        if raises:
            raise RuntimeError("simulated inference failure")
        return "output"

    monkeypatch.setattr(LTX2Pipeline, "forward", forward)
    from vllm_omni.diffusion.request import OmniDiffusionRequest

    monkeypatch.setattr(OmniDiffusionRequest, "is_dummy_run_request_id", lambda value: True)
    req = SimpleNamespace(requests=[SimpleNamespace(request_id="dummy")])
    if raises:
        with pytest.raises(RuntimeError, match="simulated"):
            pipeline.forward(req)
    else:
        assert pipeline.forward(req) == "output"
    assert released == [True]


def test_offload_sidecars_remain_mutable_for_next_policy_rpc():
    class NativeLora(Component):
        def __init__(self):
            super().__init__(2, 2)
            with torch.inference_mode():
                self.lora_a_stacked = (torch.ones(1, 2, 2),)
                self.lora_b_stacked = (torch.ones(1, 2, 2),)

        def _set_diffusion_lora_buffer_device(self, device):
            # copy=True reproduces a real accelerator->CPU transfer on CPU.
            self.lora_a_stacked = tuple(t.to(device, copy=True) for t in self.lora_a_stacked)
            self.lora_b_stacked = tuple(t.to(device, copy=True) for t in self.lora_b_stacked)

    layer = NativeLora()
    for _ in range(2):
        with torch.inference_mode():
            JointRolloutOffload._move(layer, torch.device("cpu"))
        for tensor in (*layer.lora_a_stacked, *layer.lora_b_stacked):
            assert not torch.is_inference(tensor)
            tensor[0] = 0  # native manager resets adapters outside inference_mode
            tensor[0].copy_(torch.ones(2, 2))
