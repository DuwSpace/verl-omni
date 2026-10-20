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

"""Configured initialization seeds must work independently of deterministic kernels."""

from types import SimpleNamespace

import torch

from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine


def test_engine_seed_controls_trainable_initialization_without_forcing_deterministic_kernels(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(PPODiffusersFSDPEngine, "_init_device_mesh", lambda self: None)
    cfg = SimpleNamespace(full_determinism=False, seed=123, param_offload=False, optimizer_offload=False)
    original_flag = torch.are_deterministic_algorithms_enabled()
    samples = []
    for previous_seed in (11, 999):
        torch.manual_seed(previous_seed)
        PPODiffusersFSDPEngine(SimpleNamespace(lora_rank=1), cfg, None, None)
        samples.append(torch.randn(32, 16))
    torch.testing.assert_close(samples[0], samples[1], atol=0, rtol=0)
    assert torch.are_deterministic_algorithms_enabled() == original_flag
