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

"""Actor initialization seeds remain effective without deterministic kernels."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import set_seed
from verl.workers.config import FSDPEngineConfig

from verl_omni.workers.engine.fsdp import diffusers_impl


@pytest.mark.parametrize("full_determinism", [False, True])
def test_engine_initialization_seeds_all_rngs(monkeypatch, full_determinism):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(diffusers_impl.DiffusersFSDPEngine, "_init_device_mesh", lambda self: None)
    monkeypatch.setattr(
        diffusers_impl.DiffusionModelBase,
        "get_class",
        lambda config: SimpleNamespace(validate_engine_config=lambda engine_config: None),
    )
    deterministic_calls = []

    def deterministic_seed(seed):
        deterministic_calls.append(seed)
        set_seed(seed)

    monkeypatch.setattr(diffusers_impl, "enable_full_determinism", deterministic_seed)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    try:
        set_seed(73)
        expected = random.random(), np.random.rand(), torch.rand(4)
        set_seed(99)
        diffusers_impl.NFTDiffusersFSDPEngine(
            SimpleNamespace(lora_rank=0),
            FSDPEngineConfig(seed=73, full_determinism=full_determinism),
            None,
            None,
        )
        assert random.random() == expected[0]
        assert np.random.rand() == expected[1]
        torch.testing.assert_close(torch.rand(4), expected[2], rtol=0, atol=0)
        assert deterministic_calls == ([73] if full_determinism else [])
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
