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
import os
import subprocess
import sys
import threading

import pytest
import torch

from verl_omni.utils.determinism import configure_worker_determinism, deterministic_call


@pytest.fixture(autouse=True)
def restore_flags(monkeypatch):
    for key in ("HCCL_DETERMINISTIC", "LCCL_DETERMINISTIC"):
        monkeypatch.setenv(key, os.environ.get(key, "test-placeholder"))
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn


def test_strict_mode_does_not_reset_sampling_rng(monkeypatch):
    monkeypatch.setenv("VERL_OMNI_FULL_DETERMINISM", "1")
    torch.use_deterministic_algorithms(True, warn_only=True)
    before = torch.get_rng_state().clone()
    configure_worker_determinism()
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.equal(before, torch.get_rng_state())
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark


def test_disabled_does_not_override_caller_flags(monkeypatch):
    monkeypatch.delenv("VERL_OMNI_FULL_DETERMINISM", raising=False)
    torch.use_deterministic_algorithms(False)
    configure_worker_determinism()
    assert not torch.are_deterministic_algorithms_enabled()


def test_spawned_process_applies_inherited_flag(monkeypatch):
    monkeypatch.setenv("VERL_OMNI_FULL_DETERMINISM", "1")
    code = """
import torch
from verl_omni.utils.determinism import configure_worker_determinism
torch.use_deterministic_algorithms(False)
configure_worker_determinism()
assert torch.are_deterministic_algorithms_enabled()
assert not torch.is_deterministic_algorithms_warn_only_enabled()
"""
    subprocess.run([sys.executable, "-c", code], env=dict(os.environ), check=True, timeout=60)


@pytest.mark.asyncio
async def test_reward_call_configures_in_executing_thread(monkeypatch):
    monkeypatch.setenv("VERL_OMNI_FULL_DETERMINISM", "1")
    main_thread = threading.get_ident()
    calls = []
    original = torch.use_deterministic_algorithms

    def record(*args, **kwargs):
        calls.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "use_deterministic_algorithms", record)

    def infer(value):
        assert calls[-1] == threading.get_ident() != main_thread
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        return value

    assert await asyncio.to_thread(deterministic_call, infer, value=7) == 7
