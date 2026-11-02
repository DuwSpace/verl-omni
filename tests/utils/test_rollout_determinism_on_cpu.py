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

"""Strict rollout flags do not consume or reset sampling randomness."""

from concurrent.futures import ThreadPoolExecutor

import torch

from verl_omni.utils.determinism import configure_rollout_determinism


def test_rollout_determinism_is_opt_in_and_applies_in_worker_thread(monkeypatch):
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    rng = torch.random.get_rng_state().clone()
    monkeypatch.delenv("VERL_OMNI_FULL_DETERMINISM", raising=False)
    monkeypatch.setenv("HCCL_DETERMINISTIC", "true")
    monkeypatch.setenv("LCCL_DETERMINISTIC", "0")
    try:
        torch.use_deterministic_algorithms(False)
        configure_rollout_determinism()
        assert not torch.are_deterministic_algorithms_enabled()
        monkeypatch.setenv("VERL_OMNI_FULL_DETERMINISM", "1")

        def work():
            torch.use_deterministic_algorithms(True, warn_only=True)
            configure_rollout_determinism()
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()

        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(work).result()
        assert torch.equal(rng, torch.random.get_rng_state())
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
