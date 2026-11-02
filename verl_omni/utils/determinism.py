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

"""Opt-in strict determinism for spawned rollout computation."""

import os
import threading

import torch

_thread_state = threading.local()


def rollout_determinism_env(enabled: bool) -> dict[str, str]:
    """Export before Ray workers spawn; Python torch flags are not inherited."""
    if not enabled:
        return {}
    return {
        "VERL_OMNI_FULL_DETERMINISM": "1",
        "HCCL_DETERMINISTIC": "strict",
        "LCCL_DETERMINISTIC": "1",
        "CLOSE_MATMUL_K_SHIFT": "1",
    }


def configure_rollout_determinism() -> None:
    """Apply in the executing thread without resetting the request RNG."""
    if os.environ.get("VERL_OMNI_FULL_DETERMINISM") != "1":
        return
    # Upstream initialization may reset the communication mode to 'true'.
    os.environ["HCCL_DETERMINISTIC"] = "strict"
    os.environ["LCCL_DETERMINISTIC"] = "1"
    torch.use_deterministic_algorithms(True, warn_only=False)
    if not getattr(_thread_state, "reported", False):
        print(
            f"ROLLOUT_DETERMINISM pid={os.getpid()} thread={threading.current_thread().name} "
            f"enabled={torch.are_deterministic_algorithms_enabled()} "
            f"warn_only={torch.is_deterministic_algorithms_warn_only_enabled()}",
            flush=True,
        )
        _thread_state.reported = True
