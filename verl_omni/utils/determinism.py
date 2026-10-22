# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Strict deterministic kernels in spawned inference workers and pool threads."""

import os
import threading

import torch

_thread_state = threading.local()


def configure_worker_determinism():
    """Apply before model work in each executing thread; never reseed per request.

    The driver propagates this flag and communication settings through Ray's
    runtime_env. Python torch flags do not propagate into spawned workers.
    Launch blocking is intentionally independent of algorithm determinism.
    """
    if os.environ.get("VERL_OMNI_FULL_DETERMINISM") != "1":
        return
    # Upstream actor seeding resets HCCL_DETERMINISTIC to 'true'. Keep the
    # stricter communication setting shared by training and rollout.
    os.environ["HCCL_DETERMINISTIC"] = "strict"
    os.environ["LCCL_DETERMINISTIC"] = "1"
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not getattr(_thread_state, "reported", False):
        print(
            f"VERL_OMNI_DETERMINISM pid={os.getpid()} thread={threading.current_thread().name} "
            f"enabled={torch.are_deterministic_algorithms_enabled()} "
            f"warn_only={torch.is_deterministic_algorithms_warn_only_enabled()}",
            flush=True,
        )
        _thread_state.reported = True


def deterministic_call(function, *args, **kwargs):
    """Configure in the pool thread that actually executes native reward work."""
    configure_worker_determinism()
    return function(*args, **kwargs)
