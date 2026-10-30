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
"""Process-local deterministic native reward computation."""

import os
import random

import numpy as np
import torch


def configure_reward_reproducibility(*, deterministic: bool):
    """Enable strict deterministic operators before model/device initialization.

    Settings apply to the isolated reward worker process, including its inference
    threads. This reproduces a fixed batch; it does not guarantee equality across
    different batch sizes or compositions.
    """
    if not deterministic:
        return
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    os.environ["HCCL_DETERMINISTIC"] = "true"
    os.environ["LCCL_DETERMINISTIC"] = "1"
    os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
