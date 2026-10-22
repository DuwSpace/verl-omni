# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Byte-level comparison helpers for text diagnostics and policy loading."""

import hashlib

import torch


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    result = hashlib.sha256(f"{value.dtype}:{tuple(value.shape)}".encode())
    result.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return result.hexdigest()


@torch.no_grad()
def first_different_byte_block(value, host, *, chunk_bytes=4 * 1024 * 1024):
    """Compare every byte on the working device, without copying weights D2H.

    A bounded reference chunk is copied H2D independently of the model load.
    Return only complementary all-equal/any-different flags, twice, so a zeroed
    result buffer cannot be interpreted as success. This detects differences;
    it does not prove that the device or both transfer paths are fault-free.
    No floating-point reduction, custom kernel, or full pinned model is used.
    """
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if value.shape != host.shape or value.dtype != host.dtype or host.device.type != "cpu":
        raise ValueError("Working tensor and CPU master must have the same shape and dtype")
    actual = value.detach().contiguous().reshape(-1).view(torch.uint8)
    reference = host.detach().contiguous().reshape(-1).view(torch.uint8)
    for offset in range(0, reference.numel(), chunk_bytes):
        expected = reference[offset : offset + chunk_bytes].to(value.device, non_blocking=False)
        equal = actual[offset : offset + chunk_bytes] == expected
        status = torch.stack((equal.all(), (~equal).any()))
        for _ in range(2):
            if status.to("cpu", copy=True, non_blocking=False).tolist() != [True, False]:
                return offset
    return None
