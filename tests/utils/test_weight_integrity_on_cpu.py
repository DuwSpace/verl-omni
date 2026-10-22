# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from verl_omni.utils.weight_integrity import first_different_byte_block


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64])
def test_byte_comparison_covers_all_chunks_and_detects_small_corruption(dtype):
    reference = torch.arange(40, dtype=dtype).reshape(5, 8)
    actual = reference.clone()
    assert first_different_byte_block(actual, reference, chunk_bytes=16) is None
    actual.view(-1)[-1] = 0
    assert first_different_byte_block(actual, reference, chunk_bytes=16) is not None


def test_byte_comparison_checks_bits_not_float_equality():
    reference = torch.tensor([float("nan"), 0.0])
    actual = reference.clone()
    assert first_different_byte_block(actual, reference) is None
    actual[1] = -0.0
    assert first_different_byte_block(actual, reference) == 0


def test_zeroed_status_transfer_cannot_pass(monkeypatch):
    original = torch.Tensor.to

    def zero_status(tensor, *args, **kwargs):
        result = original(tensor, *args, **kwargs)
        if tensor.dtype == torch.bool and tensor.numel() == 2:
            result.zero_()
        return result

    monkeypatch.setattr(torch.Tensor, "to", zero_status)
    value = torch.arange(8, dtype=torch.float32)
    assert first_different_byte_block(value.clone(), value) == 0
