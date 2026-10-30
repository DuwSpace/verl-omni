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

"""Bound vocabulary activations without repeating FSDP head collectives."""

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from verl.utils.torch_functional import logprobs_from_logits_torch_npu


class ChunkedLogprobHead(torch.nn.Module):
    """Preserve LM-head parameter names and enter FSDP only once per batch."""

    def __init__(self, head, chunk_size=128):
        super().__init__()
        self.weight = head.weight
        self.bias = head.bias
        self.in_features = head.in_features
        self.out_features = head.out_features
        self.chunk_size = chunk_size

    def forward(self, hidden_states, *, targets=None, temperature=1.0):
        """Return ordinary logits or selected-token FP32 log probabilities."""
        if targets is None:
            return F.linear(hidden_states, self.weight, self.bias)

        def project(hidden, target):
            logits = F.linear(hidden, self.weight, self.bias).float() / temperature
            if logits.device.type == "npu":
                return logprobs_from_logits_torch_npu(logits, target)
            return logits.gather(-1, target[:, None]).squeeze(-1) - logits.logsumexp(-1)

        values = []
        for start in range(0, hidden_states.shape[0], self.chunk_size):
            hidden = hidden_states[start : start + self.chunk_size]
            target = targets[start : start + self.chunk_size]
            if torch.is_grad_enabled():
                values.append(checkpoint(project, hidden, target, use_reentrant=False))
            else:
                values.append(project(hidden, target))
        return torch.cat(values)
