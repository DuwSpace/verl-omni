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

"""Frozen conditioning with the shared Gemma language model."""

from contextlib import contextmanager


@contextmanager
def base_encoder(model):
    """Use unmerged base weights without changing FSDP's trainable parameter set.

    PEFT enable_adapters(False) also changes requires_grad. Only scope its
    forward flag here: the text policy may already have a live autograd graph, and
    the optimizer must continue owning the same LoRA parameters.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    layers = [layer for layer in model.modules() if isinstance(layer, BaseTunerLayer)]
    if any(layer.merged for layer in layers):
        raise RuntimeError("Frozen Gemma encoding requires unmerged LoRA weights")
    states = [layer.disable_adapters for layer in layers]
    try:
        for layer in layers:
            layer._disable_adapters = True
        yield
    finally:
        for layer, disabled in zip(layers, states, strict=True):
            layer._disable_adapters = disabled
