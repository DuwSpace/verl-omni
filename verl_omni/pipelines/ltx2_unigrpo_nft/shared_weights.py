# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build native LTX components once; later workers bind the shared CPU cache."""

import threading
from contextlib import contextmanager
from pathlib import Path

import torch

from verl_omni.utils.frozen_cpu_weights import FrozenCpuWeights

_CONSTRUCTION_LOCK = threading.RLock()


@contextmanager
def _empty_parameters():
    """Construct meta parameters, keeping derived/nonpersistent buffers on CPU.

    Unlike Accelerate's constructor replay, preserve vLLM parameter subclasses
    and their loader attributes without invoking their custom __init__ again.
    The patch is scoped to this constructor thread and always restored.
    """
    original = torch.nn.Module.register_parameter
    owner = threading.get_ident()

    def register(module, name, parameter):
        original(module, name, parameter)
        if threading.get_ident() != owner or parameter is None or parameter.is_meta:
            return
        empty = torch.Tensor._make_subclass(type(parameter), parameter.detach().to("meta"), parameter.requires_grad)
        empty.__dict__.update(parameter.__dict__)
        module._parameters[name] = empty

    torch.nn.Module.register_parameter = register
    try:
        yield
    finally:
        torch.nn.Module.register_parameter = original


def _empty_component(component_cls, model, subfolder, *, dtype, revision, local_files_only):
    kwargs = dict(subfolder=subfolder, revision=revision, local_files_only=local_files_only)
    if subfolder == "text_encoder":
        from transformers import Gemma3Config, Gemma3ForConditionalGeneration, GenerationConfig

        config = Gemma3Config.from_pretrained(model, **kwargs)
        with _empty_parameters():
            component = Gemma3ForConditionalGeneration._from_config(
                config, torch_dtype=torch.float32, attn_implementation="sdpa"
            )
            component.tie_weights()
        try:
            component.generation_config = GenerationConfig.from_pretrained(model, **kwargs)
        except OSError:
            component.generation_config = GenerationConfig.from_model_config(config)
    else:
        config = component_cls.load_config(model, **kwargs)
        with _empty_parameters():
            component = component_cls.from_config(config)
        component.to(dtype=dtype)
        if subfolder == "vae" and hasattr(component, "init_distributed"):
            component.init_distributed()
    return component.eval().requires_grad_(False)


@contextmanager
def shared_component_loading(directory, *, ready):
    """Reuse the native graph/setup, replacing only its two weight factories.

    The publisher loads Gemma directly in FP32 (no throwaway BF16 copy).
    Followers create empty components and attach the published mappings.
    Native tokenizer, scheduler, attention setup and VAE geometry stay intact.
    Each diffusion worker owns one pipeline; other threads use original loaders.
    """
    from vllm_omni.diffusion.models.ltx2 import ltx2_components as native

    with _CONSTRUCTION_LOCK:
        original_load = native._load_component
        original_transformer = native.create_transformer_from_config
        owner = threading.get_ident()

        def attach(component, name):
            component.requires_grad_(False)
            component._joint_cpu_weights = FrozenCpuWeights.from_shared(component, Path(directory) / f"{name}.pt")
            return component

        def load(component_cls, model, subfolder, **kwargs):
            if threading.get_ident() != owner:
                return original_load(component_cls, model, subfolder, **kwargs)
            if ready:
                component = _empty_component(
                    component_cls,
                    model,
                    subfolder,
                    dtype=kwargs["dtype"],
                    revision=kwargs["revision"],
                    local_files_only=kwargs["local_files_only"],
                )
                return attach(component, subfolder)
            if subfolder == "text_encoder":
                from transformers import Gemma3ForConditionalGeneration

                return Gemma3ForConditionalGeneration.from_pretrained(
                    model,
                    subfolder=subfolder,
                    torch_dtype=torch.float32,
                    revision=kwargs["revision"],
                    local_files_only=kwargs["local_files_only"],
                    attn_implementation="sdpa",
                )
            return original_load(component_cls, model, subfolder, **kwargs)

        def transformer(*args, **kwargs):
            if not ready or threading.get_ident() != owner:
                return original_transformer(*args, **kwargs)
            if kwargs.get("quant_config") is not None:
                raise ValueError("Shared LTX CPU masters require an unquantized transformer")
            with _empty_parameters():
                component = original_transformer(*args, **kwargs)
            return attach(component, "transformer")

        native._load_component = load
        native.create_transformer_from_config = transformer
        try:
            yield
        finally:
            native._load_component = original_load
            native.create_transformer_from_config = original_transformer
