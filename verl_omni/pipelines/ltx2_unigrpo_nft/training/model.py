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

"""Actor model: Gemma text policy, frozen AV conditioning, and trainable DiT LoRA."""

import math

import torch
from transformers import PretrainedConfig

from verl_omni.pipelines.non_diffusers_model_base import NonDiffusersModelBase

from ..text.policy import base_encoder
from .logprob_head import ChunkedLogprobHead


class LTXJointConfig(PretrainedConfig):
    """Checkpoint metadata for the complete joint model, not just its DiT."""

    model_type = "ltx2_unigrpo_nft"

    def __init__(
        self, gemma_config=None, transformer_config=None, connectors_config=None, base_model_path=None, **kwargs
    ):
        kwargs.setdefault("architectures", ["LTXJointModel"])
        super().__init__(**kwargs)
        self.gemma_config = gemma_config or {}
        self.transformer_config = transformer_config or {}
        self.connectors_config = connectors_config or {}
        # Transformers may omit name_or_path on serialization; retain an
        # explicit provenance field for the common frozen base snapshot.
        self.base_model_path = base_model_path


class NonAffineRMSNorm(torch.nn.Module):
    """Diffusers' reference formula without the NPU kernel's required gamma."""

    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states):
        variance = hidden_states.float().square().mean(-1, keepdim=True)
        return (hidden_states * torch.rsqrt(variance + self.eps)).to(hidden_states.dtype)


def _replace_nonaffine_rmsnorm(module):
    from diffusers.models.normalization import RMSNorm

    for name, child in list(module.named_modules()):
        if isinstance(child, RMSNorm) and child.weight is None:
            parent, _, leaf = name.rpartition(".")
            setattr(module.get_submodule(parent), leaf, NonAffineRMSNorm(child.eps))


class LTXJointModel(NonDiffusersModelBase):
    """Own the exact same Gemma parameters for autoregressive and AV conditioning passes.

    Only the trainable model components are loaded; VAEs and vocoder belong to
    rollout. AV conditioning uses frozen base Gemma weights and connectors.
    """

    _no_split_modules = ["Gemma3DecoderLayer", "LTX2VideoTransformerBlock"]
    _supports_gradient_checkpointing = True
    # Match Transformers' from_pretrained buffer precision on the rollout side.
    _keep_in_fp32_buffers = ("gemma.model.language_model.rotary_emb",)
    # PEFT/Transformers scope nested Gemma checkpoint conversions from this
    # composite root; there is no additional wrapper prefix to remove.
    base_model_prefix = ""

    def __init__(self, gemma, transformer, connectors):
        from .attention import install_native_connector_attention, install_native_dit_attention

        super().__init__()
        self.gemma_compute_dtype = "bfloat16"
        self.gemma = gemma
        self.gemma.lm_head = ChunkedLogprobHead(gemma.lm_head)
        self.transformer = transformer
        self.connectors = connectors
        _replace_nonaffine_rmsnorm(self.transformer)
        _replace_nonaffine_rmsnorm(self.connectors)
        install_native_connector_attention(self.connectors)
        install_native_dit_attention(self.transformer)
        self.config = LTXJointConfig(
            gemma_config=gemma.config.to_dict(),
            transformer_config=dict(transformer.config),
            connectors_config=dict(getattr(connectors, "config", {})),
        )
        self.requires_grad_(False)

    @classmethod
    def from_pretrained(cls, model_path, torch_dtype=torch.bfloat16, **kwargs):
        from diffusers import LTX2VideoTransformer3DModel
        from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
        from transformers import Gemma3ForConditionalGeneration

        gemma_kwargs = dict(kwargs)
        # Match the rollout's explicit Transformers SDPA backend.
        gemma_kwargs.setdefault("attn_implementation", "sdpa")
        gemma = Gemma3ForConditionalGeneration.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=torch_dtype, **gemma_kwargs
        )
        transformer = LTX2VideoTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=torch_dtype, **kwargs
        )
        connectors = LTX2TextConnectors.from_pretrained(
            model_path, subfolder="connectors", torch_dtype=torch_dtype, **kwargs
        )
        module = cls(gemma, transformer, connectors)
        module.config.name_or_path = str(model_path)
        module.config.base_model_path = str(model_path)
        return module

    def add_adapter(self, adapter_config, adapter_name="default"):
        """Require both policy components and exclude frozen feature-processing modules."""
        if adapter_config.bias != "none" or adapter_config.modules_to_save:
            raise ValueError("LTX joint policy trains only LoRA, without biases or modules_to_save")
        super().add_adapter(adapter_config, adapter_name)
        selected = [name for name, _ in self.named_parameters() if f".{adapter_name}." in name and "lora_" in name]
        if not selected or any(
            not name.startswith(("gemma.model.language_model.", "transformer.")) for name in selected
        ):
            raise ValueError("Joint LoRA must target only Gemma language layers and the AV transformer")
        if not all(any(name.startswith(prefix) for name in selected) for prefix in ("gemma.", "transformer.")):
            raise ValueError("Joint LoRA must contain both Gemma and DiT parameters")
        self.set_adapter(adapter_name)
        if self.gemma_compute_dtype == "float32":
            from ..text.precision import install_fp32_gemma_compute

            install_fp32_gemma_compute(self.gemma)

    def fsdp2_mixed_precision_policy(self, module, default_policy):
        """Preserve Gemma activations and LoRA precision without changing DiT."""
        if self.gemma_compute_dtype == "float32" and any(module is child for child in self.gemma.modules()):
            from dataclasses import replace

            return replace(default_policy, param_dtype=None, cast_forward_inputs=False)
        return default_policy

    def set_adapter(self, adapter_name):
        """Select leaf adapters without invoking unrelated Transformers/Diffusers PEFT wrappers."""
        from peft.tuners.tuners_utils import BaseTunerLayer

        if adapter_name not in getattr(self, "peft_config", {}):
            raise ValueError(f"Unknown joint adapter: {adapter_name}")
        for module in self.modules():
            if isinstance(module, BaseTunerLayer):
                module.set_adapter(adapter_name)
        # The engine's generic adapter copy helper discovers parameters using
        # requires_grad. Frozen old/ref forwards must additionally use no_grad.
        self.connectors.requires_grad_(False)

    def disable_adapters(self):
        from peft.tuners.tuners_utils import BaseTunerLayer

        for layer in self.modules():
            if isinstance(layer, BaseTunerLayer):
                layer.enable_adapters(False)

    def enable_adapters(self):
        from peft.tuners.tuners_utils import BaseTunerLayer

        for layer in self.modules():
            if isinstance(layer, BaseTunerLayer):
                layer.enable_adapters(True)

    def enable_gradient_checkpointing(self, gradient_checkpointing_func=None):
        super().enable_gradient_checkpointing(gradient_checkpointing_func)
        self.gemma.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.transformer.enable_gradient_checkpointing(gradient_checkpointing_func=gradient_checkpointing_func)

    @torch.no_grad()
    def encode_condition(self, input_ids, attention_mask):
        """Match LTX's stack-all-Gemma-layers layout without computing vocabulary logits."""
        with base_encoder(self.gemma):
            output = self.gemma.model.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden = torch.stack(output.hidden_states, dim=-1).flatten(2, 3)
        hidden = hidden.to(next(self.connectors.parameters()).dtype)
        return self.connectors(hidden, attention_mask, padding_side="left")

    def recaption_logprobs(self, input_ids, attention_mask, response_mask, temperature):
        """Teacher-force exact sampled tokens with temperature-only sampling.

        Masks use full sequence positions; output columns predict input_ids[:, 1:].
        Enter the LM head once per rank. Vocabulary projections are checkpointed
        in local chunks inside that FSDP boundary, independent of response length.
        """
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Recaption temperature must be finite and positive")
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape or input_ids.shape != response_mask.shape:
            raise ValueError("Recaption token IDs and masks must share [batch, sequence] shape")
        if response_mask.dtype != torch.bool or bool(response_mask[:, 0].any()):
            raise ValueError("Response mask must be boolean and leave a preceding context token")
        if bool((response_mask & ~attention_mask.bool()).any()) or not bool(response_mask.any(dim=1).all()):
            raise ValueError("Every response needs attended generated tokens")
        if bool((response_mask[:, 1:] & ~attention_mask[:, :-1].bool()).any()):
            raise ValueError("Each response token needs an attended preceding context token")
        hidden = self.gemma.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        mask = response_mask[:, 1:]
        selected_hidden = hidden[:, :-1][mask]
        targets = input_ids[:, 1:][mask]

        values = self.gemma.lm_head(selected_hidden, targets=targets, temperature=temperature)
        result = hidden.new_zeros(mask.shape, dtype=torch.float32)
        return result.masked_scatter(mask, values)

    def forward(self, *, task, input_ids, attention_mask, response_mask=None, temperature=1.0, **dit_inputs):
        """Keep both branches behind the root forward for FSDP parameter materialization."""
        if task == "recaption":
            return self.recaption_logprobs(input_ids, attention_mask, response_mask, temperature)
        if task != "av":
            raise ValueError(f"Unknown joint training task: {task}")
        video, audio, mask = self.encode_condition(input_ids, attention_mask)
        return self.transformer(
            **dit_inputs,
            encoder_hidden_states=video,
            audio_encoder_hidden_states=audio,
            encoder_attention_mask=mask,
            audio_encoder_attention_mask=mask,
            return_dict=False,
        )
