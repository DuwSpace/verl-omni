# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import copy

import torch
from peft import LoraConfig, inject_adapter_in_model
from transformers import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel

from verl_omni.pipelines.ltx2_unigrpo_nft.text.precision import install_fp32_gemma_compute
from verl_omni.pipelines.ltx2_unigrpo_nft.training.logprob_head import ChunkedLogprobHead


class TinyGemma(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = Gemma3TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            sliding_window=16,
            layer_types=["sliding_attention", "full_attention"],
        )
        config._attn_implementation = "sdpa"
        self.model = torch.nn.Module()
        self.model.language_model = Gemma3TextModel(config)
        self.lm_head = torch.nn.Linear(32, 64, bias=False)

    def forward(self, ids):
        return self.lm_head(self.model.language_model(input_ids=ids, use_cache=False).last_hidden_state)


def test_fp32_compute_preserves_storage_names_and_autocast():
    torch.manual_seed(3)
    model = TinyGemma().to(torch.bfloat16).eval()
    reference = copy.deepcopy(model).float()
    embedding = reference.model.language_model.embed_tokens
    embedding.embed_scale.fill_(embedding.scalar_embed_scale)
    parameters = dict(model.named_parameters())
    install_fp32_gemma_compute(model)
    install_fp32_gemma_compute(model)
    assert dict(model.named_parameters()).keys() == parameters.keys()
    assert all(p is parameters[n] and p.dtype == torch.bfloat16 for n, p in model.named_parameters())
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(ids)
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, reference(ids), atol=1e-6, rtol=1e-5)


def test_nonzero_lora_and_chunked_head_checkpoint_backward():
    torch.manual_seed(4)
    model = TinyGemma().to(torch.bfloat16)
    model.requires_grad_(False)
    # Match production: install before PEFT, then reapply for newly created adapters.
    install_fp32_gemma_compute(model)
    inject_adapter_in_model(LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"]), model)
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.float()
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.01)
    model.lm_head = ChunkedLogprobHead(model.lm_head, chunk_size=2)
    install_fp32_gemma_compute(model)
    language = model.model.language_model
    language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    language.enable_input_require_grads()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        hidden = language(input_ids=ids, use_cache=False).last_hidden_state[0, :-1]
        values = model.lm_head(hidden, targets=ids[0, 1:])
        (-values.mean()).backward()
    assert values.dtype == torch.float32
    grads = [p.grad for n, p in model.named_parameters() if "lora_" in n]
    assert all(g is not None and torch.isfinite(g).all() and g.abs().max() > 0 for g in grads)


def test_fsdp_policy_only_preserves_gemma():
    from types import SimpleNamespace

    from torch.distributed.fsdp import MixedPrecisionPolicy

    from verl_omni.pipelines.ltx2_unigrpo_nft.training.model import LTXJointModel

    gemma = TinyGemma()
    model = SimpleNamespace(gemma=gemma, gemma_compute_dtype="float32")
    default = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    policy = LTXJointModel.fsdp2_mixed_precision_policy(model, gemma.model.language_model.layers[0], default)
    assert policy.param_dtype is None
    assert not policy.cast_forward_inputs
    assert policy.reduce_dtype == torch.float32
    dit = torch.nn.Linear(2, 2)
    assert LTXJointModel.fsdp2_mixed_precision_policy(model, dit, default) is default
    model.gemma_compute_dtype = "bfloat16"
    assert LTXJointModel.fsdp2_mixed_precision_policy(model, gemma.lm_head, default) is default


def test_recipe_uses_same_compute_precision_for_actor_rollout_and_val():
    from pathlib import Path

    from omegaconf import OmegaConf

    path = Path(__file__).resolve().parents[2] / "examples/unigrpo_nft_trainer/ltx2/joint_train.yaml"
    config = OmegaConf.load(path).actor_rollout_ref
    assert config.model.pipeline.gemma_compute_dtype == "float32"
    assert config.rollout.pipeline.gemma_compute_dtype == config.model.pipeline.gemma_compute_dtype
    assert config.rollout.val_kwargs.pipeline.gemma_compute_dtype == config.model.pipeline.gemma_compute_dtype
    assert (
        config.rollout.engine_kwargs.vllm_omni.additional_config.gemma_compute_dtype
        == config.model.pipeline.gemma_compute_dtype
    )
