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
"""Compare native vLLM-Omni Gemma encode vs the OmniNFT token-id mixin on one NPU."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration, masking_utils

from verl_omni.pipelines.ltx2_omni_nft.prompt_mixin import LTXTokenIdPromptMixin
from verl_omni.pipelines.ltx2_omni_nft.vllm_omni_rollout_adapter import LTX23OmniNFTPipeline

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="Ascend NPU is required",
)

_DEFAULT_MODEL = "/hub/models--diffusers--LTX-2.3-Diffusers/snapshots/8eee8edcf067e838b843f926ec4d4cc9b2be1aaf"
_PROMPT = "A cat walking on a sunny beach, cinematic lighting."


def _model_path() -> str:
    model_path = os.environ.get("MODEL_PATH", _DEFAULT_MODEL)
    if not Path(model_path).exists():
        pytest.skip(f"LTX checkpoint missing at {model_path}")
    return model_path


def _attn_impl(text_encoder) -> dict[str, str | None]:
    config = text_encoder.config
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    return {
        "model": getattr(config, "_attn_implementation", None),
        "text": getattr(text_config, "_attn_implementation", None),
    }


def _load_gemma(model_path: str, attn_implementation: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {}
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    with torch.device("cpu"):
        text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            subfolder="text_encoder",
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        )
    text_encoder.to("npu")
    text_encoder.eval()
    return tokenizer, text_encoder


def _debug_fast_all(tensor: torch.Tensor) -> bool:
    print(
        "[fast_all]",
        f"shape={tuple(tensor.shape)}",
        f"dtype={tensor.dtype}",
        f"device={tensor.device}",
        f"contiguous={tensor.is_contiguous()}",
        flush=True,
    )

    torch.npu.synchronize()
    print("[fast_all] entry sync OK", flush=True)

    summed = tensor.sum()
    torch.npu.synchronize()
    print(
        "[fast_all] sum OK",
        f"shape={tuple(summed.shape)}",
        f"dtype={summed.dtype}",
        flush=True,
    )

    result = summed == tensor.numel()
    torch.npu.synchronize()
    print("[fast_all] eq OK", flush=True)

    value = result.item()
    print(f"[fast_all] item OK: {value}", flush=True)
    return bool(value)


@pytest.mark.parametrize(
    "dtype",
    [torch.bool, torch.int64, torch.float32],
    ids=["bool", "int64", "float32"],
)
def test_fast_all_scalar_stages_on_npu(dtype: torch.dtype) -> None:
    device = torch.device("npu:0")
    print(f"[scalar_ab] dtype={dtype}", flush=True)
    tensor = torch.ones((1, 1024), dtype=dtype, device=device)
    torch.npu.synchronize()
    print("[scalar_ab] create OK", flush=True)
    assert _debug_fast_all(tensor) is True


@pytest.mark.parametrize("attn_implementation", ["sdpa", "eager"])
def test_gemma_real_padding_mask_stages_on_npu(
    attn_implementation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_size = int(os.environ.get("GEMMA_ENCODER_BATCH_SIZE", "1"))
    tokenizer, text_encoder = _load_gemma(
        _model_path(),
        attn_implementation=attn_implementation,
    )
    device = torch.device("npu:0")
    text_inputs = tokenizer(
        [_PROMPT.strip()] * batch_size,
        padding="max_length",
        max_length=1024,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    print(
        "[gemma_ab]",
        f"batch_size={batch_size}",
        f"requested={attn_implementation}",
        f"configured={_attn_impl(text_encoder)}",
        f"mask_ones={int(attention_mask.cpu().sum().item())}",
        f"mask_numel={attention_mask.numel()}",
        flush=True,
    )
    monkeypatch.setattr(masking_utils, "fast_all", _debug_fast_all)

    print("before text encoder sync", flush=True)
    torch.npu.synchronize()
    print("before text encoder sync OK", flush=True)

    hidden_states = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    ).hidden_states
    torch.npu.synchronize()
    print(
        f"[gemma_ab] encoder OK hidden_states={len(hidden_states)} shape={tuple(hidden_states[-1].shape)}",
        flush=True,
    )


def _native_encode(text_encoder, tokenizer, prompt: str, max_sequence_length: int, device: torch.device):
    text_inputs = tokenizer(
        [prompt.strip()],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    hidden_states = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    ).hidden_states
    embeds = torch.stack(hidden_states, dim=-1).flatten(2, 3).to(dtype=text_encoder.dtype)
    torch.npu.synchronize()
    return embeds, input_ids, attention_mask


def test_omninft_adapter_gemma_encode_with_per_request_attention_mask_on_npu() -> None:
    tokenizer, text_encoder = _load_gemma(_model_path())
    device = torch.device("npu")
    max_sequence_length = 1024
    text_inputs = tokenizer(
        [_PROMPT.strip()],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)

    class RecordingTextEncoder:
        def __init__(self, model) -> None:
            self.model = model
            self.dtype = model.dtype
            self.attention_mask = object()

        def __call__(self, **kwargs):
            self.attention_mask = kwargs.get("attention_mask")
            return self.model(**kwargs)

    recording_encoder = RecordingTextEncoder(text_encoder)
    pipeline = object.__new__(LTX23OmniNFTPipeline)
    pipeline.__dict__["device"] = device
    pipeline.tokenizer = tokenizer
    pipeline.text_encoder = recording_encoder
    pipeline.tokenizer_max_length = max_sequence_length

    prompt_embeds, returned_mask = pipeline._encode_token_ids(
        input_ids,
        attention_mask,
        max_sequence_length,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(recording_encoder.attention_mask, attention_mask)
    assert prompt_embeds.shape[:2] == input_ids.shape
    torch.testing.assert_close(returned_mask, attention_mask)


def test_native_vllm_omni_gemma_encode_matches_mixin_on_npu() -> None:
    model_path = _model_path()
    device = torch.device("npu")
    tokenizer, text_encoder = _load_gemma(model_path)
    attn = _attn_impl(text_encoder)
    print(f"gemma attn_implementation={attn}")
    max_sequence_length = 1024

    native_error = None
    native_embeds = native_ids = native_mask = None
    try:
        native_embeds, native_ids, native_mask = _native_encode(
            text_encoder, tokenizer, _PROMPT, max_sequence_length, device
        )
        print(f"native encode ok shape={tuple(native_embeds.shape)} ids={tuple(native_ids.shape)}")
    except Exception as exc:
        native_error = exc
        print(f"native encode failed: {type(exc).__name__}: {exc}")

    mixin = object.__new__(LTXTokenIdPromptMixin)
    mixin.device = device
    mixin.tokenizer = tokenizer
    mixin.text_encoder = text_encoder
    mixin.tokenizer_max_length = max_sequence_length

    mixin_error = None
    mixin_embeds = None
    try:
        if native_ids is None:
            text_inputs = tokenizer(
                [_PROMPT.strip()],
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            token_ids = text_inputs.input_ids[0].tolist()
            attention_mask = text_inputs.attention_mask[0]
        else:
            token_ids = native_ids[0].tolist()
            attention_mask = native_mask[0]
        mixin_embeds, mixin_mask = mixin._encode_token_ids(token_ids, attention_mask, max_sequence_length)
        torch.npu.synchronize()
        print(f"mixin encode ok shape={tuple(mixin_embeds.shape)}")
    except Exception as exc:
        mixin_error = exc
        print(f"mixin encode failed: {type(exc).__name__}: {exc}")

    if native_error is not None:
        pytest.fail(
            "Native vLLM-Omni Gemma encode failed on this NPU stack "
            f"(attn={attn}): {type(native_error).__name__}: {native_error}"
        )
    if mixin_error is not None:
        pytest.fail(
            "Mixin encode failed while native encode succeeded "
            f"(attn={attn}): {type(mixin_error).__name__}: {mixin_error}"
        )
    assert native_embeds is not None and mixin_embeds is not None
    torch.testing.assert_close(mixin_embeds, native_embeds, atol=0.0, rtol=0.0)


def test_native_gemma_encode_batch_sizes_on_one_npu() -> None:
    """Localize whether packed CFG batch [2]/[4, 1024] fails without TP."""
    tokenizer, text_encoder = _load_gemma(_model_path())
    device = torch.device("npu")
    max_sequence_length = 1024
    results: dict[int, str] = {}
    torch.npu.synchronize()
    print(f"after load allocated={torch.npu.memory_allocated() / 1024**3:.2f}GiB")
    for batch_size in (1, 2, 4):
        prompts = [_PROMPT] * batch_size
        text_inputs = tokenizer(
            [prompt.strip() for prompt in prompts],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        try:
            hidden_states = text_encoder(
                input_ids=text_inputs.input_ids.to(device),
                attention_mask=text_inputs.attention_mask.to(device),
                output_hidden_states=True,
            ).hidden_states
            torch.npu.synchronize()
            allocated = torch.npu.memory_allocated() / 1024**3
            results[batch_size] = (
                f"ok hidden={len(hidden_states)}x{tuple(hidden_states[0].shape)} allocated={allocated:.2f}GiB"
            )
            print(f"batch={batch_size} {results[batch_size]}")
            del hidden_states
        except Exception as exc:
            results[batch_size] = f"{type(exc).__name__}: {exc}"
            print(f"batch={batch_size} failed: {results[batch_size]}")
        torch.npu.empty_cache()
        torch.npu.synchronize()
    failed = {batch: msg for batch, msg in results.items() if not msg.startswith("ok")}
    if failed:
        passed = {batch: message for batch, message in results.items() if batch not in failed}
        pytest.fail(f"Gemma encode failed on 1 NPU without TP: {failed}; passed={passed}")
