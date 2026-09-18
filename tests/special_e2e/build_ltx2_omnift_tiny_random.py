#!/usr/bin/env python
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
"""Build a self-contained tiny random LTX-2.3 checkpoint for OmniNFT smoke tests.

The checkpoint uses the real Diffusers LTX-2 components with small dimensions.
It preserves the published LTX-2.3 one-stage component layout, video/audio latent
geometry, Gemma-3 text encoder contract, and BWE vocoder marker used by
vLLM-Omni's version detector. It is only suitable for execution tests, not media
quality evaluation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    FlowMatchEulerDiscreteScheduler,
    LTX2Pipeline,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2 import LTX2TextConnectors
from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    Gemma3Config,
    Gemma3ForConditionalGeneration,
    Gemma3TextConfig,
    PreTrainedTokenizerFast,
    SiglipVisionConfig,
)

DEFAULT_OUTPUT_DIR = Path.home() / "models" / "tiny-random" / "LTX-2.3-OmniNFT"
_SEED = 42
_MANIFEST_FILE = ".omnift-tiny-checkpoint.json"
_MANIFEST_SCHEMA_VERSION = 1
_GENERATOR_ID = "verl-omni.ltx2-omnift-tiny"
_BUILD_SPEC = {
    "version": 1,
    "seed": _SEED,
    "serialization": "safetensors",
    "pipeline_class": "LTX2Pipeline",
}
_REQUIRED_FILES = (
    "model_index.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "connectors/config.json",
    "connectors/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "audio_vae/config.json",
    "audio_vae/diffusion_pytorch_model.safetensors",
    "vocoder/config.json",
    "vocoder/diffusion_pytorch_model.safetensors",
    "scheduler/scheduler_config.json",
)


def _manifest_payload() -> dict:
    return {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "generator": _GENERATOR_ID,
        "build_spec": _BUILD_SPEC,
    }


def _read_manifest(path: Path) -> dict | None:
    try:
        value = json.loads((path / _MANIFEST_FILE).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _is_owned(path: Path) -> bool:
    manifest = _read_manifest(path)
    return manifest is not None and manifest.get("generator") == _GENERATOR_ID


def _is_complete(path: Path) -> bool:
    if _read_manifest(path) != _manifest_payload():
        return False
    return all((path / relative).is_file() and (path / relative).stat().st_size > 0 for relative in _REQUIRED_FILES)


def _write_manifest(path: Path) -> None:
    (path / _MANIFEST_FILE).write_text(json.dumps(_manifest_payload(), indent=2, sort_keys=True) + "\n")


def _target_state(path: Path) -> str:
    if not path.exists() and not path.is_symlink():
        return "missing"
    if path.is_symlink() or not path.is_dir():
        return "unmanaged"
    if not any(path.iterdir()):
        return "empty"
    return "owned" if _is_owned(path) else "unmanaged"


def _build_tokenizer() -> PreTrainedTokenizerFast:
    tokens = (
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>",
        "<start_of_turn>",
        "<end_of_turn>",
        "<image>",
        "user",
        "model",
        "a",
        "robot",
        "dancing",
        "with",
        "sound",
        ".",
    )
    tokenizer_impl = Tokenizer(models.WordLevel({token: index for index, token in enumerate(tokens)}, "<unk>"))
    tokenizer_impl.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    tokenizer.model_max_length = 64
    return tokenizer


def _build_text_encoder(vocab_size: int) -> Gemma3ForConditionalGeneration:
    text_config = Gemma3TextConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        sliding_window=64,
        layer_types=["full_attention", "full_attention"],
        query_pre_attn_scalar=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
    )
    vision_config = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_channels=3,
        image_size=16,
        patch_size=8,
    )
    config = Gemma3Config(
        text_config=text_config,
        vision_config=vision_config,
        mm_tokens_per_image=4,
        boi_token_index=6,
        eoi_token_index=6,
        image_token_index=6,
    )
    return Gemma3ForConditionalGeneration(config)


def _build_pipeline() -> LTX2Pipeline:
    """Build seeded random components with compatible audio/video replay geometry.

    Packed video and audio latents both have width 16 for sequence concatenation;
    the audio VAE's eight latent channels pack to 16, matching its base_channels
    normalization width. Connector outputs match video/audio cross-attention
    widths 16/8, and text projection consumes three 32-wide Gemma hidden states.
    Retain the LTX spatial/temporal scales and BWE vocoder version marker while
    reducing depth and hidden widths. Sets the process-wide PyTorch RNG seed.
    """
    torch.manual_seed(_SEED)
    tokenizer = _build_tokenizer()
    text_encoder = _build_text_encoder(len(tokenizer))
    transformer = LTX2VideoTransformer3DModel(
        in_channels=16,
        out_channels=16,
        patch_size=1,
        patch_size_t=1,
        num_attention_heads=2,
        attention_head_dim=8,
        cross_attention_dim=16,
        vae_scale_factors=(8, 32, 32),
        pos_embed_max_pos=20,
        base_height=32,
        base_width=32,
        audio_in_channels=16,
        audio_out_channels=16,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_cross_attention_dim=8,
        audio_scale_factor=4,
        audio_pos_embed_max_pos=20,
        audio_sampling_rate=16000,
        audio_hop_length=160,
        num_layers=2,
        qk_norm="rms_norm_across_heads",
        caption_channels=32,
        rope_double_precision=False,
        rope_type="split",
        use_prompt_embeddings=False,
        cross_attn_mod=True,
        audio_cross_attn_mod=True,
        norm_elementwise_affine=False,
    )
    connectors = LTX2TextConnectors(
        caption_channels=32,
        text_proj_in_factor=3,
        video_connector_num_attention_heads=2,
        video_connector_attention_head_dim=8,
        video_connector_num_layers=1,
        video_connector_num_learnable_registers=None,
        audio_connector_num_attention_heads=2,
        audio_connector_attention_head_dim=4,
        audio_connector_num_layers=1,
        audio_connector_num_learnable_registers=None,
        connector_rope_base_seq_len=32,
        rope_double_precision=False,
        rope_type="split",
        per_modality_projections=True,
        video_hidden_dim=16,
        audio_hidden_dim=8,
        proj_bias=True,
    )
    vae = AutoencoderKLLTX2Video(
        in_channels=3,
        out_channels=3,
        latent_channels=16,
        block_out_channels=(8, 16, 32, 32),
        decoder_block_out_channels=(8, 16, 16, 32),
        layers_per_block=(1, 1, 1, 1, 1),
        decoder_layers_per_block=(1, 1, 1, 1, 1),
        spatio_temporal_scaling=(True, True, True, True),
        decoder_spatio_temporal_scaling=(True, True, True, True),
        decoder_inject_noise=(False, False, False, False, False),
        downsample_type=("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
        upsample_type=("spatiotemporal", "spatiotemporal", "temporal", "spatial"),
        upsample_residual=(False, False, False, False),
        upsample_factor=(2, 2, 1, 2),
        timestep_conditioning=False,
        patch_size=4,
        patch_size_t=1,
        encoder_causal=True,
        decoder_causal=False,
        decoder_spatial_padding_mode="zeros",
        spatial_compression_ratio=32,
        temporal_compression_ratio=8,
    )
    vae.use_framewise_encoding = False
    vae.use_framewise_decoding = False
    audio_vae = AutoencoderKLLTX2Audio(
        base_channels=16,
        output_channels=2,
        ch_mult=(1,),
        num_res_blocks=1,
        attn_resolutions=None,
        in_channels=2,
        resolution=32,
        latent_channels=8,
        norm_type="pixel",
        causality_axis="height",
        dropout=0.0,
        mid_block_add_attention=False,
        sample_rate=16000,
        mel_hop_length=160,
        is_causal=True,
        mel_bins=8,
    )
    vocoder = LTX2VocoderWithBWE(
        in_channels=16,
        hidden_channels=64,
        out_channels=2,
        upsample_kernel_sizes=[11, 4, 4, 4, 4, 4],
        upsample_factors=[5, 2, 2, 2, 2, 2],
        resnet_kernel_sizes=[3],
        resnet_dilations=[[1, 3, 5]],
        antialias=False,
        bwe_in_channels=16,
        bwe_hidden_channels=32,
        bwe_out_channels=2,
        bwe_upsample_kernel_sizes=[12, 11, 4, 4, 4],
        bwe_upsample_factors=[6, 5, 2, 2, 2],
        bwe_resnet_kernel_sizes=[3],
        bwe_resnet_dilations=[[1, 3, 5]],
        bwe_antialias=False,
        filter_length=512,
        hop_length=80,
        window_length=512,
        num_mel_channels=8,
        input_sampling_rate=16000,
        output_sampling_rate=48000,
    )
    scheduler = FlowMatchEulerDiscreteScheduler(
        base_image_seq_len=1024,
        max_image_seq_len=4096,
        base_shift=0.95,
        max_shift=2.05,
        shift_terminal=0.1,
        use_dynamic_shifting=True,
    )
    return LTX2Pipeline(
        scheduler=scheduler,
        vae=vae,
        audio_vae=audio_vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        connectors=connectors,
        transformer=transformer,
        vocoder=vocoder,
        processor=None,
        prompt_enhancer=None,
    )


def ensure_tiny_ltx2_checkpoint(output_dir: str | Path, *, force: bool = False) -> Path:
    """Create or refresh a generator-owned tiny checkpoint and return its path.

    Args:
        output_dir: Destination, expanded to an absolute path. Missing/empty
            directories and caches carrying this generator's ownership marker
            are accepted; symlinks, files, and other nonempty directories are not.
        force: Rebuild an otherwise reusable owned cache. Does not authorize
            replacing an unmanaged target.

    Returns:
        The destination path. Without force, reuse requires an exact manifest
        match and nonempty required files; contents are not hash-verified.

    Notes:
        A sibling directory lock rejects concurrent builds. Build in staging
        before replacing the target; retain a backup during replacement and
        restore it if installation fails. Normal exception cleanup removes
        staging and the lock; abrupt process termination can leave them behind.
    """
    output_dir = Path(output_dir).expanduser().absolute()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = output_dir.with_name(f".{output_dir.name}.omnift-tiny.lock")
    try:
        lock_dir.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"Another tiny LTX-2.3 build is using {output_dir}: {lock_dir}") from exc

    staging_dir: Path | None = None
    backup_dir: Path | None = None
    try:
        target_state = _target_state(output_dir)
        if target_state == "unmanaged":
            raise RuntimeError(
                f"Refusing to replace unowned tiny-checkpoint target {output_dir}. "
                f"Choose an empty OMNIFT_TINY_MODEL_DIR or remove the directory yourself."
            )
        if target_state == "owned" and not force and _is_complete(output_dir):
            return output_dir

        staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.building-", dir=output_dir.parent))
        _build_pipeline().save_pretrained(staging_dir, safe_serialization=True)
        _write_manifest(staging_dir)
        if not _is_complete(staging_dir):
            raise RuntimeError(f"Tiny LTX-2.3 checkpoint is incomplete: {staging_dir}")

        if output_dir.exists():
            backup_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.previous-", dir=output_dir.parent))
            backup_dir.rmdir()
            output_dir.replace(backup_dir)
        try:
            staging_dir.replace(output_dir)
            staging_dir = None
        except BaseException:
            if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
                backup_dir.replace(output_dir)
                backup_dir = None
            raise
        if backup_dir is not None:
            shutil.rmtree(backup_dir)
            backup_dir = None
        return output_dir
    finally:
        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir)
        if backup_dir is not None and backup_dir.exists():
            if not output_dir.exists():
                backup_dir.replace(output_dir)
            else:
                shutil.rmtree(backup_dir)
        lock_dir.rmdir()


def main() -> None:
    """Build/reuse the requested checkpoint and report its path and on-disk size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild a generator-owned cache; never replace an unmarked non-empty target.",
    )
    args = parser.parse_args()
    output_dir = ensure_tiny_ltx2_checkpoint(args.output_dir, force=args.force)
    size_bytes = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
    print(f"Tiny LTX-2.3 OmniNFT checkpoint ready at {output_dir} ({size_bytes} bytes)")


if __name__ == "__main__":
    main()
