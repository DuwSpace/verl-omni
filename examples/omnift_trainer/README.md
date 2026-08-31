# OmniNFT training

This directory contains the production LTX-2.3 OmniNFT recipe.

## Prepare data

Convert the native OmniNFT VGGSound JSONL metadata to the standard RLHF parquet schema:

```bash
python3 examples/omnift_trainer/ltx2/prepare_data.py \
  --train_file /path/to/train_metadata_20k.jsonl \
  --val_file /path/to/test_metadata.jsonl \
  --output_dir ./data/omninft/vggsound/verl_omni
```

The converter writes `train.parquet` and `test.parquet`. Each row keeps the joint generation prompt, stable prompt-group
`uid`, separate video/audio reward prompts, and native source metadata. The standard `RLHFDataset` reads these files; no
runtime custom dataset or collator is used.

## Launch

```bash
bash examples/omnift_trainer/ltx2/run_ltx2_3_omninft_lora_npu.sh
```

The launcher uses online direct-preference training with the `default` and `old` policy adapters, five Native Reward
components, per-component group normalization, and explicit video/audio routing. Override `DATA_DIR`, `MODEL_PATH`,
`REWARD_ROOT`, or the individual train/validation paths for the local environment.

## Routing

The reward order is `video_align`, `hpsv3`, `audiobox`, `clap`, `desync`. Video receives weights `1.0`, `1.5`, and `1.0`
from VideoAlign, HPSv3, and DeSync. Audio receives weights `0.5`, `1.0`, and `1.0` from AudioBox, CLAP, and DeSync.

## Reward checkpoints

Pinned downloads, host/container paths, and SHA-256 checks are in
[reward_models.md](reward_models.md).

## Architecture

Rollout/reward call chains and the fields crossing the rollout → reward →
trainer-adapter boundary are in [architecture.md](architecture.md).

