# OmniNFT training

This directory contains the production LTX-2.3 OmniNFT recipe.

For environment setup, model downloads, Docker startup, data preparation, and
the commands to run after entering the container, see [RUN.md](RUN.md).

## Prepare data

Convert the native OmniNFT VGGSound
[`train_metadata_20k.jsonl`](https://github.com/zghhui/OmniNFT/blob/master/dataset/vggsound/train_metadata_20k.jsonl)
and
[`test_metadata.jsonl`](https://github.com/zghhui/OmniNFT/blob/master/dataset/vggsound/test_metadata.jsonl)
to the standard RLHF parquet schema:

```bash
python3 examples/omnift_trainer/data_process/prepare_data.py
```

By default, the converter reads both files directly from the upstream repository and writes `train.parquet` and
`test.parquet` to `./data/omninft/vggsound/verl_omni`. Local JSONL paths can be supplied with `--train_file` and
`--val_file`. Each row keeps the joint generation prompt, stable prompt-group `uid`, separate video/audio reward
prompts, and native source metadata. The standard `RLHFDataset` reads these files; no runtime custom dataset or collator
is used.

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

## Prepare rewards

```bash
bash examples/omnift_trainer/download_reward_models.sh
```

The script installs the reward-specific Python packages, downloads all pinned checkpoints and Qwen2-VL base models,
checks out the pinned OmniNFT Synchformer source, and verifies the core files. Assets are written to
`outputs/omnift-rewards` by default; override `REWARD_ROOT` when needed.
