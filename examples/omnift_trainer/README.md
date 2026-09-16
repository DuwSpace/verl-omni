# OmniNFT training

This directory contains the production LTX-2.3 OmniNFT recipe.

Use the repository's standard NPU environment, then follow the preparation and
launch commands below. The launcher accepts environment variables for local
model, reward-model, data, and output paths.

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

Each reward uses the upstream named-native-model deployment and owns an
independent worker group. The default eight-device placement is VideoAlign
`[0,1]`, HPSv3 `[2,3]`, AudioBox `[4]`, CLAP `[5]`, and DeSync `[6,7]`; override
the corresponding `*_DEVICES` environment variables after measuring each
model. Every group scores the complete generated batch, split evenly across its
replicas without padding. The recipe sets `offload=True` and
`offload_mode=cpu`: each worker loads its model once on CPU, moves it to the
assigned NPU for scoring, and moves it back to CPU on sleep. Final worker
shutdown closes the retained model. Set `REWARD_OFFLOAD_MODE=recreate` to use
the compatible load-on-wake/close-on-sleep behavior. The recipe does not
require cross-model colocation or a shared replica.

The checked-in recipe intentionally preserves the source experiment settings:
training rollouts use 20 denoising steps at 256x384 with video/audio CFG 1.5/3.0,
while the inexpensive pre-training validation probe uses one step at 512x768
with CFG 3.0/7.0, modality scale 3.0, and rescale 0.7. The one-step probe checks
the validation and reward pipeline; it is not a quality evaluation. Use a
separately reviewed validation override for quality comparisons.

The native vLLM-Omni sampler consumes `video_cfg_scale`, `audio_cfg_scale`,
`video_modality_scale`, `audio_modality_scale`, `video_rescale_scale`, and
`audio_rescale_scale`. Actor replay applies only video/audio CFG; the recipe
sets both training-side CFG values to 1 so current/old/reference policies are
compared without guidance. Modality and rescale settings are sampler-only and
must not be interpreted as actor-training controls.

## Routing

Routing is keyed by reward name; matrix column order is not a user-facing
contract. Video receives weights `1.0`, `1.5`, and `1.0` from VideoAlign,
HPSv3, and DeSync. Audio receives weights `0.5`, `1.0`, and `1.0` from
AudioBox, CLAP, and DeSync.

## Prepare rewards

```bash
bash examples/omnift_trainer/download_reward_models.sh
```

The script installs the reward-specific Python packages (including the Qwen visual-input helper used by this
multimodal stack), downloads all pinned checkpoints and Qwen2-VL base models, checks out the pinned OmniNFT
Synchformer source, and verifies the core files. Assets are written to
`outputs/omnift-rewards` by default; override `REWARD_ROOT` when needed.
