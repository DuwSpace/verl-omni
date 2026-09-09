# OmniNFT 运行说明

本说明按顺序介绍如何准备镜像、下载模型、启动 Docker、生成数据并运行
LTX-2.3 OmniNFT 训练。当前 production launcher 使用 Ascend NPU。

所有命令默认从仓库根目录执行。

## 1. 准备镜像

如果已有迁移过来的 Docker image file：

```bash
IMAGE_FILE=/path/to/verl-omni-npu-a2-omnift.tar \
IMAGE_NAME=verl-omni:npu-a2-omnift \
bash examples/omnift_trainer/setup_environment.sh
```

`IMAGE_FILE` 支持 `.tar`、`.tar.gz` 和 `.tar.zst`。`IMAGE_NAME` 必须与 image file
中保存的 tag 一致。

如果没有 image file，直接用当前代码构建：

```bash
IMAGE_NAME=verl-omni:npu-a2-omnift \
bash examples/omnift_trainer/setup_environment.sh
```

该脚本使用 [`docker/Dockerfile.a2.npu`](../../docker/Dockerfile.a2.npu)。镜像内已安装
`audiobox_aesthetics==0.0.4` 和 `timm==1.0.27`，并包含一份构建时生成的 1+1
prompt parquet，路径为 `/workspace/data/omninft/s5_2_data`。

## 2. 下载模型

宿主机先安装 Hugging Face CLI，然后执行下载脚本：

```bash
python3 -m pip install -U huggingface_hub

MODEL_ROOT=/path/to/huggingface/hub \
bash examples/omnift_trainer/download_models.sh
```

脚本下载并校验：

- LTX-2.3 Diffusers 基础模型；
- VideoReward、HPSv3、AudioBox、CLAP、Synchformer；
- Qwen2-VL-2B 和 Qwen2-VL-7B；
- OmniNFT reward 推理源码仓（本地目录名为 `OmniNFT-reference`）。

下载完成后的关键目录为：

```text
/path/to/huggingface/hub/
├── models--diffusers--LTX-2.3-Diffusers/
└── omnift-rewards/
```

完整 revision 和 checksum 见 [reward_models.md](reward_models.md)。

### 2.1 OmniNFT reward 推理代码仓

`download_models.sh` 会执行等价于下面的操作：

```bash
git clone https://github.com/zghhui/OmniNFT.git \
  "$MODEL_ROOT/omnift-rewards/OmniNFT-reference"
git -C "$MODEL_ROOT/omnift-rewards/OmniNFT-reference" checkout --detach \
  fb9237f6e74edf0d0f2a683f4d975b79fde588fe
```

这里的 `OmniNFT-reference` 就是 reward 推理代码仓，不是 Docker 镜像。关键内容为：

```text
OmniNFT-reference/
├── flow_grpo/rewards.py                  # 五类 reward 的官方调用逻辑
├── flow_grpo/server/                     # HPSv3、VideoAlign HTTP 推理服务
├── flow_grpo/HPSv3/                      # HPSv3 Python 包和配置
├── flow_grpo/videoalign/                 # VideoAlign 推理实现
└── flow_grpo/audio_video_align/           # DeSync 推理实现
```

校验代码仓来源和版本：

```bash
git -C "$MODEL_ROOT/omnift-rewards/OmniNFT-reference" remote get-url origin
git -C "$MODEL_ROOT/omnift-rewards/OmniNFT-reference" rev-parse HEAD
```

期望分别得到 `https://github.com/zghhui/OmniNFT.git` 和
`fb9237f6e74edf0d0f2a683f4d975b79fde588fe`。

## 3. 启动 Docker

设置实际路径和设备数量：

```bash
REPO=/path/to/verl-omni
MODEL_ROOT=/path/to/huggingface/hub
IMAGE_NAME=verl-omni:npu-a2-omnift
NUM_GPUS=8
```

启动容器：

```bash
DEVICES=""
for i in $(seq 0 $((NUM_GPUS - 1))); do
  DEVICES="$DEVICES --device=/dev/davinci$i"
done

docker run -dit \
  --name verl-omni-omnift-train \
  --network host \
  --ipc host \
  $DEVICES \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$REPO:/repo" \
  -v "$MODEL_ROOT:/hub:ro" \
  -w /repo \
  -e PYTHONPATH=/repo \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  "$IMAGE_NAME" \
  bash
```

如果 `npu-smi` 的宿主机路径不同，请修改对应 mount。设备数量和 `/dev/davinci*`
列表也应按目标机器调整。

## 4. Docker 启动后的操作

进入容器：

```bash
docker exec -it verl-omni-omnift-train bash
cd /repo
export PYTHONPATH=/repo
```

确认代码、设备和两个额外 Reward 包：

```bash
python3 -c "import verl_omni; print(verl_omni.__file__)"
npu-smi info
python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.device_count())"
python3 -c "from audiobox_aesthetics.model.aes import AesMultiOutput; from timm.layers import trunc_normal_; print('reward dependencies OK')"
```

第一条命令应输出 `/repo/verl_omni/...`，设备数量应与启动容器时传入的设备一致。

确认模型挂载：

```bash
test -d /hub/models--diffusers--LTX-2.3-Diffusers/snapshots/8eee8edcf067e838b843f926ec4d4cc9b2be1aaf
test -f /hub/omnift-rewards/VideoReward/checkpoint-11352/model.pth
test -f /hub/omnift-rewards/HPSv3/HPSv3.safetensors
test -d /hub/omnift-rewards/audiobox-aesthetics
test -f /hub/omnift-rewards/checkpoints/clap-htsat-unfused/pytorch_model.bin
test -f /hub/omnift-rewards/synchformer/synchformer_state_dict.pth
test -d /hub/omnift-rewards/OmniNFT-reference
```

## 5. 生成训练数据

生成 1 条训练样本和 1 条验证样本：

```bash
bash examples/omnift_trainer/ltx2/prepare_s5_2_data_1.sh
```

输出为：

```text
/repo/outputs/s5_2_data/train.parquet
/repo/outputs/s5_2_data/test.parquet
```

检查：

```bash
python3 - <<'PY'
import pandas as pd

train = pd.read_parquet("/repo/outputs/s5_2_data/train.parquet")
test = pd.read_parquet("/repo/outputs/s5_2_data/test.parquet")
print("train:", len(train), "test:", len(test))
PY
```

期望输出为 `train: 1 test: 1`。

## 6. 训练脚本参数

训练入口为：

```text
examples/omnift_trainer/ltx2/run_ltx2_3_omninft_lora_npu.sh
```

### 6.1 环境变量

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `DATA_DIR` | `<repo>/outputs/s5_2_data` | 包含训练和验证 parquet 的目录 |
| `TRAIN_FILE` | `$DATA_DIR/train.parquet` | 训练 parquet，可单独覆盖 |
| `VAL_FILE` | `$DATA_DIR/test.parquet` | 验证 parquet，可单独覆盖 |
| `MODEL_PATH` | `/hub/models--diffusers--LTX-2.3-Diffusers/snapshots/8eee...` | LTX-2.3 Diffusers snapshot |
| `REWARD_ROOT` | `/hub/omnift-rewards` | 五个 Reward 及两个 Qwen2-VL 基座的根目录 |
| `DESYNC_SOURCE_ROOT` | `$REWARD_ROOT/OmniNFT-reference` | 固定 revision 的 Synchformer 源码目录 |
| `NUM_GPUS` | `8` | 当前节点参与训练的 NPU 数量 |
| `ROLLOUT_TP` | `8` | rollout tensor parallel size，必须能整除 `NUM_GPUS` |
| `REWARD_NUM_WORKERS` | `$NUM_GPUS` | Native Reward worker 数量，不能超过可用设备数 |
| `TOTAL_TRAINING_STEPS` | `100` | 总训练 step 数 |
| `MAX_NUM_SEQS` | `4` | vLLM-Omni 同时接纳的最大请求数；显存不足时可降为 2 或 1 |
| `REQUEST_BATCH_MAX_WAIT_MS` | `100` | request batching 最长等待时间，单位毫秒 |
| `REWARD_PARALLEL_GROUPS` | `audiobox_clap` | 并行执行 AudioBox 与 CLAP；设为空字符串可关闭 |
| `OUTPUT_DIR` | `<repo>/outputs/run_ltx2_3_omninft_lora_npu` | checkpoint、日志和生成样本的根目录 |
| `ROLLOUT_DATA_SAVE_FREQ` | `10` | 每隔多少 step 保存 rollout 样本 |
| `ROLLOUT_DATA_MAX_SAMPLES` | `null` | 每次保存的 rollout 样本上限；`null` 表示不限制 |
| `WANDB_MODE` | `online` | W&B 模式，可设为 `offline` |
| `OMNIFT_ROLLOUT_PROGRESS` | `1` | 是否输出 OmniNFT rollout 进度 |
| `ASCEND_HOME_PATH` | `/usr/local/Ascend/ascend-toolkit` | 容器内 Ascend toolkit 路径 |
| `NODE_RANK` | `0` | 日志文件名使用的节点 rank；当前 recipe 的 `nnodes=1` |

这些参数写在命令前即可覆盖，例如：

```bash
WANDB_MODE=offline \
TOTAL_TRAINING_STEPS=1 \
MAX_NUM_SEQS=2 \
OUTPUT_DIR=/repo/outputs/omninft_smoke \
bash examples/omnift_trainer/ltx2/run_ltx2_3_omninft_lora_npu.sh
```

`REWARD_PARALLEL_GROUPS` 只接受空字符串或 `audiobox_clap`：

```bash
# 关闭 Reward 并行组
REWARD_PARALLEL_GROUPS="" \
bash examples/omnift_trainer/ltx2/run_ltx2_3_omninft_lora_npu.sh
```

### 6.2 关键固定训练参数

| 配置 | 当前值 | 作用 |
| --- | ---: | --- |
| `data.train_batch_size` | `1` | 每个训练 batch 包含 1 个 prompt group |
| `actor_rollout_ref.rollout.n` | `8` | 每个 prompt 生成 8 个候选 |
| `algorithm.timestep_fraction` | `0.4` | 每次更新使用 40% diffusion timesteps |
| `algorithm.timestep_selection` | `top_sigma` | 选取最高噪声区间的训练 timesteps |
| `actor_rollout_ref.model.lora_rank` | `32` | LoRA rank |
| `actor_rollout_ref.model.lora_alpha` | `64` | LoRA alpha |
| `actor_rollout_ref.actor.optim.lr` | `3e-5` | Actor 学习率 |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | `1` | PPO mini-batch size |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | `1` | 每卡 micro-batch size |
| `actor_rollout_ref.actor.fsdp_config.model_dtype` | `bfloat16` | Actor 计算 dtype |
| `actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size` | `1` | 当前关闭 rollout sequence parallel |
| `actor_rollout_ref.rollout.pipeline.height/width` | `512 / 768` | 训练生成视频分辨率 |
| `actor_rollout_ref.rollout.pipeline.num_frames` | `121` | 训练生成帧数 |
| `actor_rollout_ref.rollout.pipeline.frame_rate` | `24.0` | 视频帧率 |
| `actor_rollout_ref.rollout.pipeline.num_inference_steps` | `20` | 训练 rollout 去噪步数 |
| `actor_rollout_ref.rollout.pipeline.video_cfg_scale` | `1.5` | rollout 视频 CFG scale |
| `actor_rollout_ref.rollout.pipeline.audio_cfg_scale` | `3.0` | rollout 音频 CFG scale |
| `actor_rollout_ref.rollout.pipeline.video_modality_scale` | `1.0` | 视频分支的跨模态 guidance 强度；`1.0` 保持官方 OmniNFT recipe 的原始幅度 |
| `actor_rollout_ref.rollout.pipeline.audio_modality_scale` | `1.0` | 音频分支的跨模态 guidance 强度；`1.0` 保持官方 OmniNFT recipe 的原始幅度 |
| `actor_rollout_ref.rollout.pipeline.video_rescale_scale` | `0.0` | 视频 guidance rescale 系数；`0.0` 表示不做 rescale |
| `actor_rollout_ref.rollout.pipeline.audio_rescale_scale` | `0.0` | 音频 guidance rescale 系数；`0.0` 表示不做 rescale |
| `trainer.save_freq` | `50` | checkpoint 保存间隔 |
| `trainer.test_freq` | `10` | 验证间隔 |

五个 Reward 的顺序固定为：

```text
video_align, hpsv3, audiobox, clap, desync
```

当前 micro-batch 分别为 `2, 8, 8, 8, 2`。视频路由权重为 VideoAlign `1.0`、
HPSv3 `1.5`、DeSync `1.0`；音频路由权重为 AudioBox `0.5`、CLAP `1.0`、
DeSync `1.0`。

验证阶段另有同名的四个参数，当前同样设置为 `1.0, 1.0, 0.0, 0.0`：

```text
actor_rollout_ref.rollout.val_kwargs.pipeline.video_modality_scale
actor_rollout_ref.rollout.val_kwargs.pipeline.audio_modality_scale
actor_rollout_ref.rollout.val_kwargs.pipeline.video_rescale_scale
actor_rollout_ref.rollout.val_kwargs.pipeline.audio_rescale_scale
```

`pipeline.*` 控制训练 rollout，`val_kwargs.pipeline.*` 控制验证采样，修改一组不会自动
修改另一组。为了保证两阶段 guidance 行为一致，需要分别覆盖。

### 6.3 OmniNFT reward 推理与对齐验证

当前 Ascend launcher 不启动独立 HTTP reward server。五个 reward 通过
`pkg://verl_omni.utils.reward_score.*_native` 在 NPU worker 内直接推理：

```text
videoalign_native, hpsv3_native, audiobox_native, clap_native, desync_native
```

其中 DeSync 会通过 `DESYNC_SOURCE_ROOT` 使用挂载到
`/hub/omnift-rewards/OmniNFT-reference` 的官方源码。要用同一段 rollout 数据同时执行
verl-omni native reward 和 OmniNFT 官方 reward 推理，并检查两者分数是否对齐，运行：

```bash
OMNIFT_PARITY_MEDIA=/repo/outputs/omnift_rollout_prod/0.mp4 \
OMNIFT_PARITY_REPLAY=/repo/outputs/omnift_rollout_prod/replay_g8_81f.pkl \
OMNIFT_PARITY_REFERENCE=/hub/omnift-rewards/OmniNFT-reference \
CLAP_MODEL_PATH=/hub/omnift-rewards/checkpoints/clap-htsat-unfused \
AUDIOBOX_MODEL_PATH=/hub/omnift-rewards/audiobox-aesthetics \
HPSV3_MODEL_PATH=/hub/omnift-rewards/HPSv3/HPSv3.safetensors \
HPSV3_BASE_MODEL_PATH=/hub/omnift-rewards/Qwen2-VL-7B-Instruct \
VIDEOALIGN_MODEL_PATH=/hub/omnift-rewards/VideoReward/checkpoint-11352/model.pth \
VIDEOALIGN_BASE_MODEL_PATH=/hub/omnift-rewards/Qwen2-VL-2B-Instruct \
DESYNC_MODEL_PATH=/hub/omnift-rewards/synchformer/synchformer_state_dict.pth \
pytest -sv tests/reward_loop/test_omninft_native_parity_on_npu.py
```

`OMNIFT_PARITY_MEDIA` 和 `OMNIFT_PARITY_REPLAY` 必须来自同一次 G=8、121 帧、24 FPS
rollout。上面的路径是当前机器已有的对齐样本；迁移到新机器时应替换为实际路径。

官方代码仓中的下面两个脚本是 CUDA HTTP server 启动器：

```text
flow_grpo/server/run_remote_hpsv3.sh
flow_grpo/server/run_remote_videoalign.sh
```

它们内部检查 `torch.cuda` 并使用 `cuda:*`，不能直接用于当前 Ascend NPU recipe；当前
训练也没有配置 HTTP reward client，因此不需要在启动训练前运行它们。

脚本最后会追加命令行参数 `"$@"`，因此可以临时覆盖其他 Hydra 配置：

```bash
bash examples/omnift_trainer/ltx2/run_ltx2_3_omninft_lora_npu.sh \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  trainer.save_freq=10
```

环境变量能控制的选项优先使用环境变量，避免重复写 Hydra 参数。

## 7. 启动训练

如需在线记录 W&B，先在容器中设置：

```bash
export WANDB_API_KEY=your_key
export WANDB_MODE=online
```

不使用 W&B 时：

```bash
export WANDB_MODE=offline
```

运行训练：

```bash
DATA_DIR=/repo/outputs/s5_2_data \
MODEL_PATH=/hub/models--diffusers--LTX-2.3-Diffusers/snapshots/8eee8edcf067e838b843f926ec4d4cc9b2be1aaf \
REWARD_ROOT=/hub/omnift-rewards \
OUTPUT_DIR=/repo/outputs/run_ltx2_3_omninft_lora_npu \
NUM_GPUS=8 \
ROLLOUT_TP=8 \
bash examples/omnift_trainer/ltx2/run_ltx2_3_omninft_lora_npu.sh
```

`NUM_GPUS` 和 `ROLLOUT_TP` 应根据实际设备与显存调整；当前已验证配置为 8/8。

## 8. 查看输出

主输出目录：

```text
/repo/outputs/run_ltx2_3_omninft_lora_npu/
├── checkpoints/
└── logs/<timestamp>/
    ├── 0.log
    ├── rollout_videos/
    └── validation_videos/
```

查看最新日志：

```bash
LOG_FILE=$(find /repo/outputs/run_ltx2_3_omninft_lora_npu/logs -name 0.log | sort | tail -n 1)
tail -f "$LOG_FILE"
```

训练结束后离开容器使用 `exit`。需要停止容器时在宿主机运行：

```bash
docker stop verl-omni-omnift-train
```
