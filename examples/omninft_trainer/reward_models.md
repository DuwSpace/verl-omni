# OmniNFT Reward 模型下载与校验

本文记录 OmniNFT 五个训练 Reward 的模型来源、固定 revision、下载命令和本地路径。模型清单参考
[`zghhui/OmniNFT`](https://github.com/zghhui/OmniNFT/tree/fb9237f6e74edf0d0f2a683f4d975b79fde588fe/flow_grpo)，
聚合仓固定为
[`zghhui/OmniNFT-Reward-Series@9e30061`](https://huggingface.co/zghhui/OmniNFT-Reward-Series/tree/9e30061a1392d03bafdcf717e80a385ddf411b4d)。

核验日期：2026-08-26。

## 路径约定

宿主机模型根目录：

```bash
REWARD_ROOT=/home/model-cache/huggingface/hub/omnift-rewards
```

调试容器将 `/home/model-cache/huggingface/hub` 只读挂载为 `/hub`，所以容器内模型根目录为：

```text
/hub/omnift-rewards
```

DeSync 还需要固定 revision 的 OmniNFT Synchformer 源码；宿主机与容器内路径分别为
`$REWARD_ROOT/OmniNFT-reference` 和 `/hub/omnift-rewards/OmniNFT-reference`。运行时只读取这个本地
checkout，不访问网络。

## 五个训练 Reward

| Reward | 下载来源与 revision | 宿主机下载路径 | 容器内运行路径 | 参考配置权重 |
| --- | --- | --- | --- | --- |
| VideoAlign | `KlingTeam/VideoReward@4f26600130683e6f1de9f5d463887f28e8ef995c` | `$REWARD_ROOT/VideoReward/` | `/hub/omnift-rewards/VideoReward` | `1.0` |
| HPSv3 | `MizzenAI/HPSv3@4f81e3e09edd82fe3c5f636444c721b592a735ca` | `$REWARD_ROOT/HPSv3/HPSv3.safetensors` | `/hub/omnift-rewards/HPSv3/HPSv3.safetensors` | `1.5` |
| AudioBox Aesthetics | `facebook/audiobox-aesthetics@9b1dd8e5df9af7216e836a98974fe3b82c56ded6` | `$REWARD_ROOT/audiobox-aesthetics/` | `/hub/omnift-rewards/audiobox-aesthetics` | `0.5` |
| LAION-CLAP HTSAT unfused | `laion/clap-htsat-unfused@8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a` | `$REWARD_ROOT/checkpoints/clap-htsat-unfused/` | `/hub/omnift-rewards/checkpoints/clap-htsat-unfused` | `1.0` |
| DeSync / Synchformer | `zghhui/OmniNFT-Reward-Series@9e30061a1392d03bafdcf717e80a385ddf411b4d` | `$REWARD_ROOT/synchformer/synchformer_state_dict.pth` | `/hub/omnift-rewards/synchformer/synchformer_state_dict.pth` | `1.0` |

VideoAlign 和 HPSv3 在创建模型与 processor 时还需要对应的 Qwen2-VL 基座：

| 使用方 | 下载来源与 revision | 宿主机路径（相对 `$REWARD_ROOT`） | 容器内路径 |
| --- | --- | --- | --- |
| VideoAlign | `Qwen/Qwen2-VL-2B-Instruct@895c3a49bc3fa70a340399125c650a463535e71c` | `Qwen2-VL-2B-Instruct/` | `/hub/omnift-rewards/Qwen2-VL-2B-Instruct` |
| HPSv3 | `Qwen/Qwen2-VL-7B-Instruct@eed13092ef92e448dd6875b2a00151bd3f7db0ac` | `Qwen2-VL-7B-Instruct/` | `/hub/omnift-rewards/Qwen2-VL-7B-Instruct` |

## 固定版本下载命令

以下命令构造五 Reward 的目标模型目录。默认访问 Hugging Face 官方端点；需要镜像时，可在每条命令前设置经过验证的
`HF_ENDPOINT`。

```bash
REWARD_ROOT=/home/model-cache/huggingface/hub/omnift-rewards
mkdir -p "$REWARD_ROOT"

hf download MizzenAI/HPSv3 \
  --revision 4f81e3e09edd82fe3c5f636444c721b592a735ca \
  --local-dir "$REWARD_ROOT/HPSv3"

hf download KlingTeam/VideoReward \
  --revision 4f26600130683e6f1de9f5d463887f28e8ef995c \
  --local-dir "$REWARD_ROOT/VideoReward"

hf download facebook/audiobox-aesthetics \
  --revision 9b1dd8e5df9af7216e836a98974fe3b82c56ded6 \
  --local-dir "$REWARD_ROOT/audiobox-aesthetics"

hf download laion/clap-htsat-unfused \
  --revision 8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a \
  --local-dir "$REWARD_ROOT/checkpoints/clap-htsat-unfused"

hf download zghhui/OmniNFT-Reward-Series \
  synchformer/synchformer_state_dict.pth \
  --revision 9e30061a1392d03bafdcf717e80a385ddf411b4d \
  --local-dir "$REWARD_ROOT"

hf download Qwen/Qwen2-VL-2B-Instruct \
  --revision 895c3a49bc3fa70a340399125c650a463535e71c \
  --local-dir "$REWARD_ROOT/Qwen2-VL-2B-Instruct"

hf download Qwen/Qwen2-VL-7B-Instruct \
  --revision eed13092ef92e448dd6875b2a00151bd3f7db0ac \
  --local-dir "$REWARD_ROOT/Qwen2-VL-7B-Instruct"

git clone https://github.com/zghhui/OmniNFT.git "$REWARD_ROOT/OmniNFT-reference"
git -C "$REWARD_ROOT/OmniNFT-reference" checkout --detach \
  fb9237f6e74edf0d0f2a683f4d975b79fde588fe
```

`KwaiVGI/VideoReward` 当前会解析到 `KlingTeam/VideoReward`；这里记录解析后的仓库 ID 和固定 revision。

## 已核验的核心文件

| 文件 | 字节数 | SHA-256 | 与 Reward-Series 对比 |
| --- | ---: | --- | --- |
| `HPSv3/HPSv3.safetensors` | 16,584,387,300 | `a13d7ff5a07b7ffa0f7824e60d62e6ae144541ceefd5224b4c08fda7ab39f353` | 一致 |
| `VideoReward/checkpoint-11352/model.pth` | 5,031,072,529 | `48375908e6112de9f0248402db156a23b480709a6960b091c598c6f4c88d21b9` | 一致 |
| `VideoReward/checkpoint-11352/tokenizer/tokenizer.json` | 11,420,941 | `a05e80818b958b49531b8e1f758c4ae65f090fdec80e6d8408963f70ae3f737d` | 一致 |
| `audiobox-aesthetics/checkpoint.pt` | 415,520,834 | `a4931a7a01c3e6733352e9d85371835f03bf9135f8b31e1583c23538811d4a32` | 一致 |
| `audiobox-aesthetics/model.safetensors` | 415,472,992 | `a5a3c2412649cc2384ec525ffd5180ce6c4778f43bed6108e0a1303de04d014e` | 一致 |
| `synchformer/synchformer_state_dict.pth` | 950,058,171 | `8aff082f2df5c3bc52759db0c865c7ee772ae6400b860d1b7e90413f2defb67c` | 一致 |
| `checkpoints/clap-htsat-unfused/pytorch_model.bin` | 614,525,833 | `1cd3c601bc4afe0fa87be3de4c13dd2cfadd249fac1e29acf74a9b296c3219bb` | Reward-Series 未提供；共享缓存已校验 |

复核本地核心文件：

```bash
REWARD_ROOT=/home/model-cache/huggingface/hub/omnift-rewards
sha256sum \
  "$REWARD_ROOT/HPSv3/HPSv3.safetensors" \
  "$REWARD_ROOT/VideoReward/checkpoint-11352/model.pth" \
  "$REWARD_ROOT/VideoReward/checkpoint-11352/tokenizer/tokenizer.json" \
  "$REWARD_ROOT/audiobox-aesthetics/checkpoint.pt" \
  "$REWARD_ROOT/audiobox-aesthetics/model.safetensors" \
  "$REWARD_ROOT/checkpoints/clap-htsat-unfused/pytorch_model.bin" \
  "$REWARD_ROOT/synchformer/synchformer_state_dict.pth"
```

## 已知差异与使用约束

### CLAP

OmniNFT 参考实现的默认路径是 `checkpoints/clap-htsat-unfused`，并通过
`transformers.ClapModel.from_pretrained()` 加载。因此本项目固定使用上表中的 `laion/clap-htsat-unfused`。
当前机器只保留共享模型缓存中的固定 revision，不在代码仓内复制模型文件。

Reward-Series 的 `CLAP/AudioCLIP-Full-Training.pt` 是 537,302,068 字节，SHA-256 为
`2441d35b353352c8b1bbfb8f7c687f46314c3d2909e940eaf763b8c17f632c44`；它是另一种 AudioCLIP checkpoint，
同样不能直接替代 `clap-htsat-unfused/` 目录。

### VideoReward 配置

Reward-Series 的 `VideoReward/model_config.json` 把 `model_name_or_path` 写成作者机器上的绝对路径；当前本地
`KlingTeam/VideoReward` 配置使用可移植的 `Qwen/Qwen2-VL-2B-Instruct`。核心 `model.pth` 权重完全一致，运行时应
继续使用可移植配置，并确保上表中的 2B 基座可访问。

### DeSync / Synchformer

安装 DeSync 独立依赖使用 `uv pip install -e ".[desync]"`。Native scorer 的 `model_path` 指向
`/hub/omnift-rewards/synchformer/synchformer_state_dict.pth`，`source_root` 指向
`/hub/omnift-rewards/OmniNFT-reference`；它会在加载前校验 checkout 的 HEAD 必须精确等于
`fb9237f6e74edf0d0f2a683f4d975b79fde588fe`，并用 strict state-dict 加载。不要把源码路径替换为可变分支，
也不要在评分进程中执行 clone 或下载。

### Reward-Series 的额外目录

聚合仓还包含 `ImageBind/` 和 `LatentSync_pyav/`。它们不是当前 OmniNFT 五个训练 Reward 的组成部分，不应把
“完整下载聚合仓”与“五 Reward 运行依赖已齐全”混为一谈。
