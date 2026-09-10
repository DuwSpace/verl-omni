# verl-omni Ascend A2 Docker 镜像构建问题与解决经验

Last updated: 09/03/2026

## 1. 背景与目标

本次工作的目标是在 8 张 Ascend 910B3（Atlas A2）服务器上构建一个可运行
`verl-omni`、`vLLM`、`vLLM-Ascend` 和 `vLLM-Omni` 的训练镜像。

使用的 Dockerfile 为：

```text
docker/Dockerfile.a2.npu
```

最终成功生成的镜像为：

```text
verl-omni:npu-a2-910b3
```

镜像 ID：

```text
801d28e53932
```

## 2. 官方 Dockerfile 直接构建时遇到的问题

这些问题并非单个 Python 包安装失败，而是 CANN、PyTorch、torch-npu、vLLM、
vLLM-Ascend、vLLM-Omni 和 verl-omni 多层依赖同时变化后产生的组合兼容性问题。

### 2.1 CANN、torch-npu 与 vLLM-Ascend 版本不匹配

原 Dockerfile 使用 CANN 9.0.0，但仓库当前锁定的 vLLM-Ascend revision 需要
CANN 9.1 和 torch-npu 2.10.0.post4 这一代运行栈。继续使用旧组合容易在编译期或
运行期出现 ABI、算子和扩展不匹配。

修复方式：

- 基础镜像升级到 `quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.12`。
- PyTorch 固定为 `2.10.0+cpu`。
- torchvision 固定为 `0.25.0+cpu`。
- torchaudio 固定为 `2.10.0+cpu`。
- torch-npu 固定为 `2.10.0.post4`。
- vLLM 更新到与当前依赖栈匹配的 `v0.27.1`。

经验：Ascend 环境不能只升级 vLLM 或 torch-npu 中的一个组件。应先确定
vLLM-Ascend 的 pin，再反推 CANN、PyTorch 和 torch-npu 的完整版本矩阵。

### 2.2 国内网络环境下 apt 和 GitHub 下载不稳定

基础镜像默认 Ubuntu 软件源在国内网络中速度慢或容易超时，源码安装又需要访问
GitHub、PyPI、PyTorch wheel 源和 Ascend wheel 源。

修复方式：

- Ubuntu ports 源替换为清华镜像。
- Python 默认索引配置为清华 PyPI 镜像。
- Ascend 包使用华为云 Ascend PyPI 源。
- PyTorch CPU wheel 使用 PyTorch 官方 CPU 索引。
- 将必要域名加入 pip trusted-host。
- verl-omni 不再从 GitHub 二次 clone，直接通过 `COPY .` 使用 Docker build context
  中当前检出的代码，既减少网络依赖，也保证镜像代码与本地 revision 一致。

### 2.3 `--no-build-isolation` 下缺少隐式构建依赖

vLLM-Ascend 的依赖 `arctic-inference` 在构建 wheel 时需要 `grpcio-tools` 和
`nanobind`。使用 `--no-build-isolation` 后，pip 不再自动创建隔离环境并安装这些
构建依赖，导致构建在 vLLM-Ascend editable install 阶段失败。

典型表现是构建后端找不到 `grpcio_tools` 或 `nanobind`。

修复方式是在安装 vLLM-Ascend 之前显式安装：

```dockerfile
RUN python3 -m pip install grpcio-tools nanobind
```

同时补齐源码扩展常用工具：

```text
rustc
cargo
setuptools-rust
setuptools-scm
cmake
pybind11
```

经验：选择 `--no-build-isolation` 可以复用镜像中已经对齐的 torch/CANN 环境，但
必须主动阅读每个源码包的 `pyproject.toml`，把 build-system 依赖预装到镜像中。

### 2.4 Python 导入 vLLM 时过早加载 NPU backend

构建 editable wheel 的过程中，Python 包探测可能自动加载 torch-npu 或 Ascend
backend。此时 CANN 环境尚未完全 source，或者构建动作本身并不需要初始化设备，
容易造成不必要的构建失败。

修复方式是在 vLLM 和 vLLM-Ascend 的安装阶段设置：

```text
TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

vLLM 本体以 empty target 构建，Ascend backend 由 vLLM-Ascend 提供：

```text
VLLM_TARGET_DEVICE=empty
```

### 2.5 vLLM-Omni 与 verl-omni 的 `accelerate` 约束冲突

vLLM-Omni 当前 requirements 固定了较旧的 `accelerate`，而当前 verl-omni 需要
`accelerate>=1.14.0`。如果直接让 pip 解析全部依赖，后安装的包会覆盖前一个包的
版本，导致镜像虽然构建成功，但运行时可能缺少新 API。

修复方式：

1. 从 vLLM-Omni 的 common requirements 中移除固定的 `accelerate==...` 行。
2. 安装整理后的 NPU requirements。
3. 使用 `--no-deps` editable 安装 vLLM-Omni 本体。
4. 最后统一将 `accelerate` 对齐到 verl-omni 所需版本。

### 2.6 下游 extras 会重新改写核心运行栈

安装 vLLM-Omni、verl 和 verl-omni extras 后，pip 可能再次升级或降级 torch、
torch-npu、NumPy、setuptools、packaging 和 fsspec，使前面已经匹配的运行栈失效。

修复方式是在所有项目安装完成后增加最终对齐层：

- 强制重装固定版本的 torch、torchvision 和 torchaudio。
- 使用 `--force-reinstall --no-deps` 重装 torch-npu，避免它再次触发 torch 解析。
- NumPy 固定为 `1.26.4`，满足当前 triton-ascend 的精确约束。
- 对齐 `accelerate>=1.14.0`、`setuptools>=77.0.3,<81.0.0`、
  `packaging>=26.2` 和 `fsspec<=2026.6.0`。

经验：核心计算栈应在 Dockerfile 末尾重新锁定一次。否则“最后安装的包”会无意中
决定最终运行环境。

### 2.7 Docker build 成功不代表运行栈可用

只检查 pip 命令退出码无法发现动态库、backend 注册和 editable source 路径问题。

修复方式是在镜像构建末尾加入导入检查：

```dockerfile
RUN . /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    . /usr/local/Ascend/nnal/atb/set_env.sh && \
    python3 -c "import torch, torch_npu, vllm, vllm_omni, verl, verl_omni; print('verl-omni NPU image imports OK')"
```

## 3. 最终采用的构建策略

整体顺序如下：

1. 选择与 pin 匹配的 CANN 9.1 基础镜像。
2. 安装系统编译工具和 Python 构建工具。
3. 安装并锁定 PyTorch 2.10.0 与 torch-npu 2.10.0.post4。
4. 以 empty target 安装 vLLM 0.27.1。
5. 预装 `grpcio-tools`、`nanobind` 等非隔离构建依赖。
6. source CANN/ATB 环境并安装 pinned vLLM-Ascend。
7. 过滤冲突依赖后安装 pinned vLLM-Omni。
8. 安装 pinned verl。
9. 从本地 build context 安装 verl-omni。
10. 再次锁定 torch、torch-npu、NumPy 和关键 Python 工具包。
11. 在 Docker build 内执行全栈 import 检查。
12. 在真实 8 卡容器中执行 NPU runtime 检查。

构建命令示例：

```bash
docker build \
  -f docker/Dockerfile.a2.npu \
  -t verl-omni:npu-a2-910b3 \
  .
```

## 4. 实际验证结果

### 4.1 已通过的检查

- Docker 镜像完整构建成功。
- `verl_omni` 导入成功，版本为 `0.2.0rc1`。
- `vllm_omni` 导入成功，版本为
  `0.27.0rc2.dev62+g444485650.npu`。
- `torch==2.10.0+cpu`。
- `torch-npu==2.10.0.post4`。
- vLLM 成功识别 `vllm_ascend.platform.NPUPlatform`。
- 容器中 `torch.npu.is_available()` 返回 `True`。
- 容器识别到 8 张 NPU。
- 在 `npu:0` 上完成真实 tensor 运算，求和结果为 `28.0`。
- `npu-smi` 能看到 8 张健康的 Ascend 910B3。

### 4.2 警告与已知依赖冲突

当前镜像运行基础功能正常，但不能声称 `pip check` 完全干净。上游依赖仍存在一些
互相无法同时满足的声明：

- 通用 vLLM 与 vLLM-Ascend 对 FastAPI/Starlette 的约束不同。
- verl 声明 `numpy>=2`，当前 triton-ascend 精确要求 `numpy==1.26.4`。
- vLLM-Omni 固定的 accelerate 版本低于 verl-omni 的需求。
- triton-ascend 与部分 guardrail 依赖对 attrs 的要求不同。

这里采用的是“以 Ascend 实际运行链路为准”的版本组合，并通过 import 和真实 NPU
运算验证，而不是为了让依赖检查表面全绿而破坏底层计算栈。

另外，FlashInfer 不存在时 vLLM 会回退到其他 attention backend；这在 Ascend
环境中是警告，不是本次构建失败原因。`humming.__spec__ is None` 也会产生探测警告，
但不影响上述导入验证。

## 5. LTX2.3 训练冒烟测试结果

使用了官方入口：

```text
examples/flowgrpo_trainer/ltx2/run_ltx2_3_t2av_lora_npu.sh
```

本地 LTX2.3 Diffusers 模型可以被发现，数据预处理、Ray 初始化、8 卡通信、训练侧
FSDP 初始化和模型 checkpoint 加载均已成功。训练侧识别到
`LTX2VideoTransformer3DModel` 共 19.61B 参数，FSDP 后单卡观测约为
`7.96 / 60.96 GB`。

首次使用 TP=4 时，vLLM-Omni rollout 初始化在加载辅助组件时发生：

```text
aclrtMallocPhysical failed with acl error code: 207001
OOM: Out of Memory, allocation failed
```

将 rollout 调整为 TP=8 单副本后，模型能够完成加载和 warmup，但随后发现两个兼容
问题：

1. 当前 pinned vLLM-Omni 的 `LTXForwardContext` 新增了必填 `sampler` 字段，
   verl-omni adapter 仍按旧签名构造。修复为传入 `phase_recipe.sampler`。
2. Diffusers 在 NPU 上自动调用 fused `torch_npu.npu_rms_norm`，该算子无法正确处理
   FSDP 参数视图，报 `gamma` 为 undefined tensor。训练 worker 改用与 Diffusers
   eager 分支等价的 RMSNorm 公式，并保持到 activation-checkpoint backward 重计算
   完成。vLLM-Omni rollout 运行在独立进程中，不受该训练侧 workaround 影响。

关闭 parameter offload 虽然也能跨过 RMSNorm forward，但会让训练反向峰值与 rollout
常驻内存叠加，rollout wake-up 时再次 OOM。因此最终保留 parameter offload。

最终通过的关键组合为：

- `NUM_GPUS=8`。
- `ROLLOUT_TP=8`，单 rollout 副本。
- FSDP `use_orig_params=True`。
- 保留 `param_offload=True` 和 `optimizer_offload=True`。
- 训练侧使用 persistent eager RMSNorm。
- 冒烟参数为 128×192、9 帧、2 个推理步、8 个样本、1 个训练步。

最终结果：进程退出码为 `0`，训练进度达到 `1/1`，完整完成 rollout、常量 reward、
old log-prob、actor forward/backward、optimizer step 和 rollout 权重更新。单步耗时约
114 秒，日志中 `training/global_step=1`、`perf/total_num_images=8`。

因此，当前结论为：

- 镜像构建、Python 全栈导入和 Ascend NPU 基础运行验证通过。
- LTX2.3 TP=8 单副本可以完成真实 8 卡单步训练闭环。
- TP=4 对该 19.61B 模型显存不足，不应作为 8×910B3 的默认 rollout 配置。
- 本次使用常量 reward 和极小生成参数，只证明执行链路可用，不代表真实训练效果或
  收敛质量。

后续可优先验证 TP=8、rollout/训练分阶段驻留、辅助组件 CPU offload，以及降低
rollout 并发与显存利用率等方案。

## 6. 可复用的排查方法

遇到类似问题时，建议按层验证，而不是直接运行完整训练：

1. 检查镜像是否成功生成以及 tag 是否正确。
2. 在无设备容器中检查 Python package import。
3. 挂载驱动和设备后检查 `npu-smi`。
4. 检查 `torch.npu.is_available()` 和设备数量。
5. 执行一个最小 NPU tensor 运算。
6. 检查 vLLM 是否注册为 `NPUPlatform`。
7. 再执行低分辨率、低采样步数、单训练步的模型冒烟测试。
8. 记录首次失败的完整 traceback，优先修复最内层异常，避免被 Ray 或
   orchestrator 的外层异常误导。

最重要的经验是：镜像构建问题、Python 依赖问题、NPU runtime 问题和模型显存问题
属于四个不同层次。逐层建立可验证的通过条件，可以显著缩短定位时间。
