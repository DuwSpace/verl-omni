#!/usr/bin/env bash
# Run inside the tested NPU image. Preserve its binary engine stack.
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u
cd /repo
export PYTHONPATH="/repo${PYTHONPATH:+:$PYTHONPATH}"
constraints="$(mktemp)"
trap 'rm -f "$constraints"' EXIT
python3 - "$constraints" <<'PY'
import importlib.metadata as metadata
from pathlib import Path
import sys

protected = ('torch', 'torch-npu', 'torchvision', 'torchaudio', 'vllm', 'vllm-ascend', 'vllm-omni', 'verl')
versions = {name: metadata.version(name) for name in protected}
if versions['torch'] != '2.10.0+cpu' or versions['torch-npu'] != '2.10.0.post4':
    raise SystemExit(f'Expected the npu-a2-main-8d2a9fd image stack; got {versions}')
for name in ('triton', 'triton-ascend'):
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
Path(sys.argv[1]).write_text(''.join(f'{name}=={version}\n' for name, version in versions.items()))
print('Preserving image engine versions:', versions, flush=True)
PY
python3 -m pip install -c "$constraints" \
    'diffusers==0.40.0' 'transformers==5.14.1' 'peft==0.20.0' \
    'audiobox-aesthetics==0.0.4' 'timm==1.0.27' \
    'av==18.1.0' 'imageio-ffmpeg==0.6.0' 'soundfile==0.14.0' \
    'einops==0.8.2' 'scipy==1.13.1' 'numpy==2.2.6'
# Select the mounted branch with PYTHONPATH instead of reinstalling the repo.
# This also works for worktrees whose external .git directory is not mounted.
python3 - "$constraints" <<'PY'
import importlib.metadata as metadata
from pathlib import Path
import sys

for line in Path(sys.argv[1]).read_text().splitlines():
    name, expected = line.split('==', 1)
    assert metadata.version(name) == expected, f'Engine dependency changed: {name}'
import torch
import torch_npu
from audiobox_aesthetics.model.aes import AesMultiOutput
from verl_omni.utils.reward_score.hpsv3_reward import HPSv3Model
from vllm_omni.diffusion.lora.layers.base_linear import DiffusionBaseLinearLayerWithLoRA
import timm
import verl_omni

assert Path(verl_omni.__file__).resolve().is_relative_to(Path("/repo"))

assert torch.npu.is_available(), 'NPU runtime is unavailable; check host driver and device mounts'
assert torch.npu.device_count() >= 8, 'This launcher expects eight visible NPUs'
assert hasattr(DiffusionBaseLinearLayerWithLoRA, '_set_diffusion_lora_buffer_device')
print('LTX dependencies and NPU visibility verified.', flush=True)
PY
