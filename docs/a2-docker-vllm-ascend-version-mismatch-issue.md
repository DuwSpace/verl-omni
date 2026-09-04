# [Bug] Ascend A2 Dockerfile uses CANN 9.0 and torch-npu post2 with a vLLM-Ascend revision that requires CANN 9.1 and post4

Last updated: 09/03/2026

## System Info

- verl-omni branch: `main`
- verl-omni revision: `4abe71ecf7aa5faa7722b968c1371409123083e8`
- Hardware: Ascend Atlas A2 / 8 × 910B3
- Build file: `docker/Dockerfile.a2.npu`
- Base image declared by the unmodified Dockerfile:
  `quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.12`
- PyTorch declared by the unmodified Dockerfile: `2.10.0+cpu`
- torch-npu declared by the unmodified Dockerfile: `2.10.0.post2`
- vLLM declared by the unmodified Dockerfile: `v0.27.0`
- vLLM-Ascend revision from `.github/vllm_ascend_pin.txt`:
  `d5e9816065ede613327d93908f87fee9f5c47128`

## Information

- [x] The official Dockerfile
- [ ] My own modified scripts

## Tasks

- [x] Building the officially provided Ascend Atlas A2 image
- [ ] My own task or dataset

## Reproduction

Build the A2 image from the repository root without overriding any version
arguments:

```bash
docker build --no-cache \
  -f docker/Dockerfile.a2.npu \
  -t verl-omni:npu-a2 \
  .
```

The Dockerfile selects the following stack:

```text
CANN:      9.0.0
torch:     2.10.0+cpu
torch-npu: 2.10.0.post2
```

However, the pinned vLLM-Ascend revision declares:

```text
# vllm-ascend@d5e9816065ede613327d93908f87fee9f5c47128
requirements.txt: torch-npu==2.10.0.post4
pyproject.toml:    torch-npu==2.10.0.post4
```

The installation and quick-start documentation at the same vLLM-Ascend
revision specifies this compatibility matrix:

```text
CANN:      9.1.0
torch:     2.10.0
torch-npu: 2.10.0.post4
NNAL:      9.1.0
```

This makes the Docker build internally inconsistent. Installing the pinned
vLLM-Ascend source can replace the Dockerfile-pinned `torch-npu==2.10.0.post2`
with `2.10.0.post4`, while the image still provides the CANN 9.0 runtime. Later
dependency installation or reinstallation can change the result again. The
final environment therefore depends on pip resolution order instead of one
explicitly supported Ascend version matrix.

The mismatch can be inspected without NPU model weights:

```bash
cat .github/vllm_ascend_pin.txt

git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout d5e9816065ede613327d93908f87fee9f5c47128
grep -n "torch-npu" requirements.txt pyproject.toml
grep -nE "CANN|TorchNPU" \
  docs/source/quick_start.md \
  docs/source/installation.md
```

### Regression timeline

- `docker/Dockerfile.a2.npu` was introduced by `83b66034` on 2026-07-07 with
  the CANN 9.0 base image.
- vLLM-Ascend upgraded its main development stack to CANN 9.1 and
  `torch-npu==2.10.0.post4` in `7485331d` on 2026-08-11.
- verl-omni commit `8b0f0f99` on 2026-08-18 updated
  `.github/vllm_ascend_pin.txt` to a revision containing that new stack, while
  `docker/Dockerfile.a2.npu` retained CANN 9.0 and pinned torch-npu to post2.

## Expected behavior

The official A2 Dockerfile should use one tested and internally consistent
Ascend dependency matrix. Its CANN/NNAL base image, PyTorch, torch-npu, vLLM,
and pinned vLLM-Ascend revision should agree, and dependency installation
should not silently replace one of those core packages.

For the current vLLM-Ascend pin, the expected defaults are:

```text
CANN/NNAL: 9.1.0
torch:     2.10.0
torch-npu: 2.10.0.post4
```

Alternatively, if CANN 9.0 and torch-npu post2 must remain supported, the
vLLM-Ascend pin should point to a revision or release branch that explicitly
supports that older matrix.

## Suggested fix

1. Align `docker/Dockerfile.a2.npu` with the compatibility matrix of the pinned
   vLLM-Ascend revision, or move the pin back to a compatible revision.
2. Keep the core Ascend versions in Docker build arguments so they are defined
   once and reused during final dependency realignment.
3. Install the pinned vLLM-Ascend source without allowing pip to silently
   select a different torch stack.
4. Add a build-time import/version check for `torch`, `torch_npu`, `vllm`,
   `vllm_ascend`, `vllm_omni`, `verl`, and `verl_omni`.
5. Consider a CI check that compares the Dockerfile torch-npu version with the
   requirement declared by `.github/vllm_ascend_pin.txt`.

## Local validation

An A2 image aligned to CANN 9.1, PyTorch 2.10.0, and
`torch-npu==2.10.0.post4` was built locally. On 8 × Ascend 910B3, all devices
were visible, `torch.npu.is_available()` returned `True`, basic NPU tensor
operations succeeded, and `torch`, `torch_npu`, `vllm`, `vllm_ascend`,
`vllm_omni`, `verl`, and `verl_omni` imported successfully.

This local validation is evidence for the corrected version matrix; it is not
a substitute for the repository's A2 CI coverage.

## Duplicate-work check

A GitHub web search for combinations of `Dockerfile.a2.npu`, `CANN 9.1`,
`torch-npu post4`, and `vllm-ascend` did not find an existing verl-omni Issue
or open PR addressing this exact mismatch as of 2026-08-22. The local
environment did not have the `gh` CLI installed, so the repository-prescribed
CLI duplicate checks could not be run and should be repeated before filing or
proposing a PR.
