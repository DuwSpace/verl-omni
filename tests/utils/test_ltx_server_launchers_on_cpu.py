# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host launchers preserve user overrides and fail before detached execution."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "examples/unigrpo_nft_trainer/ltx2/start_train.sh"


@pytest.fixture
def docker_stub(tmp_path):
    executable = tmp_path / "docker"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['DOCKER_CALLS'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "if '--preflight' in sys.argv: sys.exit(int(os.environ.get('PREFLIGHT_EXIT', '0')))\n"
    )
    executable.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    return {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "DOCKER_CALLS": str(calls)}, calls


@pytest.mark.parametrize("foreground", [False, True])
def test_training_launcher_passes_literal_overrides(docker_stub, foreground):
    env, calls = docker_stub
    override = 'trainer.experiment_name="a b; $(touch must-not-run)"'
    command = ["bash", str(SCRIPT), "my-container", "--name", "test-run", "--smoke", "--no-text-check"]
    if foreground:
        command.append("--foreground")
    subprocess.run([*command, "--", override], env=env, check=True, capture_output=True)
    preflight, launch = [json.loads(line) for line in calls.read_text().splitlines()]
    assert "--preflight" in preflight
    assert ("-d" in launch) is not foreground
    assert "LTX_SMOKE=1" in launch
    assert "LTX_TEXT_CHECK=0" in launch
    assert launch[-2:] == ["--inside", override]


def test_failed_preflight_does_not_start_training(docker_stub):
    env, calls = docker_stub
    result = subprocess.run(
        ["bash", str(SCRIPT), "my-container"],
        env={**env, "PREFLIGHT_EXIT": "2"},
        capture_output=True,
    )
    assert result.returncode == 2
    assert len(calls.read_text().splitlines()) == 1


def test_run_name_cannot_escape_output_directory(docker_stub):
    env, calls = docker_stub
    result = subprocess.run(["bash", str(SCRIPT), "my-container", "--name", "../old-run"], env=env, capture_output=True)
    assert result.returncode == 2
    assert not calls.exists()


@pytest.mark.parametrize("mode", ["av_only", "joint", "ar_only"])
def test_training_launcher_passes_ablation(docker_stub, mode):
    env, calls = docker_stub
    subprocess.run(["bash", str(SCRIPT), "my-container", "--ablation", mode], env=env, check=True, capture_output=True)
    preflight, launch = [json.loads(line) for line in calls.read_text().splitlines()]
    assert f"LTX_ABLATION={mode}" in preflight
    assert f"LTX_ABLATION={mode}" in launch
    assert any(value.startswith(f"LTX_RUN_NAME=ltx-{mode}-") for value in launch)


def test_invalid_ablation_does_not_launch(docker_stub):
    env, calls = docker_stub
    result = subprocess.run(["bash", str(SCRIPT), "my-container", "--ablation", "typo"], env=env, capture_output=True)
    assert result.returncode == 2
    assert not calls.exists()
