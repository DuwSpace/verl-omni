# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock

from omegaconf import OmegaConf

from verl_omni.trainer import main_diffusion


def test_training_flag_reaches_ray_workers_without_replacing_other_env(monkeypatch):
    cfg = OmegaConf.create(
        {
            "actor_rollout_ref": {"actor": {"fsdp_config": {"full_determinism": True, "seed": 123}}},
            "ray_kwargs": {"ray_init": {"runtime_env": {"env_vars": {"KEEP_ME": "yes"}}}},
        }
    )
    for name in ("validate_separate_config", "enable_rl_insight", "_validate_grm_reward_function"):
        monkeypatch.setattr(main_diffusion, name, lambda _: None)
    monkeypatch.setattr(main_diffusion, "_determinism_requested", lambda _: False)
    monkeypatch.setattr(main_diffusion, "get_ppo_ray_runtime_env", lambda: {"env_vars": {"DEFAULT": "yes"}})
    monkeypatch.setattr(main_diffusion.ray, "is_initialized", lambda: False)
    init = MagicMock()
    monkeypatch.setattr(main_diffusion.ray, "init", init)
    monkeypatch.setattr(main_diffusion.ray, "get", lambda _: None)
    for key in main_diffusion._training_determinism_env(cfg):
        # Register restoration even when the variable was originally absent.
        monkeypatch.setenv(key, "test-placeholder")
    main_diffusion.run_diffusion(cfg, task_runner_class=MagicMock())
    env = init.call_args.kwargs["runtime_env"]["env_vars"]
    assert env["VERL_OMNI_FULL_DETERMINISM"] == "1"
    assert env["PYTHONHASHSEED"] == "123"
    assert env["HCCL_DETERMINISTIC"] == "strict"
    assert env["LCCL_DETERMINISTIC"] == "1"
    assert env["KEEP_ME"] == env["DEFAULT"] == "yes"
    assert "ASCEND_LAUNCH_BLOCKING" not in env


def test_disabled_config_adds_no_determinism_environment():
    assert main_diffusion._training_determinism_env(OmegaConf.create({})) == {}
