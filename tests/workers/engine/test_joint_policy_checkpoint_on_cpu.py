# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verify the shared engine actually saves/loads mandatory frozen policy shards."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from verl_omni.pipelines.bagel_unigrpo_nft.hooks import BagelUniGRPONFTHooks
from verl_omni.workers.engine.fsdp import diffusers_impl


def _engine(monkeypatch):
    engine = object.__new__(diffusers_impl.PPODiffusersFSDPEngine)
    engine.module = torch.nn.Linear(2, 2, bias=False)
    engine.module.weight.data.fill_(1)
    engine._engine_hooks = BagelUniGRPONFTHooks(engine.module, None, None)
    engine._is_offload_param = False
    engine._is_offload_optimizer = False
    engine._uses_fsdp2_cpu_offload_policy = True
    engine.checkpoint_manager = SimpleNamespace(save_checkpoint=Mock(), load_checkpoint=Mock())
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(diffusers_impl, "aggressive_empty_cache", lambda **kwargs: None)
    return engine


def test_engine_checkpoint_roundtrip_restores_frozen_initial_policy(tmp_path, monkeypatch):
    engine = _engine(monkeypatch)
    engine.module.weight.data.fill_(2)
    engine._engine_hooks.begin_rollout(version=4)
    engine.module.weight.data.fill_(3)
    engine.save_checkpoint(str(tmp_path), global_step=5)
    path = tmp_path / "algorithm_state_rank_0.pt"
    assert path.is_file()
    assert not (tmp_path / "algorithm_state_rank_0.pt.tmp").exists()
    state = torch.load(path, map_location="cpu", weights_only=True)
    torch.testing.assert_close(state["reference"]["weight"], torch.ones(2, 2))
    torch.testing.assert_close(state["old"]["weight"], torch.full((2, 2), 2.0))
    engine._engine_hooks.policies.reference["weight"].fill_(99)
    engine._engine_hooks.policies.old["weight"].fill_(99)
    engine.load_checkpoint(str(tmp_path), del_local_after_load=False)
    assert engine._engine_hooks.policies.version == 4
    torch.testing.assert_close(engine._engine_hooks.policies.reference["weight"], torch.ones(2, 2))
    torch.testing.assert_close(engine._engine_hooks.policies.old["weight"], torch.full((2, 2), 2.0))
    engine.checkpoint_manager.save_checkpoint.assert_called_once()
    engine.checkpoint_manager.load_checkpoint.assert_called_once()


def test_resume_without_policy_shards_fails_before_model_loading(tmp_path, monkeypatch):
    engine = _engine(monkeypatch)
    with pytest.raises(FileNotFoundError, match="old/reference"):
        engine.load_checkpoint(str(tmp_path))
    engine.checkpoint_manager.load_checkpoint.assert_not_called()


def test_algorithm_without_extra_policy_state_preserves_existing_checkpoint_api(tmp_path, monkeypatch):
    engine = _engine(monkeypatch)
    engine._engine_hooks = None
    engine.save_checkpoint(str(tmp_path))
    assert not (tmp_path / "algorithm_state_rank_0.pt").exists()
    engine.load_checkpoint(str(tmp_path))
    engine.checkpoint_manager.load_checkpoint.assert_called_once()
