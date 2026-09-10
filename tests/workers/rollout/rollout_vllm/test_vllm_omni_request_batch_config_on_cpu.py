# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest

from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer


def _make_server(omni_engine_kwargs=None, ar_mode=False):
    server = object.__new__(vLLMOmniHttpServer)
    server._ar_mode = ar_mode
    server.config = SimpleNamespace(
        step_execution=False,
        engine_kwargs={"vllm_omni": dict(omni_engine_kwargs or {})},
    )
    return server


def test_diffusion_parallel_args_reach_async_omni():
    # Mirror the production recipe: SP args live only in the Hydra-injected
    # engine_kwargs.vllm_omni dict (verl's parsed CLI namespace drops them).
    server = _make_server(
        {
            "tensor_parallel_size": 2,
            "ulysses_degree": 4,
            "ulysses_mode": "strict",
            "ring_degree": None,
        }
    )
    engine_args = {"tensor_parallel_size": 2}

    server._bridge_diffusion_parallel_args(engine_args)

    assert engine_args == {
        "tensor_parallel_size": 2,
        "ulysses_degree": 4,
        "ulysses_mode": "strict",
    }


def test_diffusion_parallel_args_absent_in_config_are_not_injected():
    server = _make_server({"tensor_parallel_size": 2})
    engine_args = {"tensor_parallel_size": 2, "ulysses_mode": "advanced_uaa"}

    server._bridge_diffusion_parallel_args(engine_args)

    assert engine_args == {"tensor_parallel_size": 2, "ulysses_mode": "advanced_uaa"}


def test_diffusion_parallel_args_skipped_in_ar_mode():
    server = _make_server({"ulysses_degree": 4})
    server._ar_mode = True
    engine_args = {}

    server._bridge_diffusion_parallel_args(engine_args)

    assert engine_args == {}


@pytest.mark.parametrize("max_num_seqs", [1, 2, 8])
def test_diffusion_request_batch_size_reaches_async_omni(max_num_seqs):
    server = _make_server()
    engine_args = {"max_num_seqs": max_num_seqs}

    server._bridge_diffusion_batch_size(engine_args)

    assert engine_args["max_num_seqs"] == max_num_seqs
    assert engine_args["diffusion_batch_size"] == max_num_seqs


@pytest.mark.parametrize(
    ("ar_mode", "step_execution"),
    [
        (True, False),
        (False, True),
    ],
)
def test_non_request_batch_modes_do_not_set_diffusion_batch_size(ar_mode, step_execution):
    server = object.__new__(vLLMOmniHttpServer)
    server._ar_mode = ar_mode
    server.config = SimpleNamespace(step_execution=step_execution)
    engine_args = {"max_num_seqs": 8}

    server._bridge_diffusion_batch_size(engine_args)

    assert "diffusion_batch_size" not in engine_args


def test_diffusion_memory_flags_reach_async_omni():
    # Mirror the production recipe: the flags live in the Hydra-injected
    # engine_kwargs.vllm_omni dict, but pin 44448565 models them only on
    # OrchestratorArgs, so OmniEngineArgs.from_cli_args drops the CLI values
    # and they never reach AsyncOmni unbridged.
    server = _make_server({"enable_cpu_offload": True, "vae_use_tiling": True})
    engine_args = {"max_num_seqs": 8}

    server._bridge_diffusion_memory_flags(engine_args)

    assert engine_args == {
        "max_num_seqs": 8,
        "enable_cpu_offload": True,
        "vae_use_tiling": True,
    }


def test_bridged_memory_flags_override_parsed_cli_values():
    # The user-specified engine_kwargs value is authoritative: it must win
    # over any stale value left in engine_args by the CLI conversion.
    server = _make_server({"enable_cpu_offload": True})

    engine_args = {"enable_cpu_offload": False}
    server._bridge_diffusion_memory_flags(engine_args)

    assert engine_args["enable_cpu_offload"] is True


def test_diffusion_memory_flags_absent_are_not_injected():
    server = _make_server({"max_num_seqs": 4})
    engine_args = {"max_num_seqs": 4}

    server._bridge_diffusion_memory_flags(engine_args)

    assert engine_args == {"max_num_seqs": 4}


def test_diffusion_memory_flags_none_values_are_not_injected():
    server = _make_server({"enable_cpu_offload": None, "vae_use_tiling": None})
    engine_args = {}

    server._bridge_diffusion_memory_flags(engine_args)

    assert engine_args == {}


def test_diffusion_memory_flags_skipped_in_ar_mode():
    server = _make_server({"enable_cpu_offload": True}, ar_mode=True)
    engine_args = {}

    server._bridge_diffusion_memory_flags(engine_args)

    assert engine_args == {}
