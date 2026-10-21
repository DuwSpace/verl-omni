# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from test_ltx2_joint_model_on_cpu import _joint_batch, model  # noqa: F401
from transformers import GenerationConfig

from verl_omni.pipelines.ltx2_unigrpo_nft.hooks import LTXJointHooks
from verl_omni.pipelines.ltx2_unigrpo_nft.loss import LTXJointLoss
from verl_omni.pipelines.ltx2_unigrpo_nft.text_diagnostics import (
    TextDiagnosticConfig,
    TextDiagnostics,
    difference,
    model_evidence,
    tensor_hash,
)
from verl_omni.workers.config.diffusion.actor import LTXJointLossConfig


def _options(root, **kwargs):
    return {"enabled": True, "output_dir": str(root), **kwargs}


def _native_evidence(model, records, root, rank=0):
    diagnostic = TextDiagnostics(_options(root), side="rollout", source_rank=rank)
    for record in records:
        record.text_diagnostic = diagnostic.rollout(
            model.gemma,
            record.prompt_token_ids,
            record.response_token_ids,
            record.rollout_logprobs,
            version=record.policy_version,
            seed=record.seed,
            temperature=record.temperature,
            request_id="request-0",
        )
    return diagnostic


def _hooks(model, root):
    hooks = LTXJointHooks(model, SimpleNamespace(ltx_joint=SimpleNamespace(text_diagnostics=_options(root))), None)
    hooks._device = lambda: torch.device("cpu")
    hooks.begin_rollout(version=0)
    return hooks


def _check_batches(hooks, records, micro_size):
    hooks._validate_text_diagnostic_batches(records, micro_size)
    values = {}
    for index in range(0, len(records), micro_size):
        group = records[index : index + micro_size]
        with hooks.policies.use("old"):
            inputs = hooks._ar_batch_inputs(group)
            values[index] = hooks.module(task="recaption", **inputs).detach().cpu()
            hooks._check_text_diagnostic_batch(group, values[index], inputs)
    return values


def test_disabled_capture_does_not_touch_model_or_disk(tmp_path):
    diagnostic = TextDiagnostics({}, side="rollout")
    assert diagnostic.rollout(None, None, None, None, version=0, seed=0, temperature=1, request_id="x") is None
    diagnostic.policy(None, 0)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "options",
    [
        {"enabled": True},
        {"enabled": True, "output_dir": "relative"},
        {"mean_abs_threshold": 0},
        {"max_abs_threshold": float("nan")},
        {"enabled": "false"},
    ],
)
def test_invalid_configuration_fails_explicitly(options):
    with pytest.raises(ValueError):
        TextDiagnosticConfig(**options)


def test_real_generate_r_replay_rprime_and_actor_t_are_paired(model, tmp_path):
    _, records = _joint_batch(model)
    record = records[0]
    ids = record.prompt_token_ids.unsqueeze(0)
    with torch.no_grad():
        generated = model.gemma.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            generation_config=GenerationConfig(
                do_sample=False,
                max_new_tokens=3,
                eos_token_id=31,
                pad_token_id=0,
                return_dict_in_generate=True,
                output_scores=True,
            ),
        )
    record.response_token_ids = generated.sequences[0, ids.shape[1] :].cpu()
    record.rollout_logprobs = torch.stack(
        [
            score[0].float().log_softmax(-1)[token]
            for score, token in zip(generated.scores, record.response_token_ids, strict=True)
        ]
    ).cpu()
    record.eos_token_ids = (31,)
    native = _native_evidence(model, [record], tmp_path, rank=5)
    hooks = _hooks(model, tmp_path)
    result = _check_batches(hooks, [record], 1)
    torch.testing.assert_close(
        result[0][0, -record.response_token_ids.numel() :], record.rollout_logprobs, atol=1e-5, rtol=1e-5
    )
    paths = list(hooks.text_diagnostics.directory.glob("sample-*.json"))
    assert len(paths) == 1
    meta = json.loads(paths[0].read_text())
    assert meta["rollout_source_rank"] == 5 and meta["sample_id"] == record.text_diagnostic["sample_id"]
    assert not meta["mismatch"]
    saved = torch.load(paths[0].with_suffix(".pt"), weights_only=True)
    assert set(("R", "Rprime", "T", "input_ids", "attention_mask", "response_mask", "buffers")) <= saved.keys()
    assert saved["buffers"]  # Includes nonpersistent RoPE; state_dict alone is insufficient.
    torch.testing.assert_close(saved["Rprime"], saved["T"], rtol=0, atol=0)
    assert not list(native.directory.glob("failure-*.json"))


@pytest.mark.parametrize("corrupt", ["tokens", "version", "R", "Rprime"])
def test_transport_identity_and_score_corruption_are_rejected(model, tmp_path, corrupt):
    _, records = _joint_batch(model)
    record = records[0]
    _native_evidence(model, [record], tmp_path)
    if corrupt == "tokens":
        record.response_token_ids[0] += 1
    elif corrupt == "version":
        record.text_diagnostic["policy_version"] += 1
    elif corrupt == "R":
        record.rollout_logprobs[0] += 0.1
    else:
        record.text_diagnostic["recomputed_logprobs"][0] += 0.1
    hooks = _hooks(model, tmp_path)
    with pytest.raises(RuntimeError, match="Unpaired|changed during transport"):
        _check_batches(hooks, [record], 1)
    assert list(hooks.text_diagnostics.directory.glob("failure-*.pt"))


def test_native_mismatch_saves_original_scores_and_live_state_before_raising(model, tmp_path):
    _, records = _joint_batch(model)
    record = records[0]
    record.rollout_logprobs += 0.2
    diagnostic = TextDiagnostics(_options(tmp_path), side="rollout", source_rank=4)
    with pytest.raises(RuntimeError, match="Native Rprime/R mismatch"):
        diagnostic.rollout(
            model.gemma,
            record.prompt_token_ids,
            record.response_token_ids,
            record.rollout_logprobs,
            version=0,
            seed=record.seed,
            temperature=1.0,
            request_id="bad",
        )
    saved = torch.load(next(diagnostic.directory.glob("sample-*.pt")), weights_only=True)
    torch.testing.assert_close(saved["R"], record.rollout_logprobs)
    assert difference(saved["R"], saved["Rprime"])["mean_abs"] == pytest.approx(0.2, abs=1e-5)
    failure = torch.load(next(diagnostic.directory.glob("failure-*.pt")), weights_only=True)
    assert len(failure["parameters"]) == len(dict(model.gemma.named_parameters()))
    assert len(failure["buffers"]) == len(dict(model.gemma.named_buffers()))


def test_actor_mismatch_stops_before_backward_and_preserves_old_snapshot(model, tmp_path):
    data, records = _joint_batch(model)
    _native_evidence(model, records, tmp_path)
    hooks = _hooks(model, tmp_path)
    with torch.no_grad():
        model.gemma.lm_head.weight.zero_()  # Simulate changed frozen live state after rollout.
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.add_(0.4)  # Current differs from old; capture must contain old.
    current = {n: p.detach().clone() for n, p in model.named_parameters()}
    cfg = SimpleNamespace(diffusion_loss=LTXJointLossConfig(), ppo_micro_batch_size_per_gpu=1)
    with pytest.raises(RuntimeError, match="text mismatch"):
        hooks.forward_backward_batch(data, partial(LTXJointLoss(), config=cfg))
    assert all(p.grad is None for p in model.parameters())
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, current[name], atol=0, rtol=0)
    captured = torch.load(next(hooks.text_diagnostics.directory.glob("failure-*.pt")), weights_only=True)
    for name, tensor in hooks.policies.old.items():
        if name.startswith("gemma."):
            torch.testing.assert_close(captured["parameters"][name.removeprefix("gemma.")], tensor, atol=0, rtol=0)


def test_buffer_hashes_do_not_depend_on_registration_order():
    left, right = torch.nn.Module(), torch.nn.Module()
    left.register_buffer("a", torch.tensor([1.0, 2.0]), persistent=False)
    left.register_buffer("b", torch.tensor([3.0, 4.0]), persistent=False)
    right.register_buffer("b", torch.tensor([3.0, 4.0]), persistent=False)
    right.register_buffer("a", torch.tensor([1.0, 2.0]), persistent=False)
    assert model_evidence(left)["buffers"] == model_evidence(right)["buffers"]
    right.a[0] += 1
    assert tensor_hash(left.a) != tensor_hash(right.a)


def _distributed_failure(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=f"file://{root}/rendezvous", timeout=timedelta(seconds=60)
    )
    try:
        tiny = model.__wrapped__()
        _, records = _joint_batch(tiny)
        _native_evidence(tiny, records, Path(root), rank=rank)
        hooks = _hooks(tiny, Path(root))
        if rank == 1:
            with torch.no_grad():
                tiny.gemma.lm_head.weight.zero_()
        try:
            _check_batches(hooks, records, 1)
        except RuntimeError as exc:
            assert "rank 1" in str(exc) and "text mismatch" in str(exc)
            assert list(hooks.text_diagnostics.directory.glob("failure-*.pt"))
            (Path(root) / f"stopped-{rank}").write_text(str(exc))
        else:
            raise AssertionError("All ranks must stop even when only rank 1 mismatches")
        assert all(p.grad is None for p in tiny.parameters())
        data, invalid = _joint_batch(tiny)
        if rank == 1:
            invalid[0].rollout_logprobs[0] = float("nan")
        cfg = SimpleNamespace(diffusion_loss=LTXJointLossConfig(), ppo_micro_batch_size_per_gpu=1)
        with pytest.raises(RuntimeError, match="rank 1: invalid text replay"):
            hooks.forward_backward_batch(data, partial(LTXJointLoss(), config=cfg))
        if rank == 1:
            saved = torch.load(hooks.text_diagnostics.directory / "invalid-replay.pt", weights_only=True)
            assert torch.isnan(saved["records"][0]["rollout_logprobs"][0])
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_two_real_gloo_ranks_stop_together_on_one_rank_mismatch(tmp_path):
    torch.multiprocessing.spawn(_distributed_failure, args=(str(tmp_path),), nprocs=2, join=True)
    assert (tmp_path / "stopped-0").exists() and (tmp_path / "stopped-1").exists()


def test_diagnostic_wrapper_forwards_options_without_launching_training(tmp_path):
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    bindir = tmp_path / "bin"
    bindir.mkdir()
    shim = bindir / "python3"
    shim.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        'open(os.environ["CAPTURE_ARGV"],"w").write(json.dumps(sys.argv[1:]))\n'
    )
    shim.chmod(0o755)
    dataset = tmp_path / "data.parquet"
    dataset.touch()
    captured = tmp_path / "args.json"
    env = {
        **os.environ,
        "PATH": str(bindir) + os.pathsep + os.environ["PATH"],
        "CAPTURE_ARGV": str(captured),
        "LTX_OUTPUT_DIR": str(tmp_path / "run"),
        "LTX_MODEL_PATH": "/unused-model",
        "LTX_REWARD_ROOT": "/unused-rewards",
        "LTX_TRAIN_FILE": str(dataset),
        "LTX_VAL_FILE": str(dataset),
    }
    subprocess.run(
        [
            "bash",
            str(root / "examples/unigrpo_nft_trainer/ltx2/run_text_diagnostics.sh"),
            "trainer.total_training_steps=3",
        ],
        env=env,
        check=True,
        capture_output=True,
    )
    args = json.loads(captured.read_text())
    assert args[:2] == ["-m", "verl_omni.trainer.main_diffusion"]
    assert "actor_rollout_ref.rollout.ltx_joint.text_diagnostics.enabled=true" in args
    assert f'actor_rollout_ref.rollout.ltx_joint.text_diagnostics.output_dir="{tmp_path}/run/text_diagnostics"' in args
    assert args[-1] == "trainer.total_training_steps=3"


def test_config_composition_carries_diagnostics_to_actor_settings(tmp_path):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    directory = Path(__file__).resolve().parents[2] / "verl_omni/trainer/config"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        config = compose(
            config_name="diffusion_trainer",
            overrides=[
                "actor_rollout_ref.rollout.ltx_joint.text_diagnostics.enabled=true",
                f"actor_rollout_ref.rollout.ltx_joint.text_diagnostics.output_dir={tmp_path}",
            ],
        )
    assert OmegaConf.to_container(
        config.actor_rollout_ref.model.ltx_joint.text_diagnostics, resolve=True
    ) == OmegaConf.to_container(config.actor_rollout_ref.rollout.ltx_joint.text_diagnostics, resolve=True)
    assert config.actor_rollout_ref.model.ltx_joint.text_diagnostics.enabled


@pytest.mark.parametrize("corrupt_load", [False, True])
def test_native_sync_verifies_actual_loaded_lora(model, tmp_path, monkeypatch, corrupt_load):
    import peft
    from test_ltx2_joint_model_on_cpu import _rollout_gemma

    from verl_omni.pipelines.ltx2_unigrpo_nft.weight_sync import write_joint_lora_bundle
    from verl_omni.workers.rollout.ltx_joint_worker_extension import LTXJointWorkerExtension

    weights = {name: value.detach().cpu() for name, value in model.named_parameters() if value.requires_grad}
    bundle = tmp_path / "bundle"
    write_joint_lora_bundle(bundle, weights, model.peft_config["default"], version=1, base_model="/model")
    native = _rollout_gemma(model)
    pipeline = SimpleNamespace(
        text_encoder=native,
        transformer=model.transformer,
        device=torch.device("cpu"),
        od_config=SimpleNamespace(model="/model"),
    )
    worker = LTXJointWorkerExtension()
    worker.model_runner = SimpleNamespace(pipeline=pipeline)
    worker.lora_manager = SimpleNamespace(set_active_adapter=lambda *args: None)
    if corrupt_load:
        original = peft.set_peft_model_state_dict

        def corrupted(module, *args, **kwargs):
            result = original(module, *args, **kwargs)
            with torch.no_grad():
                next(p for n, p in module.named_parameters() if "lora_B" in n).add_(0.5)
            return result

        monkeypatch.setattr(peft, "set_peft_model_state_dict", corrupted)
        with pytest.raises(RuntimeError, match="Actual native Gemma LoRA differs"):
            worker.load_joint_lora_bundle(str(bundle), _options(tmp_path), 5)
        assert pipeline._joint_sync_in_progress
        assert not hasattr(pipeline, "_joint_policy_version")
        assert list(pipeline._joint_text_diagnostics.directory.glob("failure-*.pt"))
    else:
        worker.load_joint_lora_bundle(str(bundle), _options(tmp_path), 5)
        meta = json.loads((pipeline._joint_text_diagnostics.directory / "policy-1.json").read_text())
        assert meta["source_rank"] == 5 and meta["policy_version"] == 1 and meta["complete"]
        assert meta["parameters"] and meta["buffers"]
        assert worker.snapshot_joint_text_failure("manual test")


def test_actor_masks_preserve_variable_length_sample_pairing(model, tmp_path):
    _, records = _joint_batch(model)
    records[1].prompt_token_ids = torch.tensor([2, 3, 6, 7])
    records[1].response_token_ids = torch.tensor([4, 5, 1])
    with torch.no_grad():
        value = model(task="recaption", **records[1].ar_inputs("cpu"))
        records[1].rollout_logprobs = value[0, -3:].cpu()
    _native_evidence(model, records, tmp_path)
    hooks = _hooks(model, tmp_path)
    _check_batches(hooks, records, 2)
    metadata = [json.loads(p.read_text()) for p in hooks.text_diagnostics.directory.glob("sample-*.json")]
    assert sorted(m["comparisons"]["T_vs_R"]["tokens"] for m in metadata) == [2, 3]
    assert all(not m["mismatch"] for m in metadata)


def test_nonfinite_native_scores_capture_and_stop_even_in_warn_mode(model, tmp_path):
    _, records = _joint_batch(model)
    record = records[0]
    record.rollout_logprobs[0] = float("nan")
    diagnostic = TextDiagnostics(_options(tmp_path, stop_on_mismatch=False), side="rollout", source_rank=4)
    with pytest.raises(RuntimeError, match="Native Rprime/R mismatch"):
        diagnostic.rollout(
            model.gemma,
            record.prompt_token_ids,
            record.response_token_ids,
            record.rollout_logprobs,
            version=0,
            seed=record.seed,
            temperature=1.0,
            request_id="nonfinite",
        )
    saved = json.loads(next(diagnostic.directory.glob("sample-*.json")).read_text())
    assert not saved["Rprime_vs_R"]["finite"] and saved["Rprime_vs_R"]["mean_abs"] is None
    assert list(diagnostic.directory.glob("failure-*.pt"))
