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
"""Run OmniNFT trainer locally so breakpoints land in ray_diffusion_trainer.py."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_FILE = REPO_ROOT / "data/omninft/vggsound/train_metadata_20k.jsonl"
DEFAULT_FAKE_REWARD = REPO_ROOT / "examples/omninft_trainer/ltx2/debug/fake_native_rewards.py"
DEFAULT_MODEL_PATH = Path(
    "/hub/models--diffusers--LTX-2.3-Diffusers/snapshots/8eee8edcf067e838b843f926ec4d4cc9b2be1aaf"
)
DEFAULT_REWARD_ROOT = Path("/hub/omnift-rewards")
DEFAULT_REPLAY_PATH = REPO_ROOT / "outputs/omnift_trainer_debug/replay.pkl"
DEFAULT_ROLLOUT_DUMP_DIR = REPO_ROOT / "outputs/omnift_trainer_debug/rollout"
COMPONENT_ORDER = ["video_align", "hpsv3", "audiobox", "clap", "desync"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--reward-root", type=Path, default=DEFAULT_REWARD_ROOT)
    parser.add_argument(
        "--desync-source-root",
        type=Path,
        default="/tmp/OmniNFT-reference",
        help="Pinned OmniNFT checkout (default: <reward-root>/OmniNFT-reference).",
    )
    parser.add_argument("--replay-path", type=Path, default=DEFAULT_REPLAY_PATH)
    parser.add_argument("--rollout-dump-dir", type=Path, default=DEFAULT_ROLLOUT_DUMP_DIR)
    parser.add_argument("--fake-rewards", action="store_true")
    parser.add_argument("--fake-reward-path", type=Path, default=DEFAULT_FAKE_REWARD)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--num-inference-steps", type=int, default=24)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--wait-before-generate", action="store_true")
    parser.add_argument("--wait-before-score", action="store_true")
    return parser.parse_args()


def _fake_reward_functions(fake_reward_path: Path) -> dict[str, dict]:
    return {
        name: {
            "path": str(fake_reward_path),
            "required": True,
            "micro_batch_size": 1,
            "name": name,
            "offset": float(index * 10),
        }
        for index, name in enumerate(COMPONENT_ORDER)
    }


def _native_reward_functions(reward_root: Path, desync_source_root: Path | None) -> dict[str, dict]:
    desync_source_root = desync_source_root or reward_root / "OmniNFT-reference"
    return {
        "video_align": {
            "path": "pkg://verl_omni.utils.reward_score.videoalign_native",
            "required": True,
            "micro_batch_size": 2,
            "model_path": str(reward_root / "VideoReward/checkpoint-11352/model.pth"),
            "base_model_path": str(reward_root / "Qwen2-VL-2B-Instruct"),
            "model_revision": "KlingTeam/VideoReward@4f26600130683e6f1de9f5d463887f28e8ef995c",
            "base_model_revision": "Qwen/Qwen2-VL-2B-Instruct@895c3a49bc3fa70a340399125c650a463535e71c",
        },
        "hpsv3": {
            "path": "pkg://verl_omni.utils.reward_score.hpsv3_native",
            "required": True,
            "micro_batch_size": 8,
            "model_path": str(reward_root / "HPSv3/HPSv3.safetensors"),
            "base_model_path": str(reward_root / "Qwen2-VL-7B-Instruct"),
            "model_revision": "MizzenAI/HPSv3@4f81e3e09edd82fe3c5f636444c721b592a735ca",
            "base_model_revision": "Qwen/Qwen2-VL-7B-Instruct@eed13092ef92e448dd6875b2a00151bd3f7db0ac",
        },
        "audiobox": {
            "path": "pkg://verl_omni.utils.reward_score.audiobox_native",
            "required": True,
            "micro_batch_size": 8,
            "model_path": str(reward_root / "audiobox-aesthetics"),
            "model_revision": "facebook/audiobox-aesthetics@9b1dd8e5df9af7216e836a98974fe3b82c56ded6",
        },
        "clap": {
            "path": "pkg://verl_omni.utils.reward_score.clap_native",
            "required": True,
            "micro_batch_size": 8,
            "model_path": str(reward_root / "checkpoints/clap-htsat-unfused"),
            "model_revision": "laion/clap-htsat-unfused@8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a",
        },
        "desync": {
            "path": "pkg://verl_omni.utils.reward_score.desync_native",
            "required": True,
            "micro_batch_size": 2,
            "model_path": str(reward_root / "synchformer/synchformer_state_dict.pth"),
            "source_root": str(desync_source_root),
            "model_revision": "zghhui/OmniNFT-Reward-Series@9e30061a1392d03bafdcf717e80a385ddf411b4d",
            "source_revision": "fb9237f6e74edf0d0f2a683f4d975b79fde588fe",
        },
    }


def _build_config(args: argparse.Namespace):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "verl_omni/trainer/config")):
        config = compose(config_name="diffusion_trainer")

    config.trainer.device = "npu"
    config.trainer.n_gpus_per_node = args.num_gpus
    config.trainer.nnodes = 1
    config.trainer.logger = ["console"]
    config.trainer.val_before_train = False
    config.trainer.save_freq = -1
    config.trainer.test_freq = -1
    config.trainer.total_epochs = 1
    config.trainer.total_training_steps = 1
    config.trainer.resume_mode = "disable"
    config.trainer.log_val_generations = 0
    config.trainer.rollout_data_dir = str(args.rollout_dump_dir)
    config.trainer.rollout_data_save_freq = 1
    config.trainer.video_fps = 24
    OmegaConf.update(config, "trainer.replay_path", str(args.replay_path), force_add=True)
    config.algorithm.trainer_type = "direct_preference"
    config.algorithm.sample_source = "online"
    config.algorithm.paired_preference = False
    config.actor_rollout_ref.model.path = str(args.model_path)
    config.actor_rollout_ref.model.tokenizer_path = str(args.model_path / "tokenizer")
    config.actor_rollout_ref.model.algorithm = "omni_nft"
    config.actor_rollout_ref.model.model_type = "omni_nft_model"
    config.actor_rollout_ref.model.attn_backend = "native"
    config.actor_rollout_ref.actor.diffusion_loss.loss_mode = "omni_nft"
    rollout = config.actor_rollout_ref.rollout
    rollout.name = "vllm_omni"
    rollout.nnodes = 1
    rollout.n_gpus_per_node = args.num_gpus
    rollout.tensor_model_parallel_size = args.tensor_parallel_size
    rollout.n = args.num_candidates
    rollout.seed = 42
    rollout.load_format = "safetensors"
    rollout.layered_summon = True
    rollout.calculate_log_probs = False
    rollout.rollout_adapter = "old"
    rollout.rollout_attn_backend = "TORCH_SDPA"
    if args.num_gpus % args.tensor_parallel_size != 0:
        raise ValueError("num_gpus must be divisible by tensor_parallel_size.")
    rollout.agent.num_workers = args.num_gpus // args.tensor_parallel_size
    rollout.agent.default_agent_loop = "ltx2_omni_nft_single_turn_agent"
    rollout.pipeline.height = args.height
    rollout.pipeline.width = args.width
    rollout.pipeline.num_frames = args.num_frames
    rollout.pipeline.frame_rate = 24.0
    rollout.pipeline.num_inference_steps = args.num_inference_steps
    rollout.pipeline.guidance_scale = 4.0
    rollout.pipeline.max_sequence_length = args.max_sequence_length
    OmegaConf.update(config, "actor_rollout_ref.rollout.pipeline.output_type", "pt", force_add=True)
    OmegaConf.update(config, "actor_rollout_ref.rollout.engine_kwargs.init_timeout", 3600, force_add=True)
    OmegaConf.update(config, "actor_rollout_ref.rollout.engine_kwargs.stage_init_timeout", 3600, force_add=True)
    config.data.train_files = str(args.data_file)
    config.data.val_files = str(args.data_file)
    config.data.custom_cls.path = "pkg://verl_omni.utils.dataset.omni_nft_dataset"
    config.data.custom_cls.name = "OmniNFTPromptDataset"
    config.data.custom_cls.collate_fn = "collate_omni_nft_prompt_groups"
    config.data.return_multi_modal_inputs = False
    config.data.train_batch_size = 1
    config.data.max_prompt_length = args.max_sequence_length
    config.reward.num_workers = 8
    config.reward.reward_model.enable = False
    config.reward.reward_model.enable_resource_pool = False
    config.reward.reward_manager.name = "MultiModalRewardManager"
    config.reward.aggregation = "preserve_components"
    config.reward.component_order = list(COMPONENT_ORDER)
    if args.fake_rewards:
        reward_functions = _fake_reward_functions(args.fake_reward_path)
    else:
        reward_functions = _native_reward_functions(args.reward_root, args.desync_source_root)
    config.reward.reward_functions = OmegaConf.create(reward_functions)
    return config


def main() -> None:
    """Compose the OmniNFT debug config and run TaskRunner in this process."""
    args = _parse_args()
    if args.wait_before_generate:
        os.environ["OMNIFT_WAIT_BEFORE_GENERATE"] = "1"
    if args.wait_before_score:
        os.environ["OMNIFT_WAIT_BEFORE_SCORE"] = "1"

    import ray
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    from verl_omni.trainer.main_diffusion import TaskRunner

    config = _build_config(args)
    if not ray.is_initialized():
        ray.init(runtime_env=get_ppo_ray_runtime_env())
    print(
        "TRAINER_DEBUG_READY pid=%s. Set breakpoints in "
        "verl_omni/trainer/diffusion/ray_diffusion_trainer.py "
        "(MultiModalDirectPreferenceRayTrainer)." % os.getpid()
    )
    TaskRunner().run(config)


if __name__ == "__main__":
    main()
