#!/usr/bin/env bash
# Shared full-weight BAGEL: AR UniGRPO + image NFT, 8 independent native MP workers.
# Overrides at the end allow tiny smoke tests without changing production sampling.
set -euo pipefail

export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_CACHE="${HF_HUB_CACHE:-/hub}"
export TOKENIZERS_PARALLELISM=false
# Leave CPU capacity for all actor, rollout and reward processes on this node.
export OMP_NUM_THREADS="${BAGEL_CPU_THREADS:-4}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export PYTHONPATH="/repo:/usr/local/Ascend/cann-9.1.0/python/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

bagel_model="${BAGEL_MODEL_PATH:-${HF_HUB_CACHE}/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/5019f57d168e5816e8f3f701b17cc816bb7cf24b}"

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=/accepted-data/pickscore/train.parquet \
    data.val_files=/accepted-data/pickscore/test.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=256 \
    data.trust_remote_code=True \
    data.dataloader_num_workers=0 \
    algorithm.trainer_type=unigrpo_nft \
    algorithm.adv_estimator=flow_grpo \
    algorithm.global_std=False \
    actor_rollout_ref.model.path="${bagel_model}" \
    actor_rollout_ref.model.tokenizer_path="${bagel_model}" \
    +actor_rollout_ref.model.architecture=OmniBagelForConditionalGeneration \
    actor_rollout_ref.model.algorithm=unigrpo_nft \
    actor_rollout_ref.model.model_type=diffusion_model \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.model.fsdp_layer_prefixes="['layers.']" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim._target_=verl_omni.workers.config.diffusion.FSDPDiffusionOptimizerConfig \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.override_optimizer_config="{foreach: false}" \
    +actor_rollout_ref.actor.optim.param_group_lrs="{moe_gen: 3e-5}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.offload_policy=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=float32 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=unigrpo_nft \
    actor_rollout_ref.actor.diffusion_loss.text_loss_weight=1.0 \
    actor_rollout_ref.actor.diffusion_loss.image_loss_weight=1.0 \
    actor_rollout_ref.actor.diffusion_loss.ar_clip_ratio=0.01 \
    actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=1.5e-5 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_seqs=1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.bagel_joint.sync_bucket_size_mb="${BAGEL_SYNC_BUCKET_MB:-1024}" \
    actor_rollout_ref.rollout.pipeline.height=512 \
    actor_rollout_ref.rollout.pipeline.width=512 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=26 \
    reward.num_workers=8 \
    reward.accelerator_workers.enabled=True \
    reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.pickscore_reward \
    reward.custom_reward_function.name=compute_score_pickscore \
    trainer.device=npu \
    trainer.logger=console \
    trainer.project_name=bagel-unigrpo-nft \
    trainer.experiment_name=bagel-joint-vllm-8npu \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.test_freq=-1 \
    trainer.total_epochs=100 \
    trainer.total_training_steps=3 \
    trainer.resume_mode=disable \
    trainer.default_local_dir=/repo/reproduction/joint-full/checkpoints \
    ray_kwargs.ray_init.num_cpus=64 \
    +ray_kwargs.ray_init.address=local \
    +ray_kwargs.ray_init.object_store_memory=8589934592 \
    "$@"
