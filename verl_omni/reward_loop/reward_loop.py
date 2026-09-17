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
import asyncio
import copy
import inspect
import logging

import numpy as np
import ray
from omegaconf import open_dict
from tensordict import TensorDict
from verl.experimental.reward_loop import RewardLoopManager
from verl.experimental.reward_loop.reward_loop import RewardLoopWorker
from verl.protocol import DataProto, pad_dataproto_to_divisor
from verl.trainer.ppo.reward import resolve_reward_manager_cls

from verl_omni.workers.config.reward import (
    accelerator_workers_enabled,
    get_reward_model_entries,
    has_reward_models,
    resolve_reward_model_name,
    streaming_reward_enabled,
    validate_reward_model_terms,
)

from .async_utils import gather_complete
from .reward_model import MultiRewardModelManager
from .reward_model_executor import (
    EngineRewardExecutor,
    NativeRewardExecutor,
    build_engine_reward_executors,
    build_native_reward_executors,
)

logger = logging.getLogger(__name__)


def _validate_named_reward_manager_cls(reward_manager_cls, *, preserve_components: bool) -> None:
    """Require callable run_batch for components, or MultiVisual inheritance for scalars."""
    from .reward_manager.multi import MultiVisualRewardManager

    if preserve_components:
        if callable(getattr(reward_manager_cls, "run_batch", None)):
            return
        raise ValueError(
            "reward.aggregation='preserve_components' requires a reward manager with an async run_batch() method; "
            f"got {reward_manager_cls.__name__!r}."
        )
    if issubclass(reward_manager_cls, MultiVisualRewardManager):
        return
    raise ValueError(
        "reward.models currently requires MultiVisualRewardManager for aggregated scoring; "
        f"got {reward_manager_cls.__name__!r}. Support for other modalities is follow-up work."
    )


class OmniRewardLoopWorker(RewardLoopWorker):
    """RewardLoopWorker with named engine and native reward executors."""

    def __init__(
        self,
        config,
        reward_router_address=None,
        reward_model_specs=None,
    ):
        self.reward_model_specs = reward_model_specs or {}
        self.engine_reward_executors: dict[str, EngineRewardExecutor] = build_engine_reward_executors(
            self.reward_model_specs
        )
        self.native_reward_executors: dict[str, NativeRewardExecutor] = build_native_reward_executors(
            self.reward_model_specs
        )
        self._closed_native_executors: set[str] = set()
        self._reward_manager_shutdown_complete = False
        self._shutdown_complete = False
        self._shutdown_lock = asyncio.Lock()
        super().__init__(config, reward_router_address)

    def _init_reward_fn(self):
        super()._init_reward_fn()
        if hasattr(self.reward_manager, "set_reward_executors"):
            self.reward_manager.set_reward_executors(
                self.engine_reward_executors,
                self.native_reward_executors,
            )

    async def compute_score_components(self, data: DataProto):
        """Delegate a local shard to ``run_batch`` and return its component output.

        The manager owns input validation, sample identity, column order, and any
        in-place batch preparation. Raise TypeError if it has no batch entrypoint.
        """
        run_batch = getattr(self.reward_manager, "run_batch", None)
        if not callable(run_batch):
            raise TypeError(f"{type(self.reward_manager).__name__} does not support component scoring.")
        return await run_batch(data)

    async def shutdown(self) -> None:
        """Close native executors and run the optional manager shutdown hook.

        Calls are serialized and successful steps are not repeated. All started
        cleanup tasks finish before propagation; external cancellation takes
        precedence over collected cleanup errors, otherwise the first error is
        raised and additional errors are logged. Failed steps remain retryable.
        """
        shutdown_lock = getattr(self, "_shutdown_lock", None)
        if shutdown_lock is None:
            shutdown_lock = self._shutdown_lock = asyncio.Lock()
        async with shutdown_lock:
            if getattr(self, "_shutdown_complete", False):
                return

            closed_executors = getattr(self, "_closed_native_executors", set())
            self._closed_native_executors = closed_executors
            errors = []
            cancellation = None
            pending_executors = [
                (name, executor)
                for name, executor in self.native_reward_executors.items()
                if name not in closed_executors
            ]
            tasks = [asyncio.ensure_future(executor.close()) for _, executor in pending_executors]
            if tasks:
                try:
                    results = await gather_complete(tasks, return_exceptions=True)
                except asyncio.CancelledError as exc:
                    cancellation = exc
                    results = [_completed_task_result(task) for task in tasks]
                for (name, _), result in zip(pending_executors, results, strict=True):
                    if isinstance(result, BaseException):
                        errors.append(result)
                    else:
                        closed_executors.add(name)

            if not getattr(self, "_reward_manager_shutdown_complete", False):
                shutdown = getattr(self.reward_manager, "shutdown", None)
                if shutdown is None:
                    self._reward_manager_shutdown_complete = True
                else:
                    try:
                        result = shutdown()
                        if inspect.isawaitable(result):
                            task = asyncio.ensure_future(result)
                            try:
                                hook_results = await gather_complete([task], return_exceptions=True)
                            except asyncio.CancelledError as exc:
                                if cancellation is None:
                                    cancellation = exc
                                hook_results = [_completed_task_result(task)]
                            result = hook_results[0]
                        if isinstance(result, BaseException):
                            errors.append(result)
                        else:
                            self._reward_manager_shutdown_complete = True
                    except BaseException as exc:
                        errors.append(exc)

            self._shutdown_complete = (
                len(closed_executors) == len(self.native_reward_executors) and self._reward_manager_shutdown_complete
            )
            primary = cancellation or (errors[0] if errors else None)
            for error in errors:
                if error is primary:
                    continue
                logger.error(
                    "Additional reward worker shutdown failure",
                    exc_info=(type(error), error, error.__traceback__),
                )
            if primary is not None:
                raise primary

    async def wake_up_reward_model(self, model_name: str) -> None:
        try:
            executor = self.native_reward_executors[model_name]
        except KeyError as exc:
            raise ValueError(f"Worker has no native reward model {model_name!r}") from exc
        await executor.wake_up()

    async def sleep_reward_model(self, model_name: str) -> None:
        try:
            executor = self.native_reward_executors[model_name]
        except KeyError as exc:
            raise ValueError(f"Worker has no native reward model {model_name!r}") from exc
        await executor.sleep()


class OmniRewardLoopManager(RewardLoopManager):
    """Coordinate reward workers, model placement, scoring mode, and results.

    Named native models are placed by ``MultiRewardModelManager`` and owned by
    worker-local executors. Named scoring uses gather_complete for RPC results
    and always attempts sleep after wake/scoring. ``weighted_sum`` preserves
    the scalar upstream contract; ``preserve_components`` returns a complete
    sample-aligned ``[B, K]`` matrix. Profiler calls still fan out to
    engine-backed reward replicas when configured.
    """

    def __init__(self, config, rm_resource_pool=None, accelerator_resource_pool=None):
        self._score_lock = asyncio.Lock()
        self._shutdown = False
        self._shutdown_worker_ids: set[int] = set()
        self._reward_worker_groups = {}
        self._reward_worker_group_configs = {}
        self.reward_loop_workers = []
        self._preserve_reward_components = config.reward.get("aggregation") == "preserve_components"
        self.accelerator_resource_pool = accelerator_resource_pool
        try:
            named_reward_manager_cls = None
            if has_reward_models(config):
                validate_reward_model_terms(config)
                named_reward_manager_cls = resolve_reward_manager_cls(config)
                _validate_named_reward_manager_cls(
                    named_reward_manager_cls,
                    preserve_components=self._preserve_reward_components,
                )
            self.multi_reward_model_manager = MultiRewardModelManager(
                config,
                # The trainer maps Role.RewardModel to global_pool or reward_pool.
                # Each named model receives a sub-pool from this one parent.
                resource_pool=rm_resource_pool,
            )
            use_accelerator_workers = accelerator_workers_enabled(config)
            if self.multi_reward_model_manager.models or use_accelerator_workers:
                if use_accelerator_workers and config.reward.reward_model.get("enable", False):
                    raise ValueError(
                        "Accelerator reward workers cannot be combined with reward.reward_model.enable=True"
                    )
                if self.multi_reward_model_manager.models and not config.reward.get("reward_functions"):
                    raise ValueError("reward.models requires non-empty reward.reward_functions")
                self.config = config
                self.reward_model_manager = None
                self.reward_router_address = None
                self.reward_loop_workers_class = ray.remote(OmniRewardLoopWorker)
                self.reward_manager_cls = named_reward_manager_cls or resolve_reward_manager_cls(config)
                self._init_reward_loop_workers()
            else:
                super().__init__(config=config, rm_resource_pool=rm_resource_pool)
        except BaseException:
            try:
                self.shutdown()
            except BaseException:
                logger.exception("Failed to clean up partially initialized reward workers")
            raise

    @property
    def reward_loop_worker_handles(self):
        if not streaming_reward_enabled(self.config):
            return None
        return super().reward_loop_worker_handles

    def _init_reward_loop_workers(self):
        self.reward_loop_workers_class = ray.remote(OmniRewardLoopWorker)
        specs = self.multi_reward_model_manager.reward_model_specs
        self._reward_worker_groups = {}
        self._reward_worker_group_configs = {}

        if self.multi_reward_model_manager.models:
            entries = self.config.reward.reward_functions
            models = get_reward_model_entries(self.config)
            native_names = {name for name, spec in specs.items() if spec.backend == "native"}
            terms_by_group = {}
            shared_terms = {}
            for term_name, term in entries.items():
                model_name = resolve_reward_model_name(term_name, term, models)
                if model_name in native_names:
                    terms_by_group.setdefault(model_name, {})[term_name] = term
                else:
                    shared_terms[term_name] = term

            if shared_terms:
                group_config = self._copy_reward_config(shared_terms)
                workers = self._create_node_affinity_workers(
                    group_config,
                    {name: spec for name, spec in specs.items() if spec.backend == "engine"},
                    "engine_reward_loop_worker",
                )
                self._register_worker_group("shared", workers, group_config)

            for model_name, terms in terms_by_group.items():
                placement = self.multi_reward_model_manager.native_device_assignments[model_name]
                group_config = self._copy_reward_config(terms)
                with open_dict(group_config.reward):
                    group_config.reward.num_workers = len(placement)
                workers = self._create_native_workers(
                    group_config,
                    {model_name: specs[model_name]},
                    model_name,
                    f"native_reward_loop_worker_{model_name}",
                )
                self._register_worker_group(model_name, workers, group_config)
                self.multi_reward_model_manager.bind_native_workers(model_name, workers)

            if not self._reward_worker_groups:
                raise ValueError("reward.models produced no reward worker groups")
            self.reward_loop_workers = self._flatten_worker_groups()
            return

        if specs:
            self._register_worker_group(
                "shared",
                self._create_node_affinity_workers(self.config, specs, "reward_loop_worker"),
                self.config,
            )
            self.reward_loop_workers = self._reward_worker_groups["shared"]
            return

        use_accelerator_workers = accelerator_workers_enabled(self.config)
        if use_accelerator_workers:
            accelerator_resource_pool = self.accelerator_resource_pool
            if accelerator_resource_pool is None:
                raise ValueError("Accelerator reward workers require an accelerator resource pool")
            from .accelerator_reward_workers import build_accelerator_reward_workers

            self.reward_loop_workers = build_accelerator_reward_workers(
                config=self.config,
                reward_loop_workers_class=self.reward_loop_workers_class,
                accelerator_resource_pool=accelerator_resource_pool,
                reward_router_address=self.reward_router_address,
                reward_model_specs=specs,
            )
            self._register_worker_group("legacy", self.reward_loop_workers, self.config)
            return
        super()._init_reward_loop_workers()

    def _copy_reward_config(self, reward_functions):
        group_config = copy.deepcopy(self.config)
        with open_dict(group_config.reward):
            group_config.reward.reward_functions = copy.deepcopy(reward_functions)
            group_config.reward.num_workers = self.config.reward.num_workers
        return group_config

    def _register_worker_group(self, name, workers, config):
        self._reward_worker_groups[name] = workers
        self._reward_worker_group_configs[name] = config

    def _flatten_worker_groups(self):
        return [worker for workers in self._reward_worker_groups.values() for worker in workers]

    def _create_node_affinity_workers(self, config, specs, name_prefix):
        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        if not node_ids:
            raise ValueError("No alive Ray node with CPU resources is available for reward workers")
        workers = []
        try:
            for index in range(config.reward.num_workers):
                worker = self.reward_loop_workers_class.options(
                    name=f"{name_prefix}_{index}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_ids[index % len(node_ids)], soft=True
                    ),
                ).remote(config, self.reward_router_address, specs)
                workers.append(worker)
        except BaseException:
            try:
                self._shutdown_workers(workers)
            except BaseException:
                logger.exception("Failed to clean up partially created reward workers")
            raise
        return workers

    def _create_native_workers(self, config, specs, model_name, name_prefix):
        from .accelerator_reward_workers import build_accelerator_reward_workers

        resource_pool = self.multi_reward_model_manager.native_resource_pools.get(model_name)
        if resource_pool is None:
            raise ValueError(f"Native reward model {model_name!r} requires an allocated resource pool")
        return build_accelerator_reward_workers(
            config=config,
            reward_loop_workers_class=self.reward_loop_workers_class,
            accelerator_resource_pool=resource_pool,
            reward_router_address=self.reward_router_address,
            reward_model_specs=specs,
            worker_name_prefix=name_prefix,
        )

    def compute_rm_score(self, data):
        """Synchronous compatibility entrypoint for current trainers."""
        if self._preserve_reward_components:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.async_compute_rm_score(data))
            raise RuntimeError(
                "compute_rm_score() cannot run inside an event loop; await async_compute_rm_score() instead"
            )
        if not self.multi_reward_model_manager.models:
            return super().compute_rm_score(data)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.async_compute_rm_score(data))
        raise RuntimeError("compute_rm_score() cannot run inside an event loop; await async_compute_rm_score() instead")

    async def async_compute_rm_score(self, data):
        """Return scalar or component rewards through the configured scoring path.

        Component mode requires named deployments, fills missing sample IDs in
        ``data``, and returns sample-aligned CPU ``[B, K]`` columns sorted by name.
        Named-model phases serialize wake, RPC completion, and sleep. Without
        named models, scalar scoring runs the upstream synchronous path in a thread.
        """
        if self._preserve_reward_components:
            from .reward_components import compute_component_rewards

            if not self.multi_reward_model_manager.models:
                raise ValueError("Component-preserving reward scoring requires reward.models deployments.")
            return await self._score_with_model_lifecycle(
                lambda: compute_component_rewards(
                    self._reward_worker_groups,
                    data,
                    set(self.config.reward.reward_functions),
                )
            )
        if not self.multi_reward_model_manager.models:
            return await asyncio.to_thread(super().compute_rm_score, data)
        return await self._score_with_model_lifecycle(lambda: self._compute_named_model_scores(data))

    async def _score_with_model_lifecycle(self, score):
        """Run a scoring coroutine under the named-model lifecycle lock.

        Always attempt sleep, including after a partial wake failure. Ordinary
        sleep exceptions are logged if wake/scoring already failed, otherwise
        propagated. ``score`` must drain its inference tasks before returning or
        raising so that sleep cannot offload a model still in use.
        """
        async with self._score_lock:
            scoring_error = None
            try:
                await self.multi_reward_model_manager.wake_up()
                return await score()
            except BaseException as exc:
                scoring_error = exc
                raise
            finally:
                try:
                    await self.multi_reward_model_manager.sleep()
                except Exception:
                    if scoring_error is None:
                        raise
                    logger.exception("Failed to sleep reward models after scoring failed")

    async def _compute_named_model_scores(self, data: DataProto) -> DataProto:
        """Score padded worker shards and sum group scalars in original row order.

        After successful dispatch, drain all RPCs before propagating inference
        errors/cancellation. Remove padding, combine extra info without duplicate
        keys, and delegate rm_scores shape/device to the configured manager's
        assembler. Synchronous dispatch errors occur before the drain.
        """
        requests_by_group = {}
        for group_name, workers in self._reward_worker_groups.items():
            num_workers = len(workers)
            padded_data, pad_size = pad_dataproto_to_divisor(data, num_workers)
            chunks = padded_data.chunk(num_workers)
            requests = [worker.compute_score_batch.remote(chunk) for worker, chunk in zip(workers, chunks, strict=True)]
            requests_by_group[group_name] = (requests, pad_size)

        all_requests = [request for requests, _ in requests_by_group.values() for request in requests]
        # Remote inference must finish before the lifecycle wrapper offloads models.
        all_outputs = await gather_complete(all_requests)
        group_outputs = {}
        offset = 0
        for group_name, (requests, pad_size) in requests_by_group.items():
            outputs = all_outputs[offset : offset + len(requests)]
            offset += len(requests)
            flattened = [item for sublist in outputs for item in sublist]
            group_outputs[group_name] = flattened[: len(data)] if pad_size else flattened

        merged_scores = []
        merged_infos = []
        for index in range(len(data)):
            total = 0.0
            info = {}
            for outputs in group_outputs.values():
                item = outputs[index]
                total += float(item["reward_score"])
                for key, value in item.get("reward_extra_info", {}).items():
                    if key == "reward/combined":
                        continue
                    if key in info:
                        raise ValueError(f"Duplicate reward extra-info key {key!r} across worker groups")
                    info[key] = value
            info["reward/combined"] = total
            merged_scores.append(total)
            merged_infos.append(info)

        rm_scores = self.reward_manager_cls.assemble_rm_scores(data, merged_scores)
        batch = TensorDict({"rm_scores": rm_scores}, batch_size=len(data))
        reward_extra_keys = list(dict.fromkeys(key for info in merged_infos for key in info))
        non_tensor_batch = {key: np.array([info.get(key) for info in merged_infos]) for key in reward_extra_keys}
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={"reward_extra_keys": reward_extra_keys},
        )

    def start_profile(self, **kwargs) -> None:
        """Start profiling on all reward-model rollout servers. No-op without a reward model."""
        self._run_on_replicas("start_profile", **kwargs)

    def stop_profile(self) -> None:
        """Stop profiling on all reward-model rollout servers. No-op without a reward model."""
        self._run_on_replicas("stop_profile")

    def _run_on_replicas(self, method: str, **kwargs) -> None:
        if self.reward_model_manager is None:
            return
        replicas = self.reward_model_manager.rollout_replicas

        async def run_all():
            await asyncio.gather(*[getattr(replica, method)(**kwargs) for replica in replicas])

        asyncio.run(run_all())

    def shutdown(self) -> None:
        """Synchronously finalize unique workers, retrying only failed workers.

        Wait for every issued shutdown RPC and raise the first failure after
        logging additional ones. Successful calls become no-ops on repetition;
        this closes worker-local resources but does not kill the Ray actors.
        """
        if self._shutdown:
            return
        workers = self._managed_reward_workers()
        self._shutdown_workers(workers)
        self._shutdown = len(self._shutdown_worker_ids) == len(workers)

    def _managed_reward_workers(self):
        workers = []
        seen = set()
        grouped_workers = getattr(self, "_reward_worker_groups", {})
        candidates = [worker for group in grouped_workers.values() for worker in group]
        candidates.extend(getattr(self, "reward_loop_workers", None) or [])
        for worker in candidates:
            worker_id = id(worker)
            if worker_id not in seen:
                seen.add(worker_id)
                workers.append(worker)
        return workers

    def _shutdown_workers(self, workers) -> None:
        """Issue unfinished shutdown RPCs, drain them, and retain successful IDs."""
        completed = getattr(self, "_shutdown_worker_ids", set())
        self._shutdown_worker_ids = completed
        requests = []
        errors = []
        for worker in workers:
            worker_id = id(worker)
            if worker_id in completed:
                continue
            try:
                requests.append((worker_id, worker.shutdown.remote()))
            except BaseException as exc:
                errors.append(exc)
        for worker_id, request in requests:
            try:
                ray.get(request)
                completed.add(worker_id)
            except BaseException as exc:
                errors.append(exc)
        if errors:
            for error in errors[1:]:
                logger.error(
                    "Additional reward manager shutdown failure",
                    exc_info=(type(error), error, error.__traceback__),
                )
            raise errors[0]


def _completed_task_result(task: asyncio.Future):
    try:
        return task.result()
    except BaseException as exc:
        return exc
