# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import argparse
import asyncio
import datetime
import errno
import fcntl
import hashlib
import io
import json
import math
import os
import signal
import shutil
import socket
import struct
import threading
import time
import traceback

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from runtime.control import Heartbeat, HeartbeatStore
from runtime.data import dataset_batch_indices as _dataset_batch_indices
from runtime.errors import RuntimeConfigError, RuntimeDependencyError, RuntimeExecutionError
from runtime.semantic import (
    FileSemanticTransport,
    SemanticBackend,
    SemanticCapability,
    SemanticOperationNames,
    SemanticRequest,
    SemanticResult,
    recv_json_frame,
    send_json_frame,
)
from runtime.reshard import (
    dp_pp_tp_to_rank,
    merge_state_dict_fragments,
    rank_to_dp_pp_tp,
    reshard_optimizer_state_by_name,
    reshard_state_dict,
    route_state_dict_fragment,
    routed_weight_replicated_source_tp_rank,
    tp_shard_dim,
)
from runtime.workers.megatron_worker import MegatronBuildConfig, MegatronWorker
from runtime.worker_entry import _rank_context


_UCX_DIRECT_INIT_LOCK = threading.Lock()
_UCX_DIRECT_INITIALIZED = False
_UCX_DIRECT_PREWARM_MAGIC = b"RPEGUCXW"
_UCX_DIRECT_LOOP_LOCAL = threading.local()


def _begin_cuda_memory_observation(worker: Any) -> tuple[Any, int] | None:
    """
    Reset per-operation CUDA peaks while retaining the persistent baseline.
    """
    try:
        import torch

        model = getattr(worker, "model", None)
        if model is None:
            return None
        parameter = next(model.parameters())
        if parameter.device.type != "cuda":
            return None
        device = parameter.device.index
        if device is None:
            device = torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(device)
        return torch, device
    except Exception:
        return None


def _finish_cuda_memory_observation(observation: tuple[Any, int] | None) -> dict[str, int]:
    if observation is None:
        return {}
    torch, device = observation
    return {
        "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "cuda_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _trim_cuda_cache() -> dict[str, int | bool]:
    """
    Return unused CUDA allocator blocks to the physical GPU.

    Frozen reference/reward workers keep their BF16 parameters resident while releasing inference
    workspaces; publisher pruning drops completed device snapshots. In both cases PyTorch
    reservations are private to a process and otherwise make free HBM unavailable to its colocated
    peer.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_cache_trimmed": False, "cuda_cache_trimmed_bytes": 0}
        device = torch.cuda.current_device()
        reserved_before = int(torch.cuda.memory_reserved(device))
        torch.cuda.empty_cache()
        reserved_after = int(torch.cuda.memory_reserved(device))
        return {
            "cuda_cache_trimmed": True,
            "cuda_cache_reserved_before_bytes": reserved_before,
            "cuda_cache_reserved_after_bytes": reserved_after,
            "cuda_cache_trimmed_bytes": max(0, reserved_before - reserved_after),
        }
    except Exception:
        return {"cuda_cache_trimmed": False, "cuda_cache_trimmed_bytes": 0}


def _trim_cuda_cache_after_frozen_inference() -> dict[str, int | bool]:
    """
    Compatibility wrapper naming the primary inference-boundary caller.
    """
    return _trim_cuda_cache()


_UCX_DIRECT_ENDPOINT_CACHE: dict[tuple[int, str, str], Any] = {}
_UCX_DIRECT_EVENT_LOOP_LOCK = threading.Lock()
_UCX_DIRECT_EVENT_LOOP: asyncio.AbstractEventLoop | None = None
_TORCH_DIRECT_PROCESS_GROUP_CACHE: dict[tuple[str, int, str], tuple[Any, Any]] = {}
_TORCH_DIRECT_PROCESS_GROUP_CACHE_LOCK = threading.Lock()
_TORCH_DIRECT_PERSISTENT_PROCESS_GROUP_IDS: set[int] = set()
_UCX_TENSOR_STREAM_SCHEMA = "rpeg.weight_reshard_tensor_stream.v1"
_UCX_TENSOR_WEIGHT_SCHEMA = "rpeg.weight_tensor_version.v1"
_ACTIVATION_TENSOR_STREAM_SCHEMA = "rpeg.activation_tensor_stream.v1"
_ACTIVATION_TENSOR_RECORD_SCHEMA = "rpeg.activation_tensor_record.v1"
_UCX_TENSOR_BUCKET_BYTES = 256 * 1024 * 1024
_CUDA_IPC_TENSOR_MAGIC = b"RPEGIPC1"
_CUDA_IPC_MAX_METADATA_BYTES = 64 * 1024 * 1024
_CUDA_IPC_INLINE_CPU_TENSOR = "__rpeg_cuda_ipc_inline_cpu_tensor_v1__"


class _CudaByteBuffer:
    """
    Expose a contiguous torch CUDA tensor as bytes to UCX.
    """
    def __init__(self, tensor: Any) -> None:
        self.tensor = tensor
        self.__cuda_array_interface__ = {
            "shape": (int(tensor.numel()) * int(tensor.element_size()),),
            "strides": None,
            "typestr": "|u1",
            "data": (int(tensor.data_ptr()), False),
            "version": 3,
        }


class MegatronSemanticBackend:
    def __init__(self, *, worker_id: str, rank: int, world_size: int, state_root: str | Path) -> None:
        self.worker_id = worker_id
        self.rank = rank
        self.world_size = world_size
        self.worker = MegatronWorker()
        self.state_root = Path(state_root) / "state"
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.model_path: str | None = None
        self._model_name: str | None = None
        self.tokenizer = None
        self.tokenizer_path: str | None = None
        self._prompt_cache: dict[str, list[str]] = {}
        self._preference_cache: dict[str, list[dict[str, str]]] = {}
        self._phased_generation_prompts: dict[str, list[str] | None] = {}
        self._activation_cache: dict[tuple[int, str], dict[str, Any]] = {}
        self._activation_layout_cache: dict[
            tuple[int, str, tuple[int, int, int]],
            dict[str, Any],
        ] = {}
        self._activation_store = "filesystem"
        self._activation_cache_device = "cpu"
        self._activation_cache_max_bytes = 0
        self._activation_cache_cuda_placements = 0
        self._activation_cache_cpu_fallbacks = 0
        self._weight_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._weight_blob_cache: dict[tuple[str, int], tuple[bytes, str]] = {}
        # Versions needed by not-yet-completed consumers are liveness pins, not part of the ordinary
        # keep-latest cache window. In asynchronous schedules an old pinned version can coexist with
        # several newer publications; evicting it by age changes the requested RPEG tau.
        self._weight_cache_pins: dict[str, set[int]] = {}
        self._active_weight_versions: dict[str, int] = {}
        self._released_weight_states: dict[str, dict[str, Any]] = {}
        self._weight_cache_keep_latest = 8
        self._background_transfer_lock = threading.Lock()
        self._background_transfers: list[threading.Thread] = []
        self._background_transfer_errors: list[BaseException] = []
        self._host_reshard_group = None
        self._test_failure_injected = False

    def capabilities(self) -> dict[str, SemanticCapability]:
        supported = {
            SemanticOperationNames.CAPABILITIES,
            SemanticOperationNames.BUILD_MODEL,
            SemanticOperationNames.LOAD_HF_CHECKPOINT,
            SemanticOperationNames.LM_INFERENCE_STEP,
            SemanticOperationNames.LM_TRAINING_STEP,
            SemanticOperationNames.SAVE_DISTRIBUTED_CHECKPOINT,
            SemanticOperationNames.LOAD_DISTRIBUTED_CHECKPOINT,
            SemanticOperationNames.SAVE_WEIGHT_VERSION,
            SemanticOperationNames.LOAD_WEIGHT_VERSION,
            SemanticOperationNames.PRUNE_WEIGHT_VERSIONS,
            SemanticOperationNames.RESHARD_WEIGHT_VERSION,
            SemanticOperationNames.SWITCH_MODEL_LAYOUT,
            SemanticOperationNames.ENSURE_OPERATIONAL_WEIGHTS,
            SemanticOperationNames.RELEASE_OPERATIONAL_WEIGHTS,
            SemanticOperationNames.SEND_WEIGHT_VERSION,
            SemanticOperationNames.RECV_WEIGHT_VERSION,
            SemanticOperationNames.TRANSFER_ACTIVATION,
            SemanticOperationNames.REPARTITION_LOCAL_ACTIVATION,
            SemanticOperationNames.SEND_ACTIVATION,
            SemanticOperationNames.RECV_ACTIVATION,
            SemanticOperationNames.RELEASE_ACTIVATIONS,
            SemanticOperationNames.SEND_DIRECT_PREWARM,
            SemanticOperationNames.RECV_DIRECT_PREWARM,
            SemanticOperationNames.SHUTDOWN,
        }
        ppo_ops = {
            SemanticOperationNames.PPO_ACTOR_GENERATE,
            SemanticOperationNames.PPO_REFERENCE_LOGPROBS,
            SemanticOperationNames.PPO_REWARD_SCORE,
            SemanticOperationNames.PPO_CRITIC_VALUES,
            SemanticOperationNames.PPO_ACTOR_TRAIN,
            SemanticOperationNames.PPO_CRITIC_TRAIN,
            SemanticOperationNames.PPO_AC_TRAIN,
            SemanticOperationNames.PPO_FUSED_ACTOR_CRITIC_TRAIN,
            SemanticOperationNames.GRPO_ACTOR_GENERATE,
            SemanticOperationNames.GRPO_REFERENCE_LOGPROBS,
            SemanticOperationNames.GRPO_REWARD_SCORE,
            SemanticOperationNames.GRPO_ACTOR_TRAIN,
            SemanticOperationNames.DPO_ACTOR_LOGPROBS,
            SemanticOperationNames.DPO_REFERENCE_LOGPROBS,
            SemanticOperationNames.DPO_ACTOR_TRAIN,
        }
        out = {
            op: SemanticCapability(op, True)
            for op in sorted(supported | ppo_ops)
        }
        return out

    def execute(self, request: SemanticRequest) -> SemanticResult:
        try:
            self._raise_background_transfer_errors()
            self._inject_rank_failure_for_test(request)
            cuda_memory = _begin_cuda_memory_observation(self.worker)
            counters = self._execute_ok(request)
            counters.update(_finish_cuda_memory_observation(cuda_memory))
            counters.update({
                "activation_cuda_cache_enabled": int(
                    getattr(self, "_activation_cache_device", "cpu") in {"cuda", "auto"}
                    and getattr(self, "_activation_cache_max_bytes", 0) > 0
                ),
                "activation_cuda_cache_bytes_current": self._cached_activation_cuda_bytes(),
                "activation_cuda_cache_bytes_limit": getattr(
                    self, "_activation_cache_max_bytes", 0
                ),
                "activation_cuda_cache_placements": getattr(
                    self, "_activation_cache_cuda_placements", 0
                ),
                "activation_cuda_cache_cpu_fallbacks": getattr(
                    self, "_activation_cache_cpu_fallbacks", 0
                ),
                "torch_direct_process_group_cache_size": len(
                    _TORCH_DIRECT_PROCESS_GROUP_CACHE
                ),
            })
            group_counters = getattr(self.worker, "_model_parallel_group_counters", {})
            counters.update(
                {
                    f"megatron_process_groups_{name}": int(value)
                    for name, value in group_counters.items()
                }
            )
        except Exception as exc:
            # The controller receives a bounded error string, while the rank's stderr retains the
            # complete stack needed to diagnose failures in Megatron/PyTorch internals on
            # distributed production runs.
            traceback.print_exc()
            return SemanticResult(
                request_id=request.request_id,
                worker_id=self.worker_id,
                rank=self.rank,
                world_size=self.world_size,
                status="error",
                message=str(exc),
            )
        return SemanticResult(
            request_id=request.request_id,
            worker_id=self.worker_id,
            rank=self.rank,
            world_size=self.world_size,
            status="ok",
            counters=counters,
        )

    def _inject_rank_failure_for_test(self, request: SemanticRequest) -> None:
        """
        Opt-in fault injection used by distributed teardown validation.
        """
        operation = os.environ.get("RPEG_TEST_FAIL_OPERATION")
        if not operation or self._test_failure_injected or request.operation != operation:
            return
        requested_rank = int(os.environ.get("RPEG_TEST_FAIL_RANK", "-1"))
        node_match = os.environ.get("RPEG_TEST_FAIL_NODE_CONTAINS", "")
        if self.rank != requested_rank or (node_match and node_match not in request.node_id):
            return
        self._test_failure_injected = True
        raise RuntimeExecutionError(
            f"injected semantic rank failure: operation={request.operation}, rank={self.rank}"
        )

    def _launch_background_transfer(self, operation: Callable[[], None], *, name: str) -> None:
        def run() -> None:
            try:
                operation()
            except BaseException as exc:
                with self._background_transfer_lock:
                    self._background_transfer_errors.append(exc)

        thread = threading.Thread(
            target=run,
            name=f"rpeg-{self.worker_id}-rank{self.rank}-{name}",
            daemon=False,
        )
        with self._background_transfer_lock:
            self._background_transfers = [item for item in self._background_transfers if item.is_alive()]
            self._background_transfers.append(thread)
        thread.start()

    def _raise_background_transfer_errors(self) -> None:
        with self._background_transfer_lock:
            if not self._background_transfer_errors:
                return
            errors = list(self._background_transfer_errors)
            self._background_transfer_errors.clear()
        details = "; ".join(str(error) for error in errors)
        raise RuntimeExecutionError(f"background semantic transfer failed: {details}")

    def execute_generation_batch(self, requests: list[SemanticRequest]) -> dict[str, SemanticResult]:
        if not requests:
            return {}
        if len(requests) == 1:
            request = requests[0]
            return {request.request_id: self.execute(request)}
        try:
            counters_by_request = self._ppo_actor_generate_batch(requests)
        except RuntimeConfigError as exc:
            if "equal padded prompt widths" in str(exc):
                return {request.request_id: self.execute(request) for request in requests}
            return {
                request.request_id: SemanticResult(
                    request_id=request.request_id,
                    worker_id=self.worker_id,
                    rank=self.rank,
                    world_size=self.world_size,
                    status="error",
                    message=str(exc),
                )
                for request in requests
            }
        except Exception as exc:
            traceback.print_exc()
            return {
                request.request_id: SemanticResult(
                    request_id=request.request_id,
                    worker_id=self.worker_id,
                    rank=self.rank,
                    world_size=self.world_size,
                    status="error",
                    message=str(exc),
                )
                for request in requests
            }
        return {
            request.request_id: SemanticResult(
                request_id=request.request_id,
                worker_id=self.worker_id,
                rank=self.rank,
                world_size=self.world_size,
                status="ok",
                counters=counters_by_request[request.request_id],
            )
            for request in requests
        }

    def generation_batch_key(self, request: SemanticRequest) -> tuple[Any, ...] | None:
        if request.operation not in {
            SemanticOperationNames.PPO_ACTOR_GENERATE,
            SemanticOperationNames.GRPO_ACTOR_GENERATE,
        }:
            return None
        payload = request.payload
        if payload.get("artifact_part") is not None:
            return None
        do_sample = bool(payload.get("do_sample", True))
        seed = self._generation_seed(request) if do_sample else None
        samples_per_prompt = int(payload.get("samples_per_prompt", 1))
        return (
            request.operation,
            int(payload["generation_length"]),
            str(payload.get("generation_backend", "full_sequence")),
            do_sample,
            float(payload.get("temperature", 1.0)),
            float(payload.get("top_p", 1.0)),
            int(payload.get("top_k", 0)),
            seed,
            samples_per_prompt,
        )

    def _execute_ok(self, request: SemanticRequest) -> dict[str, Any]:
        op = request.operation
        payload = request.payload
        if self.worker.config is not None and payload.get("microbatch") is not None:
            requested_microbatch = int(payload["microbatch"])
            if requested_microbatch <= 0:
                raise RuntimeConfigError("semantic operation microbatch must be positive")
            if requested_microbatch != self.worker.config.microbatch:
                self.worker.config = replace(self.worker.config, microbatch=requested_microbatch)
        if op == SemanticOperationNames.CAPABILITIES:
            return {
                "capabilities": [cap.to_dict() for cap in self.capabilities().values()],
            }
        if op == SemanticOperationNames.BUILD_MODEL:
            self.model_path = str(payload["model_path"])
            self._model_name = str(payload.get("model") or self.worker_id)
            if payload.get("tokenizer_path") is not None:
                self.tokenizer_path = str(payload["tokenizer_path"])
            self._activation_store = str(payload.get("activation_store", "filesystem"))
            self._activation_cache_device = str(payload.get("activation_cache_device", "cpu"))
            self._activation_cache_max_bytes = int(payload.get("activation_cache_max_bytes", 0))
            if self._activation_cache_device not in {"cpu", "cuda", "auto"}:
                raise RuntimeConfigError(
                    f"unsupported activation cache device {self._activation_cache_device!r}"
                )
            if self._activation_cache_max_bytes < 0:
                raise RuntimeConfigError("activation_cache_max_bytes must be >= 0")
            if self._activation_cache_device == "cuda" and self._activation_cache_max_bytes <= 0:
                raise RuntimeConfigError(
                    "CUDA activation caching requires activation_cache_max_bytes > 0"
                )
            self._weight_cache_keep_latest = int(payload.get("weight_cache_keep_latest", 8))
            cfg = MegatronBuildConfig(
                model_path=self.model_path,
                parallelism=(
                    int(payload["dp"]),
                    int(payload["pp"]),
                    int(payload["tp"]),
                ),
                dtype=str(payload.get("dtype", "bf16")),
                use_transformer_engine=bool(payload.get("use_transformer_engine", True)),
                perform_initialization=bool(payload.get("perform_initialization", False)),
                max_sequence_length=None
                if payload.get("max_sequence_length") is None
                else int(payload["max_sequence_length"]),
                activation_checkpointing=bool(payload.get("activation_checkpointing", False)),
                learning_rate=float(payload.get("learning_rate", 1e-5)),
                weight_decay=float(payload.get("weight_decay", 0.0)),
                grad_clip=float(payload.get("grad_clip", 1.0)),
                distributed_optimizer=str(payload.get("distributed_optimizer", "none")),
                zero2_bucket_bytes=int(payload.get("zero2_bucket_bytes", 64 * 1024 * 1024)),
                attention_backend=str(payload.get("attention_backend", "unfused")),
                zero2_max_inflight_buckets=int(payload.get("zero2_max_inflight_buckets", 2)),
                microbatch=int(payload.get("microbatch", 1)),
                auxiliary_heads=tuple(str(x) for x in payload.get("auxiliary_heads", ())),
                value_head_prefixes=tuple(str(x) for x in payload.get("value_head_prefixes", ())),
                reward_head_prefixes=tuple(str(x) for x in payload.get("reward_head_prefixes", ())),
                static_kv_cache_max_entries=int(payload.get("static_kv_cache_max_entries", 4)),
                static_kv_cache_batch_bucket=int(payload.get("static_kv_cache_batch_bucket", 0)),
                static_kv_cache_sequence_bucket=int(payload.get("static_kv_cache_sequence_bucket", 0)),
                pipeline_nccl_max_ctas=(
                    None
                    if payload.get("pipeline_nccl_max_ctas") is None
                    else int(payload["pipeline_nccl_max_ctas"])
                ),
                pipeline_nccl_cta_policy=(
                    None
                    if payload.get("pipeline_nccl_cta_policy") is None
                    else int(payload["pipeline_nccl_cta_policy"])
                ),
                seed=int(payload.get("seed", 1234)),
            )
            self.worker.build_model(cfg)
            return dict(self.worker.built_model_info().__dict__)
        if op == SemanticOperationNames.LOAD_HF_CHECKPOINT:
            report = self.worker.load_hf_checkpoint(
                str(payload["model_path"]),
                strict=bool(payload.get("strict", False)),
            )
            return {"loaded": True, **report}
        if op == SemanticOperationNames.LM_INFERENCE_STEP:
            input_ids = self._input_ids(payload)
            result = self.worker.inference_step(input_ids)
            return {"forward_items": len(result) if hasattr(result, "__len__") else 1}
        if op == SemanticOperationNames.LM_TRAINING_STEP:
            input_ids = self._input_ids(payload)
            loss = self.worker.training_step(
                input_ids,
                learning_rate=None if payload.get("learning_rate") is None else float(payload["learning_rate"]),
            )
            return {"loss": loss}
        if op == SemanticOperationNames.SAVE_DISTRIBUTED_CHECKPOINT:
            rematerialized_for_checkpoint = not self.worker.operational_weights_materialized
            self._ensure_operational_weights(
                replace(
                    request,
                    payload={**payload, "model": self._model_name or self.worker_id},
                )
            )
            path = self.worker.save_distributed_checkpoint(
                str(payload["path"]),
                step=int(payload.get("step", request.step)),
                include_optimizer_state=bool(payload.get("include_optimizer_state", True)),
            )
            released_bytes = 0
            if rematerialized_for_checkpoint:
                released_bytes = self.worker.release_operational_weights()
            return {
                "checkpoint_path": str(path),
                "operational_weights_released": rematerialized_for_checkpoint,
                "operational_weights_released_bytes": released_bytes,
            }
        if op == SemanticOperationNames.LOAD_DISTRIBUTED_CHECKPOINT:
            if not self.worker.operational_weights_materialized:
                self.worker.materialize_operational_weights()
            self.worker.load_distributed_checkpoint(
                str(payload["path"]),
                strict=bool(payload.get("strict", True)),
            )
            return {"checkpoint_path": str(payload["path"]), "loaded": True}
        if op == SemanticOperationNames.SAVE_WEIGHT_VERSION:
            return self._save_weight_version(request)
        if op == SemanticOperationNames.LOAD_WEIGHT_VERSION:
            return self._load_weight_version(request)
        if op == SemanticOperationNames.PRUNE_WEIGHT_VERSIONS:
            return self._prune_weight_versions(request)
        if op == SemanticOperationNames.RESHARD_WEIGHT_VERSION:
            return self._reshard_weight_version(request)
        if op == SemanticOperationNames.SWITCH_MODEL_LAYOUT:
            return self._switch_model_layout(request)
        if op == SemanticOperationNames.ENSURE_OPERATIONAL_WEIGHTS:
            return self._ensure_operational_weights(request)
        if op == SemanticOperationNames.RELEASE_OPERATIONAL_WEIGHTS:
            return self._release_operational_weights(request)
        if op == SemanticOperationNames.SEND_WEIGHT_VERSION:
            return self._send_weight_version_direct(request)
        if op == SemanticOperationNames.RECV_WEIGHT_VERSION:
            return self._recv_weight_version_direct(request)
        if op == SemanticOperationNames.TRANSFER_ACTIVATION:
            return self._transfer_activation(request)
        if op == SemanticOperationNames.REPARTITION_LOCAL_ACTIVATION:
            return self._repartition_local_activation(request)
        if op == SemanticOperationNames.SEND_ACTIVATION:
            return self._send_activation_direct(request)
        if op == SemanticOperationNames.RECV_ACTIVATION:
            return self._recv_activation_direct(request)
        if op == SemanticOperationNames.RELEASE_ACTIVATIONS:
            return self._release_step_activations(request)
        if op == SemanticOperationNames.SEND_DIRECT_PREWARM:
            return self._send_direct_prewarm(request)
        if op == SemanticOperationNames.RECV_DIRECT_PREWARM:
            return self._recv_direct_prewarm(request)
        if op == SemanticOperationNames.SHUTDOWN:
            self.shutdown()
            return {"shutdown": True}
        if payload.get("parallelism") is not None:
            expected_parallelism = tuple(int(x) for x in payload["parallelism"])
            actual_parallelism = self._parallelism()
            if expected_parallelism != actual_parallelism:
                raise RuntimeConfigError(
                    f"{op} requires layout {expected_parallelism}, active layout is {actual_parallelism}"
                )
        if op == SemanticOperationNames.PPO_ACTOR_GENERATE:
            return self._ppo_actor_generate(request, samples_per_prompt=int(payload.get("samples_per_prompt", 1)))
        if op == SemanticOperationNames.GRPO_ACTOR_GENERATE:
            return self._ppo_actor_generate(request, samples_per_prompt=int(payload.get("samples_per_prompt", 1)))
        if op == SemanticOperationNames.PPO_REFERENCE_LOGPROBS:
            return self._ppo_reference_logprobs(request)
        if op == SemanticOperationNames.GRPO_REFERENCE_LOGPROBS:
            return self._ppo_reference_logprobs(request)
        if op == SemanticOperationNames.PPO_REWARD_SCORE:
            return self._ppo_reward_score(request)
        if op == SemanticOperationNames.GRPO_REWARD_SCORE:
            return self._ppo_reward_score(request)
        if op == SemanticOperationNames.PPO_CRITIC_VALUES:
            return self._ppo_critic_values(request)
        if op == SemanticOperationNames.PPO_ACTOR_TRAIN:
            return self._ppo_actor_train(request)
        if op == SemanticOperationNames.PPO_AC_TRAIN:
            return self._ppo_ac_train(request)
        if op == SemanticOperationNames.PPO_FUSED_ACTOR_CRITIC_TRAIN:
            return self._ppo_fused_actor_critic_train(request)
        if op == SemanticOperationNames.GRPO_ACTOR_TRAIN:
            return self._grpo_actor_train(request)
        if op == SemanticOperationNames.PPO_CRITIC_TRAIN:
            return self._ppo_critic_train(request)
        if op == SemanticOperationNames.DPO_ACTOR_LOGPROBS:
            return self._dpo_logprobs(request, name="log_pi")
        if op == SemanticOperationNames.DPO_REFERENCE_LOGPROBS:
            return self._dpo_logprobs(request, name="log_ref")
        if op == SemanticOperationNames.DPO_ACTOR_TRAIN:
            return self._dpo_actor_train(request)

        cap = self.capabilities().get(op)
        if cap is not None and not cap.supported:
            raise RuntimeConfigError(f"unsupported semantic operation {op}: {cap.reason}")
        raise RuntimeConfigError(f"unknown semantic operation {op!r}")

    def _input_ids(self, payload: dict[str, Any]):
        import torch

        if "input_ids" in payload:
            return torch.tensor(payload["input_ids"], dtype=torch.long)
        batch = int(payload.get("batch_size", 1))
        seq = int(payload.get("seq_length", 8))
        vocab = int(payload.get("vocab_size", 128))
        return torch.randint(0, vocab, (batch, seq), dtype=torch.long)

    def _activation_name(self, name: str) -> str:
        out = str(name)
        if not out or "/" in out or "\\" in out or out in {".", ".."}:
            raise RuntimeConfigError(f"invalid activation artifact name {name!r}")
        return out

    def _activation_worker_id(self, worker_id: str) -> str:
        out = str(worker_id)
        if not out or "/" in out or "\\" in out or out in {".", ".."}:
            raise RuntimeConfigError(f"invalid activation worker id {worker_id!r}")
        return out

    def _activation_step_path(self, worker_id: str, step: int) -> Path:
        """
        Resolve an activation directory without mutating shared storage.
        """
        return self.state_root / "activations" / self._activation_worker_id(worker_id) / f"step_{step:08d}"

    def _activation_step_dir(self, worker_id: str, step: int) -> Path:
        """
        Resolve and create an activation directory for a writer path.
        """
        path = self._activation_step_path(worker_id, step)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _activation_path_for(self, worker_id: str, step: int, name: str) -> Path:
        return self._activation_step_dir(worker_id, step) / f"{self._activation_name(name)}.pt"

    def _tensor_path(self, step: int, name: str) -> Path:
        return self._activation_path_for(self.worker_id, step, name)

    def _parallelism(self) -> tuple[int, int, int]:
        cfg = getattr(getattr(self, "worker", None), "config", None)
        if cfg is None:
            return (1, 1, 1)
        return tuple(int(x) for x in cfg.parallelism)

    def _dp_pp_tp_rank(self) -> tuple[int, int, int]:
        _dp, pp, tp = self._parallelism()
        return rank_to_dp_pp_tp(self.rank, pp, tp)

    def _dp_rank(self) -> int:
        return self._dp_pp_tp_rank()[0]

    def _dp_size(self) -> int:
        return self._parallelism()[0]

    def _is_dp_artifact_writer(self) -> bool:
        _dp_rank, pp_rank, tp_rank = self._dp_pp_tp_rank()
        return pp_rank == 0 and tp_rank == 0

    def _request_parallelism(
        self,
        payload: dict[str, Any],
        role: str,
        *,
        expected_world_size: int | None = None,
    ) -> tuple[int, int, int]:
        """
        Return the layout which owns one side of an activation transfer.

        Persistent workers can switch layouts between producing and moving an artifact. Falling back
        to the active layout preserves compatibility with older control messages, while new
        direct-transfer requests carry both layouts explicitly. A receiver also parses the remote
        source layout, whose world size can legitimately differ from its own; callers doing
        local-rank arithmetic pass expected_world_size explicitly.
        """
        raw = payload.get(f"{role}_parallelism")
        if raw is None:
            return self._parallelism()
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise RuntimeConfigError(f"invalid {role}_parallelism {raw!r}")
        parallelism = tuple(int(value) for value in raw)
        if any(value <= 0 for value in parallelism):
            raise RuntimeConfigError(f"invalid {role}_parallelism {raw!r}")
        if (
            expected_world_size is not None
            and math.prod(parallelism) != int(expected_world_size)
        ):
            raise RuntimeConfigError(
                f"{role}_parallelism {parallelism} has world size {math.prod(parallelism)}, "
                f"expected worker world size {expected_world_size}"
            )
        return parallelism

    def _request_dp_pp_tp_rank(
        self,
        payload: dict[str, Any],
        role: str,
    ) -> tuple[int, int, int]:
        _dp, pp, tp = self._request_parallelism(
            payload,
            role,
            expected_world_size=self.world_size,
        )
        return rank_to_dp_pp_tp(self.rank, pp, tp)

    def _request_dp_rank(self, payload: dict[str, Any], role: str) -> int:
        return self._request_dp_pp_tp_rank(payload, role)[0]

    def _is_request_dp_artifact_writer(self, payload: dict[str, Any], role: str) -> bool:
        _dp_rank, pp_rank, tp_rank = self._request_dp_pp_tp_rank(payload, role)
        return pp_rank == 0 and tp_rank == 0

    def _local_batch_range(self, global_batch: int) -> tuple[int, int]:
        dp = self._dp_size()
        if global_batch % dp != 0:
            raise RuntimeConfigError(f"global batch {global_batch} must be divisible by data parallel size {dp}")
        per_dp = global_batch // dp
        start = self._dp_rank() * per_dp
        return start, start + per_dp

    def _artifact_shard_path_for(self, worker_id: str, step: int, name: str, dp_rank: int) -> Path:
        return self._activation_step_dir(worker_id, step) / f"{self._activation_name(name)}.dp{int(dp_rank):05d}.pt"

    def _artifact_shard_metadata_path_for(self, worker_id: str, step: int, name: str, dp_rank: int) -> Path:
        return self._activation_step_dir(worker_id, step) / f"{self._activation_name(name)}.dp{int(dp_rank):05d}.json"

    def _artifact_manifest_path_for(self, worker_id: str, step: int, name: str) -> Path:
        return self._activation_step_dir(worker_id, step) / f"{self._activation_name(name)}.manifest.json"

    def _save_local_artifact(
        self,
        step: int,
        name: str,
        data: dict[str, Any],
        *,
        prune_producer_cache: bool = False,
    ) -> None:
        if not self._is_dp_artifact_writer():
            return
        # Megatron's static-KV inference path can return PyTorch inference tensors. They are
        # immutable inference products, but downstream PPO training may use token ids/masks in
        # operations which save inputs for backward. Clone such tensors outside inference mode at
        # the semantic artifact boundary, so a colocated inference/training worker has the same
        # contract as a serialized cross-worker transfer.
        key = (int(step), self._activation_name(name))
        data = self._place_activation_artifact(key, data)
        self._activation_cache[key] = data
        layout_cache = getattr(self, "_activation_layout_cache", None)
        if layout_cache is None:
            self._activation_layout_cache = {}
            layout_cache = self._activation_layout_cache
        layout_cache[
            (key[0], key[1], self._parallelism())
        ] = data
        if prune_producer_cache:
            self._prune_activation_cache(int(step))
        if self._activation_store == "memory":
            return
        batch_size = _optional_artifact_batch_size(data)
        global_batch_size = batch_size * self._dp_size() if batch_size is not None else None
        global_start = self._dp_rank() * batch_size if batch_size is not None else None
        global_end = global_start + batch_size if global_start is not None and batch_size is not None else None
        self._write_local_artifact_payload(
            step=step,
            name=name,
            data=data,
            transport="shared_filesystem",
            global_batch_size=global_batch_size,
            global_start=global_start,
            global_end=global_end,
        )

    def _write_local_artifact_payload(
        self,
        *,
        step: int,
        name: str,
        data: dict[str, Any],
        transport: str,
        global_batch_size: int | None,
        global_start: int | None,
        global_end: int | None,
    ) -> Path:
        import torch

        if not self._is_dp_artifact_writer():
            raise RuntimeConfigError("only DP artifact writer ranks can publish activation artifacts")
        dp_rank = self._dp_rank()
        local_batch_size = _optional_artifact_batch_size(data)
        path = self._artifact_shard_path_for(self.worker_id, step, name, dp_rank)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = dict(data)
        payload["_rpeg_shard"] = {
            "schema_version": "rpeg.activation_shard.v1",
            "worker_id": self.worker_id,
            "step": step,
            "name": name,
            "dp_rank": dp_rank,
            "dp_size": self._dp_size(),
        }
        torch.save(payload, tmp)
        tmp.replace(path)

        stat = path.stat()
        self._write_artifact_shard_metadata(
            worker_id=self.worker_id,
            step=step,
            name=name,
            dp_rank=dp_rank,
            dp_size=self._dp_size(),
            bytes_=stat.st_size,
            sha256=_file_sha256(path),
            transport=transport,
            local_batch_size=local_batch_size,
            global_batch_size=global_batch_size,
            global_start=global_start,
            global_end=global_end,
        )
        if dp_rank == 0:
            self._write_artifact_manifest(
                worker_id=self.worker_id,
                step=step,
                name=name,
                dp_size=self._dp_size(),
                transport=transport,
                global_batch_size=global_batch_size,
            )
        return path

    def _cached_activation_cuda_bytes(
        self,
        *,
        exclude_key: tuple[int, str] | None = None,
    ) -> int:
        artifacts: list[dict[str, Any]] = []
        artifacts.extend(
            artifact
            for key, artifact in self._activation_cache.items()
            if key != exclude_key
        )
        artifacts.extend(
            artifact
            for key, artifact in getattr(self, "_activation_layout_cache", {}).items()
            if exclude_key is None or key[:2] != exclude_key
        )
        return _artifact_tensor_storage_bytes(artifacts, device_type="cuda")

    def _place_activation_artifact(
        self,
        key: tuple[int, str],
        artifact: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Place one semantic artifact under the configured hard HBM limit.
        """
        import torch

        cache_device = getattr(self, "_activation_cache_device", "cpu")
        cache_max_bytes = getattr(self, "_activation_cache_max_bytes", 0)
        wants_cuda = cache_device == "cuda" or (
            cache_device == "auto"
            and cache_max_bytes > 0
            and torch.cuda.is_available()
        )
        candidate_bytes = _artifact_tensor_storage_bytes([artifact])
        if wants_cuda and torch.cuda.is_available():
            current_bytes = self._cached_activation_cuda_bytes(exclude_key=key)
            if current_bytes + candidate_bytes <= cache_max_bytes:
                self._activation_cache_cuda_placements = (
                    getattr(self, "_activation_cache_cuda_placements", 0) + 1
                )
                return _materialize_trainable_artifact(
                    artifact,
                    device=torch.device("cuda", torch.cuda.current_device()),
                )
            self._activation_cache_cpu_fallbacks = (
                getattr(self, "_activation_cache_cpu_fallbacks", 0) + 1
            )
        return _materialize_trainable_artifact(artifact, device=torch.device("cpu"))

    def _write_artifact_shard_metadata(
        self,
        *,
        worker_id: str,
        step: int,
        name: str,
        dp_rank: int,
        dp_size: int,
        bytes_: int,
        sha256: str,
        transport: str,
        local_batch_size: int | None,
        global_batch_size: int | None,
        global_start: int | None,
        global_end: int | None,
    ) -> None:
        from runtime.semantic import write_json_atomic

        metadata: dict[str, Any] = {
            "schema_version": "rpeg.activation_shard_meta.v1",
            "worker_id": worker_id,
            "step": int(step),
            "name": name,
            "dp_rank": int(dp_rank),
            "dp_size": int(dp_size),
            "bytes": int(bytes_),
            "sha256": sha256,
            "transport": transport,
        }
        if local_batch_size is not None:
            metadata["local_batch_size"] = int(local_batch_size)
        if global_batch_size is not None:
            metadata["global_batch_size"] = int(global_batch_size)
        if global_start is not None and global_end is not None:
            metadata["global_start"] = int(global_start)
            metadata["global_end"] = int(global_end)
        write_json_atomic(
            self._artifact_shard_metadata_path_for(worker_id, step, name, dp_rank),
            metadata,
        )

    def _write_artifact_manifest(
        self,
        *,
        worker_id: str,
        step: int,
        name: str,
        dp_size: int,
        transport: str,
        global_batch_size: int | None,
    ) -> None:
        from runtime.semantic import write_json_atomic

        manifest: dict[str, Any] = {
            "schema_version": "rpeg.activation_manifest.v1",
            "worker_id": worker_id,
            "step": int(step),
            "name": name,
            "dp_size": int(dp_size),
            "transport": transport,
        }
        if global_batch_size is not None:
            manifest["global_batch_size"] = int(global_batch_size)
        write_json_atomic(self._artifact_manifest_path_for(worker_id, step, name), manifest)

    def _load_local_artifact(
        self,
        step: int,
        name: str,
        *,
        synchronize_model_parallel: bool = True,
    ) -> dict[str, Any]:
        import torch

        key = (int(step), self._activation_name(name))
        if synchronize_model_parallel and self._activation_store in {"memory", "hybrid"}:
            _dp, pp, tp = self._parallelism()
            if pp * tp > 1 and torch.distributed.is_initialized():
                artifact = self._activation_cache.get(key) if self._is_dp_artifact_writer() else None
                if artifact is None and self._is_dp_artifact_writer() and self._activation_store == "hybrid":
                    shard_path = self._artifact_shard_path_for(self.worker_id, step, name, self._dp_rank())
                    if shard_path.exists():
                        artifact = torch.load(shard_path, map_location="cpu", weights_only=False)
                from megatron.core import parallel_state

                group = parallel_state.get_model_parallel_group()
                source_rank = torch.distributed.get_global_rank(group, 0)
                record, _wire_bytes = _torch_model_parallel_broadcast_tensor_record(
                    None if artifact is None else _activation_artifact_to_tensor_record(artifact),
                    group=group,
                    source_rank=source_rank,
                    expected_schema=_ACTIVATION_TENSOR_STREAM_SCHEMA,
                )
                artifact = _activation_artifact_from_tensor_record(record)
                if not self._is_dp_artifact_writer():
                    artifact = self._place_activation_artifact(key, artifact)
                self._activation_cache[key] = artifact
                layout_cache = getattr(self, "_activation_layout_cache", None)
                if layout_cache is None:
                    self._activation_layout_cache = {}
                    layout_cache = self._activation_layout_cache
                layout_cache[
                    (key[0], key[1], self._parallelism())
                ] = artifact
                return artifact
        cached = self._activation_cache.get(key)
        if cached is not None:
            return cached
        shard_path = self._artifact_shard_path_for(self.worker_id, step, name, self._dp_rank())
        if shard_path.exists():
            artifact = torch.load(shard_path, map_location="cpu", weights_only=False)
        else:
            artifact = self._load_tensor_file(step, name)
        artifact = self._place_activation_artifact(key, artifact)
        self._activation_cache[key] = artifact
        layout_cache = getattr(self, "_activation_layout_cache", None)
        if layout_cache is None:
            self._activation_layout_cache = {}
            layout_cache = self._activation_layout_cache
        layout_cache[
            (key[0], key[1], self._parallelism())
        ] = artifact
        return artifact

    def _prune_activation_cache(self, current_step: int) -> None:
        """
        Legacy opt-in pruning for producers with externally proven last use.

        The static scheduler may execute future inference steps before it transfers an older
        artifact (notably for tau>0 shadow execution). Therefore age alone is not a valid liveness
        test. Consumer artifacts are released explicitly after their training operation; producer
        pruning stays opt-in until the controller has a last-use protocol.
        """
        oldest = int(current_step) - 2
        for key in tuple(self._activation_cache):
            if key[0] < oldest:
                self._activation_cache.pop(key, None)
        layout_cache = getattr(self, "_activation_layout_cache", {})
        for key in tuple(layout_cache):
            if key[0] < oldest:
                layout_cache.pop(key, None)

    def _release_local_artifacts(self, step: int, *names: str) -> None:
        """
        Release named local cache entries after graph-proven last use.
        """
        for name in names:
            key = (int(step), self._activation_name(name))
            self._activation_cache.pop(key, None)
            layout_cache = getattr(self, "_activation_layout_cache", {})
            for layout_key in tuple(layout_cache):
                if layout_key[:2] == key:
                    layout_cache.pop(layout_key, None)

    def _release_step_activations(self, request: SemanticRequest) -> dict[str, Any]:
        """
        Bound semantic-artifact lifetime at the all-trainers barrier.

        A PPO artifact can have multiple local and remote consumers. Training kernels therefore
        cannot infer last use from their own completion. The graph controller issues this operation
        only after every training consumer for the logical step has completed.
        """
        step = int(request.payload.get("step", request.step))
        cache_keys = [key for key in self._activation_cache if key[0] == step]
        layout_keys = [key for key in self._activation_layout_cache if key[0] == step]
        for key in cache_keys:
            self._activation_cache.pop(key, None)
        for key in layout_keys:
            self._activation_layout_cache.pop(key, None)

        persisted_bytes = 0
        persisted_files = 0
        # Path resolution must be side-effect free here. Every model-parallel rank executes this
        # semantic operation, while only rank 0 owns persisted-file deletion. If nonzero ranks mkdir
        # concurrently with rank-0 rmtree, pathlib can observe a parent that vanishes between its
        # existence check and mkdir and raise FileExistsError/ENOENT.
        step_dir = self._activation_step_path(self.worker_id, step)
        remove_persisted = bool(request.payload.get("remove_persisted", True))
        if self.rank == 0 and remove_persisted and step_dir.exists():
            for path in step_dir.rglob("*"):
                if path.is_file():
                    persisted_files += 1
                    persisted_bytes += path.stat().st_size
            shutil.rmtree(step_dir)
        return {
            "step": step,
            "activation_cache_entries_released": len(cache_keys),
            "activation_layout_entries_released": len(layout_keys),
            "activation_persisted_files_released": persisted_files,
            "activation_persisted_bytes_released": persisted_bytes,
            "rank": self.rank,
        }

    def _load_local_artifact_for_layout(
        self,
        step: int,
        name: str,
        parallelism: tuple[int, int, int],
    ) -> dict[str, Any]:
        key = (int(step), self._activation_name(name), tuple(parallelism))
        layout_cache = getattr(self, "_activation_layout_cache", None)
        if layout_cache is not None and key in layout_cache:
            return layout_cache[key]
        # Compatibility for artifacts produced before layout-versioned cache insertion. Register the
        # fallback immediately so a subsequent destination materialization cannot destroy its source
        # view.
        artifact = self._activation_cache.get(key[:2])
        if artifact is None:
            artifact = self._load_local_artifact(
                step,
                name,
                synchronize_model_parallel=False,
            )
        if layout_cache is None:
            self._activation_layout_cache = {}
            layout_cache = self._activation_layout_cache
        layout_cache[key] = artifact
        return artifact

    def _serialize_local_artifact(
        self,
        step: int,
        name: str,
        *,
        parallelism: tuple[int, int, int] | None = None,
    ) -> tuple[bytes, str]:
        import torch

        artifact = (
            self._load_local_artifact_for_layout(step, name, parallelism)
            if parallelism is not None
            else self._load_local_artifact(step, name, synchronize_model_parallel=False)
        )
        buffer = io.BytesIO()
        torch.save(artifact, buffer)
        payload = buffer.getvalue()
        return payload, hashlib.sha256(payload).hexdigest()

    def _store_serialized_artifact(self, step: int, name: str, payload: bytes, expected_sha256: str) -> dict[str, Any]:
        import torch

        sha256 = hashlib.sha256(payload).hexdigest()
        if sha256 != expected_sha256:
            raise RuntimeConfigError(
                f"in-memory activation checksum mismatch for {name}: {sha256} != {expected_sha256}"
            )
        artifact = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        if not isinstance(artifact, dict):
            raise RuntimeConfigError("in-memory activation payload must deserialize to a mapping")
        key = (int(step), self._activation_name(name))
        artifact = self._place_activation_artifact(key, artifact)
        self._activation_cache[key] = artifact
        layout_cache = getattr(self, "_activation_layout_cache", None)
        if layout_cache is None:
            self._activation_layout_cache = {}
            layout_cache = self._activation_layout_cache
        layout_cache[
            (key[0], key[1], self._parallelism())
        ] = artifact
        if self._activation_store == "hybrid":
            self._save_local_artifact(step, name, artifact, prune_producer_cache=False)
        return artifact

    def _store_tensor_artifact(
        self,
        step: int,
        name: str,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Materialize a received tensor record at the activation boundary.

        NCCL/UCX receives directly into CUDA tensors. The bounded placement policy below keeps them
        there when HBM was reserved by the planner, otherwise it performs the explicit host
        fallback.
        """
        artifact = _activation_artifact_from_tensor_record(record)
        self._save_local_artifact(
            step,
            name,
            artifact,
            prune_producer_cache=False,
        )
        return artifact

    def _artifact_name(self, base: str, payload: dict[str, Any], *, part_input: bool = False) -> str:
        part = payload.get("artifact_part")
        if part is None:
            return base
        if part_input or base == "rollout":
            return f"{base}.part{int(part)}"
        return f"{base}.part{int(part)}"

    def _merge_artifact_parts_if_last(self, step: int, base: str, payload: dict[str, Any]) -> None:
        part = payload.get("artifact_part")
        parts = int(payload.get("artifact_parts") or 0)
        if part is None or parts <= 0 or int(part) != parts - 1:
            return
        if not self._is_dp_artifact_writer():
            return
        shards = [
            self._load_local_artifact(step, f"{base}.part{idx}", synchronize_model_parallel=False)
            for idx in range(parts)
        ]
        merged = _sort_artifact_by_original_index(_merge_artifact_shards(shards))
        self._save_local_artifact(step, base, merged)

    def _partial_artifact_counters(
        self,
        payload: dict[str, Any],
        *,
        actual_n_seqs: int | None = None,
    ) -> dict[str, Any]:
        part = payload.get("artifact_part")
        if part is None:
            return {}
        counters: dict[str, Any] = {
            "artifact_part": int(part),
            "artifact_parts": int(payload.get("artifact_parts") or 0),
            "planned_batch_offset": int(payload.get("partial_batch_offset") or 0),
        }
        if payload.get("partial_n_seqs") is not None:
            counters["planned_n_seqs"] = int(payload["partial_n_seqs"])
        if actual_n_seqs is not None:
            counters["partial_n_seqs"] = int(actual_n_seqs)
        for key in ("partial_n_active", "partial_phase_steps", "partial_phase_t_start"):
            if payload.get(key) is not None:
                counters[key] = int(payload[key])
        if payload.get("partial_is_prefill_phase") is not None:
            counters["partial_is_prefill_phase"] = bool(payload["partial_is_prefill_phase"])
        return counters

    def _weight_version_dir_for(self, worker_id: str, model: str, version: int) -> Path:
        return self.state_root / "weights" / str(worker_id) / str(model) / f"version_{int(version):08d}"

    def _weight_version_path_for(self, worker_id: str, model: str, version: int, rank: int) -> Path:
        return self._weight_version_dir_for(worker_id, model, version) / f"rank{int(rank):05d}.pt"

    def _weight_version_metadata_path_for(self, worker_id: str, model: str, version: int, rank: int) -> Path:
        return self._weight_version_dir_for(worker_id, model, version) / f"rank{int(rank):05d}.json"

    def _weight_version_path(self, model: str, version: int) -> Path:
        return self._weight_version_path_for(self.worker_id, model, version, self.rank)

    def _cache_weight_version(self, model: str, version: int, state: dict[str, Any]) -> None:
        key = (str(model), int(version))
        self._weight_cache[key] = state
        self._weight_blob_cache.pop(key, None)
        keep = int(self._weight_cache_keep_latest)
        if keep <= 0:
            return
        versions = sorted(version_ for model_, version_ in self._weight_cache if model_ == model)
        retained = set(versions[-keep:])
        retained.update(self._weight_cache_pins.get(str(model), set()))
        for stale_version in (version_ for version_ in versions if version_ not in retained):
            stale_key = (str(model), int(stale_version))
            self._weight_cache.pop(stale_key, None)
            self._weight_blob_cache.pop(stale_key, None)

    def _prepare_weight_cache_insertion(self, model: str, version: int) -> None:
        """
        Evict entries that insertion would discard before allocating a snapshot.
        """
        keep = int(self._weight_cache_keep_latest)
        if keep <= 0:
            return
        key = (str(model), int(version))
        existing = sorted(
            version_
            for model_, version_ in self._weight_cache
            if model_ == model and (model_, version_) != key
        )
        retained = set(existing[-max(0, keep - 1):])
        retained.update(self._weight_cache_pins.get(str(model), set()))
        for stale_version in (version_ for version_ in existing if version_ not in retained):
            stale_key = (str(model), int(stale_version))
            self._weight_cache.pop(stale_key, None)
            self._weight_blob_cache.pop(stale_key, None)

    def _serialize_weight_version(self, model: str, version: int) -> tuple[bytes, str]:
        import torch

        key = (str(model), int(version))
        cached_blob = self._weight_blob_cache.get(key)
        if cached_blob is not None:
            return cached_blob
        state = self._weight_cache.get(key)
        if state is None:
            path = self._weight_version_path(model, version)
            if not path.exists():
                raise RuntimeConfigError(f"missing RPEG weight version file: {path}")
            state = torch.load(path, map_location="cpu", weights_only=False)
        stream_state = {
            name: value
            for name, value in state.items()
            if name not in {"optimizer_state_by_name", "optimizer_state_name_map"}
        }
        buffer = io.BytesIO()
        torch.save(stream_state, buffer)
        payload = buffer.getvalue()
        serialized = (payload, hashlib.sha256(payload).hexdigest())
        self._weight_blob_cache[key] = serialized
        return serialized

    def _store_serialized_weight_version(
        self,
        *,
        model: str,
        version: int,
        payload: bytes,
        expected_sha256: str,
        source_worker_id: str,
        shadow: bool,
    ) -> dict[str, Any]:
        import torch

        sha256 = hashlib.sha256(payload).hexdigest()
        if sha256 != expected_sha256:
            raise RuntimeConfigError(
                f"in-memory weight checksum mismatch for {model}:{version}: {sha256} != {expected_sha256}"
            )
        state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        if not isinstance(state, dict) or state.get("schema_version") != "rpeg.weight_version.v1":
            raise RuntimeConfigError("in-memory weight payload has an unsupported schema")
        if str(state.get("model")) != str(model) or int(state.get("version", -1)) != int(version):
            raise RuntimeConfigError(f"in-memory weight payload metadata mismatch for {model}:{version}")
        if int(state.get("world_size", -1)) != self.world_size:
            raise RuntimeConfigError(
                f"in-memory weight version {model}:{version} has world_size={state.get('world_size')}, "
                f"current worker world_size={self.world_size}"
            )
        state = dict(state)
        state["worker_id"] = self.worker_id
        state["source_worker_id"] = str(source_worker_id)
        state["shadow"] = bool(shadow)
        state["_serialized_bytes"] = len(payload)
        state.pop("optimizer_state_by_name", None)
        state.pop("optimizer_state_name_map", None)
        self._cache_weight_version(model, version, state)
        self._weight_blob_cache[(str(model), int(version))] = (payload, sha256)
        return state

    def _store_tensor_weight_version(
        self,
        *,
        model: str,
        version: int,
        source_version: int,
        tensor_record: dict[str, Any],
        serialized_bytes: int,
        source_worker_id: str,
        shadow: bool,
    ) -> dict[str, Any]:
        if tensor_record.get("schema_version") != _UCX_TENSOR_WEIGHT_SCHEMA:
            raise RuntimeConfigError("tensor-native weight payload has an unsupported schema")
        fragments = tensor_record.get("fragments")
        if not isinstance(fragments, dict):
            raise RuntimeConfigError("tensor-native weight payload has no state mapping")
        if (
            str(tensor_record.get("model")) != str(model)
            or int(tensor_record.get("version", -1)) != int(source_version)
        ):
            raise RuntimeConfigError(
                f"tensor-native weight payload metadata mismatch for {model}:{source_version}"
            )
        if int(tensor_record.get("world_size", -1)) != self.world_size:
            raise RuntimeConfigError(
                f"tensor-native weight version {model}:{version} has "
                f"world_size={tensor_record.get('world_size')}, current worker world_size={self.world_size}"
            )
        if int(tensor_record.get("rank", -1)) != self.rank:
            raise RuntimeConfigError(
                f"tensor-native weight version {model}:{version} has rank={tensor_record.get('rank')}, "
                f"current rank={self.rank}"
            )
        state = {
            key: value
            for key, value in tensor_record.items()
            if key not in {"fragments", "optimizer_state_by_name", "optimizer_state_name_map"}
        }
        state["schema_version"] = "rpeg.weight_version.v1"
        state["worker_id"] = self.worker_id
        state["version"] = int(version)
        state["source_worker_id"] = str(source_worker_id)
        state["source_version"] = int(source_version)
        state["shadow"] = bool(shadow)
        state["state_dict"] = {
            name: contribution["value"]
            for name, contribution in fragments.items()
        }
        state["_serialized_bytes"] = int(serialized_bytes)
        installed_active = False
        adopted_bytes = 0
        if not self.worker.operational_weights_materialized:
            # A shadow publication is already an independent destination-owned tensor record. If
            # liveness pruning released the dead active version, install this storage directly
            # instead of first caching a full shard and later allocating/copying another full shard.
            adopted_bytes = self.worker.adopt_rpeg_state_dict_storage(
                state["state_dict"],
                strict=True,
            )
            self._active_weight_versions[str(model)] = int(version)
            installed_active = True
        else:
            self._cache_weight_version(model, version, state)
        state["_installed_active"] = installed_active
        state["_adopted_parameter_bytes"] = adopted_bytes
        return state

    def _write_weight_version_metadata(
        self,
        *,
        worker_id: str,
        model: str,
        version: int,
        rank: int,
        world_size: int,
        step: int,
        path: Path,
        optimizer_state: bool | None,
        shadow: bool | None = None,
        source_worker_id: str | None = None,
        source_version: int | None = None,
        copy_mode: str | None = None,
        transport: str | None = None,
        sha256: str | None = None,
        parallelism: tuple[int, int, int] | None = None,
    ) -> None:
        from runtime.semantic import write_json_atomic

        metadata = {
            "schema_version": "rpeg.weight_version_meta.v1",
            "worker_id": str(worker_id),
            "rank": int(rank),
            "world_size": int(world_size),
            "model": str(model),
            "version": int(version),
            "step": int(step),
            "bytes": int(path.stat().st_size),
            "sha256": str(sha256) if sha256 is not None else _file_sha256(path),
        }
        if optimizer_state is not None:
            metadata["optimizer_state"] = bool(optimizer_state)
        if parallelism is not None:
            metadata["parallelism"] = [int(x) for x in parallelism]
        if shadow is not None:
            metadata["shadow"] = bool(shadow)
        if source_worker_id is not None:
            metadata["source_worker_id"] = str(source_worker_id)
        if source_version is not None:
            metadata["source_version"] = int(source_version)
        if copy_mode is not None:
            metadata["copy_mode"] = str(copy_mode)
        if transport is not None:
            metadata["transport"] = str(transport)
        write_json_atomic(
            self._weight_version_metadata_path_for(worker_id, model, version, rank),
            metadata,
        )

    def _load_weight_version_metadata(
        self,
        *,
        worker_id: str,
        model: str,
        version: int,
        rank: int,
    ) -> dict[str, Any]:
        from runtime.semantic import read_json

        path = self._weight_version_metadata_path_for(worker_id, model, version, rank)
        if not path.exists():
            raise RuntimeConfigError(f"missing RPEG weight version metadata: {path}")
        metadata = read_json(path)
        if metadata.get("schema_version") != "rpeg.weight_version_meta.v1":
            raise RuntimeConfigError(f"unsupported RPEG weight version metadata schema in {path}")
        expected = {
            "worker_id": str(worker_id),
            "rank": int(rank),
            "model": str(model),
            "version": int(version),
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeConfigError(
                    f"weight version metadata mismatch for {path}: {key}={metadata.get(key)!r}, expected {value!r}"
                )
        data_path = self._weight_version_path_for(worker_id, model, version, rank)
        if data_path.exists():
            actual_bytes = data_path.stat().st_size
            if int(metadata.get("bytes", -1)) != actual_bytes:
                raise RuntimeConfigError(f"weight version byte count mismatch for {data_path}")
            if metadata.get("sha256") is not None and str(metadata.get("sha256")) != _file_sha256(data_path):
                raise RuntimeConfigError(f"weight version checksum mismatch for {data_path}")
        return metadata

    def _save_rank0(self, step: int, name: str, data: dict[str, Any]) -> None:
        if self.rank != 0:
            return
        import torch

        path = self._tensor_path(step, name)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(data, tmp)
        tmp.replace(path)

    def _load_tensor_file(self, step: int, name: str) -> dict[str, Any]:
        import torch

        path = self._tensor_path(step, name)
        if not path.exists():
            raise RuntimeConfigError(f"missing PPO semantic state file: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    def _transfer_activation(self, request: SemanticRequest) -> dict[str, Any]:
        import torch

        payload = request.payload
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        source_name = self._activation_name(str(payload["source_name"]))
        destination_name = self._activation_name(str(payload.get("destination_name", source_name)))
        source_step = int(payload.get("source_step", request.step))
        destination_step = int(payload.get("destination_step", request.step))
        optional = bool(payload.get("optional", False))
        activation_transport = _normalize_file_transport(payload.get("activation_transport", "auto"))
        if not self._is_dp_artifact_writer():
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "destination_name": destination_name,
                "copied": False,
                "rank": self.rank,
            }
        source_manifest = self._artifact_manifest_path_for(source_worker_id, source_step, source_name)
        if source_manifest.exists():
            shard_metadata = self._load_artifact_shard_metadata(source_worker_id, source_step, source_name)
            direct_copy = self._try_direct_copy_artifact_shard(
                source_worker_id=source_worker_id,
                source_step=source_step,
                source_name=source_name,
                destination_step=destination_step,
                destination_name=destination_name,
                shard_metadata=shard_metadata,
                activation_transport=activation_transport,
            )
            total_bytes = sum(int(meta["bytes"]) for meta in shard_metadata)
            if direct_copy is not None:
                return {
                    "source_worker_id": source_worker_id,
                    "source_name": source_name,
                    "destination_name": destination_name,
                    "copied": direct_copy["copied"],
                    "dp_repartitioned": False,
                    "copy_mode": direct_copy["copy_mode"],
                    "destination_dp_rank": self._dp_rank(),
                    "source_dp_size": len(shard_metadata),
                    "bytes": int(direct_copy["bytes"]),
                    "sha256": direct_copy.get("sha256"),
                    "source_sha256": direct_copy.get("source_sha256"),
                    "source_shards": _artifact_shard_summary(shard_metadata),
                    "transport": direct_copy["transport"],
                    "transfer_rank_duration_s": float(direct_copy.get("transfer_rank_duration_s", 0.0)),
                    "transfer_rank_bytes_per_s": float(direct_copy.get("transfer_rank_bytes_per_s", 0.0)),
                    "rank": self.rank,
                }
            local, source_shards_read = self._repartition_artifact_for_destination_dp(
                source_worker_id=source_worker_id,
                source_step=source_step,
                source_name=source_name,
                shard_metadata=shard_metadata,
            )
            self._save_local_artifact(
                destination_step,
                destination_name,
                local,
                prune_producer_cache=False,
            )
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "destination_name": destination_name,
                "copied": True,
                "dp_repartitioned": True,
                "destination_dp_rank": self._dp_rank(),
                "source_dp_size": len(shard_metadata),
                "bytes": total_bytes,
                "source_shards": _artifact_shard_summary(shard_metadata),
                "source_shards_read": source_shards_read,
                "transport": "shared_filesystem",
                "rank": self.rank,
            }
        source_path = self._activation_path_for(source_worker_id, source_step, source_name)
        if not source_path.exists():
            if optional:
                return {
                    "source_worker_id": source_worker_id,
                    "source_name": source_name,
                    "destination_name": destination_name,
                    "copied": False,
                    "missing_optional": True,
                    "rank": self.rank,
                }
            raise RuntimeConfigError(f"missing source activation artifact: {source_path}")
        destination_path = self._activation_path_for(self.worker_id, destination_step, destination_name)
        if source_path == destination_path:
            sha256 = _file_sha256(source_path)
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "destination_name": destination_name,
                "copied": False,
                "bytes": source_path.stat().st_size,
                "sha256": sha256,
                "source_sha256": sha256,
                "copy_mode": "same_path",
                "transport": "shared_filesystem_same_path",
                "transfer_rank_duration_s": 0.0,
                "transfer_rank_bytes_per_s": 0.0,
                "rank": self.rank,
            }
        transfer = _copy_file_atomic(source_path, destination_path, mode=activation_transport)
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "destination_name": destination_name,
            "copied": True,
            "bytes": int(transfer["bytes"]),
            "sha256": str(transfer["sha256"]),
            "source_sha256": str(transfer["source_sha256"]),
            "copy_mode": str(transfer["copy_mode"]),
            "transport": str(transfer["transport"]),
            "transfer_rank_duration_s": float(transfer.get("transfer_rank_duration_s", 0.0)),
            "transfer_rank_bytes_per_s": float(transfer.get("transfer_rank_bytes_per_s", 0.0)),
            "rank": self.rank,
        }

    def _repartition_local_activation(self, request: SemanticRequest) -> dict[str, Any]:
        """
        Redistribute an in-memory artifact after a persistent layout switch.

        Source and destination are ranks of the same semantic worker, so a paired send/receive
        command would deadlock the worker control plane. The layout-switch Gloo group is already
        persistent; use one bounded variable-length byte all-gather on that group, then materialize
        only the destination DP-writer shards. This path never activates the lazy default-world NCCL
        communicator and therefore does not add CXI endpoints.
        """
        import torch
        import torch.distributed as dist

        payload = request.payload
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        if source_worker_id != self.worker_id:
            raise RuntimeConfigError(
                "local activation repartition requires identical source and destination workers"
            )
        source_name = self._activation_name(str(payload["source_name"]))
        destination_name = self._activation_name(str(payload.get("destination_name", source_name)))
        source_step = int(payload.get("source_step", request.step))
        destination_step = int(payload.get("destination_step", request.step))
        source_parallelism = self._request_parallelism(payload, "source")
        destination_parallelism = self._request_parallelism(payload, "destination")
        if source_parallelism == destination_parallelism:
            raise RuntimeConfigError("local activation repartition requires different layouts")
        if self._parallelism() != destination_parallelism:
            raise RuntimeConfigError(
                f"local activation destination layout {destination_parallelism} is not active; "
                f"worker layout is {self._parallelism()}"
            )
        source_dp, source_pp, source_tp = source_parallelism
        destination_dp, destination_pp, destination_tp = destination_parallelism
        global_batch_size = int(payload["global_batch_size"])
        global_index_repartition = bool(payload.get("global_index_repartition", False))
        source_writer = self._is_request_dp_artifact_writer(payload, "source")
        destination_writer = self._is_request_dp_artifact_writer(payload, "destination")
        source_dp_rank = self._request_dp_rank(payload, "source")
        destination_dp_rank = self._request_dp_rank(payload, "destination")

        source_shard = None
        source_start = source_end = None
        if source_writer:
            source_shard = self._load_local_artifact_for_layout(
                source_step,
                source_name,
                source_parallelism,
            )
            source_start, source_end = _batch_range_for_dp(
                global_batch_size,
                source_dp,
                source_dp_rank,
            )
            if global_index_repartition:
                _validate_global_index_artifact(
                    source_shard,
                    global_batch_size=global_batch_size,
                    expected_start=source_start,
                    expected_end=source_end,
                )
            elif _artifact_batch_size(source_shard) != source_end - source_start:
                raise RuntimeConfigError(
                    f"source activation shard DP rank {source_dp_rank} has batch "
                    f"{_artifact_batch_size(source_shard)}, expected {source_end - source_start}"
                )
        group, device, copy_mode, transport = self._local_activation_repartition_transport()
        if group is None:
            raise RuntimeConfigError("local activation repartition requires multiple worker ranks")
        if transport == "torch_dist_nccl":
            outgoing: list[tuple[dict[str, Any], dict[str, Any]]] = []
            local_slices: list[tuple[int, dict[str, Any]]] = []
            local_source_dp_ranks: list[int] = []
            if source_writer:
                assert source_shard is not None and source_start is not None and source_end is not None
                for destination_rank, overlap_start, overlap_end in _direct_repartition_destinations(
                    source_start=source_start,
                    source_end=source_end,
                    global_batch_size=global_batch_size,
                    destination_dp_size=destination_dp,
                ):
                    local = (
                        _slice_artifact_global_index_range(
                            source_shard,
                            overlap_start,
                            overlap_end,
                            global_batch_size=global_batch_size,
                        )
                        if global_index_repartition
                        else _slice_artifact_batch(
                            source_shard,
                            overlap_start - source_start,
                            overlap_end - source_start,
                        )
                    )
                    destination_world_rank = destination_rank * destination_pp * destination_tp
                    if destination_world_rank == self.rank:
                        local_slices.append((overlap_start, local))
                        local_source_dp_ranks.append(source_dp_rank)
                    else:
                        outgoing.append((
                            {"destination_group_rank": destination_world_rank},
                            _activation_artifact_to_tensor_record(local),
                        ))

            expected_remote_sources: list[tuple[int, int, int]] = []
            if destination_writer:
                destination_start, destination_end = _batch_range_for_dp(
                    global_batch_size,
                    destination_dp,
                    destination_dp_rank,
                )
                expected_sources = _direct_repartition_sources_for_destination(
                    [
                        (
                            {"dp_rank": rank},
                            *_batch_range_for_dp(global_batch_size, source_dp, rank),
                        )
                        for rank in range(source_dp)
                    ],
                    destination_start=destination_start,
                    destination_end=destination_end,
                )
                for rank, overlap_start, overlap_end in sorted(expected_sources):
                    source_world_rank = rank * source_pp * source_tp
                    if source_world_rank != self.rank:
                        expected_remote_sources.append((
                            source_world_rank,
                            overlap_start,
                            overlap_end,
                        ))

            started = time.perf_counter()
            received, sent_wire_bytes = _torch_direct_exchange_tensor_records_coalesced(
                group,
                outgoing,
                [
                    {"source_group_rank": source_world_rank}
                    for source_world_rank, _start, _end in expected_remote_sources
                ],
                device=device,
                schema_version=_ACTIVATION_TENSOR_STREAM_SCHEMA,
                expected_schema=_ACTIVATION_TENSOR_STREAM_SCHEMA,
            )
            received_wire_bytes = 0
            source_shards_read: list[int] = list(local_source_dp_ranks)
            for (
                (source_world_rank, overlap_start, overlap_end),
                (record, bytes_, _sha256),
            ) in zip(expected_remote_sources, received):
                artifact = _activation_artifact_from_tensor_record(record)
                if global_index_repartition:
                    _validate_global_index_artifact(
                        artifact,
                        global_batch_size=global_batch_size,
                        expected_start=overlap_start,
                        expected_end=overlap_end,
                    )
                elif _artifact_batch_size(artifact) != overlap_end - overlap_start:
                    raise RuntimeConfigError(
                        f"local activation slice from rank {source_world_rank} has batch "
                        f"{_artifact_batch_size(artifact)}, expected {overlap_end - overlap_start}"
                    )
                local_slices.append((overlap_start, artifact))
                received_wire_bytes += int(bytes_)
                source_shards_read.append(source_world_rank // (source_pp * source_tp))
            dist.barrier(group=group)
            duration_s = time.perf_counter() - started

            if destination_writer:
                local_slices.sort(key=lambda item: item[0])
                local = _merge_artifact_shards([artifact for _start, artifact in local_slices])
                if global_index_repartition:
                    _validate_global_index_artifact(
                        local,
                        global_batch_size=global_batch_size,
                        expected_start=destination_start,
                        expected_end=destination_end,
                    )
                    local = _sort_artifact_by_original_index(local)
                elif _artifact_batch_size(local) != destination_end - destination_start:
                    raise RuntimeConfigError(
                        f"local activation repartition produced batch {_artifact_batch_size(local)}, "
                        f"expected {destination_end - destination_start}"
                    )
                self._save_local_artifact(
                    destination_step,
                    destination_name,
                    local,
                    prune_producer_cache=False,
                )
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "destination_name": destination_name,
                "copied": destination_writer,
                "dp_repartitioned": True,
                "local_layout_transition": True,
                "copy_mode": copy_mode,
                "transport": transport,
                "wire_format": "tensor_record_v1",
                "sha256_scope": "manifest",
                "source_parallelism": list(source_parallelism),
                "destination_parallelism": list(destination_parallelism),
                "destination_dp_rank": destination_dp_rank if destination_writer else None,
                "bytes": received_wire_bytes if destination_writer else 0,
                "sent_wire_bytes": int(sent_wire_bytes),
                "logical_wire_bytes": int(sent_wire_bytes),
                "source_shards_read": sorted(source_shards_read),
                "transfer_rank_duration_s": duration_s,
                "transfer_rank_bytes_per_s": _bytes_per_second(
                    received_wire_bytes if destination_writer else int(sent_wire_bytes),
                    duration_s,
                ),
                "rank": self.rank,
            }

        artifact_payload = b""
        if source_writer:
            buffer = io.BytesIO()
            torch.save(source_shard, buffer)
            artifact_payload = buffer.getvalue()
        local_size = torch.tensor([len(artifact_payload)], dtype=torch.int64, device=device)
        gathered_sizes = [torch.zeros_like(local_size) for _ in range(self.world_size)]
        started = time.perf_counter()
        dist.all_gather(gathered_sizes, local_size, group=group)
        # Collapse the size-vector synchronization to one device-to-host copy on NCCL instead of one
        # scalar synchronization per worker rank.
        sizes = [int(value) for value in torch.cat(gathered_sizes).cpu().tolist()]
        max_size = max(sizes, default=0)
        if max_size <= 0:
            raise RuntimeConfigError(
                f"no source rank published local activation {source_name!r}@{source_step}"
            )
        local_bytes = torch.zeros(max_size, dtype=torch.uint8, device=device)
        if artifact_payload:
            encoded = torch.frombuffer(bytearray(artifact_payload), dtype=torch.uint8)
            local_bytes[: encoded.numel()].copy_(encoded.to(device=device))
        gathered_payloads = [torch.empty_like(local_bytes) for _ in range(self.world_size)]
        dist.all_gather(gathered_payloads, local_bytes, group=group)
        duration_s = time.perf_counter() - started

        source_shards: list[tuple[int, dict[str, Any]]] = []
        source_writer_ranks = {
            dp_rank * source_pp * source_tp: dp_rank
            for dp_rank in range(source_dp)
        }
        for rank, size in enumerate(sizes):
            expected_dp_rank = source_writer_ranks.get(rank)
            if expected_dp_rank is None:
                if size:
                    raise RuntimeConfigError(
                        f"non-writer source rank {rank} published {size} activation bytes"
                    )
                continue
            if size <= 0:
                raise RuntimeConfigError(
                    f"source DP writer rank {rank} published no activation bytes"
                )
            shard = torch.load(
                io.BytesIO(gathered_payloads[rank][:size].cpu().numpy().tobytes()),
                map_location="cpu",
                weights_only=False,
            )
            if not isinstance(shard, dict):
                raise RuntimeConfigError("local activation shard must be a mapping")
            expected_start, expected_end = _batch_range_for_dp(
                global_batch_size,
                source_dp,
                expected_dp_rank,
            )
            if global_index_repartition:
                _validate_global_index_artifact(
                    shard,
                    global_batch_size=global_batch_size,
                    expected_start=expected_start,
                    expected_end=expected_end,
                )
            elif _artifact_batch_size(shard) != expected_end - expected_start:
                raise RuntimeConfigError(
                    f"gathered source DP rank {expected_dp_rank} has batch "
                    f"{_artifact_batch_size(shard)}, expected {expected_end - expected_start}"
                )
            source_shards.append((expected_start, shard))

        source_shards.sort(key=lambda item: item[0])
        if destination_writer:
            global_artifact = _merge_artifact_shards([shard for _start, shard in source_shards])
            destination_start, destination_end = _batch_range_for_dp(
                global_batch_size,
                destination_dp,
                destination_dp_rank,
            )
            local = (
                _slice_artifact_global_index_range(
                    global_artifact,
                    destination_start,
                    destination_end,
                    global_batch_size=global_batch_size,
                )
                if global_index_repartition
                else _slice_artifact_batch(
                    global_artifact,
                    destination_start,
                    destination_end,
                )
            )
            if global_index_repartition:
                local = _sort_artifact_by_original_index(local)
            self._save_local_artifact(
                destination_step,
                destination_name,
                local,
                prune_producer_cache=False,
            )

        wire_bytes = sum(sizes)
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "destination_name": destination_name,
            "copied": destination_writer,
            "dp_repartitioned": True,
            "local_layout_transition": True,
            "copy_mode": copy_mode,
            "transport": transport,
            "source_parallelism": list(source_parallelism),
            "destination_parallelism": list(destination_parallelism),
            "destination_dp_rank": destination_dp_rank if destination_writer else None,
            "bytes": wire_bytes if destination_writer else 0,
            "logical_wire_bytes": wire_bytes * max(0, self.world_size - 1),
            "transfer_rank_duration_s": duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(wire_bytes, duration_s),
            "rank": self.rank,
        }

    def _local_activation_repartition_transport(self) -> tuple[Any, Any, str, str]:
        """
        Use the hot NCCL world group for same-worker activation reshaping.
        """
        import torch
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeConfigError(
                "local activation repartition requires an initialized process group"
            )
        world_group = dist.group.WORLD
        world_backend = str(dist.get_backend(world_group)).lower()
        if torch.cuda.is_available() and "nccl" in world_backend:
            return (
                world_group,
                self._weight_receive_device(),
                "in_place_nccl_memory_repartition",
                "torch_dist_nccl",
            )
        return (
            self._ensure_host_reshard_group(),
            torch.device("cpu"),
            "in_place_gloo_memory_repartition",
            "torch_dist_gloo",
        )

    def _send_activation_direct(self, request: SemanticRequest) -> dict[str, Any]:
        payload = request.payload
        direct_kind = _direct_transport_kind(payload)
        if bool(payload.get("direct_repartition", False)):
            if direct_kind == "tcp":
                return self._send_activation_tcp_repartition(request)
            if direct_kind == "torch_dist":
                return self._send_activation_torch_dist_repartition(request)
            if direct_kind == "ucx":
                return self._send_activation_ucx_repartition(request)
            raise RuntimeConfigError(
                "direct activation repartition currently requires tcp_direct, torch_dist, or ucx_direct transport"
            )
        if direct_kind == "torch_dist":
            return self._send_activation_torch_dist(request)
        if direct_kind == "ucx":
            return self._send_activation_ucx(request)
        if bool(payload.get("in_memory", False)):
            return self._send_activation_tcp_memory(request)
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        source_name = self._activation_name(str(payload["source_name"]))
        source_step = int(payload.get("source_step", request.step))
        if not self._is_request_dp_artifact_writer(payload, "source"):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "copied": False,
                "rank": self.rank,
                "transport": "tcp_direct",
            }
        source_path, source_metadata = self._direct_activation_source(source_worker_id, source_step, source_name)
        endpoint = _direct_endpoint_for_source_rank(payload, self.rank)
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        bytes_ = int(source_metadata["bytes"])
        sha256 = str(source_metadata["sha256"])
        def send() -> None:
            deadline = time.monotonic() + timeout_s
            last_error: OSError | None = None
            while time.monotonic() < deadline:
                try:
                    with socket.create_connection(
                        (str(endpoint["host"]), int(endpoint["port"])),
                        timeout=min(5.0, max(0.1, deadline - time.monotonic())),
                    ) as sock:
                        _socket_send_activation_file(sock, source_path, bytes_, sha256)
                    return
                except OSError as exc:
                    last_error = exc
                    time.sleep(0.05)
            raise RuntimeConfigError(
                f"timed out connecting direct activation sender rank {self.rank} to "
                f"{endpoint['host']}:{endpoint['port']}: {last_error}"
            )

        started = time.perf_counter()
        asynchronous = bool(payload.get("source_overlap", False))
        if asynchronous:
            self._launch_background_transfer(
                send,
                name=f"activation-{source_name}-step{source_step}",
            )
        else:
            send()
        duration_s = time.perf_counter() - started
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "copied": True,
            "bytes": bytes_,
            "sha256": sha256,
            "copy_mode": "tcp_direct",
            "transport": "tcp_direct",
            "direct_dp_writer_copy": bool(payload.get("direct_dp_writer_copy", False)),
            "transfer_rank_duration_s": duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(bytes_, duration_s),
            "source_overlap": asynchronous,
            "rank": self.rank,
        }

    def _recv_activation_direct(self, request: SemanticRequest) -> dict[str, Any]:
        payload = request.payload
        direct_kind = _direct_transport_kind(payload)
        if bool(payload.get("direct_repartition", False)):
            if direct_kind == "tcp":
                return self._recv_activation_tcp_repartition(request)
            if direct_kind == "torch_dist":
                return self._recv_activation_torch_dist_repartition(request)
            if direct_kind == "ucx":
                return self._recv_activation_ucx_repartition(request)
            raise RuntimeConfigError(
                "direct activation repartition currently requires tcp_direct, torch_dist, or ucx_direct transport"
            )
        if direct_kind == "torch_dist":
            return self._recv_activation_torch_dist(request)
        if direct_kind == "ucx":
            return self._recv_activation_ucx(request)
        if bool(payload.get("in_memory", False)):
            return self._recv_activation_tcp_memory(request)
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        source_name = self._activation_name(str(payload["source_name"]))
        destination_name = self._activation_name(str(payload.get("destination_name", source_name)))
        source_step = int(payload.get("source_step", request.step))
        destination_step = int(payload.get("destination_step", request.step))
        if not self._is_request_dp_artifact_writer(payload, "destination"):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "destination_name": destination_name,
                "copied": False,
                "rank": self.rank,
                "transport": "tcp_direct",
            }
        destination_dp_rank = self._request_dp_rank(payload, "destination")
        sharded = self._artifact_manifest_path_for(source_worker_id, source_step, source_name).exists()
        source_metadata = self._direct_activation_source_metadata(source_worker_id, source_step, source_name)
        endpoint = _direct_endpoint_for_rank(payload, self.rank)
        destination_path = self._direct_activation_destination_path(destination_step, destination_name, sharded=sharded)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination_path.with_suffix(destination_path.suffix + f".{os.getpid()}.{self.rank}.direct.tmp")
        if tmp.exists():
            tmp.unlink()
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind(("", int(endpoint["port"])))
                server.listen(1)
                server.settimeout(timeout_s)
                started = time.perf_counter()
                conn, _addr = server.accept()
                with conn:
                    bytes_, sha256 = _socket_recv_activation_file(conn, tmp)
                duration_s = time.perf_counter() - started
            expected_sha = str(source_metadata["sha256"])
            if sha256 != expected_sha:
                raise RuntimeConfigError(
                    f"direct activation checksum mismatch for {destination_path}: {sha256} != {expected_sha}"
                )
            tmp.replace(destination_path)
        finally:
            if tmp.exists():
                tmp.unlink()
        self._write_direct_activation_metadata(
            source_worker_id=source_worker_id,
            source_step=source_step,
            source_name=source_name,
            destination_step=destination_step,
            destination_name=destination_name,
            source_metadata=source_metadata,
            bytes_=bytes_,
            sha256=sha256,
        )
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "destination_name": destination_name,
            "copied": True,
            "dp_repartitioned": False,
            "destination_dp_rank": destination_dp_rank,
            "source_dp_size": self._request_parallelism(payload, "source")[0],
            "bytes": int(bytes_),
            "sha256": sha256,
            "source_sha256": expected_sha,
            "copy_mode": "tcp_direct",
            "transport": "tcp_direct",
            "direct_dp_writer_copy": bool(payload.get("direct_dp_writer_copy", False)),
            "transfer_rank_duration_s": duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(bytes_, duration_s),
            "rank": self.rank,
        }

    def _send_activation_tcp_memory(self, request: SemanticRequest) -> dict[str, Any]:
        payload = request.payload
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        source_name = self._activation_name(str(payload["source_name"]))
        source_step = int(payload.get("source_step", request.step))
        transport = "tcp_direct"
        if not self._is_request_dp_artifact_writer(payload, "source"):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        artifact_payload, sha256 = self._serialize_local_artifact(
            source_step,
            source_name,
            parallelism=self._request_parallelism(payload, "source"),
        )
        bytes_ = len(artifact_payload)
        endpoint = _direct_endpoint_for_source_rank(payload, self.rank)
        timeout_s = float(payload.get("direct_timeout_s", 120.0))

        def send() -> None:
            deadline = time.monotonic() + timeout_s
            last_error: OSError | None = None
            while time.monotonic() < deadline:
                try:
                    with socket.create_connection(
                        (str(endpoint["host"]), int(endpoint["port"])),
                        timeout=min(5.0, max(0.1, deadline - time.monotonic())),
                    ) as sock:
                        _socket_send_activation_blob(sock, artifact_payload, sha256)
                    return
                except OSError as exc:
                    last_error = exc
                    time.sleep(0.05)
            raise RuntimeConfigError(
                f"timed out connecting in-memory activation sender rank {self.rank} to "
                f"{endpoint['host']}:{endpoint['port']}: {last_error}"
            )

        started = time.perf_counter()
        asynchronous = bool(payload.get("source_overlap", False))
        if asynchronous:
            self._launch_background_transfer(
                send,
                name=f"activation-{source_name}-step{source_step}",
            )
        else:
            send()
        duration_s = time.perf_counter() - started
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "copied": True,
            "bytes": bytes_,
            "sha256": sha256,
            "copy_mode": "tcp_direct_memory",
            "transport": transport,
            "wire_format": "torch_save_memory",
            "direct_dp_writer_copy": bool(payload.get("direct_dp_writer_copy", False)),
            "transfer_rank_duration_s": duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(bytes_, duration_s),
            "source_overlap": asynchronous,
            "rank": self.rank,
        }

    def _recv_activation_tcp_memory(self, request: SemanticRequest) -> dict[str, Any]:
        payload = request.payload
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        source_name = self._activation_name(str(payload["source_name"]))
        destination_name = self._activation_name(str(payload.get("destination_name", source_name)))
        destination_step = int(payload.get("destination_step", request.step))
        transport = "tcp_direct"
        if not self._is_request_dp_artifact_writer(payload, "destination"):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "destination_name": destination_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        destination_dp_rank = self._request_dp_rank(payload, "destination")
        endpoint = _direct_endpoint_for_rank(payload, self.rank)
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("", int(endpoint["port"])))
            server.listen(1)
            server.settimeout(timeout_s)
            started = time.perf_counter()
            conn, _addr = server.accept()
            conn.settimeout(timeout_s)
            with conn:
                artifact_payload, sha256 = _socket_recv_activation_blob(conn)
        self._store_serialized_artifact(
            destination_step,
            destination_name,
            artifact_payload,
            sha256,
        )
        duration_s = time.perf_counter() - started
        bytes_ = len(artifact_payload)
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "destination_name": destination_name,
            "copied": True,
            "dp_repartitioned": False,
            "destination_dp_rank": destination_dp_rank,
            "source_dp_size": self._request_parallelism(payload, "source")[0],
            "bytes": bytes_,
            "sha256": sha256,
            "source_sha256": sha256,
            "copy_mode": "tcp_direct_memory",
            "transport": transport,
            "wire_format": "torch_save_memory",
            "direct_dp_writer_copy": bool(payload.get("direct_dp_writer_copy", False)),
            "transfer_rank_duration_s": duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(bytes_, duration_s),
            "rank": self.rank,
        }

    def _send_direct_prewarm(self, request: SemanticRequest) -> dict[str, Any]:
        payload = request.payload
        direct_kind = _direct_transport_kind(payload)
        if direct_kind != "ucx":
            raise RuntimeConfigError("direct prewarm currently supports only ucx_direct transport")
        endpoints = _direct_endpoints_for_source_rank(payload, self.rank)
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        started = time.perf_counter()
        cached_count = 0
        for endpoint in endpoints:
            cached_count += int(_run_ucx_direct(
                _ucx_direct_send_prewarm(
                    endpoint=endpoint,
                    timeout_s=timeout_s,
                    connect_grace_s=_ucx_direct_connect_grace_s(payload),
                ),
                persistent=_ucx_direct_channel_reuse(endpoint),
            ))
        duration_s = time.perf_counter() - started
        return {
            "transport": "ucx_direct",
            "prewarm_role": "send",
            "prewarm_count": len(endpoints),
            "prewarm_cached_endpoints": cached_count,
            "prewarm_rank_duration_s": duration_s,
        }

    def _recv_direct_prewarm(self, request: SemanticRequest) -> dict[str, Any]:
        payload = request.payload
        direct_kind = _direct_transport_kind(payload)
        if direct_kind != "ucx":
            raise RuntimeConfigError("direct prewarm currently supports only ucx_direct transport")
        endpoints = _direct_endpoints_for_rank(payload, self.rank)
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        started = time.perf_counter()
        cached_count = 0
        for endpoint in endpoints:
            cached_count += int(_run_ucx_direct(
                _ucx_direct_recv_prewarm(
                    endpoint=endpoint,
                    timeout_s=timeout_s,
                    ready_path=_direct_ready_marker_path(self.state_root, endpoint),
                ),
                persistent=_ucx_direct_channel_reuse(endpoint),
            ))
        duration_s = time.perf_counter() - started
        return {
            "transport": "ucx_direct",
            "prewarm_role": "recv",
            "prewarm_count": len(endpoints),
            "prewarm_cached_endpoints": cached_count,
            "prewarm_rank_duration_s": duration_s,
        }

    def _send_activation_tcp_repartition_memory(
        self,
        request: SemanticRequest,
    ) -> dict[str, Any]:
        import torch

        payload = request.payload
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        source_name = self._activation_name(str(payload["source_name"]))
        source_step = int(payload.get("source_step", request.step))
        transport = "tcp_direct"
        copy_mode = "tcp_direct_memory_repartition"
        if not self._is_request_dp_artifact_writer(payload, "source"):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        if bool(payload.get("direct_repartition_aggregate", False)):
            raise RuntimeConfigError("in-memory tcp_direct repartition does not support aggregate mode")
        source_dp_size = int(payload["source_dp_size"])
        destination_dp_size = int(payload["destination_dp_size"])
        global_batch_size = int(payload["global_batch_size"])
        global_index_repartition = bool(payload.get("global_index_repartition", False))
        source_dp_rank = self._request_dp_rank(payload, "source")
        source_start, source_end = _batch_range_for_dp(
            global_batch_size,
            source_dp_size,
            source_dp_rank,
        )
        source_shard = self._load_local_artifact_for_layout(
            source_step,
            source_name,
            self._request_parallelism(payload, "source"),
        )
        source_batch = _artifact_batch_size(source_shard)
        if global_index_repartition:
            _validate_global_index_artifact(
                source_shard,
                global_batch_size=global_batch_size,
                expected_start=source_start,
                expected_end=source_end,
            )
        elif source_batch != source_end - source_start:
            raise RuntimeConfigError(
                f"in-memory activation shard {source_worker_id}:{source_name}@{source_step} "
                f"has batch {source_batch}, expected {source_end - source_start}"
            )
        destinations = _direct_repartition_destinations(
            source_start=source_start,
            source_end=source_end,
            global_batch_size=global_batch_size,
            destination_dp_size=destination_dp_size,
        )
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        total_bytes = 0
        total_duration_s = 0.0
        sent_destinations: list[int] = []
        slice_hashes: list[str] = []
        for destination_dp_rank, overlap_start, overlap_end in destinations:
            local = (
                _slice_artifact_global_index_range(
                    source_shard,
                    overlap_start,
                    overlap_end,
                    global_batch_size=global_batch_size,
                )
                if global_index_repartition
                else _slice_artifact_batch(
                    source_shard,
                    overlap_start - source_start,
                    overlap_end - source_start,
                )
            )
            buffer = io.BytesIO()
            torch.save(local, buffer)
            artifact_payload = buffer.getvalue()
            sha256 = hashlib.sha256(artifact_payload).hexdigest()
            metadata = {
                "schema_version": "rpeg.activation_direct_slice.v1",
                "source_dp_rank": source_dp_rank,
                "destination_dp_rank": destination_dp_rank,
                "global_start": overlap_start,
                "global_end": overlap_end,
                "source_sha256": sha256,
            }
            endpoint = _direct_endpoint_for_dp_pair(
                payload,
                destination_dp_rank,
                source_dp_rank,
            )
            deadline = time.monotonic() + timeout_s
            last_error: OSError | None = None
            while time.monotonic() < deadline:
                try:
                    started = time.perf_counter()
                    with socket.create_connection(
                        (str(endpoint["host"]), int(endpoint["port"])),
                        timeout=min(5.0, max(0.1, deadline - time.monotonic())),
                    ) as sock:
                        _socket_send_activation_blob_with_metadata(
                            sock,
                            artifact_payload,
                            sha256,
                            metadata,
                        )
                    total_duration_s += time.perf_counter() - started
                    total_bytes += len(artifact_payload)
                    sent_destinations.append(destination_dp_rank)
                    slice_hashes.append(sha256)
                    break
                except OSError as exc:
                    last_error = exc
                    time.sleep(0.05)
            else:
                raise RuntimeConfigError(
                    f"timed out connecting in-memory tcp_direct repartition sender rank "
                    f"{self.rank} to {endpoint['host']}:{endpoint['port']}: {last_error}"
                )
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "copied": bool(sent_destinations),
            "bytes": total_bytes,
            "sha256": hashlib.sha256("".join(slice_hashes).encode("ascii")).hexdigest(),
            "copy_mode": copy_mode,
            "transport": transport,
            "wire_format": "torch_save_memory",
            "transfer_rank_duration_s": total_duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(total_bytes, total_duration_s),
            "destination_dp_ranks": sent_destinations,
            "rank": self.rank,
        }

    def _recv_activation_tcp_repartition_memory(
        self,
        request: SemanticRequest,
    ) -> dict[str, Any]:
        import torch

        payload = request.payload
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        source_name = self._activation_name(str(payload["source_name"]))
        destination_name = self._activation_name(str(payload.get("destination_name", source_name)))
        destination_step = int(payload.get("destination_step", request.step))
        transport = "tcp_direct"
        copy_mode = "tcp_direct_memory_repartition"
        if not self._is_request_dp_artifact_writer(payload, "destination"):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "destination_name": destination_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        if bool(payload.get("direct_repartition_aggregate", False)):
            raise RuntimeConfigError("in-memory tcp_direct repartition does not support aggregate mode")
        source_dp_size = int(payload["source_dp_size"])
        destination_dp_size = int(payload["destination_dp_size"])
        global_batch_size = int(payload["global_batch_size"])
        global_index_repartition = bool(payload.get("global_index_repartition", False))
        destination_dp_rank = self._request_dp_rank(payload, "destination")
        destination_start, destination_end = _batch_range_for_dp(
            global_batch_size,
            destination_dp_size,
            destination_dp_rank,
        )
        ranges = [
            (
                {"dp_rank": source_dp_rank},
                *_batch_range_for_dp(global_batch_size, source_dp_size, source_dp_rank),
            )
            for source_dp_rank in range(source_dp_size)
        ]
        expected_sources = _direct_repartition_sources_for_destination(
            ranges,
            destination_start=destination_start,
            destination_end=destination_end,
        )
        if not expected_sources:
            raise RuntimeConfigError(
                f"no in-memory activation shards overlap destination DP rank {destination_dp_rank}"
            )
        expected_set = set(expected_sources)
        endpoint = _direct_endpoint_for_rank(payload, self.rank)
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        slices: list[tuple[int, int, dict[str, Any], str]] = []
        seen: set[tuple[int, int, int]] = set()
        total_bytes = 0
        started = time.perf_counter()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("", int(endpoint["port"])))
            server.listen(len(expected_sources))
            server.settimeout(timeout_s)
            for _ in expected_sources:
                conn, _addr = server.accept()
                with conn:
                    conn.settimeout(timeout_s)
                    artifact_payload, sha256, slice_metadata = (
                        _socket_recv_activation_blob_with_metadata(conn)
                    )
                source_dp_rank = int(slice_metadata.get("source_dp_rank", -1))
                received_destination_dp_rank = int(
                    slice_metadata.get("destination_dp_rank", -1)
                )
                global_start = int(slice_metadata.get("global_start", -1))
                global_end = int(slice_metadata.get("global_end", -1))
                source_range = (source_dp_rank, global_start, global_end)
                if received_destination_dp_rank != destination_dp_rank:
                    raise RuntimeConfigError(
                        f"tcp_direct repartition slice targets destination DP rank "
                        f"{received_destination_dp_rank}, expected {destination_dp_rank}"
                    )
                if source_range not in expected_set or source_range in seen:
                    raise RuntimeConfigError(
                        f"unexpected or duplicate tcp_direct repartition slice from "
                        f"source_dp_rank={source_dp_rank} range={global_start}:{global_end}"
                    )
                artifact = torch.load(
                    io.BytesIO(artifact_payload),
                    map_location="cpu",
                    weights_only=False,
                )
                if not isinstance(artifact, dict):
                    raise RuntimeConfigError(
                        "in-memory activation repartition payload must be a mapping"
                    )
                artifact_batch = _artifact_batch_size(artifact)
                if global_index_repartition:
                    _validate_global_index_artifact(
                        artifact,
                        global_batch_size=global_batch_size,
                        expected_start=global_start,
                        expected_end=global_end,
                    )
                elif artifact_batch != global_end - global_start:
                    raise RuntimeConfigError(
                        f"in-memory activation slice from DP rank {source_dp_rank} has "
                        f"batch {artifact_batch}, expected {global_end - global_start}"
                    )
                seen.add(source_range)
                slices.append((global_start, source_dp_rank, artifact, sha256))
                total_bytes += len(artifact_payload)
        if seen != expected_set:
            missing = sorted(expected_set - seen)
            raise RuntimeConfigError(f"tcp_direct repartition did not receive slices {missing}")
        sorted_slices = sorted(slices, key=lambda item: item[0])
        local = _merge_artifact_shards([item[2] for item in sorted_slices])
        local_batch = _artifact_batch_size(local)
        if global_index_repartition:
            _validate_global_index_artifact(
                local,
                global_batch_size=global_batch_size,
                expected_start=destination_start,
                expected_end=destination_end,
            )
            local = _sort_artifact_by_original_index(local)
        elif local_batch != destination_end - destination_start:
            raise RuntimeConfigError(
                f"tcp_direct repartition produced batch {local_batch}, "
                f"expected {destination_end - destination_start}"
            )
        self._save_local_artifact(
            destination_step,
            destination_name,
            local,
            prune_producer_cache=False,
        )
        duration_s = time.perf_counter() - started
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "destination_name": destination_name,
            "copied": True,
            "dp_repartitioned": True,
            "destination_dp_rank": destination_dp_rank,
            "source_dp_size": source_dp_size,
            "source_rank_count": len(expected_sources),
            "bytes": total_bytes,
            "sha256": hashlib.sha256(
                "".join(item[3] for item in sorted_slices).encode("ascii")
            ).hexdigest(),
            "source_shards_read": sorted(item[1] for item in sorted_slices),
            "copy_mode": copy_mode,
            "transport": transport,
            "wire_format": "torch_save_memory",
            "transfer_rank_duration_s": duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(total_bytes, duration_s),
            "rank": self.rank,
        }

    def _send_activation_tcp_repartition(self, request: SemanticRequest) -> dict[str, Any]:
        import torch

        payload = request.payload
        if bool(payload.get("in_memory", False)):
            return self._send_activation_tcp_repartition_memory(request)
        _reject_persisted_global_index_repartition(payload, "tcp_direct")
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        source_name = self._activation_name(str(payload["source_name"]))
        source_step = int(payload.get("source_step", request.step))
        copy_mode = "tcp_direct_repartition"
        transport = "tcp_direct"
        if not self._is_request_dp_artifact_writer(payload, "source"):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        shard_metadata = self._load_artifact_shard_metadata(source_worker_id, source_step, source_name)
        ranges = _metadata_global_ranges(shard_metadata)
        if ranges is None:
            raise RuntimeConfigError("tcp_direct repartition requires activation shard global range metadata")
        source_dp_rank = self._request_dp_rank(payload, "source")
        if source_dp_rank >= len(ranges):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        source_metadata, source_start, source_end = ranges[source_dp_rank]
        global_batch_size = ranges[-1][2]
        destination_dp_size = int(payload["destination_dp_size"])
        destinations = _direct_repartition_destinations(
            source_start=source_start,
            source_end=source_end,
            global_batch_size=global_batch_size,
            destination_dp_size=destination_dp_size,
        )
        if not destinations:
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        source_path = self._artifact_shard_path_for(source_worker_id, source_step, source_name, source_dp_rank)
        source_shard = torch.load(source_path, map_location="cpu", weights_only=False)
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        total_bytes = 0
        total_duration_s = 0.0
        cached_count = 0
        sent_destinations: list[int] = []
        tmp_paths: list[Path] = []
        try:
            for destination_dp_rank, overlap_start, overlap_end in destinations:
                endpoint = _direct_endpoint_for_dp_pair(payload, destination_dp_rank, source_dp_rank)
                local = _slice_artifact_batch(source_shard, overlap_start - source_start, overlap_end - source_start)
                tmp = source_path.with_suffix(
                    source_path.suffix
                    + f".{os.getpid()}.rank{self.rank}.to_dp{destination_dp_rank}.tcp_direct_slice.tmp"
                )
                tmp_paths.append(tmp)
                if tmp.exists():
                    tmp.unlink()
                torch.save(local, tmp)
                bytes_ = tmp.stat().st_size
                sha256 = _file_sha256(tmp)
                metadata = {
                    "schema_version": "rpeg.activation_direct_slice.v1",
                    "source_dp_rank": source_dp_rank,
                    "destination_dp_rank": destination_dp_rank,
                    "global_start": overlap_start,
                    "global_end": overlap_end,
                    "source_sha256": str(source_metadata["sha256"]),
                }
                deadline = time.monotonic() + timeout_s
                last_error: OSError | None = None
                while time.monotonic() < deadline:
                    try:
                        started = time.perf_counter()
                        with socket.create_connection((str(endpoint["host"]), int(endpoint["port"])), timeout=5.0) as sock:
                            _socket_send_activation_file_with_metadata(sock, tmp, bytes_, sha256, metadata)
                        duration_s = time.perf_counter() - started
                        total_bytes += int(bytes_)
                        total_duration_s += duration_s
                        sent_destinations.append(destination_dp_rank)
                        break
                    except OSError as exc:
                        last_error = exc
                        time.sleep(0.05)
                else:
                    raise RuntimeConfigError(
                        f"timed out connecting tcp_direct repartition sender rank {self.rank} to "
                        f"{endpoint['host']}:{endpoint['port']}: {last_error}"
                    )
        finally:
            for tmp in tmp_paths:
                if tmp.exists():
                    tmp.unlink()
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "copied": bool(sent_destinations),
            "bytes": total_bytes,
            "source_sha256": str(source_metadata["sha256"]),
            "copy_mode": copy_mode,
            "transport": transport,
            "transfer_rank_duration_s": total_duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(total_bytes, total_duration_s),
            "destination_dp_ranks": sent_destinations,
            "rank": self.rank,
        }

    def _recv_activation_tcp_repartition(self, request: SemanticRequest) -> dict[str, Any]:
        import torch

        payload = request.payload
        if bool(payload.get("in_memory", False)):
            return self._recv_activation_tcp_repartition_memory(request)
        _reject_persisted_global_index_repartition(payload, "tcp_direct")
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        source_name = self._activation_name(str(payload["source_name"]))
        destination_name = self._activation_name(str(payload.get("destination_name", source_name)))
        source_step = int(payload.get("source_step", request.step))
        destination_step = int(payload.get("destination_step", request.step))
        copy_mode = "tcp_direct_repartition"
        transport = "tcp_direct"
        if not self._is_request_dp_artifact_writer(payload, "destination"):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "destination_name": destination_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        shard_metadata = self._load_artifact_shard_metadata(source_worker_id, source_step, source_name)
        ranges = _metadata_global_ranges(shard_metadata)
        if ranges is None:
            raise RuntimeConfigError("tcp_direct repartition requires activation shard global range metadata")
        global_batch_size = ranges[-1][2]
        destination_dp_rank = self._request_dp_rank(payload, "destination")
        destination_start, destination_end = _batch_range_for_dp(
            global_batch_size,
            int(payload["destination_dp_size"]),
            destination_dp_rank,
        )
        aggregate = bool(payload.get("direct_repartition_aggregate", False))
        expected_sources = (
            [(0, destination_start, destination_end)]
            if aggregate
            else _direct_repartition_sources_for_destination(
                ranges,
                destination_start=destination_start,
                destination_end=destination_end,
            )
        )
        if not expected_sources:
            raise RuntimeConfigError(
                f"no source activation shards overlap destination DP rank {destination_dp_rank} "
                f"for {source_worker_id}:{source_name}@{source_step}"
            )
        endpoint = _direct_endpoint_for_rank(payload, self.rank)
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        tmp_paths: list[Path] = []
        slices: list[tuple[int, dict[str, Any]]] = []
        total_bytes = 0
        started = time.perf_counter()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind(("", int(endpoint["port"])))
                server.listen(len(expected_sources))
                server.settimeout(timeout_s)
                for _ in expected_sources:
                    conn, _addr = server.accept()
                    with conn:
                        tmp = self._artifact_shard_path_for(
                            self.worker_id,
                            destination_step,
                            destination_name,
                            destination_dp_rank,
                        ).with_suffix(
                            f".dp{destination_dp_rank:05d}.{os.getpid()}.{len(tmp_paths)}.tcp_direct_slice.tmp"
                        )
                        tmp.parent.mkdir(parents=True, exist_ok=True)
                        if tmp.exists():
                            tmp.unlink()
                        tmp_paths.append(tmp)
                        bytes_, _sha256, slice_metadata = _socket_recv_activation_file_with_metadata(conn, tmp)
                    source_dp_rank = int(slice_metadata.get("source_dp_rank", -1))
                    global_start = int(slice_metadata.get("global_start", -1))
                    global_end = int(slice_metadata.get("global_end", -1))
                    if (source_dp_rank, global_start, global_end) not in expected_sources:
                        raise RuntimeConfigError(
                            f"unexpected tcp_direct repartition slice from source_dp_rank={source_dp_rank} "
                            f"range={global_start}:{global_end}"
                        )
                    total_bytes += int(bytes_)
                    slices.append((global_start, torch.load(tmp, map_location="cpu", weights_only=False)))
        finally:
            for tmp in tmp_paths:
                if tmp.exists():
                    tmp.unlink()
        if len(slices) != len(expected_sources):
            raise RuntimeConfigError(
                f"tcp_direct repartition received {len(slices)} slices, expected {len(expected_sources)}"
            )
        duration_s = time.perf_counter() - started
        local = _merge_artifact_shards([data for _start, data in sorted(slices, key=lambda item: item[0])])
        destination_path = self._write_local_artifact_payload(
            step=destination_step,
            name=destination_name,
            data=local,
            transport=transport,
            global_batch_size=global_batch_size,
            global_start=destination_start,
            global_end=destination_end,
        )
        sha256 = _file_sha256(destination_path)
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "destination_name": destination_name,
            "copied": True,
            "dp_repartitioned": True,
            "destination_dp_rank": destination_dp_rank,
            "source_dp_size": len(shard_metadata),
            "source_rank_count": len(expected_sources),
            "bytes": total_bytes,
            "sha256": sha256,
            "source_shards": _artifact_shard_summary(shard_metadata),
            "source_shards_read": [source_dp_rank for source_dp_rank, _start, _end in sorted(expected_sources)],
            "copy_mode": copy_mode,
            "transport": transport,
            "transfer_rank_duration_s": duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(total_bytes, duration_s),
            "rank": self.rank,
        }

    def _send_activation_ucx_repartition(self, request: SemanticRequest) -> dict[str, Any]:
        import torch

        payload = request.payload
        if bool(payload.get("in_memory", False)):
            return self._send_activation_ucx_repartition_memory(request)
        _reject_persisted_global_index_repartition(payload, "ucx_direct")
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        source_name = self._activation_name(str(payload["source_name"]))
        source_step = int(payload.get("source_step", request.step))
        copy_mode = "ucx_direct_repartition"
        transport = "ucx_direct"
        if not self._is_request_dp_artifact_writer(payload, "source"):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        shard_metadata = self._load_artifact_shard_metadata(source_worker_id, source_step, source_name)
        ranges = _metadata_global_ranges(shard_metadata)
        if ranges is None:
            raise RuntimeConfigError("ucx_direct repartition requires activation shard global range metadata")
        source_dp_rank = self._request_dp_rank(payload, "source")
        aggregate = bool(payload.get("direct_repartition_aggregate", False))
        if aggregate and source_dp_rank != 0:
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        if source_dp_rank >= len(ranges):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        source_metadata, source_start, source_end = ranges[source_dp_rank]
        global_batch_size = ranges[-1][2]
        destination_dp_size = int(payload["destination_dp_size"])
        destinations = (
            [
                (
                    destination_dp_rank,
                    *_batch_range_for_dp(global_batch_size, destination_dp_size, destination_dp_rank),
                )
                for destination_dp_rank in range(destination_dp_size)
            ]
            if aggregate
            else _direct_repartition_destinations(
                source_start=source_start,
                source_end=source_end,
                global_batch_size=global_batch_size,
                destination_dp_size=destination_dp_size,
            )
        )
        if not destinations:
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        source_path = self._artifact_shard_path_for(source_worker_id, source_step, source_name, source_dp_rank)
        source_shard = torch.load(source_path, map_location="cpu", weights_only=False)
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        total_bytes = 0
        total_duration_s = 0.0
        cached_count = 0
        sent_destinations: list[int] = []
        tmp_paths: list[Path] = []
        try:
            for destination_dp_rank, overlap_start, overlap_end in destinations:
                endpoint = _direct_endpoint_for_dp_pair(payload, destination_dp_rank, source_dp_rank)
                if aggregate:
                    aggregate_slices = []
                    source_shards_read = []
                    for metadata, shard_start, shard_end in ranges:
                        shard_overlap_start = max(overlap_start, shard_start)
                        shard_overlap_end = min(overlap_end, shard_end)
                        if shard_overlap_start >= shard_overlap_end:
                            continue
                        shard_dp_rank = int(metadata["dp_rank"])
                        source_shards_read.append(shard_dp_rank)
                        shard = torch.load(
                            self._artifact_shard_path_for(source_worker_id, source_step, source_name, shard_dp_rank),
                            map_location="cpu",
                            weights_only=False,
                        )
                        aggregate_slices.append(
                            _slice_artifact_batch(
                                shard,
                                shard_overlap_start - shard_start,
                                shard_overlap_end - shard_start,
                            )
                        )
                    if not aggregate_slices:
                        continue
                    local = _merge_artifact_shards(aggregate_slices)
                else:
                    source_shards_read = [source_dp_rank]
                    local = _slice_artifact_batch(source_shard, overlap_start - source_start, overlap_end - source_start)
                tmp = source_path.with_suffix(
                    source_path.suffix
                    + f".{os.getpid()}.rank{self.rank}.to_dp{destination_dp_rank}.ucx_direct_slice.tmp"
                )
                tmp_paths.append(tmp)
                if tmp.exists():
                    tmp.unlink()
                torch.save(local, tmp)
                bytes_ = tmp.stat().st_size
                sha256 = _file_sha256(tmp)
                metadata = {
                    "schema_version": "rpeg.activation_direct_slice.v1",
                    "source_dp_rank": source_dp_rank,
                    "destination_dp_rank": destination_dp_rank,
                    "global_start": overlap_start,
                    "global_end": overlap_end,
                    "source_sha256": str(source_metadata["sha256"]),
                    "source_shards_read": source_shards_read,
                }
                started = time.perf_counter()
                cached = _run_ucx_direct(
                    _ucx_direct_send_file_with_metadata(
                        endpoint=endpoint,
                        path=tmp,
                        bytes_=bytes_,
                        sha256=sha256,
                        metadata=metadata,
                        timeout_s=timeout_s,
                        ready_path=_direct_ready_marker_path(self.state_root, endpoint),
                        ready_wait_s=_ucx_direct_ready_wait_s(payload),
                        connect_grace_s=_ucx_direct_connect_grace_s(payload),
                    ),
                    persistent=_ucx_direct_channel_reuse(endpoint),
                )
                duration_s = time.perf_counter() - started
                total_bytes += int(bytes_)
                total_duration_s += duration_s
                sent_destinations.append(destination_dp_rank)
                cached_count += int(cached)
        finally:
            for tmp in tmp_paths:
                if tmp.exists():
                    tmp.unlink()
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "copied": bool(sent_destinations),
            "bytes": total_bytes,
            "source_sha256": str(source_metadata["sha256"]),
            "copy_mode": copy_mode,
            "transport": transport,
            "transfer_rank_duration_s": total_duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(total_bytes, total_duration_s),
            "direct_channel_reuse": bool(payload.get("direct_channel_reuse", False)),
            "ucx_direct_cached_endpoint_count": cached_count,
            "ucx_direct_cached_endpoint": cached_count > 0,
            "destination_dp_ranks": sent_destinations,
            "rank": self.rank,
        }

    def _recv_activation_ucx_repartition(self, request: SemanticRequest) -> dict[str, Any]:
        import torch

        payload = request.payload
        if bool(payload.get("in_memory", False)):
            return self._recv_activation_ucx_repartition_memory(request)
        _reject_persisted_global_index_repartition(payload, "ucx_direct")
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        source_name = self._activation_name(str(payload["source_name"]))
        destination_name = self._activation_name(str(payload.get("destination_name", source_name)))
        source_step = int(payload.get("source_step", request.step))
        destination_step = int(payload.get("destination_step", request.step))
        copy_mode = "ucx_direct_repartition"
        transport = "ucx_direct"
        if not self._is_request_dp_artifact_writer(payload, "destination"):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "destination_name": destination_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        shard_metadata = self._load_artifact_shard_metadata(source_worker_id, source_step, source_name)
        ranges = _metadata_global_ranges(shard_metadata)
        if ranges is None:
            raise RuntimeConfigError("ucx_direct repartition requires activation shard global range metadata")
        global_batch_size = ranges[-1][2]
        destination_dp_rank = self._request_dp_rank(payload, "destination")
        destination_start, destination_end = _batch_range_for_dp(
            global_batch_size,
            int(payload["destination_dp_size"]),
            destination_dp_rank,
        )
        aggregate = bool(payload.get("direct_repartition_aggregate", False))
        expected_sources = (
            [(0, destination_start, destination_end)]
            if aggregate
            else _direct_repartition_sources_for_destination(
                ranges,
                destination_start=destination_start,
                destination_end=destination_end,
            )
        )
        if not expected_sources:
            raise RuntimeConfigError(
                f"no source activation shards overlap destination DP rank {destination_dp_rank} "
                f"for {source_worker_id}:{source_name}@{source_step}"
            )
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        base_path = self._artifact_shard_path_for(
            self.worker_id,
            destination_step,
            destination_name,
            destination_dp_rank,
        )
        base_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_paths = [
            base_path.with_suffix(
                f".dp{destination_dp_rank:05d}.{os.getpid()}.{idx}.ucx_direct_slice.tmp"
            )
            for idx in range(len(expected_sources))
        ]
        for tmp in tmp_paths:
            if tmp.exists():
                tmp.unlink()
        started = time.perf_counter()
        try:
            pair_endpoints = _direct_has_all_dp_pair_endpoints(
                payload,
                destination_dp_rank=destination_dp_rank,
                source_dp_ranks=[source_dp_rank for source_dp_rank, _start, _end in expected_sources],
            )
            if aggregate or pair_endpoints:
                endpoint_destinations = []
                for idx, (expected_source_dp_rank, _expected_start, _expected_end) in enumerate(sorted(expected_sources)):
                    endpoint = _direct_endpoint_for_dp_pair(payload, destination_dp_rank, expected_source_dp_rank)
                    endpoint_destinations.append((endpoint, tmp_paths[idx]))
                received, cached_count = _run_ucx_direct(
                    _ucx_direct_recv_endpoint_files_with_metadata(
                        endpoint_destinations=endpoint_destinations,
                        timeout_s=timeout_s,
                        ready_paths=[
                            _direct_ready_marker_path(self.state_root, endpoint)
                            for endpoint, _tmp in endpoint_destinations
                        ],
                    ),
                    persistent=any(_ucx_direct_channel_reuse(endpoint) for endpoint, _tmp in endpoint_destinations),
                )
            else:
                endpoint = _direct_endpoint_for_dp_rank(payload, destination_dp_rank)
                received, cached_count = _run_ucx_direct(
                    _ucx_direct_recv_files_with_metadata(
                        endpoint=endpoint,
                        destinations=tmp_paths,
                        expected_count=len(expected_sources),
                        timeout_s=timeout_s,
                        ready_path=_direct_ready_marker_path(self.state_root, endpoint),
                    ),
                    persistent=_ucx_direct_channel_reuse(endpoint),
                )
            slices: list[tuple[int, dict[str, Any]]] = []
            total_bytes = 0
            source_shards_read: list[int] = []
            for tmp, bytes_, _sha256, slice_metadata in received:
                source_dp_rank = int(slice_metadata.get("source_dp_rank", -1))
                global_start = int(slice_metadata.get("global_start", -1))
                global_end = int(slice_metadata.get("global_end", -1))
                if (source_dp_rank, global_start, global_end) not in expected_sources:
                    raise RuntimeConfigError(
                        f"unexpected ucx_direct repartition slice from source_dp_rank={source_dp_rank} "
                        f"range={global_start}:{global_end}"
                    )
                total_bytes += int(bytes_)
                raw_source_shards_read = slice_metadata.get("source_shards_read")
                if isinstance(raw_source_shards_read, list):
                    source_shards_read.extend(int(rank) for rank in raw_source_shards_read)
                else:
                    source_shards_read.append(source_dp_rank)
                slices.append((global_start, torch.load(tmp, map_location="cpu", weights_only=False)))
        finally:
            for tmp in tmp_paths:
                if tmp.exists():
                    tmp.unlink()
        if len(slices) != len(expected_sources):
            raise RuntimeConfigError(
                f"ucx_direct repartition received {len(slices)} slices, expected {len(expected_sources)}"
            )
        duration_s = time.perf_counter() - started
        local = _merge_artifact_shards([data for _start, data in sorted(slices, key=lambda item: item[0])])
        destination_path = self._write_local_artifact_payload(
            step=destination_step,
            name=destination_name,
            data=local,
            transport=transport,
            global_batch_size=global_batch_size,
            global_start=destination_start,
            global_end=destination_end,
        )
        sha256 = _file_sha256(destination_path)
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "destination_name": destination_name,
            "copied": True,
            "dp_repartitioned": True,
            "destination_dp_rank": destination_dp_rank,
            "source_dp_size": len(shard_metadata),
            "source_rank_count": len(expected_sources),
            "bytes": total_bytes,
            "sha256": sha256,
            "source_shards": _artifact_shard_summary(shard_metadata),
            "source_shards_read": sorted(set(source_shards_read)),
            "copy_mode": copy_mode,
            "transport": transport,
            "transfer_rank_duration_s": duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(total_bytes, duration_s),
            "direct_channel_reuse": bool(payload.get("direct_channel_reuse", False)),
            "ucx_direct_cached_endpoint_count": cached_count,
            "ucx_direct_cached_endpoint": cached_count > 0,
            "rank": self.rank,
        }

    def _send_activation_ucx_repartition_memory(self, request: SemanticRequest) -> dict[str, Any]:
        payload = request.payload
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        source_name = self._activation_name(str(payload["source_name"]))
        source_step = int(payload.get("source_step", request.step))
        transport = "ucx_direct"
        if not self._is_request_dp_artifact_writer(payload, "source"):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        source_dp_size = int(payload["source_dp_size"])
        destination_dp_size = int(payload["destination_dp_size"])
        global_batch_size = int(payload["global_batch_size"])
        global_index_repartition = bool(payload.get("global_index_repartition", False))
        source_dp_rank = self._request_dp_rank(payload, "source")
        source_start, source_end = _batch_range_for_dp(
            global_batch_size,
            source_dp_size,
            source_dp_rank,
        )
        source_shard = self._load_local_artifact_for_layout(
            source_step,
            source_name,
            self._request_parallelism(payload, "source"),
        )
        if global_index_repartition:
            _validate_global_index_artifact(
                source_shard,
                global_batch_size=global_batch_size,
                expected_start=source_start,
                expected_end=source_end,
            )
        elif _artifact_batch_size(source_shard) != source_end - source_start:
            raise RuntimeConfigError(
                f"in-memory activation shard {source_worker_id}:{source_name}@{source_step} "
                f"has batch {_artifact_batch_size(source_shard)}, expected {source_end - source_start}"
            )
        destinations = _direct_repartition_destinations(
            source_start=source_start,
            source_end=source_end,
            global_batch_size=global_batch_size,
            destination_dp_size=destination_dp_size,
        )
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        total_bytes = 0
        total_duration_s = 0.0
        cached_count = 0
        sent_destinations = []
        for destination_dp_rank, overlap_start, overlap_end in destinations:
            local = (
                _slice_artifact_global_index_range(
                    source_shard,
                    overlap_start,
                    overlap_end,
                    global_batch_size=global_batch_size,
                )
                if global_index_repartition
                else _slice_artifact_batch(
                    source_shard,
                    overlap_start - source_start,
                    overlap_end - source_start,
                )
            )
            artifact_record = _activation_artifact_to_tensor_record(local)
            endpoint = _direct_endpoint_for_dp_pair(
                payload,
                destination_dp_rank,
                source_dp_rank,
            )
            started = time.perf_counter()
            cached = _run_ucx_direct(
                _ucx_direct_send_tensor_record(
                    endpoint=endpoint,
                    record=artifact_record,
                    timeout_s=timeout_s,
                    ready_path=_direct_ready_marker_path(self.state_root, endpoint),
                    ready_wait_s=_ucx_direct_ready_wait_s(payload),
                    connect_grace_s=_ucx_direct_connect_grace_s(payload),
                    schema_version=_ACTIVATION_TENSOR_STREAM_SCHEMA,
                ),
                persistent=_ucx_direct_channel_reuse(endpoint),
            )
            duration_s = time.perf_counter() - started
            manifest, tensor_bytes = _tensor_stream_manifest(
                artifact_record,
                schema_version=_ACTIVATION_TENSOR_STREAM_SCHEMA,
            )
            total_bytes += tensor_bytes + len(_tensor_stream_manifest_bytes(manifest))
            total_duration_s += duration_s
            cached_count += int(cached)
            sent_destinations.append(destination_dp_rank)
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "copied": bool(sent_destinations),
            "bytes": total_bytes,
            "copy_mode": "ucx_direct_memory_repartition",
            "transport": transport,
            "wire_format": "tensor_record_v1",
            "sha256_scope": "manifest",
            "transfer_rank_duration_s": total_duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(total_bytes, total_duration_s),
            "direct_channel_reuse": bool(payload.get("direct_channel_reuse", False)),
            "ucx_direct_cached_endpoint_count": cached_count,
            "ucx_direct_cached_endpoint": cached_count > 0,
            "destination_dp_ranks": sent_destinations,
            "rank": self.rank,
        }

    def _recv_activation_ucx_repartition_memory(self, request: SemanticRequest) -> dict[str, Any]:
        payload = request.payload
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        source_name = self._activation_name(str(payload["source_name"]))
        destination_name = self._activation_name(str(payload.get("destination_name", source_name)))
        source_step = int(payload.get("source_step", request.step))
        destination_step = int(payload.get("destination_step", request.step))
        transport = "ucx_direct"
        if not self._is_request_dp_artifact_writer(payload, "destination"):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "destination_name": destination_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        source_dp_size = int(payload["source_dp_size"])
        destination_dp_size = int(payload["destination_dp_size"])
        global_batch_size = int(payload["global_batch_size"])
        global_index_repartition = bool(payload.get("global_index_repartition", False))
        destination_dp_rank = self._request_dp_rank(payload, "destination")
        destination_start, destination_end = _batch_range_for_dp(
            global_batch_size,
            destination_dp_size,
            destination_dp_rank,
        )
        ranges = [
            (
                {
                    "dp_rank": source_dp_rank,
                    "global_start": source_start,
                    "global_end": source_end,
                },
                source_start,
                source_end,
            )
            for source_dp_rank in range(source_dp_size)
            for source_start, source_end in [
                _batch_range_for_dp(global_batch_size, source_dp_size, source_dp_rank)
            ]
        ]
        expected_sources = _direct_repartition_sources_for_destination(
            ranges,
            destination_start=destination_start,
            destination_end=destination_end,
        )
        if not expected_sources:
            raise RuntimeConfigError(
                f"no in-memory activation shards overlap destination DP rank {destination_dp_rank}"
            )
        endpoint_sources = [
            (
                _direct_endpoint_for_dp_pair(payload, destination_dp_rank, source_dp_rank),
                source_dp_rank,
                overlap_start,
                overlap_end,
            )
            for source_dp_rank, overlap_start, overlap_end in sorted(expected_sources)
        ]
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        started = time.perf_counter()
        received = _run_ucx_direct(
            _ucx_direct_recv_endpoint_tensor_records(
                endpoint_sources=endpoint_sources,
                timeout_s=timeout_s,
                state_root=self.state_root,
                device=self._weight_receive_device(),
                expected_schema=_ACTIVATION_TENSOR_STREAM_SCHEMA,
            ),
            persistent=any(
                _ucx_direct_channel_reuse(endpoint)
                for endpoint, _source_rank, _start, _end in endpoint_sources
            ),
        )
        slices = []
        total_bytes = 0
        cached_count = 0
        slice_hashes = []
        for (
            source_dp_rank,
            overlap_start,
            overlap_end,
            artifact_record,
            bytes_,
            sha256,
            cached,
        ) in received:
            artifact = _activation_artifact_from_tensor_record(artifact_record)
            if global_index_repartition:
                _validate_global_index_artifact(
                    artifact,
                    global_batch_size=global_batch_size,
                    expected_start=overlap_start,
                    expected_end=overlap_end,
                )
            elif _artifact_batch_size(artifact) != overlap_end - overlap_start:
                raise RuntimeConfigError(
                    f"in-memory activation slice from DP rank {source_dp_rank} has batch "
                    f"{_artifact_batch_size(artifact)}, expected {overlap_end - overlap_start}"
                )
            slices.append((overlap_start, artifact))
            total_bytes += int(bytes_)
            slice_hashes.append(sha256)
            cached_count += int(cached)
        local = _merge_artifact_shards(
            [artifact for _start, artifact in sorted(slices, key=lambda item: item[0])]
        )
        if global_index_repartition:
            _validate_global_index_artifact(
                local,
                global_batch_size=global_batch_size,
                expected_start=destination_start,
                expected_end=destination_end,
            )
            local = _sort_artifact_by_original_index(local)
        elif _artifact_batch_size(local) != destination_end - destination_start:
            raise RuntimeConfigError(
                f"ucx_direct repartition produced batch {_artifact_batch_size(local)}, "
                f"expected {destination_end - destination_start}"
            )
        self._save_local_artifact(
            destination_step,
            destination_name,
            local,
            prune_producer_cache=False,
        )
        duration_s = time.perf_counter() - started
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "destination_name": destination_name,
            "copied": True,
            "dp_repartitioned": True,
            "destination_dp_rank": destination_dp_rank,
            "source_dp_size": source_dp_size,
            "source_rank_count": len(expected_sources),
            "bytes": total_bytes,
            "sha256": hashlib.sha256("".join(slice_hashes).encode("ascii")).hexdigest(),
            "source_shards_read": sorted(item[0] for item in received),
            "copy_mode": "ucx_direct_memory_repartition",
            "transport": transport,
            "wire_format": "tensor_record_v1",
            "sha256_scope": "manifest",
            "transfer_rank_duration_s": duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(total_bytes, duration_s),
            "direct_channel_reuse": bool(payload.get("direct_channel_reuse", False)),
            "ucx_direct_cached_endpoint_count": cached_count,
            "ucx_direct_cached_endpoint": cached_count > 0,
            "rank": self.rank,
        }

    def _send_activation_ucx(self, request: SemanticRequest) -> dict[str, Any]:
        payload = request.payload
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        source_name = self._activation_name(str(payload["source_name"]))
        source_step = int(payload.get("source_step", request.step))
        transport = "ucx_direct"
        if not self._is_request_dp_artifact_writer(payload, "source"):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        endpoint = _direct_endpoint_for_source_rank(payload, self.rank)
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        in_memory = bool(payload.get("in_memory", False))
        if in_memory:
            artifact_record = _activation_artifact_to_tensor_record(
                self._load_local_artifact_for_layout(
                    source_step,
                    source_name,
                    self._request_parallelism(payload, "source"),
                )
            )
            manifest, tensor_bytes = _tensor_stream_manifest(
                artifact_record,
                schema_version=_ACTIVATION_TENSOR_STREAM_SCHEMA,
            )
            manifest_bytes = _tensor_stream_manifest_bytes(manifest)
            bytes_ = tensor_bytes + len(manifest_bytes)
            sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        else:
            source_path, source_metadata = self._direct_activation_source(source_worker_id, source_step, source_name)
            bytes_ = int(source_metadata["bytes"])
            sha256 = str(source_metadata["sha256"])
        started = time.perf_counter()
        send = (
            _ucx_direct_send_tensor_record(
                endpoint=endpoint,
                record=artifact_record,
                timeout_s=timeout_s,
                ready_path=_direct_ready_marker_path(self.state_root, endpoint),
                ready_wait_s=_ucx_direct_ready_wait_s(payload),
                connect_grace_s=_ucx_direct_connect_grace_s(payload),
                schema_version=_ACTIVATION_TENSOR_STREAM_SCHEMA,
            )
            if in_memory
            else _ucx_direct_send_file(
                endpoint=endpoint,
                path=source_path,
                bytes_=bytes_,
                sha256=sha256,
                timeout_s=timeout_s,
                ready_path=_direct_ready_marker_path(self.state_root, endpoint),
                ready_wait_s=_ucx_direct_ready_wait_s(payload),
                connect_grace_s=_ucx_direct_connect_grace_s(payload),
            )
        )
        asynchronous = in_memory and bool(payload.get("source_overlap", False))
        if asynchronous:
            self._launch_background_transfer(
                lambda: _run_ucx_direct(send, persistent=_ucx_direct_channel_reuse(endpoint)),
                name=f"activation-{source_name}-step{source_step}",
            )
            cached = False
        else:
            cached = _run_ucx_direct(
                send,
                persistent=_ucx_direct_channel_reuse(endpoint),
            )
        duration_s = time.perf_counter() - started
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "copied": True,
            "bytes": bytes_,
            "sha256": sha256,
            "copy_mode": "ucx_direct_memory" if in_memory else transport,
            "transport": transport,
            "wire_format": "tensor_record_v1" if in_memory else "serialized_file",
            "sha256_scope": "manifest" if in_memory else "payload",
            "direct_dp_writer_copy": bool(payload.get("direct_dp_writer_copy", False)),
            "transfer_rank_duration_s": duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(bytes_, duration_s),
            "direct_channel_reuse": bool(payload.get("direct_channel_reuse", False)),
            "source_overlap": asynchronous,
            "ucx_direct_cached_endpoint_count": int(cached),
            "ucx_direct_cached_endpoint": bool(cached),
            "rank": self.rank,
        }

    def _recv_activation_ucx(self, request: SemanticRequest) -> dict[str, Any]:
        payload = request.payload
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        source_name = self._activation_name(str(payload["source_name"]))
        destination_name = self._activation_name(str(payload.get("destination_name", source_name)))
        source_step = int(payload.get("source_step", request.step))
        destination_step = int(payload.get("destination_step", request.step))
        transport = "ucx_direct"
        if not self._is_request_dp_artifact_writer(payload, "destination"):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "destination_name": destination_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        destination_dp_rank = self._request_dp_rank(payload, "destination")
        in_memory = bool(payload.get("in_memory", False))
        endpoint = _direct_endpoint_for_rank(payload, self.rank)
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        if in_memory:
            started = time.perf_counter()
            artifact_record, bytes_, sha256, cached = _run_ucx_direct(
                _ucx_direct_recv_tensor_record(
                    endpoint=endpoint,
                    timeout_s=timeout_s,
                    ready_path=_direct_ready_marker_path(self.state_root, endpoint),
                    device=self._weight_receive_device(),
                    expected_schema=_ACTIVATION_TENSOR_STREAM_SCHEMA,
                ),
                persistent=_ucx_direct_channel_reuse(endpoint),
            )
            self._store_tensor_artifact(
                destination_step,
                destination_name,
                artifact_record,
            )
            duration_s = time.perf_counter() - started
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "destination_name": destination_name,
                "copied": True,
                "dp_repartitioned": False,
                "destination_dp_rank": destination_dp_rank,
                "source_dp_size": self._request_parallelism(payload, "source")[0],
                "bytes": bytes_,
                "sha256": sha256,
                "source_sha256": sha256,
                "copy_mode": "ucx_direct_memory",
                "transport": transport,
                "wire_format": "tensor_record_v1",
                "sha256_scope": "manifest",
                "direct_dp_writer_copy": bool(payload.get("direct_dp_writer_copy", False)),
                "transfer_rank_duration_s": duration_s,
                "transfer_rank_bytes_per_s": _bytes_per_second(bytes_, duration_s),
                "direct_channel_reuse": bool(payload.get("direct_channel_reuse", False)),
                "ucx_direct_cached_endpoint_count": int(cached),
                "ucx_direct_cached_endpoint": bool(cached),
                "rank": self.rank,
            }
        sharded = self._artifact_manifest_path_for(source_worker_id, source_step, source_name).exists()
        source_metadata = self._direct_activation_source_metadata(source_worker_id, source_step, source_name)
        destination_path = self._direct_activation_destination_path(destination_step, destination_name, sharded=sharded)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination_path.with_suffix(destination_path.suffix + f".{os.getpid()}.{self.rank}.ucx_direct.tmp")
        if tmp.exists():
            tmp.unlink()
        expected_bytes = int(source_metadata["bytes"])
        started = time.perf_counter()
        try:
            bytes_, sha256, cached = _run_ucx_direct(
                _ucx_direct_recv_file(
                    endpoint=endpoint,
                    destination=tmp,
                    bytes_=expected_bytes,
                    timeout_s=timeout_s,
                    ready_path=_direct_ready_marker_path(self.state_root, endpoint),
                ),
                persistent=_ucx_direct_channel_reuse(endpoint),
            )
            expected_sha = str(source_metadata["sha256"])
            if sha256 != expected_sha:
                raise RuntimeConfigError(
                    f"ucx_direct activation checksum mismatch for {destination_path}: {sha256} != {expected_sha}"
                )
            tmp.replace(destination_path)
        finally:
            if tmp.exists():
                tmp.unlink()
        duration_s = time.perf_counter() - started
        self._write_direct_activation_metadata(
            source_worker_id=source_worker_id,
            source_step=source_step,
            source_name=source_name,
            destination_step=destination_step,
            destination_name=destination_name,
            source_metadata=source_metadata,
            bytes_=bytes_,
            sha256=sha256,
            transport=transport,
        )
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "destination_name": destination_name,
            "copied": True,
            "dp_repartitioned": False,
            "destination_dp_rank": destination_dp_rank,
            "source_dp_size": self._request_parallelism(payload, "source")[0],
            "bytes": int(bytes_),
            "sha256": sha256,
            "source_sha256": expected_sha,
            "copy_mode": transport,
            "transport": transport,
            "direct_dp_writer_copy": bool(payload.get("direct_dp_writer_copy", False)),
            "transfer_rank_duration_s": duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(bytes_, duration_s),
            "direct_channel_reuse": bool(payload.get("direct_channel_reuse", False)),
            "ucx_direct_cached_endpoint_count": int(cached),
            "ucx_direct_cached_endpoint": bool(cached),
            "rank": self.rank,
        }

    def _send_activation_torch_dist(self, request: SemanticRequest) -> dict[str, Any]:
        payload = request.payload
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        source_name = self._activation_name(str(payload["source_name"]))
        source_step = int(payload.get("source_step", request.step))
        backend = _torch_direct_backend(payload)
        transport = f"torch_dist_{backend}"
        if not self._is_request_dp_artifact_writer(payload, "source"):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        endpoint = _direct_endpoint_for_source_rank(payload, self.rank)
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        in_memory = bool(payload.get("in_memory", False))
        if in_memory:
            artifact_record = _activation_artifact_to_tensor_record(
                self._load_local_artifact_for_layout(
                    source_step,
                    source_name,
                    self._request_parallelism(payload, "source"),
                )
            )
            manifest, tensor_bytes = _tensor_stream_manifest(
                artifact_record,
                schema_version=_ACTIVATION_TENSOR_STREAM_SCHEMA,
            )
            manifest_bytes = _tensor_stream_manifest_bytes(manifest)
            bytes_ = tensor_bytes + len(manifest_bytes)
            sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        else:
            source_path, source_metadata = self._direct_activation_source(
                source_worker_id,
                source_step,
                source_name,
            )
            bytes_ = int(source_metadata["bytes"])
            sha256 = str(source_metadata["sha256"])
        started = time.perf_counter()
        pg = _torch_direct_process_group(
            endpoint=endpoint,
            backend=backend,
            pg_rank=0,
            timeout_s=timeout_s,
        )
        try:
            if in_memory:
                sent_bytes, sent_sha256 = _torch_direct_send_tensor_record(
                    pg,
                    artifact_record,
                    backend,
                    schema_version=_ACTIVATION_TENSOR_STREAM_SCHEMA,
                )
                if sent_bytes != bytes_ or sent_sha256 != sha256:
                    raise RuntimeConfigError(
                        "activation tensor manifest changed while sending"
                    )
            else:
                _torch_direct_send_file(
                    pg,
                    backend,
                    source_path,
                    bytes_,
                    ack_endpoint=endpoint,
                    timeout_s=timeout_s,
                )
            _torch_direct_barrier(pg)
        finally:
            _torch_direct_close(pg)
        duration_s = time.perf_counter() - started
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "copied": True,
            "bytes": bytes_,
            "sha256": sha256,
            "copy_mode": f"{transport}_memory" if in_memory else transport,
            "transport": transport,
            "wire_format": "tensor_record_v1" if in_memory else "serialized_file",
            "sha256_scope": "manifest" if in_memory else "payload",
            "direct_dp_writer_copy": bool(payload.get("direct_dp_writer_copy", False)),
            "transfer_rank_duration_s": duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(bytes_, duration_s),
            "rank": self.rank,
        }

    def _recv_activation_torch_dist(self, request: SemanticRequest) -> dict[str, Any]:
        import torch

        payload = request.payload
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        source_name = self._activation_name(str(payload["source_name"]))
        destination_name = self._activation_name(str(payload.get("destination_name", source_name)))
        source_step = int(payload.get("source_step", request.step))
        destination_step = int(payload.get("destination_step", request.step))
        backend = _torch_direct_backend(payload)
        transport = f"torch_dist_{backend}"
        if not self._is_request_dp_artifact_writer(payload, "destination"):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "destination_name": destination_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        destination_dp_rank = self._request_dp_rank(payload, "destination")
        endpoint = _direct_endpoint_for_rank(payload, self.rank)
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        in_memory = bool(payload.get("in_memory", False))
        if not in_memory:
            sharded = self._artifact_manifest_path_for(source_worker_id, source_step, source_name).exists()
            source_metadata = self._direct_activation_source_metadata(
                source_worker_id,
                source_step,
                source_name,
            )
            destination_path = self._direct_activation_destination_path(
                destination_step,
                destination_name,
                sharded=sharded,
            )
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = destination_path.with_suffix(
                destination_path.suffix + f".{os.getpid()}.{self.rank}.torch_dist.tmp"
            )
            if tmp.exists():
                tmp.unlink()
            expected_bytes = int(source_metadata["bytes"])
        started = time.perf_counter()
        pg = _torch_direct_process_group(
            endpoint=endpoint,
            backend=backend,
            pg_rank=1,
            timeout_s=timeout_s,
        )
        try:
            if in_memory:
                receive_device = (
                    torch.device("cuda", torch.cuda.current_device())
                    if backend == "nccl"
                    else torch.device("cpu")
                )
                artifact_record, bytes_, sha256 = _torch_direct_recv_tensor_record(
                    pg,
                    backend,
                    device=receive_device,
                    expected_schema=_ACTIVATION_TENSOR_STREAM_SCHEMA,
                )
                expected_sha = sha256
                self._store_tensor_artifact(
                    destination_step,
                    destination_name,
                    artifact_record,
                )
            else:
                bytes_, sha256 = _torch_direct_recv_file(
                    pg,
                    backend,
                    tmp,
                    expected_bytes,
                    ack_endpoint=endpoint,
                    timeout_s=timeout_s,
                )
                expected_sha = str(source_metadata["sha256"])
                if sha256 != expected_sha:
                    raise RuntimeConfigError(
                        f"torch_dist activation checksum mismatch for {destination_path}: "
                        f"{sha256} != {expected_sha}"
                    )
                tmp.replace(destination_path)
            _torch_direct_barrier(pg)
        finally:
            _torch_direct_close(pg)
            if not in_memory and tmp.exists():
                tmp.unlink()
        duration_s = time.perf_counter() - started
        if not in_memory:
            self._write_direct_activation_metadata(
                source_worker_id=source_worker_id,
                source_step=source_step,
                source_name=source_name,
                destination_step=destination_step,
                destination_name=destination_name,
                source_metadata=source_metadata,
                bytes_=bytes_,
                sha256=sha256,
                transport=transport,
            )
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "destination_name": destination_name,
            "copied": True,
            "dp_repartitioned": False,
            "destination_dp_rank": destination_dp_rank,
            "source_dp_size": self._request_parallelism(payload, "source")[0],
            "bytes": int(bytes_),
            "sha256": sha256,
            "source_sha256": expected_sha,
            "copy_mode": f"{transport}_memory" if in_memory else transport,
            "transport": transport,
            "wire_format": "tensor_record_v1" if in_memory else "serialized_file",
            "sha256_scope": "manifest" if in_memory else "payload",
            "direct_dp_writer_copy": bool(payload.get("direct_dp_writer_copy", False)),
            "transfer_rank_duration_s": duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(bytes_, duration_s),
            "rank": self.rank,
        }

    def _send_activation_torch_dist_collective_repartition_memory(
        self,
        request: SemanticRequest,
    ) -> dict[str, Any]:
        import torch

        payload = request.payload
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        source_name = self._activation_name(str(payload["source_name"]))
        source_step = int(payload.get("source_step", request.step))
        backend = _torch_direct_backend(payload)
        transport = f"torch_dist_{backend}"
        copy_mode = (
            f"{transport}_collective_memory_repartition"
            if bool(payload.get("in_memory", False))
            else f"{transport}_collective_repartition"
        )
        if not self._is_request_dp_artifact_writer(payload, "source"):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        endpoint = payload.get("direct_collective_endpoint")
        if not isinstance(endpoint, dict):
            raise RuntimeConfigError("collective activation repartition requires a direct collective endpoint")
        source_dp_size = int(payload["source_dp_size"])
        destination_dp_size = int(payload["destination_dp_size"])
        world_size = source_dp_size + destination_dp_size
        if int(endpoint.get("world_size", -1)) != world_size:
            raise RuntimeConfigError(
                f"collective activation group has world_size={endpoint.get('world_size')}, expected {world_size}"
            )
        global_batch_size = int(payload["global_batch_size"])
        global_index_repartition = bool(payload.get("global_index_repartition", False))
        source_dp_rank = self._request_dp_rank(payload, "source")
        source_start, source_end = _batch_range_for_dp(
            global_batch_size,
            source_dp_size,
            source_dp_rank,
        )
        source_shard = self._load_local_artifact_for_layout(
            source_step,
            source_name,
            self._request_parallelism(payload, "source"),
        )
        expected_batch = source_end - source_start
        if global_index_repartition:
            _validate_global_index_artifact(
                source_shard,
                global_batch_size=global_batch_size,
                expected_start=source_start,
                expected_end=source_end,
            )
        elif _artifact_batch_size(source_shard) != expected_batch:
            raise RuntimeConfigError(
                f"in-memory activation shard {source_worker_id}:{source_name}@{source_step} "
                f"has batch {_artifact_batch_size(source_shard)}, expected {expected_batch}"
            )
        destinations = _direct_repartition_destinations(
            source_start=source_start,
            source_end=source_end,
            global_batch_size=global_batch_size,
            destination_dp_size=destination_dp_size,
        )
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        started = time.perf_counter()
        received_bytes = 0
        pg = _torch_direct_process_group(
            endpoint=endpoint,
            backend=backend,
            pg_rank=source_dp_rank,
            timeout_s=timeout_s,
        )
        try:
            if backend == "nccl":
                outgoing = []
                slice_hashes = []
                for destination_dp_rank, overlap_start, overlap_end in destinations:
                    local = (
                        _slice_artifact_global_index_range(
                            source_shard,
                            overlap_start,
                            overlap_end,
                            global_batch_size=global_batch_size,
                        )
                        if global_index_repartition
                        else _slice_artifact_batch(
                            source_shard,
                            overlap_start - source_start,
                            overlap_end - source_start,
                        )
                    )
                    record = _activation_artifact_to_tensor_record(local)
                    manifest, _tensor_bytes = _tensor_stream_manifest(
                        record,
                        schema_version=_ACTIVATION_TENSOR_STREAM_SCHEMA,
                    )
                    slice_hashes.append(
                        hashlib.sha256(_tensor_stream_manifest_bytes(manifest)).hexdigest()
                    )
                    outgoing.append((
                        {"destination_group_rank": source_dp_size + destination_dp_rank},
                        record,
                    ))
                _received, sent_bytes = _torch_direct_exchange_tensor_records_coalesced(
                    pg,
                    outgoing,
                    [],
                    device=torch.device("cuda", torch.cuda.current_device()),
                    schema_version=_ACTIVATION_TENSOR_STREAM_SCHEMA,
                    expected_schema=_ACTIVATION_TENSOR_STREAM_SCHEMA,
                )
                local_bytes = int(sent_bytes)
                local_sha256 = hashlib.sha256(
                    "".join(slice_hashes).encode("ascii")
                ).hexdigest()
                logical_network_bytes = local_bytes
                wire_format = "tensor_record_v1"
                sha256_scope = "manifest"
            else:
                buffer = io.BytesIO()
                torch.save(source_shard, buffer)
                local_payload = buffer.getvalue()
                local_sha256 = hashlib.sha256(local_payload).hexdigest()
                for root_rank in range(source_dp_size):
                    received, _sha256 = _torch_direct_collective_broadcast_blob(
                        pg,
                        backend,
                        pg_rank=source_dp_rank,
                        root_rank=root_rank,
                        payload=local_payload if root_rank == source_dp_rank else None,
                        sha256=local_sha256 if root_rank == source_dp_rank else None,
                    )
                    if root_rank != source_dp_rank:
                        received_bytes += len(received)
                local_bytes = len(local_payload)
                logical_network_bytes = local_bytes * max(0, world_size - 1)
                wire_format = "torch_save_memory"
                sha256_scope = "payload"
            _torch_direct_barrier(pg)
        finally:
            _torch_direct_close(pg)
        duration_s = time.perf_counter() - started
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "copied": bool(destinations),
            "bytes": local_bytes,
            "sha256": local_sha256,
            "copy_mode": copy_mode,
            "transport": transport,
            "wire_format": wire_format,
            "sha256_scope": sha256_scope,
            "transfer_rank_duration_s": duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(local_bytes, duration_s),
            "logical_network_bytes": logical_network_bytes,
            "collective_received_bytes": received_bytes,
            "collective_participant_count": world_size,
            "direct_channel_reuse": bool(endpoint.get("direct_channel_reuse", False)),
            "destination_dp_ranks": [rank for rank, _start, _end in destinations],
            "rank": self.rank,
        }

    def _recv_activation_torch_dist_collective_repartition_memory(
        self,
        request: SemanticRequest,
    ) -> dict[str, Any]:
        import torch

        payload = request.payload
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        source_name = self._activation_name(str(payload["source_name"]))
        destination_name = self._activation_name(str(payload.get("destination_name", source_name)))
        destination_step = int(payload.get("destination_step", request.step))
        backend = _torch_direct_backend(payload)
        transport = f"torch_dist_{backend}"
        copy_mode = (
            f"{transport}_collective_memory_repartition"
            if bool(payload.get("in_memory", False))
            else f"{transport}_collective_repartition"
        )
        if not self._is_request_dp_artifact_writer(payload, "destination"):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "destination_name": destination_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        endpoint = payload.get("direct_collective_endpoint")
        if not isinstance(endpoint, dict):
            raise RuntimeConfigError("collective activation repartition requires a direct collective endpoint")
        source_dp_size = int(payload["source_dp_size"])
        destination_dp_size = int(payload["destination_dp_size"])
        world_size = source_dp_size + destination_dp_size
        if int(endpoint.get("world_size", -1)) != world_size:
            raise RuntimeConfigError(
                f"collective activation group has world_size={endpoint.get('world_size')}, expected {world_size}"
            )
        destination_dp_rank = self._request_dp_rank(payload, "destination")
        pg_rank = source_dp_size + destination_dp_rank
        global_batch_size = int(payload["global_batch_size"])
        global_index_repartition = bool(payload.get("global_index_repartition", False))
        destination_start, destination_end = _batch_range_for_dp(
            global_batch_size,
            destination_dp_size,
            destination_dp_rank,
        )
        expected_sources = {
            source_dp_rank: (overlap_start, overlap_end)
            for source_dp_rank, overlap_start, overlap_end in _direct_repartition_sources_for_destination(
                [
                    (
                        {"dp_rank": source_dp_rank},
                        *_batch_range_for_dp(global_batch_size, source_dp_size, source_dp_rank),
                    )
                    for source_dp_rank in range(source_dp_size)
                ],
                destination_start=destination_start,
                destination_end=destination_end,
            )
        }
        if not expected_sources:
            raise RuntimeConfigError(
                f"no in-memory activation shards overlap destination DP rank {destination_dp_rank}"
            )
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        started = time.perf_counter()
        slices: list[tuple[int, dict[str, Any]]] = []
        total_wire_bytes = 0
        useful_bytes = 0
        source_shards_read: list[int] = []
        slice_hashes: list[str] = []
        pg = _torch_direct_process_group(
            endpoint=endpoint,
            backend=backend,
            pg_rank=pg_rank,
            timeout_s=timeout_s,
        )
        try:
            if backend == "nccl":
                ordered_sources = sorted(expected_sources)
                received_records, _sent_bytes = _torch_direct_exchange_tensor_records_coalesced(
                    pg,
                    [],
                    [
                        {"source_group_rank": source_dp_rank}
                        for source_dp_rank in ordered_sources
                    ],
                    device=torch.device("cuda", torch.cuda.current_device()),
                    schema_version=_ACTIVATION_TENSOR_STREAM_SCHEMA,
                    expected_schema=_ACTIVATION_TENSOR_STREAM_SCHEMA,
                )
                for source_dp_rank, (record, bytes_, sha256) in zip(
                    ordered_sources,
                    received_records,
                ):
                    overlap_start, overlap_end = expected_sources[source_dp_rank]
                    artifact = _activation_artifact_from_tensor_record(record)
                    if global_index_repartition:
                        _validate_global_index_artifact(
                            artifact,
                            global_batch_size=global_batch_size,
                            expected_start=overlap_start,
                            expected_end=overlap_end,
                        )
                    elif _artifact_batch_size(artifact) != overlap_end - overlap_start:
                        raise RuntimeConfigError(
                            f"collective activation slice from DP rank {source_dp_rank} has batch "
                            f"{_artifact_batch_size(artifact)}, expected {overlap_end - overlap_start}"
                        )
                    slices.append((overlap_start, artifact))
                    total_wire_bytes += int(bytes_)
                    useful_bytes += int(bytes_)
                    source_shards_read.append(source_dp_rank)
                    slice_hashes.append(sha256)
                wire_format = "tensor_record_v1"
                sha256_scope = "manifest"
            else:
                for source_dp_rank in range(source_dp_size):
                    artifact_payload, sha256 = _torch_direct_collective_broadcast_blob(
                        pg,
                        backend,
                        pg_rank=pg_rank,
                        root_rank=source_dp_rank,
                    )
                    total_wire_bytes += len(artifact_payload)
                    overlap = expected_sources.get(source_dp_rank)
                    if overlap is None:
                        continue
                    source_start, source_end = _batch_range_for_dp(
                        global_batch_size,
                        source_dp_size,
                        source_dp_rank,
                    )
                    overlap_start, overlap_end = overlap
                    artifact = torch.load(
                        io.BytesIO(artifact_payload),
                        map_location="cpu",
                        weights_only=False,
                    )
                    expected_source_batch = source_end - source_start
                    if not isinstance(artifact, dict):
                        raise RuntimeConfigError("in-memory activation repartition payload must be a mapping")
                    if global_index_repartition:
                        _validate_global_index_artifact(
                            artifact,
                            global_batch_size=global_batch_size,
                            expected_start=source_start,
                            expected_end=source_end,
                        )
                    elif _artifact_batch_size(artifact) != expected_source_batch:
                        raise RuntimeConfigError(
                            f"collective activation shard from DP rank {source_dp_rank} has batch "
                            f"{_artifact_batch_size(artifact)}, expected {expected_source_batch}"
                        )
                    local = (
                        _slice_artifact_global_index_range(
                            artifact,
                            overlap_start,
                            overlap_end,
                            global_batch_size=global_batch_size,
                        )
                        if global_index_repartition
                        else _slice_artifact_batch(
                            artifact,
                            overlap_start - source_start,
                            overlap_end - source_start,
                        )
                    )
                    slices.append((overlap_start, local))
                    useful_bytes += len(artifact_payload)
                    source_shards_read.append(source_dp_rank)
                    slice_hashes.append(sha256)
                wire_format = "torch_save_memory"
                sha256_scope = "payload"
            _torch_direct_barrier(pg)
        finally:
            _torch_direct_close(pg)
        local = _merge_artifact_shards(
            [artifact for _start, artifact in sorted(slices, key=lambda item: item[0])]
        )
        if global_index_repartition:
            _validate_global_index_artifact(
                local,
                global_batch_size=global_batch_size,
                expected_start=destination_start,
                expected_end=destination_end,
            )
            local = _sort_artifact_by_original_index(local)
        elif _artifact_batch_size(local) != destination_end - destination_start:
            raise RuntimeConfigError(
                f"collective activation repartition produced batch {_artifact_batch_size(local)}, "
                f"expected {destination_end - destination_start}"
            )
        self._save_local_artifact(
            destination_step,
            destination_name,
            local,
            prune_producer_cache=False,
        )
        duration_s = time.perf_counter() - started
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "destination_name": destination_name,
            "copied": True,
            "dp_repartitioned": True,
            "destination_dp_rank": destination_dp_rank,
            "source_dp_size": source_dp_size,
            "source_rank_count": len(source_shards_read),
            "bytes": total_wire_bytes,
            "useful_source_bytes": useful_bytes,
            "sha256": hashlib.sha256("".join(slice_hashes).encode("ascii")).hexdigest(),
            "source_shards_read": sorted(source_shards_read),
            "copy_mode": copy_mode,
            "transport": transport,
            "wire_format": wire_format,
            "sha256_scope": sha256_scope,
            "transfer_rank_duration_s": duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(total_wire_bytes, duration_s),
            "collective_participant_count": world_size,
            "direct_channel_reuse": bool(endpoint.get("direct_channel_reuse", False)),
            "rank": self.rank,
        }

    def _send_activation_torch_dist_repartition_memory(
        self,
        request: SemanticRequest,
    ) -> dict[str, Any]:
        import torch

        payload = request.payload
        if isinstance(payload.get("direct_collective_endpoint"), dict):
            return self._send_activation_torch_dist_collective_repartition_memory(request)
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        source_name = self._activation_name(str(payload["source_name"]))
        source_step = int(payload.get("source_step", request.step))
        backend = _torch_direct_backend(payload)
        transport = f"torch_dist_{backend}"
        copy_mode = f"{transport}_memory_repartition"
        if not self._is_request_dp_artifact_writer(payload, "source"):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        source_dp_size = int(payload["source_dp_size"])
        destination_dp_size = int(payload["destination_dp_size"])
        global_batch_size = int(payload["global_batch_size"])
        global_index_repartition = bool(payload.get("global_index_repartition", False))
        source_dp_rank = self._request_dp_rank(payload, "source")
        source_start, source_end = _batch_range_for_dp(
            global_batch_size,
            source_dp_size,
            source_dp_rank,
        )
        source_shard = self._load_local_artifact_for_layout(
            source_step,
            source_name,
            self._request_parallelism(payload, "source"),
        )
        if global_index_repartition:
            _validate_global_index_artifact(
                source_shard,
                global_batch_size=global_batch_size,
                expected_start=source_start,
                expected_end=source_end,
            )
        elif _artifact_batch_size(source_shard) != source_end - source_start:
            raise RuntimeConfigError(
                f"in-memory activation shard {source_worker_id}:{source_name}@{source_step} "
                f"has batch {_artifact_batch_size(source_shard)}, expected {source_end - source_start}"
            )
        destinations = _direct_repartition_destinations(
            source_start=source_start,
            source_end=source_end,
            global_batch_size=global_batch_size,
            destination_dp_size=destination_dp_size,
        )
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        total_bytes = 0
        total_duration_s = 0.0
        sent_destinations: list[int] = []
        for destination_dp_rank, overlap_start, overlap_end in destinations:
            local = (
                _slice_artifact_global_index_range(
                    source_shard,
                    overlap_start,
                    overlap_end,
                    global_batch_size=global_batch_size,
                )
                if global_index_repartition
                else _slice_artifact_batch(
                    source_shard,
                    overlap_start - source_start,
                    overlap_end - source_start,
                )
            )
            artifact_record = _activation_artifact_to_tensor_record(local)
            endpoint = _direct_endpoint_for_dp_pair(
                payload,
                destination_dp_rank,
                source_dp_rank,
            )
            started = time.perf_counter()
            pg = _torch_direct_process_group(
                endpoint=endpoint,
                backend=backend,
                pg_rank=0,
                timeout_s=timeout_s,
            )
            try:
                bytes_, _sha256 = _torch_direct_send_tensor_record(
                    pg,
                    artifact_record,
                    backend,
                    schema_version=_ACTIVATION_TENSOR_STREAM_SCHEMA,
                )
                _torch_direct_barrier(pg)
            finally:
                _torch_direct_close(pg)
            total_duration_s += time.perf_counter() - started
            total_bytes += int(bytes_)
            sent_destinations.append(destination_dp_rank)
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "copied": bool(sent_destinations),
            "bytes": total_bytes,
            "copy_mode": copy_mode,
            "transport": transport,
            "wire_format": "tensor_record_v1",
            "sha256_scope": "manifest",
            "transfer_rank_duration_s": total_duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(total_bytes, total_duration_s),
            "direct_channel_reuse": bool(payload.get("direct_channel_reuse", False)),
            "destination_dp_ranks": sent_destinations,
            "rank": self.rank,
        }

    def _recv_activation_torch_dist_repartition_memory(
        self,
        request: SemanticRequest,
    ) -> dict[str, Any]:
        import torch

        payload = request.payload
        if isinstance(payload.get("direct_collective_endpoint"), dict):
            return self._recv_activation_torch_dist_collective_repartition_memory(request)
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        source_name = self._activation_name(str(payload["source_name"]))
        destination_name = self._activation_name(str(payload.get("destination_name", source_name)))
        destination_step = int(payload.get("destination_step", request.step))
        backend = _torch_direct_backend(payload)
        transport = f"torch_dist_{backend}"
        copy_mode = f"{transport}_memory_repartition"
        if not self._is_request_dp_artifact_writer(payload, "destination"):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "destination_name": destination_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        source_dp_size = int(payload["source_dp_size"])
        destination_dp_size = int(payload["destination_dp_size"])
        global_batch_size = int(payload["global_batch_size"])
        global_index_repartition = bool(payload.get("global_index_repartition", False))
        destination_dp_rank = self._request_dp_rank(payload, "destination")
        destination_start, destination_end = _batch_range_for_dp(
            global_batch_size,
            destination_dp_size,
            destination_dp_rank,
        )
        ranges = [
            (
                {"dp_rank": source_dp_rank},
                *_batch_range_for_dp(global_batch_size, source_dp_size, source_dp_rank),
            )
            for source_dp_rank in range(source_dp_size)
        ]
        expected_sources = _direct_repartition_sources_for_destination(
            ranges,
            destination_start=destination_start,
            destination_end=destination_end,
        )
        if not expected_sources:
            raise RuntimeConfigError(
                f"no in-memory activation shards overlap destination DP rank {destination_dp_rank}"
            )
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        started = time.perf_counter()
        slices: list[tuple[int, dict[str, Any]]] = []
        total_bytes = 0
        source_shards_read: list[int] = []
        slice_hashes: list[str] = []
        receive_device = (
            torch.device("cuda", torch.cuda.current_device())
            if backend == "nccl"
            else torch.device("cpu")
        )
        for source_dp_rank, overlap_start, overlap_end in sorted(expected_sources):
            endpoint = _direct_endpoint_for_dp_pair(
                payload,
                destination_dp_rank,
                source_dp_rank,
            )
            pg = _torch_direct_process_group(
                endpoint=endpoint,
                backend=backend,
                pg_rank=1,
                timeout_s=timeout_s,
            )
            try:
                artifact_record, bytes_, sha256 = _torch_direct_recv_tensor_record(
                    pg,
                    backend,
                    device=receive_device,
                    expected_schema=_ACTIVATION_TENSOR_STREAM_SCHEMA,
                )
                _torch_direct_barrier(pg)
            finally:
                _torch_direct_close(pg)
            artifact = _activation_artifact_from_tensor_record(artifact_record)
            if not isinstance(artifact, dict):
                raise RuntimeConfigError("in-memory activation repartition payload must be a mapping")
            if global_index_repartition:
                _validate_global_index_artifact(
                    artifact,
                    global_batch_size=global_batch_size,
                    expected_start=overlap_start,
                    expected_end=overlap_end,
                )
            elif _artifact_batch_size(artifact) != overlap_end - overlap_start:
                raise RuntimeConfigError(
                    f"in-memory activation slice from DP rank {source_dp_rank} has batch "
                    f"{_artifact_batch_size(artifact)}, expected {overlap_end - overlap_start}"
                )
            slices.append((overlap_start, artifact))
            total_bytes += int(bytes_)
            source_shards_read.append(source_dp_rank)
            slice_hashes.append(sha256)
        local = _merge_artifact_shards(
            [artifact for _start, artifact in sorted(slices, key=lambda item: item[0])]
        )
        if global_index_repartition:
            _validate_global_index_artifact(
                local,
                global_batch_size=global_batch_size,
                expected_start=destination_start,
                expected_end=destination_end,
            )
            local = _sort_artifact_by_original_index(local)
        elif _artifact_batch_size(local) != destination_end - destination_start:
            raise RuntimeConfigError(
                f"torch_dist repartition produced batch {_artifact_batch_size(local)}, "
                f"expected {destination_end - destination_start}"
            )
        self._save_local_artifact(
            destination_step,
            destination_name,
            local,
            prune_producer_cache=False,
        )
        duration_s = time.perf_counter() - started
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "destination_name": destination_name,
            "copied": True,
            "dp_repartitioned": True,
            "destination_dp_rank": destination_dp_rank,
            "source_dp_size": source_dp_size,
            "source_rank_count": len(expected_sources),
            "bytes": total_bytes,
            "sha256": hashlib.sha256("".join(slice_hashes).encode("ascii")).hexdigest(),
            "source_shards_read": sorted(source_shards_read),
            "copy_mode": copy_mode,
            "transport": transport,
            "wire_format": "tensor_record_v1",
            "sha256_scope": "manifest",
            "transfer_rank_duration_s": duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(total_bytes, duration_s),
            "direct_channel_reuse": bool(payload.get("direct_channel_reuse", False)),
            "rank": self.rank,
        }

    def _send_activation_torch_dist_repartition(self, request: SemanticRequest) -> dict[str, Any]:
        import torch

        payload = request.payload
        if isinstance(payload.get("direct_collective_endpoint"), dict):
            return self._send_activation_torch_dist_collective_repartition_memory(request)
        if bool(payload.get("in_memory", False)):
            return self._send_activation_torch_dist_repartition_memory(request)
        _reject_persisted_global_index_repartition(payload, "torch_dist")
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        source_name = self._activation_name(str(payload["source_name"]))
        source_step = int(payload.get("source_step", request.step))
        backend = _torch_direct_backend(payload)
        transport = f"torch_dist_{backend}"
        copy_mode = f"{transport}_repartition"
        if not self._is_request_dp_artifact_writer(payload, "source"):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        shard_metadata = self._load_artifact_shard_metadata(source_worker_id, source_step, source_name)
        ranges = _metadata_global_ranges(shard_metadata)
        if ranges is None:
            raise RuntimeConfigError("torch_dist repartition requires activation shard global range metadata")
        source_dp_rank = self._request_dp_rank(payload, "source")
        if source_dp_rank >= len(ranges):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        source_metadata, source_start, source_end = ranges[source_dp_rank]
        global_batch_size = ranges[-1][2]
        destination_dp_size = int(payload["destination_dp_size"])
        destinations = _direct_repartition_destinations(
            source_start=source_start,
            source_end=source_end,
            global_batch_size=global_batch_size,
            destination_dp_size=destination_dp_size,
        )
        if not destinations:
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        source_path = self._artifact_shard_path_for(source_worker_id, source_step, source_name, source_dp_rank)
        source_shard = torch.load(source_path, map_location="cpu", weights_only=False)
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        total_bytes = 0
        total_duration_s = 0.0
        sent_destinations: list[int] = []
        tmp_paths: list[Path] = []
        try:
            for destination_dp_rank, overlap_start, overlap_end in destinations:
                endpoint = _direct_endpoint_for_dp_pair(payload, destination_dp_rank, source_dp_rank)
                local = _slice_artifact_batch(source_shard, overlap_start - source_start, overlap_end - source_start)
                tmp = source_path.with_suffix(
                    source_path.suffix
                    + f".{os.getpid()}.rank{self.rank}.to_dp{destination_dp_rank}.torch_dist_slice.tmp"
                )
                tmp_paths.append(tmp)
                if tmp.exists():
                    tmp.unlink()
                torch.save(local, tmp)
                bytes_ = tmp.stat().st_size
                sha256 = _file_sha256(tmp)
                metadata = {
                    "schema_version": "rpeg.activation_direct_slice.v1",
                    "source_dp_rank": source_dp_rank,
                    "destination_dp_rank": destination_dp_rank,
                    "global_start": overlap_start,
                    "global_end": overlap_end,
                    "payload_sha256": sha256,
                    "source_sha256": str(source_metadata["sha256"]),
                    "source_shards_read": [source_dp_rank],
                }
                started = time.perf_counter()
                pg = _torch_direct_process_group(
                    endpoint=endpoint,
                    backend=backend,
                    pg_rank=0,
                    timeout_s=timeout_s,
                )
                try:
                    _torch_direct_send_file_with_metadata(
                        pg,
                        backend,
                        tmp,
                        bytes_,
                        sha256,
                        metadata,
                        ack_endpoint=endpoint,
                        timeout_s=timeout_s,
                    )
                    _torch_direct_barrier(pg)
                finally:
                    _torch_direct_close(pg)
                duration_s = time.perf_counter() - started
                total_bytes += int(bytes_)
                total_duration_s += duration_s
                sent_destinations.append(destination_dp_rank)
        finally:
            for tmp in tmp_paths:
                if tmp.exists():
                    tmp.unlink()
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "copied": bool(sent_destinations),
            "bytes": total_bytes,
            "source_sha256": str(source_metadata["sha256"]),
            "copy_mode": copy_mode,
            "transport": transport,
            "transfer_rank_duration_s": total_duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(total_bytes, total_duration_s),
            "destination_dp_ranks": sent_destinations,
            "rank": self.rank,
        }

    def _recv_activation_torch_dist_repartition(self, request: SemanticRequest) -> dict[str, Any]:
        import torch

        payload = request.payload
        if isinstance(payload.get("direct_collective_endpoint"), dict):
            return self._recv_activation_torch_dist_collective_repartition_memory(request)
        if bool(payload.get("in_memory", False)):
            return self._recv_activation_torch_dist_repartition_memory(request)
        _reject_persisted_global_index_repartition(payload, "torch_dist")
        source_worker_id = self._activation_worker_id(str(payload["source_worker_id"]))
        source_name = self._activation_name(str(payload["source_name"]))
        destination_name = self._activation_name(str(payload.get("destination_name", source_name)))
        source_step = int(payload.get("source_step", request.step))
        destination_step = int(payload.get("destination_step", request.step))
        backend = _torch_direct_backend(payload)
        transport = f"torch_dist_{backend}"
        copy_mode = f"{transport}_repartition"
        if not self._is_request_dp_artifact_writer(payload, "destination"):
            return {
                "source_worker_id": source_worker_id,
                "source_name": source_name,
                "destination_name": destination_name,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        shard_metadata = self._load_artifact_shard_metadata(source_worker_id, source_step, source_name)
        ranges = _metadata_global_ranges(shard_metadata)
        if ranges is None:
            raise RuntimeConfigError("torch_dist repartition requires activation shard global range metadata")
        global_batch_size = ranges[-1][2]
        destination_dp_rank = self._request_dp_rank(payload, "destination")
        destination_start, destination_end = _batch_range_for_dp(
            global_batch_size,
            int(payload["destination_dp_size"]),
            destination_dp_rank,
        )
        expected_sources = _direct_repartition_sources_for_destination(
            ranges,
            destination_start=destination_start,
            destination_end=destination_end,
        )
        if not expected_sources:
            raise RuntimeConfigError(
                f"no source activation shards overlap destination DP rank {destination_dp_rank} "
                f"for {source_worker_id}:{source_name}@{source_step}"
            )
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        base_path = self._artifact_shard_path_for(
            self.worker_id,
            destination_step,
            destination_name,
            destination_dp_rank,
        )
        base_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_paths = [
            base_path.with_suffix(
                f".dp{destination_dp_rank:05d}.{os.getpid()}.{idx}.torch_dist_slice.tmp"
            )
            for idx in range(len(expected_sources))
        ]
        for tmp in tmp_paths:
            if tmp.exists():
                tmp.unlink()
        started = time.perf_counter()
        slices: list[tuple[int, dict[str, Any]]] = []
        total_bytes = 0
        source_shards_read: list[int] = []
        try:
            for idx, (expected_source_dp_rank, _expected_start, _expected_end) in enumerate(sorted(expected_sources)):
                endpoint = _direct_endpoint_for_dp_pair(payload, destination_dp_rank, expected_source_dp_rank)
                pg = _torch_direct_process_group(
                    endpoint=endpoint,
                    backend=backend,
                    pg_rank=1,
                    timeout_s=timeout_s,
                )
                try:
                    bytes_, _sha256, slice_metadata = _torch_direct_recv_file_with_metadata(
                        pg,
                        backend,
                        tmp_paths[idx],
                        ack_endpoint=endpoint,
                        timeout_s=timeout_s,
                    )
                    _torch_direct_barrier(pg)
                finally:
                    _torch_direct_close(pg)
                source_dp_rank = int(slice_metadata.get("source_dp_rank", -1))
                global_start = int(slice_metadata.get("global_start", -1))
                global_end = int(slice_metadata.get("global_end", -1))
                if (source_dp_rank, global_start, global_end) not in expected_sources:
                    raise RuntimeConfigError(
                        f"unexpected torch_dist repartition slice from source_dp_rank={source_dp_rank} "
                        f"range={global_start}:{global_end}"
                    )
                total_bytes += int(bytes_)
                raw_source_shards_read = slice_metadata.get("source_shards_read")
                if isinstance(raw_source_shards_read, list):
                    source_shards_read.extend(int(rank) for rank in raw_source_shards_read)
                else:
                    source_shards_read.append(source_dp_rank)
                slices.append((global_start, torch.load(tmp_paths[idx], map_location="cpu", weights_only=False)))
        finally:
            for tmp in tmp_paths:
                if tmp.exists():
                    tmp.unlink()
        if len(slices) != len(expected_sources):
            raise RuntimeConfigError(
                f"torch_dist repartition received {len(slices)} slices, expected {len(expected_sources)}"
            )
        duration_s = time.perf_counter() - started
        local = _merge_artifact_shards([data for _start, data in sorted(slices, key=lambda item: item[0])])
        destination_path = self._write_local_artifact_payload(
            step=destination_step,
            name=destination_name,
            data=local,
            transport=transport,
            global_batch_size=global_batch_size,
            global_start=destination_start,
            global_end=destination_end,
        )
        sha256 = _file_sha256(destination_path)
        return {
            "source_worker_id": source_worker_id,
            "source_name": source_name,
            "destination_name": destination_name,
            "copied": True,
            "dp_repartitioned": True,
            "destination_dp_rank": destination_dp_rank,
            "source_dp_size": len(shard_metadata),
            "source_rank_count": len(expected_sources),
            "bytes": total_bytes,
            "sha256": sha256,
            "source_shards": _artifact_shard_summary(shard_metadata),
            "source_shards_read": sorted(set(source_shards_read)),
            "copy_mode": copy_mode,
            "transport": transport,
            "transfer_rank_duration_s": duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(total_bytes, duration_s),
            "rank": self.rank,
        }

    def _direct_activation_source(self, worker_id: str, step: int, name: str) -> tuple[Path, dict[str, Any]]:
        metadata = self._direct_activation_source_metadata(worker_id, step, name)
        if self._artifact_manifest_path_for(worker_id, step, name).exists():
            path = self._artifact_shard_path_for(worker_id, step, name, self._dp_rank())
        else:
            path = self._activation_path_for(worker_id, step, name)
        return path, metadata

    def _direct_activation_source_metadata(self, worker_id: str, step: int, name: str) -> dict[str, Any]:
        manifest_path = self._artifact_manifest_path_for(worker_id, step, name)
        if manifest_path.exists():
            shard_metadata = self._load_artifact_shard_metadata(worker_id, step, name)
            dp_rank = self._dp_rank()
            if dp_rank >= len(shard_metadata):
                raise RuntimeConfigError(f"direct activation source missing dp rank {dp_rank} for {worker_id}:{name}")
            return dict(shard_metadata[dp_rank])
        path = self._activation_path_for(worker_id, step, name)
        if not path.exists():
            raise RuntimeConfigError(f"missing source activation artifact: {path}")
        return {
            "worker_id": worker_id,
            "step": step,
            "name": name,
            "dp_rank": self._dp_rank(),
            "dp_size": self._dp_size(),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
            "transport": "shared_filesystem",
        }

    def _direct_activation_destination_path(self, step: int, name: str, *, sharded: bool) -> Path:
        if sharded:
            return self._artifact_shard_path_for(self.worker_id, step, name, self._dp_rank())
        return self._activation_path_for(self.worker_id, step, name)

    def _write_direct_activation_metadata(
        self,
        *,
        source_worker_id: str,
        source_step: int,
        source_name: str,
        destination_step: int,
        destination_name: str,
        source_metadata: dict[str, Any],
        bytes_: int,
        sha256: str,
        transport: str = "tcp_direct",
    ) -> None:
        manifest_path = self._artifact_manifest_path_for(source_worker_id, source_step, source_name)
        if not manifest_path.exists():
            return
        shard_metadata = self._load_artifact_shard_metadata(source_worker_id, source_step, source_name)
        ranges = _metadata_global_ranges(shard_metadata)
        if ranges is None:
            raise RuntimeConfigError("direct activation transfer requires global range metadata for sharded artifacts")
        dp_rank = self._dp_rank()
        global_batch_size = ranges[-1][2]
        _metadata, source_start, source_end = ranges[dp_rank]
        self._write_artifact_shard_metadata(
            worker_id=self.worker_id,
            step=destination_step,
            name=destination_name,
            dp_rank=dp_rank,
            dp_size=self._dp_size(),
            bytes_=bytes_,
            sha256=sha256,
            transport=transport,
            local_batch_size=_metadata_int(source_metadata, "local_batch_size"),
            global_batch_size=global_batch_size,
            global_start=source_start,
            global_end=source_end,
        )
        if dp_rank == 0:
            self._write_artifact_manifest(
                worker_id=self.worker_id,
                step=destination_step,
                name=destination_name,
                dp_size=self._dp_size(),
                transport=transport,
                global_batch_size=global_batch_size,
            )

    def _load_artifact_shard_metadata(self, worker_id: str, step: int, name: str) -> list[dict[str, Any]]:
        from runtime.semantic import read_json

        manifest = read_json(self._artifact_manifest_path_for(worker_id, step, name))
        dp_size = int(manifest["dp_size"])
        shard_metadata = []
        for dp_rank in range(dp_size):
            path = self._artifact_shard_path_for(worker_id, step, name, dp_rank)
            if not path.exists():
                raise RuntimeConfigError(f"missing activation shard: {path}")
            metadata_path = self._artifact_shard_metadata_path_for(worker_id, step, name, dp_rank)
            if not metadata_path.exists():
                raise RuntimeConfigError(f"missing activation shard metadata: {metadata_path}")
            metadata = read_json(metadata_path)
            _validate_artifact_shard_metadata(
                metadata,
                path=path,
                worker_id=worker_id,
                step=step,
                name=name,
                dp_rank=dp_rank,
                dp_size=dp_size,
            )
            shard_metadata.append(metadata)
        return shard_metadata

    def _load_all_artifact_shards(self, worker_id: str, step: int, name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        import torch

        shard_metadata = self._load_artifact_shard_metadata(worker_id, step, name)
        shards = [
            torch.load(
                self._artifact_shard_path_for(worker_id, step, name, int(metadata["dp_rank"])),
                map_location="cpu",
                weights_only=False,
            )
            for metadata in shard_metadata
        ]
        return shards, shard_metadata

    def _try_direct_copy_artifact_shard(
        self,
        *,
        source_worker_id: str,
        source_step: int,
        source_name: str,
        destination_step: int,
        destination_name: str,
        shard_metadata: list[dict[str, Any]],
        activation_transport: str,
    ) -> dict[str, Any] | None:
        ranges = _metadata_global_ranges(shard_metadata)
        if ranges is None:
            return None
        source_dp_size = len(shard_metadata)
        destination_dp_size = self._dp_size()
        dp_rank = self._dp_rank()
        if source_dp_size != destination_dp_size or dp_rank >= source_dp_size:
            return None
        source_metadata = shard_metadata[dp_rank]
        global_batch_size = ranges[-1][2]
        destination_start, destination_end = self._local_batch_range(global_batch_size)
        _metadata, source_start, source_end = ranges[dp_rank]
        if source_start != destination_start or source_end != destination_end:
            return None
        source_path = self._artifact_shard_path_for(source_worker_id, source_step, source_name, dp_rank)
        destination_path = self._artifact_shard_path_for(self.worker_id, destination_step, destination_name, dp_rank)
        if source_path == destination_path:
            return {
                "copied": False,
                "copy_mode": "same_path",
                "bytes": int(source_metadata["bytes"]),
                "transport": str(source_metadata.get("transport", "shared_filesystem")),
            }
        transfer = _copy_file_atomic(
            source_path,
            destination_path,
            mode=activation_transport,
            source_metadata=source_metadata,
            tmp_suffix=f".{os.getpid()}.{self.rank}.tmp",
        )
        stat = destination_path.stat()
        copy_mode = str(transfer["copy_mode"])
        transport = str(transfer["transport"])
        destination_sha256 = str(transfer["sha256"])
        self._write_artifact_shard_metadata(
            worker_id=self.worker_id,
            step=destination_step,
            name=destination_name,
            dp_rank=dp_rank,
            dp_size=destination_dp_size,
            bytes_=stat.st_size,
            sha256=destination_sha256,
            transport=transport,
            local_batch_size=_metadata_int(source_metadata, "local_batch_size"),
            global_batch_size=global_batch_size,
            global_start=source_start,
            global_end=source_end,
        )
        if dp_rank == 0:
            self._write_artifact_manifest(
                worker_id=self.worker_id,
                step=destination_step,
                name=destination_name,
                dp_size=destination_dp_size,
                transport=transport,
                global_batch_size=global_batch_size,
            )
        return {
            "copied": True,
            "copy_mode": copy_mode,
            "bytes": stat.st_size,
            "sha256": destination_sha256,
            "source_sha256": str(source_metadata["sha256"]),
            "transport": transport,
            "transfer_rank_duration_s": float(transfer.get("transfer_rank_duration_s", 0.0)),
            "transfer_rank_bytes_per_s": float(transfer.get("transfer_rank_bytes_per_s", 0.0)),
        }

    def _repartition_artifact_for_destination_dp(
        self,
        *,
        source_worker_id: str,
        source_step: int,
        source_name: str,
        shard_metadata: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[int]]:
        import torch

        ranges = _metadata_global_ranges(shard_metadata)
        if ranges is None:
            shards, _ = self._load_all_artifact_shards(source_worker_id, source_step, source_name)
            merged = _merge_artifact_shards(shards)
            start, end = self._local_batch_range(_artifact_batch_size(merged))
            return _slice_artifact_batch(merged, start, end), [int(meta["dp_rank"]) for meta in shard_metadata]

        global_batch_size = ranges[-1][2]
        destination_start, destination_end = self._local_batch_range(global_batch_size)
        slices = []
        source_shards_read = []
        for metadata, source_start, source_end in ranges:
            overlap_start = max(destination_start, source_start)
            overlap_end = min(destination_end, source_end)
            if overlap_start >= overlap_end:
                continue
            dp_rank = int(metadata["dp_rank"])
            source_shards_read.append(dp_rank)
            shard = torch.load(
                self._artifact_shard_path_for(source_worker_id, source_step, source_name, dp_rank),
                map_location="cpu",
                weights_only=False,
            )
            slices.append(_slice_artifact_batch(shard, overlap_start - source_start, overlap_end - source_start))
        if not slices:
            raise RuntimeConfigError(
                f"no source activation shards overlap destination DP rank {self._dp_rank()} "
                f"for {source_worker_id}:{source_name}@{source_step}"
            )
        return _merge_artifact_shards(slices), source_shards_read

    def _save_weight_version(self, request: SemanticRequest) -> dict[str, Any]:
        import torch

        self.worker._require_model()
        model = str(request.payload["model"])
        version = int(request.payload["version"])
        if bool(request.payload.get("active_only", False)):
            self._active_weight_versions[model] = version
            return {
                "model": model,
                "version": version,
                "weight_path": f"active://{self.worker_id}/{model}/{version}/rank{self.rank}",
                "bytes": 0,
                "rank": self.rank,
                "optimizer_state": False,
                "shadow": bool(request.payload.get("shadow", False)),
                "persisted": False,
                "active_only": True,
            }
        path = self._weight_version_path(model, version)
        path.parent.mkdir(parents=True, exist_ok=True)
        optimizer_state = self._optimizer_state_by_name() if bool(request.payload.get("include_optimizer_state", True)) else None
        shadow = bool(request.payload.get("shadow", False))
        state_name_map = self._state_dict_name_map()
        device_snapshot = bool(request.payload.get("device_snapshot", False))
        if device_snapshot and bool(request.payload.get("persist", True)):
            raise RuntimeConfigError("device_snapshot weight versions must be ephemeral")
        self._prepare_weight_cache_insertion(model, version)
        state = {
            "schema_version": "rpeg.weight_version.v1",
            "worker_id": self.worker_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "model": model,
            "version": version,
            "step": int(request.payload.get("step", request.step)),
            "shadow": shadow,
            "state_dict": (
                self.worker.rpeg_device_state_dict()
                if device_snapshot
                else self.worker.rpeg_state_dict()
            ),
            "state_dict_name_map": state_name_map,
            "state_dict_shard_specs": self._state_dict_shard_specs(),
        }
        self._cache_weight_version(model, version, dict(state))
        self._active_weight_versions[model] = version
        if optimizer_state is not None:
            state["optimizer_state_by_name"] = optimizer_state
            state["optimizer_state_name_map"] = {
                name: state_name_map.get(name, name)
                for name in optimizer_state
            }
        if not bool(request.payload.get("persist", True)):
            payload_bytes = _weight_state_payload_bytes(state)
            return {
                "model": model,
                "version": version,
                "weight_path": f"memory://{self.worker_id}/{model}/{version}/rank{self.rank}",
                "bytes": payload_bytes,
                "rank": self.rank,
                "optimizer_state": optimizer_state is not None,
                "shadow": shadow,
                "persisted": False,
            }
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            torch.save(state, tmp)
            tmp.replace(path)
        finally:
            if tmp.exists():
                tmp.unlink()
        self._write_weight_version_metadata(
            worker_id=self.worker_id,
            model=model,
            version=version,
            rank=self.rank,
            world_size=self.world_size,
            step=int(request.payload.get("step", request.step)),
            path=path,
            optimizer_state=optimizer_state is not None,
            shadow=shadow,
            parallelism=self._parallelism(),
        )
        return {
            "model": model,
            "version": version,
            "weight_path": str(path),
            "bytes": path.stat().st_size,
            "rank": self.rank,
            "optimizer_state": optimizer_state is not None,
            "shadow": shadow,
        }

    def _load_weight_version(self, request: SemanticRequest) -> dict[str, Any]:
        import torch

        self.worker._require_model()
        model = str(request.payload["model"])
        version = int(request.payload["version"])
        strict = bool(request.payload.get("strict", True))
        if self._active_weight_versions.get(model) == version:
            materialization = self._ensure_operational_weights(request)
            return {
                "model": model,
                "version": version,
                "weight_path": f"active://{self.worker_id}/{model}/{version}/rank{self.rank}",
                "bytes": 0,
                "rank": self.rank,
                "optimizer_state": False,
                "shadow": bool(request.payload.get("shadow", False)),
                "active_model": True,
                "materialized": bool(materialization["materialized"]),
            }
        path = self._weight_version_path(model, version)
        state = self._weight_cache.get((model, version))
        metadata: dict[str, Any] = {}
        if state is None:
            if not path.exists():
                raise RuntimeConfigError(f"missing RPEG weight version file: {path}")
            metadata = self._load_weight_version_metadata(
                worker_id=self.worker_id,
                model=model,
                version=version,
                rank=self.rank,
            )
            state = torch.load(path, map_location="cpu", weights_only=False)
            cached_state = {
                name: value
                for name, value in state.items()
                if name not in {"optimizer_state_by_name", "optimizer_state_name_map"}
            }
            self._cache_weight_version(model, version, cached_state)
        if state.get("schema_version") != "rpeg.weight_version.v1":
            raise RuntimeConfigError(f"unsupported RPEG weight version schema in {path}")
        if str(state.get("model")) != model or int(state.get("version", -1)) != version:
            raise RuntimeConfigError(f"weight version metadata mismatch in {path}")
        if int(state.get("world_size", -1)) != self.world_size:
            raise RuntimeConfigError(
                f"weight version {model}:{version} was saved with world_size={state.get('world_size')}, "
                f"current worker world_size={self.world_size}"
            )
        loaded = state["state_dict"]
        adopted_bytes = 0
        if not self.worker.operational_weights_materialized:
            adopted_bytes = self.worker.adopt_rpeg_state_dict_storage(loaded, strict=strict)
            # The module now owns the received tensor storage. Retaining the cache record would keep
            # a logically duplicate full version live and would also make subsequent pruning account
            # it as a snapshot.
            self._weight_cache.pop((model, version), None)
            self._weight_blob_cache.pop((model, version), None)
        else:
            self.worker.load_rpeg_state_dict(loaded, strict=strict)
        self._active_weight_versions[model] = version
        optimizer_loaded = False
        if "optimizer_state_by_name" in state and bool(request.payload.get("load_optimizer_state", False)):
            self._load_optimizer_state_by_name(state["optimizer_state_by_name"])
            optimizer_loaded = True
        shadow = bool(state.get("shadow", metadata.get("shadow", False)))
        return {
            "model": model,
            "version": version,
            "weight_path": str(path) if path.exists() else f"memory://{self.worker_id}/{model}/{version}/rank{self.rank}",
            "bytes": path.stat().st_size if path.exists() else _weight_state_payload_bytes(state),
            "rank": self.rank,
            "optimizer_state": optimizer_loaded,
            "shadow": shadow,
            "adopted_parameter_bytes": adopted_bytes,
        }

    def _prune_weight_versions(self, request: SemanticRequest) -> dict[str, Any]:
        keep_by_model = request.payload.get("keep_versions_by_model")
        if not isinstance(keep_by_model, dict):
            raise RuntimeConfigError("prune_weight_versions requires keep_versions_by_model")
        normalized: dict[str, set[int]] = {}
        for model, versions in keep_by_model.items():
            if not isinstance(versions, (list, tuple)):
                raise RuntimeConfigError(
                    f"prune_weight_versions versions for {model!r} must be a list"
                )
            normalized[str(model)] = {int(version) for version in versions}

        preserve_latest = bool(request.payload.get("preserve_latest", False))
        for model, versions in normalized.items():
            self._weight_cache_pins[model] = set(versions)
            if preserve_latest:
                cached = sorted(
                    version
                    for cached_model, version in self._weight_cache
                    if cached_model == model
                )
                keep = int(self._weight_cache_keep_latest)
                if keep > 0:
                    normalized[model].update(cached[-keep:])

        released_active: list[tuple[str, int]] = []
        released_active_bytes = 0
        if bool(request.payload.get("release_unrequired_active", False)):
            for model, live_versions in normalized.items():
                active_version = self._active_weight_versions.get(model)
                if active_version is None or active_version in live_versions:
                    continue
                if not self.worker.operational_weights_materialized:
                    self._active_weight_versions.pop(model, None)
                    continue
                if self.worker._optimizer is not None:
                    raise RuntimeConfigError(
                        "release_unrequired_active is invalid for an optimizer-owning worker"
                    )
                released_active_bytes += int(self.worker.release_operational_weights())
                self._active_weight_versions.pop(model, None)
                released_active.append((model, int(active_version)))

        removed: list[tuple[str, int]] = []
        removed_bytes = 0
        for key, state in tuple(self._weight_cache.items()):
            model, version = key
            if model not in normalized or version in normalized[model]:
                continue
            removed_bytes += _weight_state_payload_bytes(state)
            self._weight_cache.pop(key, None)
            self._weight_blob_cache.pop(key, None)
            removed.append(key)
        # The loop variable retains the last removed state until this method returns. Drop it before
        # empty_cache, so another colocated process can immediately reuse the released publication
        # HBM.
        state = None
        cache_trim = _trim_cuda_cache() if removed or released_active else {}
        return {
            "pruned_weight_version_count": len(removed),
            "pruned_weight_version_bytes": removed_bytes,
            "remaining_weight_version_count": len(self._weight_cache),
            "remaining_weight_versions": {
                model: sorted(
                    version
                    for cached_model, version in self._weight_cache
                    if cached_model == model
                )
                for model in sorted(normalized)
            },
            "released_active_weight_version_count": len(released_active),
            "released_active_weight_version_bytes": released_active_bytes,
            "released_active_weight_versions": {
                model: version for model, version in released_active
            },
            **cache_trim,
        }

    def _reshard_weight_version(self, request: SemanticRequest) -> dict[str, Any]:
        import torch

        payload = request.payload
        model = str(payload["model"])
        source_worker_id = str(payload["source_worker_id"])
        source_version = int(payload["source_version"])
        destination_version = int(payload.get("destination_version", source_version))
        source_parallelism = tuple(int(x) for x in payload["source_parallelism"])
        destination_parallelism = tuple(int(x) for x in payload["destination_parallelism"])
        source_world_size = int(payload["source_world_size"])
        shadow = bool(payload.get("shadow", False))
        if not self.worker.operational_weights_materialized:
            self._ensure_operational_weights(request)
        if source_parallelism == destination_parallelism and destination_version == source_version:
            return self._copy_same_layout_weight_version(
                model=model,
                source_worker_id=source_worker_id,
                source_version=source_version,
                destination_version=destination_version,
                source_world_size=source_world_size,
                step=int(payload.get("step", request.step)),
                shadow=shadow,
            )
        source_states = []
        source_state_name_maps = []
        source_state_shard_specs = []
        source_optimizer_states = []
        source_optimizer_name_maps = []
        have_optimizer_state = True
        for source_rank in range(source_world_size):
            source_path = self._weight_version_path_for(source_worker_id, model, source_version, source_rank)
            if not source_path.exists():
                raise RuntimeConfigError(f"missing source RPEG weight version file: {source_path}")
            self._load_weight_version_metadata(
                worker_id=source_worker_id,
                model=model,
                version=source_version,
                rank=source_rank,
            )
            state = torch.load(source_path, map_location="cpu", weights_only=False)
            if state.get("schema_version") != "rpeg.weight_version.v1":
                raise RuntimeConfigError(f"unsupported RPEG weight version schema in {source_path}")
            if str(state.get("model")) != model or int(state.get("version", -1)) != source_version:
                raise RuntimeConfigError(f"source weight version metadata mismatch in {source_path}")
            source_states.append(state["state_dict"])
            source_state_name_maps.append(dict(state.get("state_dict_name_map") or {name: name for name in state["state_dict"]}))
            source_state_shard_specs.append(dict(state.get("state_dict_shard_specs") or {}))
            if "optimizer_state_by_name" in state:
                source_optimizer_states.append(state["optimizer_state_by_name"])
                source_optimizer_name_maps.append(
                    dict(state.get("optimizer_state_name_map") or {
                        name: name for name in state["optimizer_state_by_name"]
                    })
                )
            else:
                have_optimizer_state = False
        destination_state_name_map = self._state_dict_name_map() if self.worker.model is not None else None
        destination_state_shard_specs = self._state_dict_shard_specs() if self.worker.model is not None else None
        if source_parallelism == destination_parallelism:
            if self.rank >= len(source_states):
                raise RuntimeConfigError(
                    f"destination rank {self.rank} outside source state count {len(source_states)} for same-layout transfer"
                )
            destination_state = {
                key: value.detach().clone() if hasattr(value, "detach") else value
                for key, value in source_states[self.rank].items()
            }
        else:
            destination_state = reshard_state_dict(
                source_states,
                source_parallelism=source_parallelism,
                destination_parallelism=destination_parallelism,
                destination_rank=self.rank,
                source_name_maps=source_state_name_maps,
                destination_name_map=destination_state_name_map,
                source_shard_specs=source_state_shard_specs,
                destination_shard_specs=destination_state_shard_specs,
            )
        destination_optimizer_state = None
        if have_optimizer_state and source_optimizer_states:
            if source_parallelism == destination_parallelism:
                destination_optimizer_state = _clone_optimizer_state(source_optimizer_states[self.rank])
            else:
                if destination_state_name_map is None:
                    destination_optimizer_name_map = None
                else:
                    destination_optimizer_name_map = {
                        name: destination_state_name_map.get(name, name)
                        for name in self._optimizer_parameter_names()
                    }
                destination_optimizer_state = reshard_optimizer_state_by_name(
                    source_optimizer_states,
                    source_parallelism=source_parallelism,
                    destination_parallelism=destination_parallelism,
                    destination_rank=self.rank,
                    source_name_maps=source_optimizer_name_maps,
                    destination_name_map=destination_optimizer_name_map,
                    source_shard_specs=source_state_shard_specs,
                    destination_shard_specs=destination_state_shard_specs,
                )
            destination_optimizer_state = self._filter_destination_optimizer_state(destination_optimizer_state)
        destination_path = self._weight_version_path_for(self.worker_id, model, destination_version, self.rank)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": "rpeg.weight_version.v1",
            "worker_id": self.worker_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "model": model,
            "version": destination_version,
            "step": int(payload.get("step", request.step)),
            "source_worker_id": source_worker_id,
            "source_version": source_version,
            "source_parallelism": list(source_parallelism),
            "destination_parallelism": list(destination_parallelism),
            "shadow": shadow,
            "state_dict": destination_state,
            "state_dict_name_map": destination_state_name_map or {name: name for name in destination_state},
            "state_dict_shard_specs": destination_state_shard_specs or {},
        }
        if destination_optimizer_state is not None:
            state["optimizer_state_by_name"] = destination_optimizer_state
            state["optimizer_state_name_map"] = {
                name: (destination_state_name_map or {}).get(name, name)
                for name in destination_optimizer_state
            }
        tmp = destination_path.with_suffix(destination_path.suffix + ".tmp")
        try:
            torch.save(state, tmp)
            tmp.replace(destination_path)
        finally:
            if tmp.exists():
                tmp.unlink()
        self._write_weight_version_metadata(
            worker_id=self.worker_id,
            model=model,
            version=destination_version,
            rank=self.rank,
            world_size=self.world_size,
            step=int(payload.get("step", request.step)),
            path=destination_path,
            optimizer_state=destination_optimizer_state is not None,
            shadow=shadow,
            source_worker_id=source_worker_id,
            source_version=source_version,
            copy_mode="reshard",
            transport="shared_filesystem_torch",
            parallelism=destination_parallelism,
        )
        return {
            "model": model,
            "source_worker_id": source_worker_id,
            "source_version": source_version,
            "destination_version": destination_version,
            "weight_path": str(destination_path),
            "bytes": destination_path.stat().st_size,
            "sha256": _file_sha256(destination_path),
            "copy_mode": "reshard",
            "transport": "shared_filesystem_torch",
            "rank": self.rank,
            "optimizer_state": destination_optimizer_state is not None,
            "shadow": shadow,
            "source_rank_count": source_world_size,
        }

    def _switch_model_layout(self, request: SemanticRequest) -> dict[str, Any]:
        """
        Redistribute one colocated model in place without a second persistent model.
        """
        import torch

        payload = request.payload
        source_parallelism = self._parallelism()
        destination_parallelism = tuple(int(x) for x in payload["destination_parallelism"])
        if source_parallelism == destination_parallelism:
            return {
                "switched": False,
                "source_parallelism": source_parallelism,
                "destination_parallelism": destination_parallelism,
                "rank": self.rank,
            }
        destination_world_size = destination_parallelism[0] * destination_parallelism[1] * destination_parallelism[2]
        if self.world_size != destination_world_size:
            raise RuntimeConfigError(
                f"in-place layout {destination_parallelism} requires "
                f"{destination_world_size} ranks, worker has {self.world_size}"
            )
        requested_source = tuple(int(x) for x in payload.get("source_parallelism", source_parallelism))
        if requested_source != source_parallelism:
            raise RuntimeConfigError(
                f"layout switch expected source {requested_source}, active layout is {source_parallelism}"
            )

        started = time.perf_counter()
        layout_collective_mode = _in_place_layout_collective_mode(
            source_parallelism,
            destination_parallelism,
        )
        if layout_collective_mode == "pp_to_dp_broadcast":
            from megatron.core import parallel_state

            transfer_group = parallel_state.get_pipeline_model_parallel_group()
            if str(torch.distributed.get_backend(transfer_group)).lower() != "nccl":
                raise RuntimeConfigError(
                    "PP-to-DP layout switch requires the existing NCCL PP group"
                )
            transfer_backend = "nccl"
            receive_device = torch.device("cuda", torch.cuda.current_device())
            copy_mode = "in_place_pp_to_dp_host_staged_broadcast"
            transport = "torch_dist_nccl_model_parallel_broadcast"
        elif layout_collective_mode == "dp_to_pp_local":
            transfer_group = None
            transfer_backend = "nccl"
            receive_device = torch.device("cuda", torch.cuda.current_device())
            copy_mode = "in_place_dp_to_pp_local_extract"
            transport = "local_tensor"
        else:
            transfer_group, transfer_backend, receive_device, copy_mode, transport = (
                self._layout_switch_transport()
            )
        # NCCL receives materialize fragments on the GPU. Keep the local contribution on that same
        # device as detached views; mixing the old CPU snapshot with received CUDA fragments fails
        # during concatenation and needlessly stages a full model shard through host memory.
        source_state = (
            self.worker.rpeg_device_state_dict(clone=False)
            if transfer_backend == "nccl"
            else self.worker.rpeg_state_dict()
        )
        source_name_map = self._state_dict_name_map()
        source_shard_specs = self._state_dict_shard_specs()
        source_bytes = _weight_state_payload_bytes({"state_dict": source_state})
        architecture = getattr(self.worker, "architecture", None)
        num_layers = None if architecture is None else int(architecture.num_layers)
        destination_fragments: list[dict[str, dict[str, Any]]] | None = None
        sent_wire_bytes = 0
        received_wire_bytes = 0
        routed_records: list[tuple[int, dict[str, dict[str, Any]], dict[str, Any]]] = []
        expected_fragment_count = self.world_size
        if layout_collective_mode == "pp_to_dp_broadcast":
            _src_dp, src_pp, src_tp = source_parallelism
            destination_fragments = []
            expected_fragment_count = src_pp
            expected_group_ranks = _in_place_pp_group_ranks(
                self.rank,
                source_parallelism,
            )
            observed_group_ranks = tuple(
                torch.distributed.get_process_group_ranks(transfer_group)
            )
            if observed_group_ranks != expected_group_ranks:
                raise RuntimeConfigError(
                    f"PP-to-DP broadcast group ranks {observed_group_ranks} do not match "
                    f"expected {expected_group_ranks}"
                )
            destination_mp_width = (
                destination_parallelism[1] * destination_parallelism[2]
            )
            for source_rank in expected_group_ranks:
                record = None
                if self.rank == source_rank:
                    fragment = route_state_dict_fragment(
                        source_state,
                        source_parallelism=source_parallelism,
                        destination_parallelism=destination_parallelism,
                        source_rank=self.rank,
                        destination_rank=self.rank,
                        source_name_map=source_name_map,
                        source_shard_specs=source_shard_specs,
                        num_layers=num_layers,
                        replicated_source_tp_rank=routed_weight_replicated_source_tp_rank(
                            source_parallelism,
                            destination_parallelism,
                            self.rank,
                        ),
                        clone_values=False,
                    )
                    record = {
                        "schema_version": "rpeg.weight_reshard_fragment.v1",
                        "worker_id": self.worker_id,
                        "rank": self.rank,
                        "world_size": self.world_size,
                        "model": str(payload["model"]),
                        "version": int(payload.get("source_version", 0)),
                        "destination_rank": self.rank,
                        "fragments": fragment,
                    }
                received_record, broadcast_bytes = (
                    _torch_model_parallel_broadcast_tensor_record(
                        record,
                        group=transfer_group,
                        source_rank=source_rank,
                        expected_schema="rpeg.weight_reshard_fragment.v1",
                        stream_non_source_to_cpu=True,
                        stream_chunk_bytes=16 * 1024 * 1024,
                    )
                )
                if (
                    int(received_record.get("rank", -1)) != source_rank
                    or str(received_record.get("worker_id")) != self.worker_id
                    or int(received_record.get("world_size", -1)) != self.world_size
                    or int(received_record.get("destination_rank", -1))
                    % destination_mp_width
                    != self.rank % destination_mp_width
                ):
                    raise RuntimeConfigError(
                        f"invalid PP-to-DP layout fragment from source rank {source_rank}"
                    )
                # Stage each received PP shard to Grace memory immediately. NCCL remains the network
                # data plane while peak HBM stays below the destination model plus persistent
                # optimizer state.
                received_record = _tensor_record_to_cpu(received_record)
                destination_fragments.append(
                    dict(received_record.get("fragments") or {})
                )
                if self.rank == source_rank:
                    sent_wire_bytes += int(broadcast_bytes)
                else:
                    received_wire_bytes += int(broadcast_bytes)
        elif layout_collective_mode == "dp_to_pp_local":
            source_dp_rank, _source_pp_rank, _source_tp_rank = rank_to_dp_pp_tp(
                self.rank,
                source_parallelism[1],
                source_parallelism[2],
            )
            fragment = route_state_dict_fragment(
                source_state,
                source_parallelism=source_parallelism,
                destination_parallelism=destination_parallelism,
                source_rank=self.rank,
                destination_rank=self.rank,
                source_name_map=source_name_map,
                source_shard_specs=source_shard_specs,
                num_layers=num_layers,
                replicated_source_tp_rank=routed_weight_replicated_source_tp_rank(
                    source_parallelism,
                    destination_parallelism,
                    self.rank,
                ),
                # Every source DP replica owns a complete model shard. The local DP->PP transpose
                # deliberately consumes the replica whose DP rank becomes this rank's destination PP
                # stage; route_state_dict_fragment otherwise defaults to DP replica zero and returns
                # an empty fragment on the other N-1 ranks.
                active_source_dp_rank=source_dp_rank,
                clone_values=False,
            )
            destination_fragments = [
                _tensor_record_to_cpu({"fragments": fragment})["fragments"]
            ]
            expected_fragment_count = 1
        else:
            for destination_rank in range(self.world_size):
                fragment = route_state_dict_fragment(
                    source_state,
                    source_parallelism=source_parallelism,
                    destination_parallelism=destination_parallelism,
                    source_rank=self.rank,
                    destination_rank=destination_rank,
                    source_name_map=source_name_map,
                    source_shard_specs=source_shard_specs,
                    num_layers=num_layers,
                    replicated_source_tp_rank=routed_weight_replicated_source_tp_rank(
                        source_parallelism,
                        destination_parallelism,
                        destination_rank,
                    ),
                    clone_values=False,
                )
                if self.world_size == 1:
                    destination_fragments = [fragment]
                    continue
                record = {
                    "schema_version": "rpeg.weight_reshard_fragment.v1",
                    "worker_id": self.worker_id,
                    "rank": self.rank,
                    "world_size": self.world_size,
                    "model": str(payload["model"]),
                    "version": int(payload.get("source_version", 0)),
                    "destination_rank": destination_rank,
                    "fragments": fragment,
                }
                routed_records.append((destination_rank, fragment, record))

        if (
            layout_collective_mode is None
            and self.world_size > 1
            and transfer_backend == "nccl"
        ):
            local_fragment = next(
                fragment
                for destination_rank, fragment, _record in routed_records
                if destination_rank == self.rank
            )
            received, sent_wire_bytes = _torch_direct_exchange_tensor_records_coalesced(
                transfer_group,
                [
                    ({"destination_group_rank": destination_rank}, record)
                    for destination_rank, _fragment, record in routed_records
                    if destination_rank != self.rank
                ],
                [
                    {"source_group_rank": source_rank}
                    for source_rank in range(self.world_size)
                    if source_rank != self.rank
                ],
                device=receive_device,
            )
            received_by_rank = {
                int(record.get("rank", -1)): (record, int(bytes_))
                for record, bytes_, _sha256 in received
            }
            destination_fragments = []
            for source_rank in range(self.world_size):
                if source_rank == self.rank:
                    destination_fragments.append(local_fragment)
                    continue
                item = received_by_rank.get(source_rank)
                if item is None:
                    raise RuntimeConfigError(
                        f"in-place layout switch received no NCCL fragment from rank {source_rank}"
                    )
                received_record, received_bytes = item
                self._validate_layout_switch_fragment_record(
                    received_record,
                    source_rank=source_rank,
                    destination_rank=self.rank,
                )
                destination_fragments.append(dict(received_record.get("fragments") or {}))
                received_wire_bytes += received_bytes
        elif layout_collective_mode is None and self.world_size > 1:
            for destination_rank, fragment, record in routed_records:
                if self.rank == destination_rank:
                    destination_fragments = []
                    for source_rank in range(self.world_size):
                        if source_rank == self.rank:
                            destination_fragments.append(fragment)
                            continue
                        received, received_bytes, _sha256 = _torch_direct_recv_tensor_record(
                            transfer_group,
                            transfer_backend,
                            device=receive_device,
                            source_rank=source_rank,
                        )
                        self._validate_layout_switch_fragment_record(
                            received,
                            source_rank=source_rank,
                            destination_rank=destination_rank,
                        )
                        destination_fragments.append(dict(received.get("fragments") or {}))
                        received_wire_bytes += int(received_bytes)
                else:
                    sent_bytes, _sha256 = _torch_direct_send_tensor_record(
                        transfer_group,
                        record,
                        transfer_backend,
                        destination_rank=destination_rank,
                    )
                    sent_wire_bytes += int(sent_bytes)
        if (
            destination_fragments is None
            or len(destination_fragments) != expected_fragment_count
        ):
            raise RuntimeConfigError(
                f"layout switch rank {self.rank} gathered "
                f"{0 if destination_fragments is None else len(destination_fragments)} fragments, "
                f"expected {expected_fragment_count}"
            )
        del source_state

        # Device snapshots retain BF16 tensors from the old module. A colocated transition consumes
        # those versions directly and must release them.
        model = str(payload["model"])
        for key in tuple(self._weight_cache):
            if key[0] == model:
                self._weight_cache.pop(key, None)
                self._weight_blob_cache.pop(key, None)

        source_config = self.worker.release_model_for_layout_switch()
        self.worker.build_switched_layout(
            source_config,
            parallelism=destination_parallelism,
            microbatch=int(payload.get("microbatch", source_config.microbatch)),
        )
        destination_state = merge_state_dict_fragments(
            destination_fragments,
            destination_name_map=self._state_dict_name_map(),
            destination_shard_specs=self._state_dict_shard_specs(),
            source_parallelism=source_parallelism,
            destination_parallelism=destination_parallelism,
            destination_rank=self.rank,
            clone_complete_values=False,
            consume_inputs=True,
        )
        # The merged destination shard already owns complete tensor storage. Drop the freshly
        # rebuilt module's placeholder BF16 allocations and adopt the merged tensors instead of
        # holding two full destination shards during load. The canonical ZeRO-2 optimizer is
        # deliberately unbound by release_model_for_layout_switch and remains persistent.
        released_destination_bytes = self.worker.release_operational_weights()
        adopted_destination_bytes = self.worker.adopt_rpeg_state_dict_storage(
            destination_state,
            strict=True,
        )
        self._active_weight_versions[model] = int(payload.get("source_version", 0))
        restored_optimizer_binding = self.worker.restore_canonical_optimizer_binding()
        duration_s = time.perf_counter() - started
        destination_bytes = _weight_state_payload_bytes({"state_dict": destination_state})
        # The rebuilt module owns a copied canonical state. Temporary routed records and merge
        # output otherwise retain an extra full destination shard until this request returns,
        # starving colocated workers.
        del destination_state
        del destination_fragments
        del routed_records
        cache_trim = _trim_cuda_cache()
        return {
            "switched": True,
            "model": model,
            "source_parallelism": source_parallelism,
            "destination_parallelism": destination_parallelism,
            "copy_mode": copy_mode,
            "transport": transport,
            "bytes": sent_wire_bytes,
            "source_rank_bytes": source_bytes,
            "destination_rank_bytes": destination_bytes,
            "routed_wire_bytes": sent_wire_bytes + received_wire_bytes,
            "routed_wire_bytes_sent": sent_wire_bytes,
            "routed_wire_bytes_received": received_wire_bytes,
            "wire_format": "tensor_native",
            "transfer_rank_duration_s": duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(source_bytes, duration_s),
            "optimizer_persistent": self.worker._optimizer is not None,
            "optimizer_rebound": restored_optimizer_binding,
            "layout_destination_placeholder_released_bytes": released_destination_bytes,
            "layout_destination_storage_adopted_bytes": adopted_destination_bytes,
            "rank": self.rank,
            **cache_trim,
        }

    def _validate_layout_switch_fragment_record(
        self,
        record: dict[str, Any],
        *,
        source_rank: int,
        destination_rank: int,
    ) -> None:
        if (
            record.get("schema_version") != "rpeg.weight_reshard_fragment.v1"
            or str(record.get("worker_id")) != self.worker_id
            or int(record.get("rank", -1)) != source_rank
            or int(record.get("world_size", -1)) != self.world_size
            or int(record.get("destination_rank", -1)) != destination_rank
        ):
            raise RuntimeConfigError(
                f"in-place layout switch received invalid fragment metadata "
                f"from rank {source_rank} for rank {destination_rank}"
            )

    def _layout_switch_transport(self) -> tuple[Any, str, Any, str, str]:
        """
        Prefer the existing GPU world communicator for an in-place reshard.
        """
        import torch
        import torch.distributed as dist

        if self.world_size == 1:
            return (
                None,
                "local",
                self._weight_receive_device(),
                "in_place_routed_device_reshard",
                "local_tensor",
            )
        if not dist.is_initialized():
            raise RuntimeConfigError("in-place layout switching requires an initialized process group")
        world_group = dist.group.WORLD
        world_backend = str(dist.get_backend(world_group)).lower()
        if torch.cuda.is_available() and "nccl" in world_backend:
            return (
                world_group,
                "nccl",
                self._weight_receive_device(),
                "in_place_routed_device_reshard",
                "torch_dist_nccl_tensor",
            )
        host_group = self._ensure_host_reshard_group()
        return (
            host_group,
            "gloo",
            torch.device("cpu"),
            "in_place_routed_host_reshard",
            "torch_dist_gloo_tensor",
        )

    def _ensure_operational_weights(self, request: SemanticRequest) -> dict[str, Any]:
        model = str(request.payload["model"])
        if self.worker.operational_weights_materialized:
            return {
                "model": model,
                "materialized": False,
                "materialized_bytes": 0,
                "optimizer_source": False,
                "rank": self.rank,
            }
        materialized_bytes = self.worker.materialize_operational_weights()
        optimizer_source = self.worker.restore_canonical_optimizer_binding()
        if not optimizer_source:
            state = self._released_weight_states.get(model)
            if state is None:
                raise RuntimeConfigError(
                    f"worker {self.worker_id} has no retained transient state for {model}"
                )
            self.worker.load_rpeg_state_dict(state, strict=True)
        return {
            "model": model,
            "materialized": True,
            "materialized_bytes": materialized_bytes,
            "optimizer_source": optimizer_source,
            "rank": self.rank,
        }

    def _discard_released_weight_state(self, model: str) -> dict[str, Any]:
        """
        Drop the restorable old version before receiving its replacement.

        _release_operational_weights deliberately retains detached tensor views, so an interrupted
        inference lifecycle can rematerialize the same version. Those views still own the original
        device storage. A non-shadow tensor-native publication supersedes that version, so retaining
        it while receiving the replacement would require two full model shards on the destination
        GPU.
        """
        present = str(model) in self._released_weight_states
        state = self._released_weight_states.pop(str(model), None)
        discarded_bytes = (
            _weight_state_payload_bytes({"state_dict": state})
            if isinstance(state, dict)
            else 0
        )
        # Do not let this local retain the last device tensors across empty_cache: colocated
        # publisher and receiver processes must be able to reuse the storage immediately.
        state = None
        cache_trim = _trim_cuda_cache() if present else {}
        return {
            "discarded_released_weight_state": present,
            "discarded_released_weight_state_bytes": discarded_bytes,
            **cache_trim,
        }

    def _release_operational_weights(self, request: SemanticRequest) -> dict[str, Any]:
        model = str(request.payload["model"])
        if not self.worker.operational_weights_materialized:
            return {
                "model": model,
                "released": False,
                "released_bytes": 0,
                "optimizer_persistent": self.worker._optimizer is not None,
                "rank": self.rank,
            }
        optimizer = self.worker._optimizer
        optimizer_source = (
            optimizer is not None
            and self.worker._optimizer_parallelism is not None
            and self.worker.config is not None
            and tuple(int(value) for value in self.worker.config.parallelism)
            == self.worker._optimizer_parallelism
        )
        if not optimizer_source:
            self._released_weight_states[model] = self.worker.rpeg_state_dict()
            for key in tuple(self._weight_cache):
                if key[0] == model:
                    self._weight_cache.pop(key, None)
                    self._weight_blob_cache.pop(key, None)
        released_bytes = self.worker.release_operational_weights()
        return {
            "model": model,
            "released": True,
            "released_bytes": released_bytes,
            "optimizer_persistent": optimizer is not None,
            "optimizer_source": optimizer_source,
            "rank": self.rank,
        }

    def _ensure_host_reshard_group(self):
        if self.world_size == 1:
            return None
        if self._host_reshard_group is not None:
            return self._host_reshard_group
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeConfigError("in-place layout switching requires an initialized process group")
        if not dist.is_gloo_available():
            raise RuntimeDependencyError("tensor-native host layout switching requires Gloo")
        self._host_reshard_group = dist.new_group(
            ranks=list(range(self.world_size)),
            backend="gloo",
        )
        return self._host_reshard_group

    def _copy_same_layout_weight_version(
        self,
        *,
        model: str,
        source_worker_id: str,
        source_version: int,
        destination_version: int,
        source_world_size: int,
        step: int,
        shadow: bool = False,
    ) -> dict[str, Any]:
        if self.world_size != source_world_size:
            raise RuntimeConfigError(
                f"same-layout weight transfer requires destination world_size={source_world_size}, "
                f"got {self.world_size}"
            )
        if self.rank >= source_world_size:
            raise RuntimeConfigError(
                f"destination rank {self.rank} outside source world_size {source_world_size}"
            )
        source_path = self._weight_version_path_for(source_worker_id, model, source_version, self.rank)
        if not source_path.exists():
            raise RuntimeConfigError(f"missing source RPEG weight version file: {source_path}")
        source_metadata = self._load_weight_version_metadata(
            worker_id=source_worker_id,
            model=model,
            version=source_version,
            rank=self.rank,
        )
        destination_path = self._weight_version_path_for(self.worker_id, model, destination_version, self.rank)
        if source_path == destination_path:
            copy_mode = "same_path"
            transport = "shared_filesystem_same_path"
        else:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = destination_path.with_suffix(destination_path.suffix + f".{os.getpid()}.{self.rank}.tmp")
            if tmp.exists():
                tmp.unlink()
            copy_mode = "hardlink"
            try:
                copy_mode = _fast_clone_or_copy_file(source_path, tmp)
                tmp.replace(destination_path)
            finally:
                if tmp.exists():
                    tmp.unlink()
            transport = f"shared_filesystem_{copy_mode}"
        optimizer_state = None
        if "optimizer_state" in source_metadata:
            optimizer_state = bool(source_metadata["optimizer_state"])
        destination_sha256 = _cloned_file_sha256(copy_mode, source_metadata, destination_path)
        self._write_weight_version_metadata(
            worker_id=self.worker_id,
            model=model,
            version=destination_version,
            rank=self.rank,
            world_size=self.world_size,
            step=step,
            path=destination_path,
            optimizer_state=optimizer_state,
            shadow=shadow,
            source_worker_id=source_worker_id,
            source_version=source_version,
            copy_mode=copy_mode,
            transport=transport,
            sha256=destination_sha256,
            parallelism=self._parallelism(),
        )
        return {
            "model": model,
            "source_worker_id": source_worker_id,
            "source_version": source_version,
            "destination_version": destination_version,
            "weight_path": str(destination_path),
            "rank": self.rank,
            "bytes": destination_path.stat().st_size,
            "sha256": destination_sha256,
            "source_sha256": str(source_metadata["sha256"]),
            "copy_mode": copy_mode,
            "transport": transport,
            "optimizer_state": optimizer_state,
            "shadow": shadow,
        }

    def _send_weight_version_direct(self, request: SemanticRequest) -> dict[str, Any]:
        payload = request.payload
        direct_kind = _direct_transport_kind(payload)
        if direct_kind not in {"ucx", "torch_dist"}:
            raise RuntimeConfigError("direct weight-version transfer requires ucx_direct or torch_dist transport")
        model = str(payload["model"])
        source_worker_id = str(payload["source_worker_id"])
        source_version = int(payload["source_version"])
        source_world_size = int(payload["source_world_size"])
        backend = _torch_direct_backend(payload) if direct_kind == "torch_dist" else None
        transport = f"torch_dist_{backend}" if backend is not None else "ucx_direct"
        if bool(payload.get("in_memory_reshard", False)):
            if direct_kind != "ucx" and backend != "nccl":
                raise RuntimeConfigError("in-memory weight reshard requires ucx_direct or nccl_direct")
            return self._send_weight_version_reshard_direct(request)
        if self.worker_id != source_worker_id:
            return {
                "model": model,
                "source_worker_id": source_worker_id,
                "source_version": source_version,
                "copied": False,
                "rank": self.rank,
                "transport": transport,
            }
        if self.world_size != source_world_size:
            raise RuntimeConfigError(
                f"direct weight send requires source world_size={source_world_size}, got {self.world_size}"
            )
        in_memory = bool(payload.get("in_memory", False)) and (
            direct_kind == "ucx" or backend == "nccl"
        )
        if in_memory:
            source_record = self._weight_cache.get((model, source_version))
            if source_record is None:
                source_path = self._weight_version_path_for(
                    source_worker_id,
                    model,
                    source_version,
                    self.rank,
                )
                if not source_path.exists():
                    raise RuntimeConfigError(f"missing source RPEG weight version file: {source_path}")
                import torch

                source_record = torch.load(source_path, map_location="cpu", weights_only=False)
            tensor_record = _tensor_weight_version_record(source_record)
            manifest, tensor_bytes = _tensor_stream_manifest(tensor_record)
            manifest_bytes = _tensor_stream_manifest_bytes(manifest)
            bytes_ = tensor_bytes + len(manifest_bytes)
            sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            source_metadata: dict[str, Any] = {
                "optimizer_state": False,
                "shadow": bool(payload.get("shadow", False)),
            }
        else:
            source_path = self._weight_version_path_for(source_worker_id, model, source_version, self.rank)
            if not source_path.exists():
                raise RuntimeConfigError(f"missing source RPEG weight version file: {source_path}")
            source_metadata = self._load_weight_version_metadata(
                worker_id=source_worker_id,
                model=model,
                version=source_version,
                rank=self.rank,
            )
            bytes_ = int(source_metadata["bytes"])
            sha256 = str(source_metadata["sha256"])
        endpoint = _direct_endpoint_for_rank(payload, self.rank)
        local_endpoint = bool(endpoint.get("same_physical_device", False))
        local_transport = (
            _direct_endpoint_backend(endpoint, backend)
            if local_endpoint and direct_kind == "torch_dist"
            else None
        )
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        started = time.perf_counter()
        cached = False
        if direct_kind == "ucx":
            send = (
                _ucx_direct_send_tensor_record(
                    endpoint=endpoint,
                    record=tensor_record,
                    timeout_s=timeout_s,
                    ready_path=_direct_ready_marker_path(self.state_root, endpoint),
                    ready_wait_s=_ucx_direct_ready_wait_s(payload),
                    connect_grace_s=_ucx_direct_connect_grace_s(payload),
                )
                if in_memory
                else _ucx_direct_send_file(
                    endpoint=endpoint,
                    path=source_path,
                    bytes_=bytes_,
                    sha256=sha256,
                    timeout_s=timeout_s,
                    ready_path=_direct_ready_marker_path(self.state_root, endpoint),
                    ready_wait_s=_ucx_direct_ready_wait_s(payload),
                    connect_grace_s=_ucx_direct_connect_grace_s(payload),
                )
            )
            if in_memory and bool(payload.get("source_overlap", False)):
                self._launch_background_transfer(
                    lambda: _run_ucx_direct(send, persistent=_ucx_direct_channel_reuse(endpoint)),
                    name=f"weight-{model}-v{source_version}-to-rank{endpoint.get('rank', 0)}",
                )
                cached = False
                asynchronous = True
            else:
                cached = _run_ucx_direct(send, persistent=_ucx_direct_channel_reuse(endpoint))
                asynchronous = False
        elif in_memory and backend == "nccl":
            pg = _torch_direct_process_group(
                endpoint=endpoint,
                backend=str(backend),
                pg_rank=0,
                timeout_s=timeout_s,
            )
            ack_sock = _torch_direct_connect_ack_socket(endpoint, timeout_s=timeout_s)
            try:
                _torch_direct_send_tensor_record(pg, tensor_record, str(backend))
                if ack_sock is not None:
                    _torch_direct_recv_socket_ack(ack_sock)
            finally:
                if ack_sock is not None:
                    ack_sock.close()
                _torch_direct_close(pg)
            asynchronous = False
        else:
            pg = _torch_direct_process_group(
                endpoint=endpoint,
                backend=str(backend),
                pg_rank=0,
                timeout_s=timeout_s,
            )
            try:
                _torch_direct_send_file(
                    pg,
                    str(backend),
                    source_path,
                    bytes_,
                    ack_endpoint=endpoint,
                    timeout_s=timeout_s,
                )
                _torch_direct_barrier(pg)
            finally:
                _torch_direct_close(pg)
            asynchronous = False
        duration_s = time.perf_counter() - started
        return {
            "model": model,
            "source_worker_id": source_worker_id,
            "source_version": source_version,
            "copied": True,
            "rank": self.rank,
            "bytes": bytes_,
            "network_bytes": 0 if local_endpoint else bytes_,
            "local_ipc_bytes": bytes_ if local_endpoint else 0,
            "local_endpoint_count": int(local_endpoint),
            "local_transport": local_transport,
            "sha256": sha256,
            "source_sha256": sha256,
            "copy_mode": "ucx_direct_memory" if in_memory else transport,
            "transport": transport,
            "wire_format": "tensor_native" if in_memory else "serialized_file",
            "optimizer_state": bool(source_metadata.get("optimizer_state", False)),
            "shadow": bool(source_metadata.get("shadow", False)),
            "transfer_rank_duration_s": duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(bytes_, duration_s),
            "direct_channel_reuse": bool(payload.get("direct_channel_reuse", False)),
            "source_overlap": asynchronous,
            "ucx_direct_cached_endpoint_count": int(cached),
            "ucx_direct_cached_endpoint": bool(cached),
        }

    def _send_weight_version_reshard_direct(self, request: SemanticRequest) -> dict[str, Any]:
        import torch

        payload = request.payload
        direct_kind = _direct_transport_kind(payload)
        backend = _torch_direct_backend(payload) if direct_kind == "torch_dist" else None
        if direct_kind != "ucx" and backend != "nccl":
            raise RuntimeConfigError("tensor-native weight reshard requires ucx_direct or nccl_direct")
        transport = "ucx_direct" if direct_kind == "ucx" else "torch_dist_nccl"
        model = str(payload["model"])
        source_worker_id = str(payload["source_worker_id"])
        source_version = int(payload["source_version"])
        source_world_size = int(payload["source_world_size"])
        if self.worker_id != source_worker_id:
            raise RuntimeConfigError(
                f"in-memory reshard source worker mismatch: {self.worker_id} != {source_worker_id}"
            )
        if self.world_size != source_world_size:
            raise RuntimeConfigError(
                f"in-memory reshard send requires source world_size={source_world_size}, got {self.world_size}"
            )
        source_transfer_ranks = tuple(
            int(rank) for rank in payload.get("source_transfer_ranks", range(source_world_size))
        )
        if self.rank not in source_transfer_ranks:
            return {
                "model": model,
                "source_worker_id": source_worker_id,
                "source_version": source_version,
                "copied": False,
                "rank": self.rank,
                "bytes": 0,
                "transport": transport,
                "source_overlap": False,
            }
        endpoints = _direct_endpoints_for_source_rank(payload, self.rank)
        if not endpoints:
            raise RuntimeConfigError(f"in-memory reshard has no destinations for source rank {self.rank}")
        source_record = self._weight_cache.get((model, source_version))
        if source_record is None:
            source_path = self._weight_version_path(model, source_version)
            if not source_path.exists():
                raise RuntimeConfigError(f"missing RPEG weight version for routed reshard: {source_path}")
            source_record = torch.load(source_path, map_location="cpu", weights_only=False)
        source_parallelism = tuple(int(x) for x in payload["source_parallelism"])
        destination_parallelism = tuple(int(x) for x in payload["destination_parallelism"])
        source_state = source_record["state_dict"]
        source_name_map = dict(
            source_record.get("state_dict_name_map") or {name: name for name in source_state}
        )
        source_shard_specs = dict(source_record.get("state_dict_shard_specs") or {})
        architecture = getattr(self.worker, "architecture", None)
        num_layers = None if architecture is None else int(architecture.num_layers)
        routed_payloads: list[tuple[dict[str, Any], dict[str, Any], int, str]] = []
        # Destination DP replicas have identical PP/TP shards. Reuse the routed tensor views/storage
        # across replicas instead of materializing one full model shard per replica before the NCCL
        # send begins.
        routed_fragments: dict[int, dict[str, dict[str, Any]]] = {}
        destination_mp_width = destination_parallelism[1] * destination_parallelism[2]
        for endpoint in endpoints:
            destination_rank = int(endpoint["rank"])
            destination_mp_rank = destination_rank % destination_mp_width
            fragment = routed_fragments.get(destination_mp_rank)
            if fragment is None:
                fragment = route_state_dict_fragment(
                    source_state,
                    source_parallelism=source_parallelism,
                    destination_parallelism=destination_parallelism,
                    source_rank=self.rank,
                    destination_rank=destination_rank,
                    source_name_map=source_name_map,
                    source_shard_specs=source_shard_specs,
                    num_layers=num_layers,
                    replicated_source_tp_rank=routed_weight_replicated_source_tp_rank(
                        source_parallelism,
                        destination_parallelism,
                        destination_rank,
                    ),
                    clone_values=False,
                    active_source_dp_rank=int(payload.get("source_dp_rank", 0)),
                )
                routed_fragments[destination_mp_rank] = fragment
            record = {
                "schema_version": "rpeg.weight_reshard_fragment.v1",
                "worker_id": source_worker_id,
                "rank": self.rank,
                "world_size": source_world_size,
                "model": model,
                "version": source_version,
                "destination_rank": destination_rank,
                "fragments": fragment,
            }
            if direct_kind == "torch_dist" and _direct_endpoint_backend(endpoint, backend) == "gloo":
                record = _tensor_record_to_cpu(record)
            manifest, tensor_bytes = _tensor_stream_manifest(record)
            manifest_bytes = _tensor_stream_manifest_bytes(manifest)
            routed_payloads.append(
                (
                    endpoint,
                    record,
                    tensor_bytes + len(manifest_bytes),
                    hashlib.sha256(manifest_bytes).hexdigest(),
                )
            )
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        started = time.perf_counter()
        cached_count = 0

        def send_all() -> None:
            nonlocal cached_count
            if direct_kind == "ucx":
                for endpoint, record, _bytes, _sha256 in routed_payloads:
                    cached_count += int(_run_ucx_direct(
                        _ucx_direct_send_tensor_record(
                            endpoint=endpoint,
                            record=record,
                            timeout_s=timeout_s,
                            ready_path=_direct_ready_marker_path(self.state_root, endpoint),
                            ready_wait_s=_ucx_direct_ready_wait_s(payload),
                            connect_grace_s=_ucx_direct_connect_grace_s(payload),
                        ),
                        persistent=_ucx_direct_channel_reuse(endpoint),
                    ))
                return


            def send_one(item: tuple[dict[str, Any], dict[str, Any], int, str]) -> None:
                endpoint, record, _bytes, _sha256 = item
                endpoint_backend = _direct_endpoint_backend(endpoint, backend)
                if endpoint_backend == "cuda_ipc":
                    _cuda_ipc_send_tensor_record(
                        endpoint=endpoint,
                        record=record,
                        timeout_s=timeout_s,
                    )
                    return
                pg = _torch_direct_process_group(
                    endpoint=endpoint,
                    backend=endpoint_backend,
                    pg_rank=0,
                    timeout_s=timeout_s,
                )
                ack_sock = _torch_direct_connect_ack_socket(endpoint, timeout_s=timeout_s)
                try:
                    _torch_direct_send_tensor_record(pg, record, endpoint_backend)
                    if ack_sock is not None:
                        _torch_direct_recv_socket_ack(ack_sock)
                finally:
                    if ack_sock is not None:
                        ack_sock.close()
                    _torch_direct_close(pg)

            coalesced_groups: dict[str, list[tuple[dict[str, Any], dict[str, Any], int, str]]] = {}
            individual = []
            for item in routed_payloads:
                endpoint = item[0]
                if bool(endpoint.get("coalesced_weight_group", False)):
                    coalesced_groups.setdefault(str(endpoint["direct_channel_id"]), []).append(item)
                else:
                    individual.append(item)
            for _channel_id, items in sorted(coalesced_groups.items()):
                endpoint = items[0][0]
                source_group_rank = int(endpoint["source_group_rank"])
                if any(int(item[0]["source_group_rank"]) != source_group_rank for item in items):
                    raise RuntimeConfigError("one source rank has inconsistent coalesced weight-group ranks")
                pg = _torch_direct_process_group(
                    endpoint=endpoint,
                    backend="nccl",
                    pg_rank=source_group_rank,
                    timeout_s=timeout_s,
                )
                try:
                    _torch_direct_send_tensor_records_coalesced(
                        pg,
                        [(item[0], item[1]) for item in items],
                    )
                finally:
                    _torch_direct_close(pg)
            for item in individual:
                send_one(item)

        asynchronous = direct_kind == "ucx" and bool(payload.get("source_overlap", False))
        if asynchronous:
            self._launch_background_transfer(
                send_all,
                name=f"reshard-{model}-v{source_version}",
            )
        else:
            send_all()
        duration_s = time.perf_counter() - started
        transferred_bytes = sum(bytes_ for _endpoint, _record, bytes_, _sha256 in routed_payloads)
        local_ipc_bytes = sum(
            bytes_
            for endpoint, _record, bytes_, _sha256 in routed_payloads
            if direct_kind == "torch_dist"
            and _direct_endpoint_backend(endpoint, backend) in {"gloo", "cuda_ipc"}
        )
        combined_sha256 = hashlib.sha256(
            "".join(sha256 for _endpoint, _record, _bytes, sha256 in routed_payloads).encode("ascii")
        ).hexdigest()
        return {
            "model": model,
            "source_worker_id": source_worker_id,
            "source_version": source_version,
            "copied": True,
            "rank": self.rank,
            "bytes": transferred_bytes,
            "network_bytes": transferred_bytes - local_ipc_bytes,
            "local_ipc_bytes": local_ipc_bytes,
            "local_endpoint_count": sum(
                int(
                    direct_kind == "torch_dist"
                    and _direct_endpoint_backend(endpoint, backend) in {"gloo", "cuda_ipc"}
                )
                for endpoint in endpoints
            ),
            "local_transport": (
                _local_endpoint_transport(endpoints, backend)
                if direct_kind == "torch_dist"
                else None
            ),
            "source_payload_bytes": transferred_bytes,
            "destination_rank_count": len(endpoints),
            "sha256": combined_sha256,
            "copy_mode": f"{transport}_memory_reshard_source",
            "transport": transport,
            "wire_format": "tensor_native",
            "optimizer_state": False,
            "shadow": bool(payload.get("shadow", False)),
            "transfer_rank_duration_s": duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(transferred_bytes, duration_s),
            "direct_channel_reuse": any(
                _ucx_direct_channel_reuse(endpoint) for endpoint in endpoints
            ),
            "source_dp_rank": int(payload.get("source_dp_rank", 0)),
            "coalesced_weight_group": bool(payload.get("coalesced_weight_group", False)),
            "coalesced_weight_group_count": int(
                payload.get("coalesced_weight_group_count", 0)
            ),
            "coalesced_weight_group_world_sizes": tuple(
                int(value)
                for value in payload.get("coalesced_weight_group_world_sizes", ())
            ),
            "source_overlap": asynchronous,
            "ucx_direct_cached_endpoint_count": cached_count,
        }

    def _recv_weight_version_direct(self, request: SemanticRequest) -> dict[str, Any]:
        payload = request.payload
        direct_kind = _direct_transport_kind(payload)
        if direct_kind not in {"ucx", "torch_dist"}:
            raise RuntimeConfigError("direct weight-version transfer requires ucx_direct or torch_dist transport")
        model = str(payload["model"])
        source_worker_id = str(payload["source_worker_id"])
        source_version = int(payload["source_version"])
        destination_version = int(payload.get("destination_version", source_version))
        source_world_size = int(payload["source_world_size"])
        shadow = bool(payload.get("shadow", False))
        backend = _torch_direct_backend(payload) if direct_kind == "torch_dist" else None
        transport = f"torch_dist_{backend}" if backend is not None else "ucx_direct"
        if bool(payload.get("in_memory_reshard", False)):
            if direct_kind != "ucx" and backend != "nccl":
                raise RuntimeConfigError("in-memory weight reshard requires ucx_direct or nccl_direct")
            return self._recv_weight_version_reshard_direct(request)
        if self.world_size != source_world_size:
            raise RuntimeConfigError(
                f"direct same-layout weight receive requires destination world_size={source_world_size}, "
                f"got {self.world_size}"
            )
        in_memory = bool(payload.get("in_memory", False)) and (
            direct_kind == "ucx" or backend == "nccl"
        )
        endpoint = _direct_endpoint_for_rank(payload, self.rank)
        local_endpoint = bool(endpoint.get("same_physical_device", False))
        local_transport = (
            _direct_endpoint_backend(endpoint, backend)
            if local_endpoint and direct_kind == "torch_dist"
            else None
        )
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        if in_memory:
            discarded = (
                self._discard_released_weight_state(model)
                if not self.worker.operational_weights_materialized
                else {}
            )
            started = time.perf_counter()
            if direct_kind == "ucx":
                tensor_record, bytes_, sha256, cached = _run_ucx_direct(
                    _ucx_direct_recv_tensor_record(
                        endpoint=endpoint,
                        timeout_s=timeout_s,
                        ready_path=_direct_ready_marker_path(self.state_root, endpoint),
                        device=self._weight_receive_device(),
                    ),
                    persistent=_ucx_direct_channel_reuse(endpoint),
                )
            elif backend == "nccl":
                pg = _torch_direct_process_group(
                    endpoint=endpoint,
                    backend="nccl",
                    pg_rank=1,
                    timeout_s=timeout_s,
                )
                ack_conn = _torch_direct_accept_ack_socket(endpoint, timeout_s=timeout_s)
                try:
                    tensor_record, bytes_, sha256 = _torch_direct_recv_tensor_record(
                        pg,
                        "nccl",
                        device=self._weight_receive_device(),
                    )
                    if ack_conn is not None:
                        _torch_direct_send_socket_ack(ack_conn)
                finally:
                    if ack_conn is not None:
                        ack_conn.close()
                    _torch_direct_close(pg)
                cached = False
            else:
                raise RuntimeConfigError("tensor-native weight receive requires ucx_direct or nccl_direct")
            stored = self._store_tensor_weight_version(
                model=model,
                version=destination_version,
                source_version=source_version,
                tensor_record=tensor_record,
                serialized_bytes=bytes_,
                source_worker_id=source_worker_id,
                shadow=shadow,
            )
            duration_s = time.perf_counter() - started
            return {
                "model": model,
                "source_worker_id": source_worker_id,
                "source_version": source_version,
                "destination_version": destination_version,
                "weight_path": f"memory://{self.worker_id}/{model}/{destination_version}/rank{self.rank}",
                "rank": self.rank,
                "bytes": bytes_,
                "network_bytes": 0 if local_endpoint else bytes_,
                "local_ipc_bytes": bytes_ if local_endpoint else 0,
                "local_endpoint_count": int(local_endpoint),
                "local_transport": local_transport,
                "sha256": sha256,
                "source_sha256": sha256,
                "copy_mode": f"{transport}_memory",
                "transport": transport,
                "wire_format": "tensor_native",
                "optimizer_state": False,
                "shadow": shadow,
                "transfer_rank_duration_s": duration_s,
                "transfer_rank_bytes_per_s": _bytes_per_second(bytes_, duration_s),
                "direct_channel_reuse": bool(payload.get("direct_channel_reuse", False)),
                "ucx_direct_cached_endpoint_count": int(cached),
                "ucx_direct_cached_endpoint": bool(cached),
                "installed_active_weight_version": bool(stored.get("_installed_active", False)),
                "adopted_parameter_bytes": int(stored.get("_adopted_parameter_bytes", 0)),
                **discarded,
            }
        source_metadata = self._load_weight_version_metadata(
            worker_id=source_worker_id,
            model=model,
            version=source_version,
            rank=self.rank,
        )
        destination_path = self._weight_version_path_for(self.worker_id, model, destination_version, self.rank)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination_path.with_suffix(destination_path.suffix + f".{os.getpid()}.{self.rank}.{transport}.tmp")
        if tmp.exists():
            tmp.unlink()
        bytes_ = int(source_metadata["bytes"])
        started = time.perf_counter()
        try:
            cached = False
            if direct_kind == "ucx":
                received_bytes, sha256, cached = _run_ucx_direct(
                    _ucx_direct_recv_file(
                        endpoint=endpoint,
                        destination=tmp,
                        bytes_=bytes_,
                        timeout_s=timeout_s,
                        ready_path=_direct_ready_marker_path(self.state_root, endpoint),
                    ),
                    persistent=_ucx_direct_channel_reuse(endpoint),
                )
            else:
                pg = _torch_direct_process_group(
                    endpoint=endpoint,
                    backend=str(backend),
                    pg_rank=1,
                    timeout_s=timeout_s,
                )
                try:
                    received_bytes, sha256 = _torch_direct_recv_file(
                        pg,
                        str(backend),
                        tmp,
                        bytes_,
                        ack_endpoint=endpoint,
                        timeout_s=timeout_s,
                    )
                    _torch_direct_barrier(pg)
                finally:
                    _torch_direct_close(pg)
            expected_sha = str(source_metadata["sha256"])
            if sha256 != expected_sha:
                raise RuntimeConfigError(
                    f"{transport} weight checksum mismatch for {destination_path}: {sha256} != {expected_sha}"
                )
            tmp.replace(destination_path)
        finally:
            if tmp.exists():
                tmp.unlink()
        duration_s = time.perf_counter() - started
        self._write_weight_version_metadata(
            worker_id=self.worker_id,
            model=model,
            version=destination_version,
            rank=self.rank,
            world_size=self.world_size,
            step=int(payload.get("step", request.step)),
            path=destination_path,
            optimizer_state=bool(source_metadata.get("optimizer_state", False)),
            shadow=shadow,
            source_worker_id=source_worker_id,
            source_version=source_version,
            copy_mode=transport,
            transport=transport,
            sha256=sha256,
        )
        return {
            "model": model,
            "source_worker_id": source_worker_id,
            "source_version": source_version,
            "destination_version": destination_version,
            "weight_path": str(destination_path),
            "rank": self.rank,
            "bytes": int(received_bytes),
            "network_bytes": 0 if local_endpoint else int(received_bytes),
            "local_ipc_bytes": int(received_bytes) if local_endpoint else 0,
            "local_endpoint_count": int(local_endpoint),
            "local_transport": local_transport,
            "sha256": sha256,
            "source_sha256": str(source_metadata["sha256"]),
            "copy_mode": transport,
            "transport": transport,
            "optimizer_state": bool(source_metadata.get("optimizer_state", False)),
            "shadow": shadow,
            "transfer_rank_duration_s": duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(received_bytes, duration_s),
            "direct_channel_reuse": bool(payload.get("direct_channel_reuse", False)),
            "ucx_direct_cached_endpoint_count": int(cached),
            "ucx_direct_cached_endpoint": bool(cached),
        }

    def _recv_weight_version_reshard_direct(self, request: SemanticRequest) -> dict[str, Any]:
        import torch

        payload = request.payload
        direct_kind = _direct_transport_kind(payload)
        backend = _torch_direct_backend(payload) if direct_kind == "torch_dist" else None
        if direct_kind != "ucx" and backend != "nccl":
            raise RuntimeConfigError("tensor-native weight reshard requires ucx_direct or nccl_direct")
        transport = "ucx_direct" if direct_kind == "ucx" else "torch_dist_nccl"
        model = str(payload["model"])
        source_worker_id = str(payload["source_worker_id"])
        source_version = int(payload["source_version"])
        destination_version = int(payload.get("destination_version", source_version))
        source_world_size = int(payload["source_world_size"])
        source_parallelism = tuple(int(x) for x in payload["source_parallelism"])
        destination_parallelism = tuple(int(x) for x in payload["destination_parallelism"])
        shadow = bool(payload.get("shadow", False))
        destination_world_size = (
            destination_parallelism[0] * destination_parallelism[1] * destination_parallelism[2]
        )
        if self.world_size != destination_world_size:
            raise RuntimeConfigError(
                f"in-memory reshard destination world_size={self.world_size} does not match "
                f"parallelism={destination_parallelism}"
            )
        discarded = (
            self._discard_released_weight_state(model)
            if not self.worker.operational_weights_materialized
            else {}
        )
        endpoints = _direct_endpoints_for_rank(payload, self.rank)
        destination_dp_relay = bool(payload.get("destination_dp_relay", False))
        destination_mp_width = destination_parallelism[1] * destination_parallelism[2]
        relay_routes = tuple(
            route
            for route in payload.get("destination_dp_relay_routes", ())
            if isinstance(route, dict)
            and int(route.get("destination_mp_rank", -1)) == self.rank % destination_mp_width
        )
        source_transfer_ranks = tuple(sorted(
            int(route.get("source_rank", -1))
            for route in relay_routes
        )) if destination_dp_relay else tuple(
            sorted(int(endpoint.get("source_rank", -1)) for endpoint in endpoints)
        )
        if not source_transfer_ranks or source_transfer_ranks[0] < 0:
            raise RuntimeConfigError(
                f"in-memory reshard destination rank {self.rank} has invalid "
                f"{'relay routes' if destination_dp_relay else 'source endpoints'}"
            )
        if len(set(source_transfer_ranks)) != len(source_transfer_ranks):
            raise RuntimeConfigError(
                f"in-memory reshard destination rank {self.rank} has duplicate source endpoints"
            )
        timeout_s = float(payload.get("direct_timeout_s", 120.0))
        started = time.perf_counter()
        fragments_by_source_rank: dict[int, dict[str, dict[str, Any]]] = {}
        received_bytes = 0
        cached_count = 0
        payload_hashes: list[str] = []
        receive_device = self._weight_receive_device()
        def receive_one(
            endpoint: dict[str, Any],
        ) -> tuple[int, dict[str, Any], int, str, bool, bool, str | None]:
            source_rank = int(endpoint.get("source_rank", -1))
            local_endpoint = False
            local_transport = None
            if direct_kind == "ucx":
                state, received_payload_bytes, sha256, cached = _run_ucx_direct(
                    _ucx_direct_recv_tensor_record(
                        endpoint=endpoint,
                        timeout_s=timeout_s,
                        ready_path=_direct_ready_marker_path(self.state_root, endpoint),
                        device=receive_device,
                    ),
                    persistent=_ucx_direct_channel_reuse(endpoint),
                )
            else:
                endpoint_backend = _direct_endpoint_backend(endpoint, backend)
                local_endpoint = endpoint_backend in {"gloo", "cuda_ipc"}
                local_transport = endpoint_backend if local_endpoint else None
                if endpoint_backend == "cuda_ipc":
                    state, received_payload_bytes, sha256 = _cuda_ipc_recv_tensor_record(
                        endpoint=endpoint,
                        timeout_s=timeout_s,
                        device=receive_device,
                    )
                    return (
                        source_rank,
                        state,
                        int(received_payload_bytes),
                        sha256,
                        False,
                        True,
                        "cuda_ipc",
                    )
                pg = _torch_direct_process_group(
                    endpoint=endpoint,
                    backend=endpoint_backend,
                    pg_rank=1,
                    timeout_s=timeout_s,
                )
                ack_conn = _torch_direct_accept_ack_socket(endpoint, timeout_s=timeout_s)
                try:
                    state, received_payload_bytes, sha256 = _torch_direct_recv_tensor_record(
                        pg,
                        endpoint_backend,
                        device="cpu" if local_endpoint else receive_device,
                    )
                    if ack_conn is not None:
                        _torch_direct_send_socket_ack(ack_conn)
                finally:
                    if ack_conn is not None:
                        ack_conn.close()
                    _torch_direct_close(pg)
                cached = False
            return (
                source_rank,
                state,
                int(received_payload_bytes),
                sha256,
                bool(cached),
                local_endpoint,
                local_transport,
            )

        coalesced_groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        individual = []
        for index, endpoint in enumerate(endpoints):
            if bool(endpoint.get("coalesced_weight_group", False)):
                coalesced_groups.setdefault(str(endpoint["direct_channel_id"]), []).append(
                    (index, endpoint)
                )
            else:
                individual.append((index, endpoint))
        indexed_received = []
        for _channel_id, indexed_endpoints in sorted(coalesced_groups.items()):
            group_endpoints = [endpoint for _index, endpoint in indexed_endpoints]
            endpoint = group_endpoints[0]
            destination_group_rank = int(endpoint["destination_group_rank"])
            if any(
                int(item["destination_group_rank"]) != destination_group_rank
                for item in group_endpoints
            ):
                raise RuntimeConfigError(
                    "one destination rank has inconsistent coalesced weight-group ranks"
                )
            pg = _torch_direct_process_group(
                endpoint=endpoint,
                backend="nccl",
                pg_rank=destination_group_rank,
                timeout_s=timeout_s,
            )
            try:
                grouped = _torch_direct_recv_tensor_records_coalesced(
                    pg,
                    group_endpoints,
                    device=receive_device,
                )
            finally:
                _torch_direct_close(pg)
            indexed_received.extend(
                (
                    index,
                    (
                        int(item_endpoint["source_rank"]),
                        state,
                        received_payload_bytes,
                        sha256,
                        False,
                        False,
                        None,
                    ),
                )
                for (index, item_endpoint), (state, received_payload_bytes, sha256)
                in zip(indexed_endpoints, grouped)
            )
        indexed_received.extend(
            (index, receive_one(endpoint))
            for index, endpoint in individual
        )
        received = [item for _index, item in sorted(indexed_received)]
        relay_direct_received_bytes = sum(int(item[2]) for item in received)
        relay_direct_local_ipc_bytes = sum(
            int(item[2]) for item in received if bool(item[5])
        )
        relay_direct_local_endpoint_count = sum(int(bool(item[5])) for item in received)
        relay_direct_local_transports = {
            str(item[6]) for item in received if item[6] is not None
        }
        relay_broadcast_bytes = 0
        if destination_dp_relay:
            if direct_kind != "torch_dist" or backend != "nccl":
                raise RuntimeConfigError(
                    "destination-DP weight relays require torch_dist NCCL"
                )
            received_by_source: dict[int, dict[str, Any]] = {}
            for source_rank, state, _bytes, _sha256, *_rest in received:
                if source_rank in received_by_source:
                    raise RuntimeConfigError(
                        f"duplicate relay payload from source rank {source_rank}"
                    )
                if (
                    not isinstance(state, dict)
                    or state.get("schema_version") != "rpeg.weight_reshard_fragment.v1"
                    or str(state.get("worker_id")) != source_worker_id
                    or str(state.get("model")) != model
                    or int(state.get("version", -1)) != source_version
                    or int(state.get("rank", -1)) != source_rank
                    or int(state.get("world_size", -1)) != source_world_size
                    or int(state.get("destination_rank", -1)) != self.rank
                ):
                    raise RuntimeConfigError(
                        f"invalid destination-DP relay payload from source rank {source_rank}"
                    )
                received_by_source[source_rank] = state

            from megatron.core import parallel_state

            dp_group = parallel_state.get_data_parallel_group()
            if str(torch.distributed.get_backend(dp_group)).lower() != "nccl":
                raise RuntimeConfigError(
                    "destination-DP weight relay requires an NCCL data-parallel group"
                )
            dp_size = destination_parallelism[0]
            relayed = []
            for route in sorted(
                relay_routes,
                key=lambda item: (
                    int(item["source_rank"]),
                    int(item["relay_destination_rank"]),
                ),
            ):
                source_rank = int(route["source_rank"])
                relay_rank = int(route["relay_destination_rank"])
                if relay_rank % destination_mp_width != self.rank % destination_mp_width:
                    raise RuntimeConfigError(
                        f"weight relay rank {relay_rank} is outside destination MP shard "
                        f"{self.rank % destination_mp_width}"
                    )
                relay_dp_rank = relay_rank // destination_mp_width
                if not 0 <= relay_dp_rank < dp_size:
                    raise RuntimeConfigError(f"invalid destination DP relay rank {relay_rank}")
                source_global_rank = torch.distributed.get_global_rank(
                    dp_group,
                    relay_dp_rank,
                )
                source_record = received_by_source.pop(source_rank, None)
                if self.rank == relay_rank:
                    if source_record is None:
                        raise RuntimeConfigError(
                            f"destination relay rank {relay_rank} received no fragment "
                            f"from source rank {source_rank}"
                        )
                    source_record = _tensor_record_to_device(
                        source_record,
                        receive_device,
                    )
                elif source_record is not None:
                    raise RuntimeConfigError(
                        f"non-relay destination rank {self.rank} unexpectedly received "
                        f"source rank {source_rank}"
                    )
                state, broadcast_bytes = _torch_model_parallel_broadcast_tensor_record(
                    source_record,
                    group=dp_group,
                    source_rank=source_global_rank,
                    expected_schema="rpeg.weight_reshard_fragment.v1",
                )
                state = dict(state)
                state["destination_rank"] = self.rank
                manifest, _tensor_bytes = _tensor_stream_manifest(state)
                sha256 = hashlib.sha256(
                    _tensor_stream_manifest_bytes(manifest)
                ).hexdigest()
                relay_broadcast_bytes += int(broadcast_bytes)
                relayed.append((
                    source_rank,
                    state,
                    int(broadcast_bytes),
                    sha256,
                    False,
                    False,
                    None,
                ))
            if received_by_source:
                raise RuntimeConfigError(
                    f"destination relay rank {self.rank} received unexpected source ranks "
                    f"{sorted(received_by_source)}"
                )
            received = relayed
        mixed_host_merge = _tensor_records_span_devices(
            [state for _source_rank, state, *_rest in received]
        )
        local_ipc_bytes = relay_direct_local_ipc_bytes if destination_dp_relay else 0
        local_endpoint_count = (
            relay_direct_local_endpoint_count if destination_dp_relay else 0
        )
        local_transports: set[str] = (
            set(relay_direct_local_transports) if destination_dp_relay else set()
        )
        for (
            source_rank,
            state,
            received_payload_bytes,
            sha256,
            cached,
            local_endpoint,
            local_transport,
        ) in received:
            if mixed_host_merge:
                state = _tensor_record_to_cpu(state)
            if not isinstance(state, dict) or state.get("schema_version") != "rpeg.weight_reshard_fragment.v1":
                raise RuntimeConfigError(
                    f"in-memory reshard source rank {source_rank} has an unsupported weight schema"
                )
            if (
                str(state.get("worker_id")) != source_worker_id
                or str(state.get("model")) != model
                or int(state.get("version", -1)) != source_version
                or int(state.get("rank", -1)) != source_rank
                or int(state.get("world_size", -1)) != source_world_size
                or int(state.get("destination_rank", -1)) != self.rank
            ):
                raise RuntimeConfigError(
                    f"in-memory reshard source metadata mismatch for {source_worker_id}:{model}:"
                    f"{source_version} rank {source_rank}"
                )
            if source_rank in fragments_by_source_rank:
                raise RuntimeConfigError(f"duplicate in-memory reshard payload for source rank {source_rank}")
            fragments_by_source_rank[source_rank] = dict(state.get("fragments") or {})
            received_bytes += int(received_payload_bytes)
            if local_endpoint:
                local_ipc_bytes += int(received_payload_bytes)
                local_endpoint_count += 1
                if local_transport is not None:
                    local_transports.add(local_transport)
            cached_count += int(cached)
            payload_hashes.append(sha256)
        missing = sorted(set(source_transfer_ranks) - set(fragments_by_source_rank))
        if missing:
            raise RuntimeConfigError(f"in-memory reshard is missing source ranks {missing}")
        destination_name_map = self._state_dict_name_map()
        destination_shard_specs = self._state_dict_shard_specs()
        destination_state = merge_state_dict_fragments(
            [fragments_by_source_rank[rank] for rank in source_transfer_ranks],
            destination_name_map=destination_name_map,
            destination_shard_specs=destination_shard_specs,
            source_parallelism=source_parallelism,
            destination_parallelism=destination_parallelism,
            destination_rank=self.rank,
            clone_complete_values=False,
            consume_inputs=True,
        )
        destination_record = {
            "schema_version": "rpeg.weight_version.v1",
            "worker_id": self.worker_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "model": model,
            "version": destination_version,
            "step": int(payload.get("step", request.step)),
            "source_worker_id": source_worker_id,
            "source_version": source_version,
            "source_parallelism": list(source_parallelism),
            "destination_parallelism": list(destination_parallelism),
            "shadow": shadow,
            "state_dict": destination_state,
            "state_dict_name_map": destination_name_map,
            "state_dict_shard_specs": destination_shard_specs,
        }
        destination_record["_serialized_bytes"] = _weight_state_payload_bytes(destination_record)
        installed_active = False
        adopted_bytes = 0
        if not self.worker.operational_weights_materialized:
            # The routed destination record owns complete tensors regardless of whether RPEG labels
            # the version as a shadow copy. Adopt them when pruning has released the dead active
            # version, avoiding a second destination-sized allocation during the later load
            # operation.
            adopted_bytes = self.worker.adopt_rpeg_state_dict_storage(
                destination_state,
                strict=True,
            )
            self._active_weight_versions[model] = destination_version
            installed_active = True
        else:
            self._cache_weight_version(model, destination_version, destination_record)
        duration_s = time.perf_counter() - started
        combined_sha256 = hashlib.sha256("".join(payload_hashes).encode("ascii")).hexdigest()
        return {
            "model": model,
            "source_worker_id": source_worker_id,
            "source_version": source_version,
            "destination_version": destination_version,
            "weight_path": f"memory://{self.worker_id}/{model}/{destination_version}/rank{self.rank}",
            "rank": self.rank,
            "bytes": received_bytes,
            "network_bytes": received_bytes - local_ipc_bytes,
            "local_ipc_bytes": local_ipc_bytes,
            "local_endpoint_count": local_endpoint_count,
            "local_transport": (
                next(iter(local_transports))
                if len(local_transports) == 1
                else tuple(sorted(local_transports))
                if local_transports
                else None
            ),
            "sha256": combined_sha256,
            "source_sha256": combined_sha256,
            "copy_mode": f"{transport}_memory_reshard",
            "transport": transport,
            "wire_format": "tensor_native",
            "optimizer_state": False,
            "shadow": shadow,
            "source_rank_count": len(source_transfer_ranks),
            "destination_dp_relay": destination_dp_relay,
            "relay_direct_received_bytes": relay_direct_received_bytes,
            "relay_broadcast_bytes": relay_broadcast_bytes,
            "transfer_rank_duration_s": duration_s,
            "transfer_rank_bytes_per_s": _bytes_per_second(received_bytes, duration_s),
            "direct_channel_reuse": any(
                _ucx_direct_channel_reuse(endpoint) for endpoint in endpoints
            ),
            "source_dp_rank": int(payload.get("source_dp_rank", 0)),
            "coalesced_weight_group": bool(payload.get("coalesced_weight_group", False)),
            "coalesced_weight_group_count": int(
                payload.get("coalesced_weight_group_count", 0)
            ),
            "coalesced_weight_group_world_sizes": tuple(
                int(value)
                for value in payload.get("coalesced_weight_group_world_sizes", ())
            ),
            "ucx_direct_cached_endpoint_count": cached_count,
            "installed_active_weight_version": installed_active,
            "adopted_parameter_bytes": adopted_bytes,
            **discarded,
        }

    def _weight_receive_device(self):
        import torch

        fallback_device = None
        for module in (self.worker.model, self.worker.value_head, self.worker.reward_head):
            if module is None:
                continue
            for parameter in module.parameters():
                device = getattr(parameter, "device", None)
                if device is not None and device.type == "cuda":
                    return device
                if device is not None and fallback_device is None:
                    fallback_device = device
            for value in module.state_dict().values():
                device = getattr(value, "device", None)
                if device is not None and device.type == "cuda":
                    return device
                if device is not None and fallback_device is None:
                    fallback_device = device
        # Pipeline stages can have only CPU-resident metadata or an empty local shard. Semantic GPU
        # workers must still receive NCCL tensors on their bound CUDA device.
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return fallback_device or torch.device("cpu")

    def _optimizer_state_by_name(self) -> dict[str, Any] | None:
        optimizer = getattr(self.worker, "_optimizer", None)
        if optimizer is None:
            return None
        name_by_param = {param: name for name, param in self._named_parameters().items()}
        out = {}
        for param, entry in optimizer.state.items():
            name = name_by_param.get(param)
            if name is None or not entry:
                continue
            out[name] = {
                key: value.detach().cpu().clone() if hasattr(value, "detach") else value
                for key, value in entry.items()
            }
        return out

    def _load_optimizer_state_by_name(self, state_by_name: dict[str, Any]) -> None:
        optimizer = self.worker._ensure_optimizer(None)
        param_by_name = self._named_parameters()
        optimizer.state.clear()
        for name, entry in state_by_name.items():
            param = param_by_name.get(name)
            if param is None:
                continue
            restored = {}
            for key, value in entry.items():
                if hasattr(value, "to"):
                    restored[key] = value.to(device=param.device)
                else:
                    restored[key] = value
            optimizer.state[param] = restored

    def _state_dict_name_map(self) -> dict[str, str]:
        mapping = self._local_to_global_layer_indices()
        out = {}
        for name in self._state_dict_view():
            out[name] = _global_layer_name(name, mapping)
        return out

    def _state_dict_shard_specs(self) -> dict[str, dict[str, Any]]:
        params = self._named_parameters()
        released_shapes = {
            id(parameter): tuple(int(dim) for dim in shape)
            for parameter, shape, _dtype, _device
            in getattr(self.worker, "_released_operational_parameter_specs", ())
        }
        tp = self._parallelism()[2]
        out: dict[str, dict[str, Any]] = {}
        for name, value in self._state_dict_view().items():
            param = params.get(name)
            is_tp = bool(getattr(param, "tensor_model_parallel", False)) if param is not None else False
            dim = int(getattr(param, "partition_dim", 0)) if is_tp else None
            stride = int(getattr(param, "partition_stride", 1)) if is_tp else 1
            if not is_tp and value is not None:
                fallback_dim = tp_shard_dim(name, int(getattr(value, "ndim", 0)))
                if fallback_dim is not None and tp > 1:
                    is_tp = True
                    dim = fallback_dim
                    stride = 2 if name.endswith(("mlp.linear_fc1.weight", "mlp.linear_fc1.bias")) else 1
            spec: dict[str, Any] = {"is_tp": is_tp, "stride": stride}
            if dim is not None:
                spec["dim"] = dim
            if value is not None and hasattr(value, "shape"):
                global_shape = list(released_shapes.get(id(param), tuple(value.shape)))
                if is_tp and dim is not None:
                    global_shape[dim] *= tp
                spec["global_shape"] = global_shape
            out[name] = spec
        return out

    def _state_dict_view(self) -> dict[str, Any]:
        state = dict(self.worker.model.state_dict())
        if self.worker.value_head is not None:
            state.update({f"rpeg_value_head.{name}": value for name, value in self.worker.value_head.state_dict().items()})
        if self.worker.reward_head is not None:
            state.update({f"rpeg_reward_head.{name}": value for name, value in self.worker.reward_head.state_dict().items()})
        return state

    def _optimizer_parameter_names(self) -> tuple[str, ...]:
        return tuple(self._named_parameters())

    def _filter_destination_optimizer_state(self, state_by_name: dict[str, Any] | None) -> dict[str, Any] | None:
        from runtime.optimizer_sharding import balanced_parameter_owners

        if not state_by_name:
            return state_by_name
        cfg = getattr(self.worker, "config", None)
        if cfg is None:
            return state_by_name
        mode = str(getattr(cfg, "distributed_optimizer", "none") or "none").strip().lower().replace("-", "_")
        if mode not in {
            "zero1", "zero_1", "zero_stage_1", "stage1",
            "zero2", "zero_2", "zero_stage_2", "stage2",
        }:
            return state_by_name
        dp = self._dp_size()
        if dp <= 1:
            return state_by_name
        named_parameters = self._named_parameters()
        names = tuple(named_parameters)
        if not names:
            return state_by_name
        owned = {
            name
            for name, owner in zip(
                names,
                balanced_parameter_owners(named_parameters.values(), dp),
                strict=True,
            )
            if owner == self._dp_rank()
        }
        return {name: entry for name, entry in state_by_name.items() if name in owned}

    def _named_parameters(self) -> dict[str, Any]:
        if not hasattr(self.worker.model, "named_parameters"):
            return {}
        params = dict(self.worker.model.named_parameters())
        if self.worker.value_head is not None:
            params.update({f"rpeg_value_head.{name}": param for name, param in self.worker.value_head.named_parameters()})
        if self.worker.reward_head is not None:
            params.update({f"rpeg_reward_head.{name}": param for name, param in self.worker.reward_head.named_parameters()})
        return params

    def _local_to_global_layer_indices(self) -> dict[int, int]:
        decoder = getattr(self.worker.model, "decoder", None)
        layers = getattr(decoder, "layers", None)
        if layers is None:
            return {}
        out = {}
        for local_idx, layer in enumerate(layers):
            layer_number = getattr(layer, "layer_number", None)
            out[local_idx] = int(layer_number) - 1 if layer_number is not None else local_idx
        return out

    def _ppo_prompt_ids(self, request: SemanticRequest):
        import torch

        payload = request.payload
        batch_offset = int(payload.get("partial_batch_offset", 0))
        partial_n = payload.get("partial_n_seqs")
        if "input_ids" in payload:
            input_ids = torch.tensor(payload["input_ids"], dtype=torch.long)
            if partial_n is not None:
                input_ids = input_ids[batch_offset:batch_offset + int(partial_n)].contiguous()
            start, end = self._local_batch_range(int(input_ids.shape[0]))
            input_ids = input_ids[start:end].contiguous()
            prompt_lens = payload.get("prompt_lens")
            if prompt_lens is None:
                prompt_lens = [int(input_ids.shape[1])] * int(input_ids.shape[0])
            else:
                prompt_values = list(prompt_lens)
                if partial_n is not None:
                    prompt_values = prompt_values[batch_offset:batch_offset + int(partial_n)]
                prompt_lens = [int(x) for x in prompt_values[start:end]]
            return input_ids, prompt_lens, None
        if payload.get("train_prompts"):
            global_batch = int(payload.get("batch_size", 1))
            effective_batch = int(partial_n) if partial_n is not None else global_batch
            start, end = self._local_batch_range(effective_batch)
            samples_per_prompt = int(payload.get("samples_per_prompt", 1))
            prompt_batch_size = global_batch
            if samples_per_prompt > 1:
                if global_batch % samples_per_prompt != 0:
                    raise RuntimeConfigError(
                        f"rollout batch {global_batch} must be divisible by samples_per_prompt={samples_per_prompt}"
                    )
                prompt_batch_size = global_batch // samples_per_prompt
            prompts = self._prompt_batch(
                str(payload["train_prompts"]),
                step=request.step,
                batch_size=prompt_batch_size,
                offset=int(payload.get("prompt_offset", 0)),
                shuffle=bool(payload.get("data_shuffle", False)),
                seed=int(payload.get("data_seed", 0)),
                shard_index=int(payload.get("data_shard_index", 0)),
                num_shards=int(payload.get("data_num_shards", 1)),
            )
            if samples_per_prompt > 1:
                prompts = [prompt for prompt in prompts for _ in range(samples_per_prompt)]
            prompts = prompts[batch_offset:batch_offset + effective_batch][start:end]
            tokenizer = self._load_tokenizer(payload)
            max_length = int(payload.get("max_prompt_length") or payload.get("context_length") or 0)
            if max_length <= 0:
                raise RuntimeConfigError("PPO prompt tokenization requires context_length or max_prompt_length")
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = encoded["input_ids"].to(dtype=torch.long)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                prompt_lens = [int(input_ids.shape[1])] * int(input_ids.shape[0])
            else:
                # Responses are appended after the padded prompt tensor, so token statistics start
                # at the physical prompt width on every row.
                prompt_lens = [int(input_ids.shape[1])] * int(input_ids.shape[0])
            return input_ids, prompt_lens, prompts
        global_batch = int(payload.get("batch_size", 1))
        effective_batch = int(partial_n) if partial_n is not None else global_batch
        start, end = self._local_batch_range(effective_batch)
        batch = end - start
        context_length = int(payload.get("context_length", payload.get("seq_length", 8)))
        vocab = self.worker.architecture.vocab_size if self.worker.architecture is not None else int(payload.get("vocab_size", 128))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(payload.get("seed", 1234)) + int(payload.get("data_seed", 0)) + int(request.step))
        all_ids = torch.randint(0, max(vocab, 2), (global_batch, context_length), generator=generator, dtype=torch.long)
        input_ids = all_ids[batch_offset:batch_offset + effective_batch][start:end].contiguous()
        prompt_lens = [context_length] * batch
        return input_ids, prompt_lens, None

    def _load_tokenizer(self, payload: dict[str, Any]):
        tokenizer_path = str(payload.get("tokenizer_path") or self.tokenizer_path or self.model_path or "")
        if not tokenizer_path:
            raise RuntimeConfigError("PPO semantic worker needs tokenizer_path or model_path to tokenize prompts")
        if self.tokenizer is not None and self.tokenizer_path == tokenizer_path:
            return self.tokenizer
        try:
            from transformers import AutoTokenizer  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeDependencyError("PPO semantic prompt ingestion requires transformers") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=bool(payload.get("tokenizer_trust_remote_code", False)),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = str(payload.get("tokenizer_padding_side", "left"))
        tokenizer.truncation_side = str(payload.get("tokenizer_truncation_side", "left"))
        self.tokenizer = tokenizer
        self.tokenizer_path = tokenizer_path
        return tokenizer

    def _prompt_batch(
        self,
        path: str,
        *,
        step: int,
        batch_size: int,
        offset: int = 0,
        shuffle: bool = False,
        seed: int = 0,
        shard_index: int = 0,
        num_shards: int = 1,
    ) -> list[str]:
        prompts = self._load_prompts(path)
        if batch_size <= 0:
            raise RuntimeConfigError("batch_size must be positive for PPO prompt ingestion")
        indices = _dataset_batch_indices(
            len(prompts),
            step=step,
            batch_size=batch_size,
            offset=offset,
            shuffle=shuffle,
            seed=seed,
            shard_index=shard_index,
            num_shards=num_shards,
        )
        return [prompts[idx] for idx in indices]

    def _load_prompts(self, path: str) -> list[str]:
        cached = self._prompt_cache.get(path)
        if cached is not None:
            return cached
        prompts: list[str] = []
        prompt_path = Path(path)
        with prompt_path.open() as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "prompt" not in row:
                    raise RuntimeConfigError(f"{prompt_path}:{line_no} missing prompt field")
                prompts.append(str(row["prompt"]))
        if not prompts:
            raise RuntimeConfigError(f"PPO prompt file is empty: {prompt_path}")
        self._prompt_cache[path] = prompts
        return prompts

    def _ppo_actor_generate(self, request: SemanticRequest, *, samples_per_prompt: int = 1) -> dict[str, Any]:
        if request.payload.get("artifact_part") is not None:
            return self._ppo_actor_generate_phase(request, samples_per_prompt=samples_per_prompt)
        return self._ppo_actor_generate_batch([request], samples_per_prompt=samples_per_prompt)[request.request_id]

    def _ppo_actor_generate_phase(self, request: SemanticRequest, *, samples_per_prompt: int) -> dict[str, Any]:
        payload = request.payload
        part = int(payload["artifact_part"])
        parts = int(payload.get("artifact_parts") or 0)
        if parts <= 0 or not 0 <= part < parts:
            raise RuntimeConfigError(f"invalid generation artifact part {part}/{parts}")
        input_ids = prompt_lens = prompts = None
        state_key = f"{request.operation}:{request.step}"
        if part == 0:
            full_payload = dict(payload)
            full_payload.pop("partial_n_seqs", None)
            full_payload.pop("partial_batch_offset", None)
            full_request = SemanticRequest(
                request_id=request.request_id,
                worker_id=request.worker_id,
                vertex_id=request.vertex_id,
                node_id=request.node_id,
                operation=request.operation,
                step=request.step,
                payload=full_payload,
            )
            input_ids, prompt_lens, prompts = self._ppo_prompt_ids(full_request)
            self._phased_generation_prompts[state_key] = prompts
        else:
            prompts = self._phased_generation_prompts.get(state_key)
        tokenizer = self._load_tokenizer(payload)
        eos_token_ids = getattr(tokenizer, "eos_token_id", None)
        phase_prompt_tokens = sum(prompt_lens or ())
        global_batch_size = int(payload.get("batch_size", 0))
        original_index_offset = self._local_batch_range(global_batch_size)[0]
        completion_ids, old_logps, response_mask, completed_prompt_lens, original_indices, counters = (
            self.worker.generate_quantile_phase(
                state_key,
                input_ids=input_ids,
                prompt_lens=prompt_lens,
                target_tokens=int(payload["generation_length"]),
                max_new_tokens=int(payload.get("max_generation_length", payload["generation_length"])),
                final_phase=part == parts - 1,
                do_sample=bool(payload.get("do_sample", True)),
                temperature=float(payload.get("temperature", 1.0)),
                top_p=float(payload.get("top_p", 1.0)),
                top_k=int(payload.get("top_k", 0)),
                seed=self._generation_seed(request),
                generation_backend=str(payload.get("generation_backend", "full_sequence")),
                eos_token_ids=() if eos_token_ids is None else (int(eos_token_ids),),
                original_index_offset=original_index_offset,
            )
        )
        rollout = {
            "completion_ids": completion_ids.detach(),
            "old_logps": old_logps.detach(),
            "response_mask": response_mask.detach(),
            "prompt_lens": completed_prompt_lens,
            "generation_length": int(payload.get("max_generation_length", payload["generation_length"])),
            "original_indices": original_indices,
        }
        if prompts is not None and original_indices:
            rollout["prompts"] = [
                prompts[index - original_index_offset]
                for index in original_indices
            ]
        elif prompts is not None:
            rollout["prompts"] = []
        rollout_name = self._artifact_name("rollout", payload)
        writer = self._is_dp_artifact_writer()
        self._save_local_artifact(request.step, rollout_name, rollout)
        self._merge_artifact_parts_if_last(request.step, "rollout", payload)
        if part == parts - 1:
            self._phased_generation_prompts.pop(state_key, None)
        return {
            "batch_size": len(original_indices),
            "global_batch_size": int(payload.get("batch_size", len(original_indices))),
            "sequence_length": int(completion_ids.shape[1]) if completion_ids.ndim == 2 else 0,
            "generation_length": int(payload["generation_length"]),
            "generated_tokens": int(response_mask.sum().item()),
            "prompt_tokens": int(phase_prompt_tokens),
            "samples_per_prompt": int(samples_per_prompt),
            "dp_rank": self._dp_rank(),
            "dp_size": self._dp_size(),
            "rank": self.rank,
            "artifact_numeric_fingerprint": (
                _artifact_numeric_fingerprint(rollout, name=rollout_name)
                if writer
                else None
            ),
            **counters,
            **self._partial_artifact_counters(payload, actual_n_seqs=len(original_indices)),
        }

    def _generation_seed(self, request: SemanticRequest) -> int:
        return int(request.payload.get("seed", 1234)) + request.step * 100_000 + self._dp_rank()

    def _ppo_actor_generate_batch(
        self,
        requests: list[SemanticRequest],
        *,
        samples_per_prompt: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        import torch

        if not requests:
            return {}
        if any(request.payload.get("artifact_part") is not None for request in requests):
            raise RuntimeConfigError("continuation generation phases must execute in graph dependency order")
        first_key = self.generation_batch_key(requests[0])
        if first_key is None:
            raise RuntimeConfigError(f"operation {requests[0].operation!r} is not generation-batchable")
        for request in requests[1:]:
            key = self.generation_batch_key(request)
            if key != first_key:
                raise RuntimeConfigError(
                    "dynamic generation batch received incompatible requests: "
                    f"{requests[0].request_id} key={first_key!r}, {request.request_id} key={key!r}"
                )

        max_new_tokens = int(requests[0].payload["generation_length"])
        prepared: list[dict[str, Any]] = []
        prompt_width: int | None = None
        for request in requests:
            prompt_ids, prompt_lens, prompts = self._ppo_prompt_ids(request)
            if prompt_width is None:
                prompt_width = int(prompt_ids.shape[1])
            elif int(prompt_ids.shape[1]) != prompt_width:
                raise RuntimeConfigError(
                    "dynamic generation batching requires equal padded prompt widths; "
                    f"{requests[0].request_id} width={prompt_width}, "
                    f"{request.request_id} width={int(prompt_ids.shape[1])}"
                )
            prepared.append({
                "request": request,
                "prompt_ids": prompt_ids,
                "prompt_lens": prompt_lens,
                "prompts": prompts,
                "batch_size": int(prompt_ids.shape[0]),
            })
        batched_prompt_ids = torch.cat([item["prompt_ids"] for item in prepared], dim=0)
        batched_prompt_lens: list[int] = []
        for item in prepared:
            batched_prompt_lens.extend(int(x) for x in item["prompt_lens"])
        first_payload = requests[0].payload
        completion_ids, old_logps, response_mask, prompt_lens = self.worker.generate_with_logps(
            batched_prompt_ids,
            max_new_tokens=max_new_tokens,
            prompt_lens=batched_prompt_lens,
            do_sample=bool(first_payload.get("do_sample", True)),
            temperature=float(first_payload.get("temperature", 1.0)),
            top_p=float(first_payload.get("top_p", 1.0)),
            top_k=int(first_payload.get("top_k", 0)),
            seed=self._generation_seed(requests[0]),
            generation_backend=str(first_payload.get("generation_backend", "full_sequence")),
        )
        generation_counters = dict(getattr(self.worker, "_last_generation_counters", {}) or {})
        counters_by_request: dict[str, dict[str, Any]] = {}
        offset = 0
        peer_ids = [request.request_id for request in requests]
        physical_batch_size = int(completion_ids.shape[0])
        physical_generated_tokens = int(response_mask.detach().float().sum().cpu())
        for batch_index, item in enumerate(prepared):
            request = item["request"]
            batch_size = int(item["batch_size"])
            end = offset + batch_size
            request_completion_ids = completion_ids[offset:end]
            request_old_logps = old_logps[offset:end]
            request_response_mask = response_mask[offset:end]
            request_prompt_lens = [int(x) for x in prompt_lens[offset:end]]
            request_prompts = item["prompts"]
            offset = end
            rollout = {
                "prompt_ids": item["prompt_ids"].detach(),
                "completion_ids": request_completion_ids.detach(),
                "old_logps": request_old_logps.detach(),
                "response_mask": request_response_mask.detach(),
                "prompt_lens": request_prompt_lens,
                "generation_length": max_new_tokens,
            }
            if request_prompts is not None:
                rollout["prompts"] = request_prompts
            rollout_name = self._artifact_name("rollout", request.payload)
            writer = self._is_dp_artifact_writer()
            self._save_local_artifact(request.step, rollout_name, rollout)
            self._merge_artifact_parts_if_last(request.step, "rollout", request.payload)
            request_samples_per_prompt = samples_per_prompt
            if request_samples_per_prompt is None:
                request_samples_per_prompt = int(request.payload.get("samples_per_prompt", 1))
            counters_by_request[request.request_id] = {
                "batch_size": int(request_completion_ids.shape[0]),
                "global_batch_size": int(request.payload.get("batch_size", request_completion_ids.shape[0])),
                "sequence_length": int(request_completion_ids.shape[1]),
                "generation_length": max_new_tokens,
                "generated_tokens": int(request_response_mask.detach().float().sum().cpu()),
                "prompt_tokens": sum(request_prompt_lens),
                "prompt_source": "jsonl" if request_prompts is not None else "synthetic",
                "samples_per_prompt": int(request_samples_per_prompt),
                "do_sample": bool(request.payload.get("do_sample", True)),
                "temperature": float(request.payload.get("temperature", 1.0)),
                "top_p": float(request.payload.get("top_p", 1.0)),
                "top_k": int(request.payload.get("top_k", 0)),
                "generation_backend": str(
                    generation_counters.get(
                        "generation_backend",
                        request.payload.get("generation_backend", "full_sequence"),
                    )
                ),
                "kv_cache_enabled": bool(generation_counters.get("kv_cache_enabled", False)),
                "prefill_forward_steps": int(generation_counters.get("prefill_forward_steps", 0)),
                "decode_forward_steps": int(generation_counters.get("decode_forward_steps", max_new_tokens)),
                "generation_forward_steps": int(generation_counters.get("generation_forward_steps", max_new_tokens)),
                "kv_cache_reused": bool(generation_counters.get("kv_cache_reused", False)),
                "kv_cache_pool_entries": int(generation_counters.get("kv_cache_pool_entries", 0)),
                "kv_cache_pool_hits": int(generation_counters.get("kv_cache_pool_hits", 0)),
                "kv_cache_pool_misses": int(generation_counters.get("kv_cache_pool_misses", 0)),
                "kv_cache_pool_evictions": int(generation_counters.get("kv_cache_pool_evictions", 0)),
                "kv_cache_max_batch_size": int(generation_counters.get("kv_cache_max_batch_size", completion_ids.shape[0])),
                "kv_cache_max_sequence_length": int(
                    generation_counters.get("kv_cache_max_sequence_length", completion_ids.shape[1])
                ),
                "old_logps_from_generation": bool(generation_counters.get("old_logps_from_generation", False)),
                "old_logp_forward_steps": int(generation_counters.get("old_logp_forward_steps", 1)),
                "dynamic_generation_batch_request_count": len(requests),
                "dynamic_generation_batch_size": physical_batch_size,
                "dynamic_generation_batch_generated_tokens": physical_generated_tokens,
                "dynamic_generation_batch_index": batch_index,
                "dynamic_generation_batch_peer_request_ids": peer_ids,
                "dp_rank": self._dp_rank(),
                "dp_size": self._dp_size(),
                "rank": self.rank,
                "artifact_numeric_fingerprint": (
                    _artifact_numeric_fingerprint(rollout, name=rollout_name)
                    if writer
                    else None
                ),
                **self._partial_artifact_counters(request.payload),
            }
        return counters_by_request

    def _ppo_reference_logprobs(self, request: SemanticRequest) -> dict[str, Any]:
        rollout = self._load_local_artifact(request.step, self._artifact_name("rollout", request.payload, part_input=True))
        if int(rollout["completion_ids"].shape[0]) == 0:
            import torch

            logps = torch.empty((0, int(rollout["generation_length"])), dtype=torch.float32)
            response_mask = logps.clone()
        else:
            logps, response_mask = self.worker.response_token_logps(
                rollout["completion_ids"],
                rollout["prompt_lens"],
                int(rollout["generation_length"]),
                grad=False,
            )
        output_name = self._artifact_name("ref_logps", request.payload)
        writer = self._is_dp_artifact_writer()
        artifact = {
            "ref_logps": logps.detach(),
            "response_mask": response_mask.detach(),
            "original_indices": list(rollout.get("original_indices", ())),
        }
        if writer:
            self._save_local_artifact(
                request.step,
                output_name,
                artifact,
            )
        self._merge_artifact_parts_if_last(request.step, "ref_logps", request.payload)
        counters = {
            "tokens": int(response_mask.detach().float().sum().cpu()) if writer else None,
            "dp_rank": self._dp_rank(),
            "dp_size": self._dp_size(),
            "rank": self.rank,
            "artifact_numeric_fingerprint": (
                _artifact_numeric_fingerprint(artifact, name=output_name)
                if writer
                else None
            ),
            **self._partial_artifact_counters(
                request.payload,
                actual_n_seqs=int(rollout["completion_ids"].shape[0]),
            ),
        }
        counters.update(_trim_cuda_cache_after_frozen_inference())
        return counters

    def _ppo_reward_score(self, request: SemanticRequest) -> dict[str, Any]:
        rollout = self._load_local_artifact(request.step, self._artifact_name("rollout", request.payload, part_input=True))
        if int(rollout["completion_ids"].shape[0]) == 0:
            import torch

            rewards = torch.empty((0,), dtype=torch.float32)
        else:
            rewards = self.worker.reward_scores(
                rollout["completion_ids"],
                rollout["prompt_lens"],
                int(rollout["generation_length"]),
                response_mask=rollout["response_mask"],
            )
        writer = self._is_dp_artifact_writer()
        output_name = self._artifact_name("rewards", request.payload)
        artifact = {
            "rewards": rewards.detach(),
            "original_indices": list(rollout.get("original_indices", ())),
        }
        if writer:
            self._save_local_artifact(
                request.step,
                output_name,
                artifact,
            )
        self._merge_artifact_parts_if_last(request.step, "rewards", request.payload)
        counters = {
            "reward_mean": (
                float(rewards.detach().float().mean().cpu())
                if writer and rewards.numel()
                else None
            ),
            "dp_rank": self._dp_rank(),
            "dp_size": self._dp_size(),
            "rank": self.rank,
            "artifact_numeric_fingerprint": (
                _artifact_numeric_fingerprint(artifact, name=output_name)
                if writer
                else None
            ),
            **self._partial_artifact_counters(
                request.payload,
                actual_n_seqs=int(rollout["completion_ids"].shape[0]),
            ),
        }
        counters.update(_trim_cuda_cache_after_frozen_inference())
        return counters

    def _ppo_critic_values(self, request: SemanticRequest) -> dict[str, Any]:
        rollout = self._load_local_artifact(request.step, self._artifact_name("rollout", request.payload, part_input=True))
        if int(rollout["completion_ids"].shape[0]) == 0:
            import torch

            values = torch.empty((0, int(rollout["generation_length"])), dtype=torch.float32)
            response_mask = values.clone()
        else:
            values, response_mask = self.worker.response_token_values(
                rollout["completion_ids"],
                rollout["prompt_lens"],
                int(rollout["generation_length"]),
                grad=False,
            )
        writer = self._is_dp_artifact_writer()
        output_name = self._artifact_name("values", request.payload)
        artifact = {
            "values": values.detach(),
            "response_mask": response_mask.detach(),
            "original_indices": list(rollout.get("original_indices", ())),
        }
        if writer:
            self._save_local_artifact(
                request.step,
                output_name,
                artifact,
            )
        self._merge_artifact_parts_if_last(request.step, "values", request.payload)
        return {
            "value_mean": (
                float(values.detach().float().mean().cpu())
                if writer and values.numel()
                else None
            ),
            "dp_rank": self._dp_rank(),
            "dp_size": self._dp_size(),
            "rank": self.rank,
            "artifact_numeric_fingerprint": (
                _artifact_numeric_fingerprint(artifact, name=output_name)
                if writer
                else None
            ),
            **self._partial_artifact_counters(
                request.payload,
                actual_n_seqs=int(rollout["completion_ids"].shape[0]),
            ),
        }

    def _ppo_training_batches(self, request: SemanticRequest, rollout, ref, rewards, values):
        from runtime.ppo_math import build_ppo_targets, minibatch_indices

        _token_rewards, advantages, returns = build_ppo_targets(
            rollout["old_logps"],
            ref["ref_logps"],
            rewards["rewards"],
            values["values"],
            rollout["response_mask"],
            beta=float(request.payload.get("beta", 0.1)),
            gamma=float(request.payload.get("gamma", 1.0)),
            gae_lambda=float(request.payload.get("gae_lambda", 0.95)),
            normalize_advantages=bool(request.payload.get("normalize_advantages", False)),
        )
        local_batch = int(rollout["completion_ids"].shape[0])
        global_batch = int(request.payload.get("batch_size", local_batch * self._dp_size()))
        if local_batch * self._dp_size() != global_batch:
            raise RuntimeConfigError(
                f"PPO local rollout batch {local_batch} across DP={self._dp_size()} does not match global batch {global_batch}"
            )
        configured_minibatch = request.payload.get("minibatch_size")
        global_minibatch = global_batch if configured_minibatch is None else int(configured_minibatch)
        if global_minibatch % self._dp_size() != 0:
            raise RuntimeConfigError(
                f"PPO minibatch_size={global_minibatch} must be divisible by DP={self._dp_size()}"
            )
        local_minibatch = global_minibatch // self._dp_size()
        batches = tuple(
            minibatch_indices(
                local_batch,
                local_minibatch,
                epochs=int(request.payload.get("epochs", 1)),
                seed=int(request.payload.get("seed", 1234)) + request.step * 100_000 + self._dp_rank(),
            )
        )
        return advantages, returns, batches

    @staticmethod
    def _ppo_index_rollout(rollout, indices):
        return (
            _index_select_batch(rollout["completion_ids"], indices),
            [int(rollout["prompt_lens"][int(i)]) for i in indices.tolist()],
            _index_select_batch(rollout["old_logps"], indices),
            _index_select_batch(rollout["response_mask"], indices),
        )

    def _ppo_actor_train(self, request: SemanticRequest) -> dict[str, Any]:
        rollout = self._load_local_artifact(request.step, "rollout")
        ref = self._load_local_artifact(request.step, "ref_logps")
        rewards = self._load_local_artifact(request.step, "rewards")
        values = self._load_local_artifact(request.step, "values")
        advantages, _returns, batches = self._ppo_training_batches(request, rollout, ref, rewards, values)
        losses = []
        for _epoch, indices in batches:
            input_ids, prompt_lens, old_logps, response_mask = self._ppo_index_rollout(rollout, indices)
            losses.append(self.worker.ppo_actor_step(
                input_ids,
                prompt_lens,
                old_logps,
                _index_select_batch(advantages, indices),
                response_mask,
                max_new_tokens=int(rollout["generation_length"]),
                clip_range=float(request.payload.get("clip_range", 0.2)),
                entropy_coef=float(request.payload.get("entropy_coef", 0.0)),
                learning_rate=None
                if request.payload.get("learning_rate") is None
                else float(request.payload["learning_rate"]),
            ))
        loss = sum(losses) / len(losses)
        self._save_local_artifact(request.step, "actor_train", {"loss": loss})
        return {
            "loss": loss,
            "optimizer_updates": len(losses),
            "rank": self.rank,
            **self.worker.last_optimizer_counters(),
        }

    def _ppo_critic_train(self, request: SemanticRequest) -> dict[str, Any]:
        rollout = self._load_local_artifact(request.step, "rollout")
        ref = self._load_local_artifact(request.step, "ref_logps")
        rewards = self._load_local_artifact(request.step, "rewards")
        values = self._load_local_artifact(request.step, "values")
        _advantages, returns, batches = self._ppo_training_batches(request, rollout, ref, rewards, values)
        losses = []
        for _epoch, indices in batches:
            input_ids, prompt_lens, _old_logps, response_mask = self._ppo_index_rollout(rollout, indices)
            losses.append(self.worker.ppo_critic_step(
                input_ids,
                prompt_lens,
                _index_select_batch(returns, indices),
                response_mask,
                max_new_tokens=int(rollout["generation_length"]),
                learning_rate=None
                if request.payload.get("learning_rate") is None
                else float(request.payload["learning_rate"]),
            ))
        loss = sum(losses) / len(losses)
        self._save_local_artifact(request.step, "critic_train", {"loss": loss})
        return {
            "loss": loss,
            "optimizer_updates": len(losses),
            "rank": self.rank,
            **self.worker.last_optimizer_counters(),
        }

    def _ppo_fused_actor_critic_train(self, request: SemanticRequest) -> dict[str, Any]:
        from runtime.fused_pipeline import FusedPipelineSchedule

        fused_model = str(request.payload.get("fused_model") or "")
        if fused_model not in {"actor", "critic"}:
            raise RuntimeConfigError(
                f"fused PPO training requires fused_model actor or critic, got {fused_model!r}"
            )
        raw_schedule = request.payload.get("fused_pipeline_schedule")
        if not isinstance(raw_schedule, dict):
            raise RuntimeConfigError("fused PPO training requires fused_pipeline_schedule")
        schedule = FusedPipelineSchedule.from_dict(raw_schedule)
        expected_hash = str(request.payload.get("fused_pipeline_schedule_hash") or "")
        if expected_hash != schedule.schedule_hash:
            raise RuntimeConfigError(
                f"fused pipeline schedule hash mismatch: {expected_hash} != {schedule.schedule_hash}"
            )
        fusion_group = str(request.payload.get("fusion_group") or "")
        if not fusion_group:
            raise RuntimeConfigError("fused PPO training requires fusion_group")
        fusion_sync_root = str(request.payload.get("fusion_sync_root") or "")
        if not fusion_sync_root:
            raise RuntimeConfigError("fused PPO training requires fusion_sync_root")
        fusion_sync_backend = str(request.payload.get("fusion_sync_backend") or "cuda_ipc")

        rollout = self._load_local_artifact(request.step, "rollout")
        ref = self._load_local_artifact(request.step, "ref_logps")
        rewards = self._load_local_artifact(request.step, "rewards")
        values = self._load_local_artifact(request.step, "values")
        advantages, returns, batches = self._ppo_training_batches(
            request,
            rollout,
            ref,
            rewards,
            values,
        )
        losses = []
        timeout_s = float(request.payload.get("fusion_timeout_s", 900.0))
        attempt = int(request.payload.get("fusion_attempt", 0))
        for update_index, (_epoch, indices) in enumerate(batches):
            input_ids, prompt_lens, old_logps, response_mask = self._ppo_index_rollout(
                rollout,
                indices,
            )
            sync_token = (
                f"{fusion_group}-step{request.step}-attempt{attempt}-update{update_index}"
            )
            common = {
                "fused_pipeline_schedule": raw_schedule,
                "fused_model": fused_model,
                "fusion_sync_root": fusion_sync_root,
                "fusion_sync_token": sync_token,
                "fusion_sync_backend": fusion_sync_backend,
                "fusion_timeout_s": timeout_s,
                "learning_rate": None
                if request.payload.get("learning_rate") is None
                else float(request.payload["learning_rate"]),
            }
            if fused_model == "actor":
                losses.append(self.worker.ppo_actor_step(
                    input_ids,
                    prompt_lens,
                    old_logps,
                    _index_select_batch(advantages, indices),
                    response_mask,
                    max_new_tokens=int(rollout["generation_length"]),
                    clip_range=float(request.payload.get("clip_range", 0.2)),
                    entropy_coef=float(request.payload.get("entropy_coef", 0.0)),
                    **common,
                ))
            else:
                losses.append(self.worker.ppo_critic_step(
                    input_ids,
                    prompt_lens,
                    _index_select_batch(returns, indices),
                    response_mask,
                    max_new_tokens=int(rollout["generation_length"]),
                    **common,
                ))
        loss = sum(losses) / len(losses)
        self._save_local_artifact(request.step, f"{fused_model}_train", {"loss": loss})
        return {
            "loss": loss,
            "optimizer_updates": len(losses),
            "rank": self.rank,
            "intra_stage_fusion": True,
            "coordinated_intra_stage_1f1b": True,
            "fused_model": fused_model,
            "fused_pipeline_schedule_hash": schedule.schedule_hash,
            "fused_pipeline_tasks": len(schedule.tasks),
            "fused_pipeline_makespan_units": schedule.makespan_s,
            "fused_pipeline_optimizer": schedule.optimizer,
            "fused_pipeline_optimizer_candidates": schedule.optimizer_candidates,
            "fused_pipeline_baseline_makespan_units": schedule.baseline_makespan_s,
            **self.worker.last_optimizer_counters(),
            **self.worker.last_fusion_counters(),
        }

    def _ppo_ac_train(self, request: SemanticRequest) -> dict[str, Any]:
        rollout = self._load_local_artifact(request.step, "rollout")
        ref = self._load_local_artifact(request.step, "ref_logps")
        rewards = self._load_local_artifact(request.step, "rewards")
        values = self._load_local_artifact(request.step, "values")
        advantages, returns, batches = self._ppo_training_batches(request, rollout, ref, rewards, values)
        losses = []
        for _epoch, indices in batches:
            input_ids, prompt_lens, old_logps, response_mask = self._ppo_index_rollout(rollout, indices)
            losses.append(self.worker.ppo_ac_step(
                input_ids,
                prompt_lens,
                old_logps,
                _index_select_batch(advantages, indices),
                _index_select_batch(returns, indices),
                response_mask,
                max_new_tokens=int(rollout["generation_length"]),
                clip_range=float(request.payload.get("clip_range", 0.2)),
                entropy_coef=float(request.payload.get("entropy_coef", 0.0)),
                value_coef=float(request.payload.get("value_coef", 1.0)),
                learning_rate=None
                if request.payload.get("learning_rate") is None
                else float(request.payload["learning_rate"]),
            ))
        loss = sum(losses) / len(losses)
        self._save_local_artifact(request.step, "ac_train", {"loss": loss})
        return {
            "loss": loss,
            "optimizer_updates": len(losses),
            "rank": self.rank,
            **self.worker.last_optimizer_counters(),
        }

    def _grpo_actor_train(self, request: SemanticRequest) -> dict[str, Any]:
        rollout = self._load_local_artifact(request.step, "rollout")
        ref = self._load_local_artifact(request.step, "ref_logps")
        rewards = self._load_local_artifact(request.step, "rewards")
        loss = self.worker.grpo_actor_step(
            rollout["completion_ids"],
            rollout["prompt_lens"],
            rollout["old_logps"],
            ref["ref_logps"],
            rewards["rewards"],
            rollout["response_mask"],
            max_new_tokens=int(rollout["generation_length"]),
            samples_per_prompt=int(request.payload.get("samples_per_prompt", 1)),
            clip_range=float(request.payload.get("clip_range", 0.2)),
            kl_coef=float(request.payload.get("kl_coef", 0.0)),
            learning_rate=None
            if request.payload.get("learning_rate") is None
            else float(request.payload["learning_rate"]),
        )
        self._save_local_artifact(request.step, "actor_train", {"loss": loss})
        return {"loss": loss, "rank": self.rank, **self.worker.last_optimizer_counters()}

    def _dpo_logprobs(self, request: SemanticRequest, *, name: str) -> dict[str, Any]:
        input_ids, prompt_lens, rows = self._dpo_input_ids(request)
        sequence_logps = self.worker.sequence_response_logps(
            input_ids,
            prompt_lens,
            int(request.payload["generation_length"]),
            grad=False,
        ).view(-1, 2)
        writer = self._is_dp_artifact_writer()
        if writer:
            self._save_local_artifact(
                request.step,
                name,
                {"sequence_logps": sequence_logps.detach().cpu(), "rows": rows},
            )
        counters = {
            "batch_size": int(sequence_logps.shape[0]),
            "kind": name,
            "logp_mean": (
                float(sequence_logps.detach().float().mean().cpu())
                if writer and sequence_logps.numel()
                else None
            ),
            "rank": self.rank,
        }
        if name == "log_ref":
            counters.update(_trim_cuda_cache_after_frozen_inference())
        return counters

    def _dpo_actor_train(self, request: SemanticRequest) -> dict[str, Any]:
        input_ids, prompt_lens, _rows = self._dpo_input_ids(request)
        ref = self._load_local_artifact(request.step, "log_ref")
        loss = self.worker.dpo_actor_step(
            input_ids,
            prompt_lens,
            ref["sequence_logps"],
            max_new_tokens=int(request.payload["generation_length"]),
            beta=float(request.payload.get("beta", 0.1)),
            learning_rate=None
            if request.payload.get("learning_rate") is None
            else float(request.payload["learning_rate"]),
        )
        self._save_local_artifact(request.step, "actor_train", {"loss": loss})
        return {"loss": loss, "rank": self.rank, **self.worker.last_optimizer_counters()}

    def _dpo_input_ids(self, request: SemanticRequest):
        import torch

        payload = request.payload
        if not payload.get("preferences"):
            raise RuntimeConfigError("DPO semantic operation requires data.preferences")
        global_batch = int(payload.get("batch_size", 1))
        start, end = self._local_batch_range(global_batch)
        rows = self._preference_batch(
            str(payload["preferences"]),
            step=request.step,
            batch_size=global_batch,
            offset=int(payload.get("preference_offset", 0)),
            shuffle=bool(payload.get("data_shuffle", False)),
            seed=int(payload.get("data_seed", 0)),
            shard_index=int(payload.get("data_shard_index", 0)),
            num_shards=int(payload.get("data_num_shards", 1)),
        )[start:end]
        tokenizer = self._load_tokenizer(payload)
        max_length = int(payload.get("context_length", 0)) + int(payload.get("generation_length", 0))
        if max_length <= 0:
            raise RuntimeConfigError("DPO tokenization requires context_length + generation_length")
        texts: list[str] = []
        prompt_lens: list[int] = []
        for row in rows:
            prompt = str(row["prompt"])
            chosen = _concat_text(prompt, str(row["chosen"]))
            rejected = _concat_text(prompt, str(row["rejected"]))
            prompt_encoded = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_length)
            prompt_len = len(prompt_encoded["input_ids"])
            texts.extend([chosen, rejected])
            prompt_lens.extend([prompt_len, prompt_len])
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        return encoded["input_ids"].to(dtype=torch.long), prompt_lens, rows

    def _preference_batch(
        self,
        path: str,
        *,
        step: int,
        batch_size: int,
        offset: int = 0,
        shuffle: bool = False,
        seed: int = 0,
        shard_index: int = 0,
        num_shards: int = 1,
    ) -> list[dict[str, str]]:
        rows = self._load_preferences(path)
        if batch_size <= 0:
            raise RuntimeConfigError("batch_size must be positive for DPO preference ingestion")
        indices = _dataset_batch_indices(
            len(rows),
            step=step,
            batch_size=batch_size,
            offset=offset,
            shuffle=shuffle,
            seed=seed,
            shard_index=shard_index,
            num_shards=num_shards,
        )
        return [rows[idx] for idx in indices]

    def _load_preferences(self, path: str) -> list[dict[str, str]]:
        cache_key = f"preferences:{path}"
        cached = getattr(self, "_preference_cache", {}).get(cache_key) if hasattr(self, "_preference_cache") else None
        if cached is not None:
            return cached
        rows: list[dict[str, str]] = []
        pref_path = Path(path)
        with pref_path.open() as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                try:
                    rows.append({
                        "prompt": str(row["prompt"]),
                        "chosen": str(row["chosen"]),
                        "rejected": str(row["rejected"]),
                    })
                except KeyError as exc:
                    raise RuntimeConfigError(f"{pref_path}:{line_no} missing DPO field {exc.args[0]!r}") from exc
        if not rows:
            raise RuntimeConfigError(f"DPO preference file is empty: {pref_path}")
        if not hasattr(self, "_preference_cache"):
            self._preference_cache = {}
        self._preference_cache[cache_key] = rows
        return rows

    def shutdown(self) -> None:
        with self._background_transfer_lock:
            transfers = list(self._background_transfers)
        for transfer in transfers:
            transfer.join()
        self._raise_background_transfer_errors()
        try:
            _run_ucx_direct(_ucx_direct_close_cached_endpoints(), persistent=True)
        except Exception:
            pass
        _torch_direct_shutdown_cached_process_groups()
        if self._host_reshard_group is not None:
            try:
                import torch.distributed as dist

                if dist.is_initialized():
                    dist.destroy_process_group(self._host_reshard_group)
            except Exception:
                pass
            self._host_reshard_group = None
        self.worker.shutdown()


def _write_heartbeat(
    store: HeartbeatStore,
    args: argparse.Namespace,
    state: str,
    rank: int,
    world_size: int,
    local_rank: int,
    message: str | None = None,
) -> None:
    store.write(
        Heartbeat(
            worker_id=args.worker_id,
            run_id=args.run_id,
            state=state,
            timestamp=time.time(),
            pid=os.getpid(),
            rank=rank if world_size > 1 else None,
            world_size=world_size,
            local_rank=local_rank,
            message=message,
            metadata={
                "vertex_id": args.vertex_id,
                "backend": args.backend,
                "command_root": args.command_root,
                "state_root": args.state_root,
                "hostname": os.uname().nodename,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "semantic_control_transport": args.control_transport,
                "semantic_control_port": int(getattr(args, "semantic_control_port", 0)),
            },
        )
    )


def _backend(args: argparse.Namespace, rank: int, world_size: int) -> SemanticBackend:
    if args.backend == "megatron":
        return MegatronSemanticBackend(
            worker_id=args.worker_id,
            rank=rank,
            world_size=world_size,
            state_root=args.state_root,
        )
    raise RuntimeConfigError(f"unsupported semantic backend {args.backend!r}")


def _parse_bool_arg(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def _semantic_result_ready(transport: FileSemanticTransport, request: SemanticRequest, rank: int) -> bool:
    return transport.result_path(request.request_id, rank).exists()


def _load_unfinished_requests(
    transport: FileSemanticTransport,
    paths: list[Path],
    *,
    rank: int,
    last_request_id: str | None,
) -> list[tuple[Path, SemanticRequest]]:
    out: list[tuple[Path, SemanticRequest]] = []
    for path in paths:
        request = transport.load_request(path)
        if request.request_id == last_request_id:
            continue
        if _semantic_result_ready(transport, request, rank):
            continue
        out.append((path, request))
    return out


def _select_generation_batch(
    backend: SemanticBackend,
    pending: list[tuple[Path, SemanticRequest]],
    start_index: int,
    *,
    max_requests: int,
) -> list[tuple[Path, SemanticRequest]]:
    if max_requests <= 1 or not hasattr(backend, "generation_batch_key"):
        return [pending[start_index]]
    _path, first = pending[start_index]
    try:
        first_key = backend.generation_batch_key(first)  # type: ignore[attr-defined]
    except Exception:
        return [pending[start_index]]
    if first_key is None:
        return [pending[start_index]]
    selected = [pending[start_index]]
    for candidate in pending[start_index + 1:]:
        if len(selected) >= max_requests:
            break
        _candidate_path, request = candidate
        try:
            candidate_key = backend.generation_batch_key(request)  # type: ignore[attr-defined]
        except Exception:
            continue
        if candidate_key == first_key:
            selected.append(candidate)
    return selected


def _execute_request_or_batch(
    backend: SemanticBackend,
    requests: list[SemanticRequest],
) -> dict[str, SemanticResult]:
    if len(requests) > 1 and hasattr(backend, "execute_generation_batch"):
        return backend.execute_generation_batch(requests)  # type: ignore[attr-defined]
    request = requests[0]
    return {request.request_id: backend.execute(request)}


def _serve_socket_control(
    server: socket.socket,
    backend: SemanticBackend,
    heartbeat_store: HeartbeatStore,
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    local_rank: int,
    should_stop,
) -> None:
    connection: socket.socket | None = None
    server.settimeout(0.2)
    while not should_stop():
        if connection is None:
            try:
                connection, _peer = server.accept()
                connection.settimeout(0.2)
            except socket.timeout:
                continue
        try:
            envelope = recv_json_frame(connection)
        except socket.timeout:
            continue
        except (OSError, RuntimeExecutionError, json.JSONDecodeError):
            connection.close()
            connection = None
            continue
        if envelope is None:
            connection.close()
            connection = None
            continue
        if envelope.get("schema_version") != "rpeg.semantic_control.v1":
            raise RuntimeConfigError("unsupported semantic control envelope")
        requests = [SemanticRequest.from_dict(item) for item in envelope.get("requests") or []]
        if not requests:
            raise RuntimeConfigError("semantic control envelope contains no requests")
        results = _execute_request_or_batch(backend, requests)
        send_json_frame(
            connection,
            {
                "schema_version": "rpeg.semantic_control_result.v1",
                "results": [results[request.request_id].to_dict() for request in requests],
            },
        )
        failed = [result for result in results.values() if result.status != "ok"]
        if failed:
            message = "; ".join(result.message or result.status for result in failed)
            _write_heartbeat(
                heartbeat_store,
                args,
                "error",
                rank,
                world_size,
                local_rank,
                message,
            )
        if any(request.operation == SemanticOperationNames.SHUTDOWN for request in requests):
            break
    if connection is not None:
        connection.close()


def _serve_file_control(
    transport: FileSemanticTransport,
    backend: SemanticBackend,
    heartbeat_store: HeartbeatStore,
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    local_rank: int,
    should_stop,
) -> None:
    last_request_id: str | None = None
    while not should_stop():
        commands = sorted(transport.command_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
        pending = _load_unfinished_requests(transport, commands, rank=rank, last_request_id=last_request_id)
        consumed: set[str] = set()
        for index, (path, request) in enumerate(pending):
            if request.request_id in consumed:
                continue
            selected = [(path, request)]
            if args.dynamic_generation_batching:
                selected = _select_generation_batch(
                    backend,
                    pending,
                    index,
                    max_requests=max(1, int(args.dynamic_generation_batch_max_requests)),
                )
            batch_requests = [item_request for _item_path, item_request in selected]
            operation_label = request.operation
            if len(batch_requests) > 1:
                operation_label = f"{request.operation}:dynamic_batch={len(batch_requests)}"
            _write_heartbeat(heartbeat_store, args, "running", rank, world_size, local_rank, operation_label)
            results = _execute_request_or_batch(backend, batch_requests)
            for item_request in batch_requests:
                result = results[item_request.request_id]
                transport.write_result(result)
                consumed.add(item_request.request_id)
                last_request_id = item_request.request_id
            failed = [result for result in results.values() if result.status != "ok"]
            state = "ready" if not failed else "error"
            message = "; ".join(result.message or result.status for result in failed) if failed else None
            _write_heartbeat(heartbeat_store, args, state, rank, world_size, local_rank, message)
            if any(item_request.operation == SemanticOperationNames.SHUTDOWN for item_request in batch_requests):
                return
        time.sleep(args.poll_interval_s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Persistent RPEG semantic worker")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--vertex-id", required=True)
    parser.add_argument("--heartbeat-path", required=True)
    parser.add_argument("--command-root", required=True)
    parser.add_argument("--state-root", default=None)
    parser.add_argument("--backend", default="megatron")
    parser.add_argument("--control-transport", choices=("tcp", "file"), default="tcp")
    parser.add_argument("--control-port", type=int, default=0)
    parser.add_argument("--poll-interval-s", type=float, default=0.05)
    parser.add_argument(
        "--dynamic-generation-batching",
        type=_parse_bool_arg,
        default=_parse_bool_arg(os.getenv("RPEG_DYNAMIC_GENERATION_BATCHING", "1")),
    )
    parser.add_argument(
        "--dynamic-generation-batch-max-requests",
        type=int,
        default=int(os.getenv("RPEG_DYNAMIC_GENERATION_BATCH_MAX_REQUESTS", "8")),
    )
    args = parser.parse_args(argv)
    if args.state_root is None:
        args.state_root = args.command_root

    rank, world_size, local_rank = _rank_context()
    assert rank is not None and world_size is not None and local_rank is not None
    heartbeat_store = HeartbeatStore(Path(args.heartbeat_path).parent)
    transport = None
    control_server = None
    if args.control_transport == "file":
        transport = FileSemanticTransport(args.command_root, _PseudoWorker(args.worker_id, args.vertex_id, world_size))
        args.semantic_control_port = 0
    else:
        control_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        control_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        control_server.bind(("0.0.0.0", int(args.control_port)))
        control_server.listen(1)
        args.semantic_control_port = int(control_server.getsockname()[1])
    backend = _backend(args, rank, world_size)
    stopping = False

    def _stop(signum, _frame) -> None:
        nonlocal stopping
        stopping = True
        _write_heartbeat(heartbeat_store, args, "stopping", rank, world_size, local_rank, f"signal {signum}")

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    _write_heartbeat(heartbeat_store, args, "ready", rank, world_size, local_rank)

    try:
        if control_server is not None:
            _serve_socket_control(
                control_server,
                backend,
                heartbeat_store,
                args,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                should_stop=lambda: stopping,
            )
        else:
            assert transport is not None
            _serve_file_control(
                transport,
                backend,
                heartbeat_store,
                args,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                should_stop=lambda: stopping,
            )
    except (RuntimeConfigError, RuntimeDependencyError, RuntimeExecutionError, OSError, RuntimeError) as exc:
        _write_heartbeat(heartbeat_store, args, "error", rank, world_size, local_rank, str(exc))
        return 2
    finally:
        if control_server is not None:
            control_server.close()
        backend.shutdown()
    _write_heartbeat(heartbeat_store, args, "stopped", rank, world_size, local_rank)
    return 0


class _PseudoDevice:
    def __init__(self, rank: int) -> None:
        self.rank = rank


class _PseudoWorker:
    def __init__(self, worker_id: str, vertex_id: str, world_size: int) -> None:
        self.worker_id = worker_id
        self.vertex_id = vertex_id
        self.devices = tuple(_PseudoDevice(rank) for rank in range(world_size))


def _clone_optimizer_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            key: value.detach().clone() if hasattr(value, "detach") else value
            for key, value in entry.items()
        }
        for name, entry in state.items()
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_artifact_batch_size(data: dict[str, Any]) -> int | None:
    try:
        return _artifact_batch_size(data)
    except RuntimeConfigError:
        return None


def _materialize_trainable_artifact(value: Any, *, device: Any = None) -> Any:
    """
    Detach a semantic artifact and materialize it on one owned device.
    """
    try:
        import torch
    except ImportError:
        return value
    if isinstance(value, torch.Tensor):
        is_inference = getattr(value, "is_inference", None)
        target = value.device if device is None else torch.device(device)
        with torch.inference_mode(False):
            detached = value.detach().to(device=target, non_blocking=False)
            # Inference tensors retain their restricted identity through a device conversion.
            # Training consumers need an ordinary owned tensor, so clone at this semantic boundary.
            if callable(is_inference) and is_inference():
                detached = detached.clone()
            return detached
    if isinstance(value, dict):
        return {
            key: _materialize_trainable_artifact(item, device=device)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_materialize_trainable_artifact(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize_trainable_artifact(item, device=device) for item in value)
    return value


def _artifact_tensor_storage_bytes(
    artifacts: list[Any],
    *,
    device_type: str | None = None,
) -> int:
    """
    Count unique tensor storages conservatively across artifact aliases.
    """
    try:
        import torch
    except ImportError:
        return 0
    storages: dict[tuple[str, int, int], int] = {}

    def visit(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            if device_type is not None and value.device.type != device_type:
                return
            try:
                storage = value.untyped_storage()
                nbytes = int(storage.nbytes())
                pointer = int(storage.data_ptr())
            except (AttributeError, RuntimeError):
                nbytes = int(value.numel()) * int(value.element_size())
                pointer = int(value.data_ptr())
            # Device index is part of the key because CUDA virtual addresses are process-local and
            # may coincide across devices.
            key = (str(value.device), pointer, nbytes)
            storages[key] = max(storages.get(key, 0), nbytes)
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    for artifact in artifacts:
        visit(artifact)
    return sum(storages.values())


def _artifact_numeric_fingerprint(artifact: dict[str, Any], *, name: str) -> dict[str, Any]:
    """
    Return compact, DP-mergeable numeric moments for tensor artifact leaves.

    Serialized hashes prove byte integrity on one route, but equivalent tensors can have different
    torch.save encodings. These moments make cross-run and cross-layout comparisons independent of
    serialization while keeping all tensor data on the worker rank.
    """
    import torch

    tensors: dict[str, dict[str, Any]] = {}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, torch.Tensor):
            with torch.inference_mode(False):
                detached = value.detach().to(device="cpu")
                if detached.is_inference():
                    detached = detached.clone()
            if detached.is_complex():
                detached = torch.view_as_real(detached)
            numeric = detached.to(dtype=torch.float64).reshape(-1)
            finite_mask = torch.isfinite(numeric)
            finite = numeric[finite_mask]
            entry: dict[str, Any] = {
                "dtype": str(value.dtype),
                "shape": [int(size) for size in value.shape],
                "numel": int(numeric.numel()),
                "finite_count": int(finite.numel()),
                "nan_count": int(torch.isnan(numeric).sum().item()),
                "posinf_count": int(torch.isposinf(numeric).sum().item()),
                "neginf_count": int(torch.isneginf(numeric).sum().item()),
                "sum": float(finite.sum().item()),
                "sum_abs": float(finite.abs().sum().item()),
                "sum_squares": float((finite * finite).sum().item()),
                "min": float(finite.min().item()) if finite.numel() else None,
                "max": float(finite.max().item()) if finite.numel() else None,
            }
            tensors[path or "<root>"] = entry
            return
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                child = f"{path}.{key}" if path else str(key)
                visit(value[key], child)
            return
        if isinstance(value, (list, tuple)):
            for index, child_value in enumerate(value):
                child = f"{path}[{index}]" if path else f"[{index}]"
                visit(child_value, child)

    visit(artifact, "")
    return {
        "schema_version": "rpeg.artifact_numeric_fingerprint.v1",
        "artifact": str(name),
        "tensor_count": len(tensors),
        "tensors": tensors,
    }


def _metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None
    return int(value)


def _artifact_shard_summary(shard_metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dp_rank": int(metadata["dp_rank"]),
            "bytes": int(metadata["bytes"]),
            "sha256": str(metadata["sha256"]),
        }
        for metadata in shard_metadata
    ]


def _metadata_global_ranges(
    shard_metadata: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], int, int]] | None:
    ranges: list[tuple[dict[str, Any], int, int]] = []
    for metadata in sorted(shard_metadata, key=lambda item: int(item["dp_rank"])):
        start = _metadata_int(metadata, "global_start")
        end = _metadata_int(metadata, "global_end")
        if start is None or end is None:
            return None
        if end < start:
            raise RuntimeConfigError(
                f"invalid activation shard global range for dp_rank={metadata.get('dp_rank')}: {start}:{end}"
            )
        ranges.append((metadata, start, end))
    expected = 0
    for metadata, start, end in ranges:
        if start != expected:
            raise RuntimeConfigError(
                f"activation shard ranges are not contiguous at dp_rank={metadata.get('dp_rank')}: "
                f"start={start}, expected={expected}"
            )
        expected = end
    for metadata, _start, end in ranges:
        declared = _metadata_int(metadata, "global_batch_size")
        if declared is not None and declared != expected:
            raise RuntimeConfigError(
                f"activation shard global_batch_size mismatch at dp_rank={metadata.get('dp_rank')}: "
                f"{declared}, expected {expected}"
            )
        local = _metadata_int(metadata, "local_batch_size")
        start = _metadata_int(metadata, "global_start")
        if local is not None and start is not None and start + local != end:
            raise RuntimeConfigError(
                f"activation shard local_batch_size mismatch at dp_rank={metadata.get('dp_rank')}"
            )
    return ranges


def _batch_range_for_dp(global_batch_size: int, dp_size: int, dp_rank: int) -> tuple[int, int]:
    if dp_size <= 0:
        raise RuntimeConfigError(f"invalid data parallel size {dp_size}")
    if dp_rank < 0 or dp_rank >= dp_size:
        raise RuntimeConfigError(f"destination dp_rank {dp_rank} outside dp_size={dp_size}")
    if global_batch_size % dp_size != 0:
        raise RuntimeConfigError(f"global batch {global_batch_size} must be divisible by data parallel size {dp_size}")
    per_dp = global_batch_size // dp_size
    start = dp_rank * per_dp
    return start, start + per_dp


def _direct_repartition_destinations(
    *,
    source_start: int,
    source_end: int,
    global_batch_size: int,
    destination_dp_size: int,
) -> list[tuple[int, int, int]]:
    destinations = []
    for destination_dp_rank in range(destination_dp_size):
        destination_start, destination_end = _batch_range_for_dp(
            global_batch_size,
            destination_dp_size,
            destination_dp_rank,
        )
        overlap_start = max(source_start, destination_start)
        overlap_end = min(source_end, destination_end)
        if overlap_start < overlap_end:
            destinations.append((destination_dp_rank, overlap_start, overlap_end))
    return destinations


def _direct_repartition_sources_for_destination(
    ranges: list[tuple[dict[str, Any], int, int]],
    *,
    destination_start: int,
    destination_end: int,
) -> list[tuple[int, int, int]]:
    sources = []
    for metadata, source_start, source_end in ranges:
        overlap_start = max(source_start, destination_start)
        overlap_end = min(source_end, destination_end)
        if overlap_start < overlap_end:
            sources.append((int(metadata["dp_rank"]), overlap_start, overlap_end))
    return sources


def _validate_artifact_shard_metadata(
    metadata: dict[str, Any],
    *,
    path: Path,
    worker_id: str,
    step: int,
    name: str,
    dp_rank: int,
    dp_size: int,
) -> None:
    if metadata.get("schema_version") != "rpeg.activation_shard_meta.v1":
        raise RuntimeConfigError(f"unsupported activation shard metadata schema in {path}")
    expected = {
        "worker_id": worker_id,
        "step": int(step),
        "name": name,
        "dp_rank": int(dp_rank),
        "dp_size": int(dp_size),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeConfigError(
                f"activation shard metadata mismatch for {path}: {key}={metadata.get(key)!r}, expected {value!r}"
            )
    actual_bytes = path.stat().st_size
    if int(metadata.get("bytes", -1)) != actual_bytes:
        raise RuntimeConfigError(
            f"activation shard byte count mismatch for {path}: "
            f"{metadata.get('bytes')!r}, expected {actual_bytes}"
        )
    actual_sha256 = _file_sha256(path)
    if str(metadata.get("sha256")) != actual_sha256:
        raise RuntimeConfigError(f"activation shard checksum mismatch for {path}")


def _global_layer_name(name: str, layer_map: dict[int, int]) -> str:
    prefix = "decoder.layers."
    if not name.startswith(prefix):
        return name
    rest = name[len(prefix):]
    local_raw, sep, suffix = rest.partition(".")
    if not sep:
        return name
    try:
        local_idx = int(local_raw)
    except ValueError:
        return name
    global_idx = layer_map.get(local_idx, local_idx)
    return f"{prefix}{global_idx}.{suffix}"


def _concat_text(prompt: str, response: str) -> str:
    if not prompt:
        return response
    if prompt.endswith((" ", "\n", "\t")) or response.startswith((" ", "\n", "\t")):
        return prompt + response
    return prompt + " " + response


def _merge_artifact_shards(shards: list[dict[str, Any]]) -> dict[str, Any]:
    if not shards:
        raise RuntimeConfigError("cannot merge empty activation shard list")
    keys = tuple(key for key in shards[0] if key != "_rpeg_shard")
    key_set = set(keys)
    if any((set(shard) - {"_rpeg_shard"}) != key_set for shard in shards[1:]):
        raise RuntimeConfigError("activation shard keys differ across DP ranks")
    return {key: _merge_artifact_values([shard[key] for shard in shards]) for key in keys}


def _sort_artifact_by_original_index(artifact: dict[str, Any]) -> dict[str, Any]:
    """
    Restore prompt-major order after completion-phase/DP concatenation.
    """
    import torch

    indices = artifact.get("original_indices")
    if indices is None:
        return artifact
    index_list = (
        [int(value) for value in indices.detach().cpu().tolist()]
        if isinstance(indices, torch.Tensor)
        else [int(value) for value in indices]
    )
    if len(index_list) != len(set(index_list)):
        raise RuntimeConfigError("activation artifact contains duplicate original_indices")
    order = sorted(range(len(index_list)), key=index_list.__getitem__)
    batch = len(order)
    sorted_artifact: dict[str, Any] = {}
    order_tensor = torch.tensor(order, dtype=torch.long)
    for key, value in artifact.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == batch:
            sorted_artifact[key] = value.index_select(
                0, order_tensor.to(device=value.device)
            ).contiguous()
        elif isinstance(value, list) and len(value) == batch:
            sorted_artifact[key] = [value[idx] for idx in order]
        elif isinstance(value, tuple) and len(value) == batch:
            sorted_artifact[key] = tuple(value[idx] for idx in order)
        else:
            sorted_artifact[key] = value
    return sorted_artifact


def _merge_artifact_values(values: list[Any]) -> Any:
    import torch

    first = values[0]
    if isinstance(first, torch.Tensor):
        if first.ndim == 0:
            if any(not torch.equal(first, value) for value in values[1:]):
                return torch.stack(tuple(value.reshape(()) for value in values)).max()
            return first
        values = _pad_tensors_for_batch_cat(values)
        return torch.cat(tuple(values), dim=0)
    if isinstance(first, list):
        out = []
        for value in values:
            out.extend(value)
        return out
    if isinstance(first, tuple):
        out = []
        for value in values:
            out.extend(list(value))
        return tuple(out)
    if isinstance(first, dict):
        keys = tuple(first)
        key_set = set(keys)
        if any(set(value) != key_set for value in values[1:]):
            raise RuntimeConfigError("nested activation metadata keys differ across DP shards")
        return {key: _merge_artifact_values([value[key] for value in values]) for key in keys}
    if any(value != first for value in values[1:]):
        if all(isinstance(value, (int, float, bool)) for value in values):
            return max(values)
        raise RuntimeConfigError("non-batch activation metadata differs across DP shards")
    return first


def _pad_tensors_for_batch_cat(values: list[Any]) -> list[Any]:
    import torch

    if not values or not isinstance(values[0], torch.Tensor) or values[0].ndim <= 1:
        return values
    max_shape = list(values[0].shape)
    for value in values[1:]:
        for dim in range(1, value.ndim):
            max_shape[dim] = max(max_shape[dim], int(value.shape[dim]))
    padded = []
    for value in values:
        pad = []
        needs_pad = False
        for dim in range(value.ndim - 1, 0, -1):
            delta = max_shape[dim] - int(value.shape[dim])
            pad.extend([0, delta])
            needs_pad = needs_pad or delta > 0
        padded.append(torch.nn.functional.pad(value, pad) if needs_pad else value)
    return padded


def _artifact_batch_size(data: dict[str, Any]) -> int:
    import torch

    saw_empty = False
    for value in data.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            size = int(value.shape[0])
            if size > 0:
                return size
            saw_empty = True
        if isinstance(value, (list, tuple)):
            size = len(value)
            if size > 0:
                return size
            saw_empty = True
    if saw_empty:
        return 0
    raise RuntimeConfigError("activation artifact has no batch dimension to repartition")


def _index_select_batch(value: Any, indices: Any) -> Any:
    """
    Index a tensor with deterministic CPU minibatch indices on its device.

    PPO permutations deliberately originate from a CPU generator, so they are reproducible across
    ranks. Semantic activation caching can keep the selected tensors on CUDA, where PyTorch requires
    the index tensor to be on the same device. Moving only the tiny index vector preserves both
    determinism and the zero-host-copy activation path.
    """
    return value.index_select(0, indices.to(device=value.device))


def _slice_artifact_batch(data: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    batch = _artifact_batch_size(data)
    return {key: _slice_artifact_value(value, start, end, batch) for key, value in data.items()}


def _artifact_original_indices(data: dict[str, Any]) -> list[int] | None:
    import torch

    raw = data.get("original_indices")
    if raw is None:
        return None
    indices = (
        [int(value) for value in raw.detach().cpu().tolist()]
        if isinstance(raw, torch.Tensor)
        else [int(value) for value in raw]
    )
    if len(indices) != _artifact_batch_size(data):
        raise RuntimeConfigError(
            "activation original_indices length does not match artifact batch"
        )
    if len(indices) != len(set(indices)):
        raise RuntimeConfigError("activation artifact contains duplicate original_indices")
    return indices


def _validate_global_index_artifact(
    data: dict[str, Any],
    *,
    global_batch_size: int,
    expected_start: int | None = None,
    expected_end: int | None = None,
) -> list[int]:
    indices = _artifact_original_indices(data)
    if indices is None:
        raise RuntimeConfigError(
            "global-index activation repartition requires original_indices"
        )
    if any(index < 0 or index >= int(global_batch_size) for index in indices):
        raise RuntimeConfigError(
            f"activation original_indices fall outside global batch {global_batch_size}"
        )
    if expected_start is not None and expected_end is not None and any(
        index < int(expected_start) or index >= int(expected_end)
        for index in indices
    ):
        raise RuntimeConfigError(
            "activation original_indices fall outside the assigned DP range "
            f"{expected_start}:{expected_end}"
        )
    return indices


def _slice_artifact_global_index_range(
    data: dict[str, Any],
    start: int,
    end: int,
    *,
    global_batch_size: int,
) -> dict[str, Any]:
    import torch

    indices = _validate_global_index_artifact(
        data,
        global_batch_size=global_batch_size,
    )
    positions = [
        position
        for position, index in enumerate(indices)
        if int(start) <= index < int(end)
    ]
    batch = len(indices)
    index_tensor = torch.tensor(positions, dtype=torch.long)

    def select(value: Any) -> Any:
        if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == batch:
            return value.index_select(
                0, index_tensor.to(device=value.device)
            ).contiguous()
        if isinstance(value, list) and len(value) == batch:
            return [value[position] for position in positions]
        if isinstance(value, tuple) and len(value) == batch:
            return tuple(value[position] for position in positions)
        if isinstance(value, dict):
            return {key: select(item) for key, item in value.items()}
        return value

    return {key: select(value) for key, value in data.items()}


def _slice_artifact_value(value: Any, start: int, end: int, batch: int) -> Any:
    import torch

    if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == batch:
        return value[start:end].contiguous()
    if isinstance(value, list) and len(value) == batch:
        return value[start:end]
    if isinstance(value, tuple) and len(value) == batch:
        return tuple(value[start:end])
    if isinstance(value, dict):
        return {key: _slice_artifact_value(item, start, end, batch) for key, item in value.items()}
    return value


def _fast_clone_or_copy_file(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        pass
    try:
        _reflink_file(source, destination)
        return "reflink"
    except OSError:
        if destination.exists():
            destination.unlink()
    shutil.copyfile(source, destination)
    return "copy"


def _normalize_file_transport(value: Any) -> str:
    normalized = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "": "auto",
        "auto": "auto",
        "shared_filesystem": "auto",
        "clone": "auto",
        "hardlink": "auto",
        "reflink": "auto",
        "kernel": "kernel",
        "kernel_copy": "kernel",
        "copy_file_range": "kernel",
        "sendfile": "kernel",
        "stream": "stream",
        "streaming": "stream",
        "copy": "copy",
        "byte_copy": "copy",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise RuntimeConfigError("activation_transport must be one of: auto, kernel, stream, copy") from exc


def _copy_file_atomic(
    source: Path,
    destination: Path,
    *,
    mode: str = "auto",
    source_metadata: dict[str, Any] | None = None,
    tmp_suffix: str | None = None,
) -> dict[str, Any]:
    mode = _normalize_file_transport(mode)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = tmp_suffix or f".{os.getpid()}.tmp"
    tmp = destination.with_suffix(destination.suffix + suffix)
    if tmp.exists():
        tmp.unlink()
    started = time.perf_counter()
    try:
        if mode == "stream":
            bytes_, sha256 = _stream_copy_file_with_sha256(source, tmp)
            copy_mode = "stream"
        elif mode == "kernel":
            try:
                bytes_ = _kernel_copy_file(source, tmp)
                sha256 = _file_sha256(tmp)
                copy_mode = "kernel"
            except OSError:
                if tmp.exists():
                    tmp.unlink()
                bytes_, sha256 = _stream_copy_file_with_sha256(source, tmp)
                copy_mode = "stream"
        elif mode == "copy":
            shutil.copyfile(source, tmp)
            bytes_ = tmp.stat().st_size
            sha256 = _file_sha256(tmp)
            copy_mode = "copy"
        else:
            copy_mode = _fast_clone_or_copy_file(source, tmp)
            bytes_ = tmp.stat().st_size
            sha256 = _cloned_file_sha256(copy_mode, source_metadata or {}, tmp)
        tmp.replace(destination)
    finally:
        if tmp.exists():
            tmp.unlink()
    source_sha256 = str((source_metadata or {}).get("sha256") or "")
    if len(source_sha256) != 64:
        if copy_mode in {"kernel", "stream"}:
            source_sha256 = sha256
        else:
            source_sha256 = _file_sha256(source)
    elif copy_mode in {"kernel", "stream", "copy"} and sha256 != source_sha256:
        raise RuntimeConfigError(
            f"copied file checksum mismatch for {destination}: destination sha256={sha256}, source metadata sha256={source_sha256}"
        )
    duration_s = time.perf_counter() - started
    return {
        "bytes": int(bytes_),
        "sha256": sha256,
        "source_sha256": source_sha256,
        "copy_mode": copy_mode,
        "transport": f"shared_filesystem_{copy_mode}",
        "transfer_rank_duration_s": duration_s,
        "transfer_rank_bytes_per_s": _bytes_per_second(bytes_, duration_s),
    }


def _stream_copy_file_with_sha256(source: Path, destination: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with source.open("rb") as src, destination.open("wb") as dst:
        for chunk in iter(lambda: src.read(16 * 1024 * 1024), b""):
            total += len(chunk)
            digest.update(chunk)
            dst.write(chunk)
    return total, digest.hexdigest()


def _direct_endpoint_for_rank(payload: dict[str, Any], rank: int) -> dict[str, Any]:
    endpoints = _direct_endpoints_for_rank(payload, rank)
    if endpoints:
        return endpoints[0]
    raise RuntimeConfigError(f"missing direct endpoint for rank {rank}")


def _direct_endpoints_for_rank(payload: dict[str, Any], rank: int) -> list[dict[str, Any]]:
    endpoints = payload.get("direct_endpoints")
    if not isinstance(endpoints, (list, tuple)):
        raise RuntimeConfigError("direct activation transfer requires direct_endpoints")
    matched: list[dict[str, Any]] = []
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        if int(endpoint.get("rank", -1)) == int(rank):
            if endpoint.get("host") is None or endpoint.get("port") is None:
                raise RuntimeConfigError(f"direct endpoint for rank {rank} is missing host or port")
            matched.append(endpoint)
    return matched


def _direct_endpoint_for_source_rank(payload: dict[str, Any], rank: int) -> dict[str, Any]:
    endpoints = _direct_endpoints_for_source_rank(payload, rank)
    if endpoints:
        return endpoints[0]
    raise RuntimeConfigError(f"missing direct endpoint for source rank {rank}")


def _direct_endpoints_for_source_rank(payload: dict[str, Any], rank: int) -> list[dict[str, Any]]:
    endpoints = payload.get("direct_endpoints")
    if not isinstance(endpoints, (list, tuple)):
        raise RuntimeConfigError("direct activation transfer requires direct_endpoints")
    saw_source_rank = False
    matched: list[dict[str, Any]] = []
    for endpoint in endpoints:
        if not isinstance(endpoint, dict) or "source_rank" not in endpoint:
            continue
        saw_source_rank = True
        if int(endpoint.get("source_rank", -1)) == int(rank):
            if endpoint.get("host") is None or endpoint.get("port") is None:
                raise RuntimeConfigError(f"direct endpoint for source rank {rank} is missing host or port")
            matched.append(endpoint)
    if matched:
        return matched
    if saw_source_rank:
        return []
    return _direct_endpoints_for_rank(payload, rank)


def _direct_endpoint_for_dp_rank(payload: dict[str, Any], dp_rank: int) -> dict[str, Any]:
    endpoints = payload.get("direct_endpoints")
    if not isinstance(endpoints, (list, tuple)):
        raise RuntimeConfigError("direct activation transfer requires direct_endpoints")
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        if int(endpoint.get("dp_rank", -1)) == int(dp_rank):
            if endpoint.get("host") is None or endpoint.get("port") is None:
                raise RuntimeConfigError(f"direct endpoint for dp_rank {dp_rank} is missing host or port")
            return endpoint
    raise RuntimeConfigError(f"missing direct endpoint for destination dp_rank {dp_rank}")


def _direct_endpoint_for_dp_pair(payload: dict[str, Any], dp_rank: int, source_dp_rank: int) -> dict[str, Any]:
    endpoints = payload.get("direct_endpoints")
    if not isinstance(endpoints, (list, tuple)):
        raise RuntimeConfigError("direct activation transfer requires direct_endpoints")
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        if (
            int(endpoint.get("dp_rank", -1)) == int(dp_rank)
            and int(endpoint.get("source_dp_rank", -1)) == int(source_dp_rank)
        ):
            if endpoint.get("host") is None or endpoint.get("port") is None:
                raise RuntimeConfigError(
                    f"direct endpoint for dp_rank {dp_rank}, source_dp_rank {source_dp_rank} is missing host or port"
                )
            return endpoint
    return _direct_endpoint_for_dp_rank(payload, dp_rank)


def _direct_has_all_dp_pair_endpoints(
    payload: dict[str, Any],
    *,
    destination_dp_rank: int,
    source_dp_ranks: list[int],
) -> bool:
    endpoints = payload.get("direct_endpoints")
    if not isinstance(endpoints, (list, tuple)):
        return False
    pairs = {
        int(endpoint.get("source_dp_rank", -1))
        for endpoint in endpoints
        if isinstance(endpoint, dict) and int(endpoint.get("dp_rank", -1)) == int(destination_dp_rank)
    }
    return all(int(source_dp_rank) in pairs for source_dp_rank in source_dp_ranks)


def _direct_transport_kind(payload: dict[str, Any]) -> str:
    kind = str(payload.get("direct_transport", "tcp")).strip().lower().replace("-", "_")
    if kind in {"tcp", "tcp_direct", "socket"}:
        return "tcp"
    if kind in {"torch", "torch_dist", "torch_direct", "gloo", "nccl"}:
        return "torch_dist"
    if kind in {"ucx", "ucx_direct", "rdma", "rdma_direct"}:
        return "ucx"
    raise RuntimeConfigError(f"unsupported direct activation transport {kind!r}")


def _run_ucx_direct(awaitable, *, persistent: bool = False):
    del persistent
    loop = _ucx_direct_event_loop()
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if running_loop is loop:
        raise RuntimeConfigError("synchronous ucx_direct dispatch cannot run on its transport event loop")
    return asyncio.run_coroutine_threadsafe(awaitable, loop).result()


def _ucx_direct_event_loop() -> asyncio.AbstractEventLoop:
    global _UCX_DIRECT_EVENT_LOOP

    with _UCX_DIRECT_EVENT_LOOP_LOCK:
        if _UCX_DIRECT_EVENT_LOOP is not None and not _UCX_DIRECT_EVENT_LOOP.is_closed():
            return _UCX_DIRECT_EVENT_LOOP
        started = threading.Event()

        def run_loop() -> None:
            global _UCX_DIRECT_EVENT_LOOP

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _UCX_DIRECT_LOOP_LOCAL.loop = loop
            _UCX_DIRECT_EVENT_LOOP = loop
            started.set()
            loop.run_forever()

        thread = threading.Thread(target=run_loop, name="rpeg-ucx-event-loop", daemon=True)
        thread.start()
        if not started.wait(timeout=10.0):
            raise RuntimeDependencyError("timed out starting the ucx_direct transport event loop")
        assert _UCX_DIRECT_EVENT_LOOP is not None
        return _UCX_DIRECT_EVENT_LOOP


def _ucx_direct_channel_reuse(endpoint: dict[str, Any]) -> bool:
    return bool(endpoint.get("direct_channel_reuse", False)) and bool(endpoint.get("direct_channel_id"))


def _ucx_direct_cache_key(role: str, endpoint: dict[str, Any]) -> tuple[int, str, str] | None:
    if not _ucx_direct_channel_reuse(endpoint):
        return None
    loop = getattr(_UCX_DIRECT_LOOP_LOCAL, "loop", None)
    if loop is None:
        return None
    return (id(loop), str(role), str(endpoint["direct_channel_id"]))


def _ucx_direct_cached_endpoint(role: str, endpoint: dict[str, Any]) -> Any | None:
    key = _ucx_direct_cache_key(role, endpoint)
    if key is None:
        return None
    return _UCX_DIRECT_ENDPOINT_CACHE.get(key)


def _ucx_direct_store_cached_endpoint(role: str, endpoint: dict[str, Any], ep: Any) -> bool:
    key = _ucx_direct_cache_key(role, endpoint)
    if key is None:
        return False
    old = _UCX_DIRECT_ENDPOINT_CACHE.get(key)
    if old is not None and old is not ep:
        asyncio.create_task(_ucx_close(old))
    _UCX_DIRECT_ENDPOINT_CACHE[key] = ep
    return True


def _ucx_direct_discard_cached_endpoint(role: str, endpoint: dict[str, Any]) -> None:
    key = _ucx_direct_cache_key(role, endpoint)
    if key is None:
        return
    ep = _UCX_DIRECT_ENDPOINT_CACHE.pop(key, None)
    if ep is not None:
        asyncio.create_task(_ucx_close(ep))


async def _ucx_direct_close_cached_endpoints() -> None:
    loop = getattr(_UCX_DIRECT_LOOP_LOCAL, "loop", None)
    loop_id = None if loop is None else id(loop)
    endpoints = [
        _UCX_DIRECT_ENDPOINT_CACHE.pop(key)
        for key in list(_UCX_DIRECT_ENDPOINT_CACHE)
        if loop_id is None or key[0] == loop_id
    ]
    for endpoint in endpoints:
        await _ucx_close(endpoint)


def _ucx_direct_module():
    try:
        import ucxx

        return ucxx
    except ModuleNotFoundError:
        pass
    try:
        import ucp

        return ucp
    except ModuleNotFoundError as exc:
        raise RuntimeDependencyError(
            "ucx_direct activation transfer requires the ucxx or ucp Python bindings"
        ) from exc


async def _ucx_direct_send_file(
    *,
    endpoint: dict[str, Any],
    path: Path,
    bytes_: int,
    sha256: str,
    timeout_s: float,
    ready_path: Path | None = None,
    ready_wait_s: float = 0.0,
    connect_grace_s: float = 0.0,
) -> bool:
    ucx = _ucx_direct_module()
    _ucx_direct_init_once(ucx)
    await _wait_direct_ready_marker_async(ready_path, timeout_s=ready_wait_s)
    ep = _ucx_direct_cached_endpoint("send", endpoint)
    cached = ep is not None
    if ep is None:
        ep = await _ucx_connect_with_retry(
            ucx,
            str(endpoint["host"]),
            int(endpoint["port"]),
            timeout_s,
            initial_delay_s=connect_grace_s,
        )
    try:
        header = struct.pack("!Q", int(bytes_)) + bytes.fromhex(sha256)
        await _ucx_send(ep, header)
        remaining = int(bytes_)
        with path.open("rb") as src:
            while remaining > 0:
                chunk = src.read(min(16 * 1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeConfigError(f"ucx_direct activation send ended early for {path}")
                await _ucx_send(ep, chunk)
                remaining -= len(chunk)
        return cached
    except Exception:
        if cached:
            _ucx_direct_discard_cached_endpoint("send", endpoint)
        raise
    finally:
        if not cached:
            await _ucx_close(ep)


async def _ucx_direct_send_blob(
    *,
    endpoint: dict[str, Any],
    payload: bytes,
    sha256: str,
    timeout_s: float,
    ready_path: Path | None = None,
    ready_wait_s: float = 0.0,
    connect_grace_s: float = 0.0,
) -> bool:
    ucx = _ucx_direct_module()
    _ucx_direct_init_once(ucx)
    await _wait_direct_ready_marker_async(ready_path, timeout_s=ready_wait_s)
    ep = _ucx_direct_cached_endpoint("send", endpoint)
    cached = ep is not None
    if ep is None:
        ep = await _ucx_connect_with_retry(
            ucx,
            str(endpoint["host"]),
            int(endpoint["port"]),
            timeout_s,
            initial_delay_s=connect_grace_s,
        )
    stored = False
    try:
        await _ucx_send(ep, struct.pack("!Q", len(payload)) + bytes.fromhex(sha256))
        view = memoryview(payload)
        for offset in range(0, len(payload), 16 * 1024 * 1024):
            await _ucx_send(ep, view[offset : offset + 16 * 1024 * 1024])
        if not cached:
            stored = _ucx_direct_store_cached_endpoint("send", endpoint, ep)
        return cached
    except Exception:
        if cached:
            _ucx_direct_discard_cached_endpoint("send", endpoint)
        raise
    finally:
        if not cached and not stored:
            await _ucx_close(ep)


def _tensor_stream_manifest(
    record: dict[str, Any],
    *,
    schema_version: str = _UCX_TENSOR_STREAM_SCHEMA,
) -> tuple[dict[str, Any], int]:
    fragments = record.get("fragments")
    if not isinstance(fragments, dict):
        raise RuntimeConfigError("tensor-native weight record must contain a fragment mapping")
    metadata = {key: value for key, value in record.items() if key != "fragments"}
    entries = []
    tensor_bytes = 0
    for name in sorted(fragments):
        contribution = fragments[name]
        if not isinstance(contribution, dict) or "value" not in contribution:
            raise RuntimeConfigError(f"invalid routed tensor contribution {name!r}")
        value = contribution["value"]
        item = {
            "name": str(name),
            "mode": str(contribution["mode"]),
            "order": int(contribution["order"]),
            "stride": int(contribution.get("stride", 1)),
            "dim": None if contribution.get("dim") is None else int(contribution["dim"]),
        }
        if value is None:
            item["kind"] = "none"
        elif all(hasattr(value, attr) for attr in ("dtype", "shape", "numel", "element_size")):
            if not value.is_contiguous():
                contribution["value"] = value = value.contiguous()
            nbytes = int(value.numel()) * int(value.element_size())
            item.update({
                "kind": "tensor",
                "dtype": str(value.dtype).removeprefix("torch."),
                "shape": [int(dim) for dim in value.shape],
                "nbytes": nbytes,
            })
            tensor_bytes += nbytes
        else:
            raise RuntimeConfigError(
                f"tensor-native weight transfer does not support {name!r} value type "
                f"{type(value).__name__}"
            )
        entries.append(item)
    return {
        "schema_version": schema_version,
        "record": metadata,
        "entries": entries,
        "bucket_bytes": _ucx_tensor_bucket_bytes(),
    }, tensor_bytes


def _activation_artifact_to_tensor_record(artifact: dict[str, Any]) -> dict[str, Any]:
    """
    Encode one structured activation as JSON metadata plus tensor leaves.

    The tree is deliberately explicit instead of relying on pickle. It preserves mapping order,
    lists, tuples, scalar types, and tensor alias independence, while keeping every tensor available
    to NCCL/UCX as a typed contiguous buffer.
    """
    import torch

    if not isinstance(artifact, dict):
        raise RuntimeConfigError("activation artifact must be a mapping")
    fragments: dict[str, dict[str, Any]] = {}

    def encode(value: Any) -> dict[str, Any]:
        if isinstance(value, torch.Tensor):
            name = f"tensor_{len(fragments):08d}"
            fragments[name] = {
                "value": value.detach(),
                "mode": "complete",
                "order": 0,
                "stride": 1,
                "dim": None,
            }
            return {"kind": "tensor", "name": name}
        if value is None:
            return {"kind": "none"}
        if isinstance(value, bool):
            return {"kind": "bool", "value": value}
        if isinstance(value, int):
            return {"kind": "int", "value": value}
        if isinstance(value, float):
            # hex preserves signed zero and gives explicit spellings for infinities/NaNs without
            # relying on non-standard JSON numbers.
            return {"kind": "float", "hex": value.hex()}
        if isinstance(value, str):
            return {"kind": "str", "value": value}
        if isinstance(value, dict):
            items = []
            for key, item in value.items():
                if not isinstance(key, str):
                    raise RuntimeConfigError(
                        "activation tensor records require string mapping keys, "
                        f"got {type(key).__name__}"
                    )
                items.append([key, encode(item)])
            return {"kind": "dict", "items": items}
        if isinstance(value, list):
            return {"kind": "list", "items": [encode(item) for item in value]}
        if isinstance(value, tuple):
            return {"kind": "tuple", "items": [encode(item) for item in value]}
        raise RuntimeConfigError(
            "activation tensor records do not support value type "
            f"{type(value).__name__}"
        )

    return {
        "schema_version": _ACTIVATION_TENSOR_RECORD_SCHEMA,
        "tree": encode(artifact),
        "fragments": fragments,
    }


def _activation_artifact_from_tensor_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Decode and strictly validate a structured activation tensor record.
    """
    import torch

    if record.get("schema_version") != _ACTIVATION_TENSOR_RECORD_SCHEMA:
        raise RuntimeConfigError(
            "unsupported activation tensor record schema: "
            f"{record.get('schema_version')!r}"
        )
    fragments = record.get("fragments")
    if not isinstance(fragments, dict):
        raise RuntimeConfigError("activation tensor record has no fragment mapping")
    used: set[str] = set()

    def decode(node: Any) -> Any:
        if not isinstance(node, dict) or not isinstance(node.get("kind"), str):
            raise RuntimeConfigError("malformed activation tensor tree node")
        kind = node["kind"]
        if kind == "tensor":
            name = node.get("name")
            if not isinstance(name, str) or name in used or name not in fragments:
                raise RuntimeConfigError(
                    f"invalid or duplicate activation tensor leaf {name!r}"
                )
            contribution = fragments[name]
            if not isinstance(contribution, dict):
                raise RuntimeConfigError(f"malformed activation tensor fragment {name!r}")
            value = contribution.get("value")
            if not isinstance(value, torch.Tensor):
                raise RuntimeConfigError(f"activation tensor fragment {name!r} is not a tensor")
            used.add(name)
            return value
        if kind == "none":
            return None
        if kind == "bool" and isinstance(node.get("value"), bool):
            return node["value"]
        if kind == "int" and isinstance(node.get("value"), int) and not isinstance(node.get("value"), bool):
            return node["value"]
        if kind == "float" and isinstance(node.get("hex"), str):
            try:
                return float.fromhex(node["hex"])
            except ValueError as exc:
                raise RuntimeConfigError("invalid activation float encoding") from exc
        if kind == "str" and isinstance(node.get("value"), str):
            return node["value"]
        if kind in {"list", "tuple"}:
            items = node.get("items")
            if not isinstance(items, list):
                raise RuntimeConfigError(f"malformed activation {kind} node")
            decoded = [decode(item) for item in items]
            return decoded if kind == "list" else tuple(decoded)
        if kind == "dict":
            items = node.get("items")
            if not isinstance(items, list):
                raise RuntimeConfigError("malformed activation dict node")
            decoded_dict: dict[str, Any] = {}
            for item in items:
                if (
                    not isinstance(item, list)
                    or len(item) != 2
                    or not isinstance(item[0], str)
                    or item[0] in decoded_dict
                ):
                    raise RuntimeConfigError("malformed or duplicate activation mapping entry")
                decoded_dict[item[0]] = decode(item[1])
            return decoded_dict
        raise RuntimeConfigError(f"unsupported activation tensor tree kind {kind!r}")

    artifact = decode(record.get("tree"))
    unused = sorted(set(fragments) - used)
    if unused:
        raise RuntimeConfigError(f"unreferenced activation tensor fragments: {unused}")
    if not isinstance(artifact, dict):
        raise RuntimeConfigError("activation tensor record root must decode to a mapping")
    return artifact


def _tensor_record_to_cpu(record: dict[str, Any]) -> dict[str, Any]:
    """
    Move a tensor-stream record to CPU without mutating cached weight state.
    """
    import torch

    return _tensor_record_to_device(record, torch.device("cpu"))


def _tensor_record_to_device(record: dict[str, Any], device: Any) -> dict[str, Any]:
    """
    Move tensor fragments to one device without mutating cached weight state.
    """
    fragments = record.get("fragments")
    if not isinstance(fragments, dict):
        raise RuntimeConfigError("tensor-native weight record must contain a fragment mapping")
    device_fragments: dict[str, dict[str, Any]] = {}
    for name, contribution in fragments.items():
        if not isinstance(contribution, dict):
            raise RuntimeConfigError(f"invalid routed tensor contribution {name!r}")
        value = contribution.get("value")
        device_contribution = dict(contribution)
        if value is not None and hasattr(value, "device") and value.device != device:
            device_contribution["value"] = value.detach().to(device=device, non_blocking=False)
        device_fragments[str(name)] = device_contribution
    device_record = dict(record)
    device_record["fragments"] = device_fragments
    return device_record


def _tensor_records_span_devices(records: list[dict[str, Any]]) -> bool:
    devices_by_name: dict[str, set[str]] = {}
    for record in records:
        fragments = record.get("fragments")
        if not isinstance(fragments, dict):
            continue
        for name, contribution in fragments.items():
            if not isinstance(contribution, dict):
                continue
            value = contribution.get("value")
            if value is not None and hasattr(value, "device"):
                devices_by_name.setdefault(str(name), set()).add(str(value.device))
    return any(len(devices) > 1 for devices in devices_by_name.values())


def _local_endpoint_transport(
    endpoints: list[dict[str, Any]],
    default_backend: str | None,
) -> str | tuple[str, ...] | None:
    transports = {
        _direct_endpoint_backend(endpoint, default_backend)
        for endpoint in endpoints
        if _direct_endpoint_backend(endpoint, default_backend) in {"gloo", "cuda_ipc"}
    }
    if not transports:
        return None
    if len(transports) == 1:
        return next(iter(transports))
    return tuple(sorted(transports))


def _cuda_ipc_send_tensor_record(
    *,
    endpoint: dict[str, Any],
    record: dict[str, Any],
    timeout_s: float,
) -> tuple[int, str]:
    import multiprocessing.reduction
    import torch

    cuda_record = _tensor_record_to_device(
        record,
        torch.device("cuda", torch.cuda.current_device()),
    )
    manifest, tensor_bytes = _tensor_stream_manifest(cuda_record)
    manifest_bytes = _tensor_stream_manifest_bytes(manifest)
    sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    torch.cuda.synchronize()
    wire_record = _cuda_ipc_inline_cpu_tensors(cuda_record)
    payload = bytes(multiprocessing.reduction.ForkingPickler.dumps(wire_record))
    if len(payload) > _CUDA_IPC_MAX_METADATA_BYTES:
        raise RuntimeConfigError(
            f"CUDA IPC tensor metadata is too large: {len(payload)} bytes"
        )
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    last_exc: BaseException | None = None
    address = (str(endpoint["host"]), int(endpoint["port"]))
    while time.monotonic() < deadline:
        try:
            sock = socket.create_connection(
                address,
                timeout=min(5.0, max(0.1, deadline - time.monotonic())),
            )
            break
        except OSError as exc:
            last_exc = exc
            time.sleep(0.05)
    else:
        raise RuntimeConfigError(
            f"timed out connecting CUDA IPC weight sender to {address}: {last_exc}"
        )
    with sock:
        sock.settimeout(max(1.0, float(timeout_s)))
        sock.sendall(
            _CUDA_IPC_TENSOR_MAGIC
            + struct.pack("!Q", len(payload))
            + bytes.fromhex(sha256)
            + payload
        )
        _torch_direct_recv_socket_ack(sock)
    return tensor_bytes + len(manifest_bytes), sha256


def _cuda_ipc_recv_tensor_record(
    *,
    endpoint: dict[str, Any],
    timeout_s: float,
    device: Any,
) -> tuple[dict[str, Any], int, str]:
    import multiprocessing.reduction
    import torch

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.settimeout(max(1.0, float(timeout_s)))
        server.bind(("", int(endpoint["port"])))
        server.listen(1)
        conn, _peer = server.accept()
        with conn:
            conn.settimeout(max(1.0, float(timeout_s)))
            magic = _socket_recv_exact(conn, len(_CUDA_IPC_TENSOR_MAGIC))
            if magic != _CUDA_IPC_TENSOR_MAGIC:
                raise RuntimeConfigError("invalid CUDA IPC tensor stream magic")
            payload_len = struct.unpack("!Q", _socket_recv_exact(conn, 8))[0]
            if payload_len > _CUDA_IPC_MAX_METADATA_BYTES:
                raise RuntimeConfigError(
                    f"invalid CUDA IPC tensor metadata length {payload_len}"
                )
            expected_sha256 = _socket_recv_exact(conn, 32).hex()
            payload = _socket_recv_exact(conn, int(payload_len))
            try:
                shared_record = multiprocessing.reduction.ForkingPickler.loads(payload)
            except Exception as exc:
                raise RuntimeConfigError(
                    f"invalid CUDA IPC tensor metadata: {type(exc).__name__}: {exc}"
                ) from exc
            shared_record = _cuda_ipc_restore_inline_cpu_tensors(shared_record)
            manifest, tensor_bytes = _tensor_stream_manifest(shared_record)
            manifest_bytes = _tensor_stream_manifest_bytes(manifest)
            sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            if sha256 != expected_sha256:
                raise RuntimeConfigError(
                    f"CUDA IPC tensor manifest checksum mismatch: {sha256} != {expected_sha256}"
                )
            destination = torch.device(device)
            if destination.type != "cuda":
                raise RuntimeConfigError("CUDA IPC weight endpoints require a CUDA destination model")
            cloned_fragments: dict[str, dict[str, Any]] = {}
            for name, contribution in shared_record["fragments"].items():
                cloned = dict(contribution)
                value = contribution.get("value")
                if value is not None:
                    if value.device.type == "cuda":
                        cloned["value"] = value.detach().to(device=destination).clone()
                    elif value.device.type == "cpu":
                        cloned["value"] = value.detach().clone()
                    else:
                        raise RuntimeConfigError(
                            f"CUDA IPC tensor {name!r} uses unsupported device {value.device}"
                        )
                cloned_fragments[str(name)] = cloned
            record = dict(shared_record)
            record["fragments"] = cloned_fragments
            torch.cuda.synchronize(destination)
            _torch_direct_send_socket_ack(conn)
    finally:
        server.close()
    return record, tensor_bytes + len(manifest_bytes), sha256


def _cuda_ipc_inline_cpu_tensors(record: dict[str, Any]) -> dict[str, Any]:
    import torch

    wire_record = dict(record)
    fragments: dict[str, dict[str, Any]] = {}
    for name, contribution in record["fragments"].items():
        wire_contribution = dict(contribution)
        value = contribution.get("value")
        if value is not None and value.device.type == "cpu":
            buffer = io.BytesIO()
            torch.save(value.detach(), buffer)
            wire_contribution["value"] = {
                _CUDA_IPC_INLINE_CPU_TENSOR: buffer.getvalue(),
            }
        fragments[str(name)] = wire_contribution
    wire_record["fragments"] = fragments
    return wire_record


def _cuda_ipc_restore_inline_cpu_tensors(record: dict[str, Any]) -> dict[str, Any]:
    import torch

    restored_record = dict(record)
    fragments: dict[str, dict[str, Any]] = {}
    for name, contribution in record["fragments"].items():
        restored_contribution = dict(contribution)
        value = contribution.get("value")
        if isinstance(value, dict) and set(value) == {_CUDA_IPC_INLINE_CPU_TENSOR}:
            payload = value[_CUDA_IPC_INLINE_CPU_TENSOR]
            if not isinstance(payload, bytes):
                raise RuntimeConfigError(f"invalid inline CPU tensor payload for {name!r}")
            restored_contribution["value"] = torch.load(
                io.BytesIO(payload),
                map_location="cpu",
                weights_only=False,
            )
        fragments[str(name)] = restored_contribution
    restored_record["fragments"] = fragments
    return restored_record


def _tensor_weight_version_record(state: dict[str, Any]) -> dict[str, Any]:
    state_dict = state.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeConfigError("tensor-native weight version must contain a state_dict mapping")
    record = {
        key: value
        for key, value in state.items()
        if key not in {
            "state_dict",
            "optimizer_state_by_name",
            "optimizer_state_name_map",
            "_serialized_bytes",
        }
    }
    record["schema_version"] = _UCX_TENSOR_WEIGHT_SCHEMA
    record["fragments"] = {
        name: {
            "value": value,
            "mode": "complete",
            "order": 0,
            "stride": 1,
            "dim": None,
        }
        for name, value in state_dict.items()
    }
    return record


def _in_place_layout_collective_mode(
    source_parallelism: tuple[int, int, int],
    destination_parallelism: tuple[int, int, int],
) -> str | None:
    """
    Select endpoint-free collectives for the RPEG DP/PP layout transpose.

    For a DP=D/PP=P source, every existing PP group contains one complete TP-sharded model replica.
    A DP=D*P/PP=1 destination maps those same ranks to independent serving replicas, so reuse the
    hot NCCL PP communicator to broadcast each source pipeline shard within its original DP replica.
    The reverse switch is local: every source serving DP replica already owns the complete TP shard
    and extracts the destination pipeline stage encoded by its rank. This covers 70B DP=2/PP=8 <->
    DP=16/PP=1 transpose without creating an all-worker communicator or staging tensor traffic
    through Gloo.
    """
    src_dp, src_pp, src_tp = source_parallelism
    dst_dp, dst_pp, dst_tp = destination_parallelism
    if src_tp != dst_tp:
        return None
    if dst_pp == 1 and src_dp * src_pp == dst_dp:
        return "pp_to_dp_broadcast"
    if src_pp == 1 and src_dp == dst_dp * dst_pp:
        return "dp_to_pp_local"
    return None


def _in_place_pp_group_ranks(
    rank: int,
    source_parallelism: tuple[int, int, int],
) -> tuple[int, ...]:
    """
    Return the canonical PP communicator membership for rank.
    """
    src_dp, src_pp, src_tp = source_parallelism
    world_size = src_dp * src_pp * src_tp
    if rank < 0 or rank >= world_size:
        raise RuntimeConfigError(
            f"rank {rank} is outside source layout {source_parallelism}"
        )
    dp_rank, _pp_rank, tp_rank = rank_to_dp_pp_tp(rank, src_pp, src_tp)
    return tuple(
        dp_pp_tp_to_rank(dp_rank, pp_rank, tp_rank, src_pp, src_tp)
        for pp_rank in range(src_pp)
    )


def _tensor_stream_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(payload) > 16 * 1024 * 1024:
        raise RuntimeConfigError(f"tensor-native weight manifest is too large: {len(payload)} bytes")
    return payload


def _torch_model_parallel_broadcast_tensor_record(
    record: dict[str, Any] | None,
    *,
    group: Any,
    source_rank: int,
    expected_schema: str,
    stream_non_source_to_cpu: bool = False,
    stream_chunk_bytes: int | None = None,
) -> tuple[dict[str, Any], int]:
    """
    Broadcast a structured activation without pickle/object collectives.

    NCCL carries the tensor leaves as typed device buffers. Only the compact JSON tree manifest
    crosses the host boundary. Gloo follows the identical protocol on CPU, which keeps the semantic
    contract testable without GPUs.

    stream_non_source_to_cpu is the bounded-HBM layout-switch path. Each non-source rank drains
    small NCCL byte chunks directly into a CPU tensor instead of first materializing the complete
    source fragment on its GPU. The source broadcasts views of its existing device tensors, so NCCL
    remains the tensor data plane and neither side creates a model-sized HBM staging allocation.
    """
    import torch

    dist = torch.distributed
    global_rank = dist.get_rank()
    is_source = global_rank == int(source_rank)
    backend = str(dist.get_backend(group)).lower()
    transport_device = (
        torch.device("cuda", torch.cuda.current_device())
        if backend == "nccl"
        else torch.device("cpu")
    )
    if stream_chunk_bytes is not None and not stream_non_source_to_cpu:
        raise RuntimeConfigError(
            "model-parallel stream_chunk_bytes requires stream_non_source_to_cpu"
        )
    if is_source:
        if not isinstance(record, dict):
            raise RuntimeConfigError("DP artifact writer has no in-memory activation record")
        manifest, tensor_bytes = _tensor_stream_manifest(
            record,
            schema_version=expected_schema,
        )
        manifest_bytes = _tensor_stream_manifest_bytes(manifest)
        manifest_length = len(manifest_bytes)
    else:
        manifest = None
        manifest_bytes = b""
        manifest_length = 0
        tensor_bytes = 0

    length_tensor = torch.tensor(
        [manifest_length],
        dtype=torch.int64,
        device=transport_device,
    )
    dist.broadcast(length_tensor, src=int(source_rank), group=group)
    manifest_length = int(length_tensor.item())
    if manifest_length < 2 or manifest_length > 16 * 1024 * 1024:
        raise RuntimeConfigError(
            f"invalid model-parallel activation manifest length {manifest_length}"
        )
    if is_source:
        manifest_tensor = torch.frombuffer(
            bytearray(manifest_bytes),
            dtype=torch.uint8,
        ).to(device=transport_device, non_blocking=False)
    else:
        manifest_tensor = torch.empty(
            manifest_length,
            dtype=torch.uint8,
            device=transport_device,
        )
    dist.broadcast(manifest_tensor, src=int(source_rank), group=group)
    if not is_source:
        manifest_bytes = manifest_tensor.cpu().numpy().tobytes()
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeConfigError("invalid model-parallel activation manifest") from exc
    assert isinstance(manifest, dict)
    if manifest.get("schema_version") != expected_schema:
        raise RuntimeConfigError(
            "unsupported model-parallel activation schema: "
            f"expected {expected_schema!r}, got {manifest.get('schema_version')!r}"
        )
    metadata = manifest.get("record")
    entries = manifest.get("entries")
    if not isinstance(metadata, dict) or not isinstance(entries, list):
        raise RuntimeConfigError("malformed model-parallel activation manifest")

    bounded_cpu_stream = bool(stream_non_source_to_cpu)
    destination_device = torch.device("cpu") if bounded_cpu_stream else transport_device
    if is_source:
        assert record is not None
        fragments = record["fragments"]
    else:
        fragments = {}
        tensor_bytes = 0
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                raise RuntimeConfigError("malformed model-parallel activation entry")
            name = entry["name"]
            if name in fragments:
                raise RuntimeConfigError(
                    f"duplicate model-parallel activation entry {name!r}"
                )
            kind = str(entry.get("kind"))
            if kind == "none":
                value = None
            elif kind == "tensor":
                shape = entry.get("shape")
                if not isinstance(shape, list) or any(
                    not isinstance(dim, int) or dim < 0 for dim in shape
                ):
                    raise RuntimeConfigError(
                        f"invalid model-parallel activation shape for {name!r}"
                    )
                value = torch.empty(
                    tuple(shape),
                    dtype=_torch_dtype_from_wire(str(entry.get("dtype"))),
                    device=destination_device,
                )
                nbytes = int(entry.get("nbytes", -1))
                expected_nbytes = int(value.numel()) * int(value.element_size())
                if nbytes != expected_nbytes:
                    raise RuntimeConfigError(
                        f"model-parallel activation byte count mismatch for {name!r}: "
                        f"{nbytes} != {expected_nbytes}"
                    )
                tensor_bytes += nbytes
            else:
                raise RuntimeConfigError(
                    f"unsupported model-parallel activation entry kind {kind!r}"
                )
            fragments[name] = {
                "value": value,
                "mode": str(entry.get("mode")),
                "order": int(entry.get("order", 0)),
                "stride": int(entry.get("stride", 1)),
                "dim": None if entry.get("dim") is None else int(entry["dim"]),
            }

    bucket_bytes = int(manifest.get("bucket_bytes", 0))
    if bucket_bytes < 1024 * 1024:
        raise RuntimeConfigError(
            f"invalid model-parallel activation bucket size {bucket_bytes}"
        )
    if bounded_cpu_stream:
        requested_chunk_bytes = (
            16 * 1024 * 1024
            if stream_chunk_bytes is None
            else int(stream_chunk_bytes)
        )
        if requested_chunk_bytes < 1024 * 1024:
            raise RuntimeConfigError(
                "model-parallel stream_chunk_bytes must be at least 1 MiB"
            )
        chunk_bytes = min(bucket_bytes, requested_chunk_bytes)
        tensor_entries = [
            entry
            for entry in entries
            if entry.get("kind") == "tensor" and int(entry.get("nbytes", 0)) > 0
        ]
        max_chunk_bytes = min(
            chunk_bytes,
            max((int(entry["nbytes"]) for entry in tensor_entries), default=0),
        )
        # A layout boundary is the right point to return inactive allocator blocks before reserving
        # the single bounded wire buffer. This is important on colocated actor/critic workers with
        # very little spare HBM. The source uses views of its resident parameters and therefore
        # needs no wire allocation.
        wire_buffer = None
        if not is_source and max_chunk_bytes:
            if transport_device.type == "cuda":
                torch.cuda.empty_cache()
            wire_buffer = torch.empty(
                max_chunk_bytes,
                dtype=torch.uint8,
                device=transport_device,
            )
        for entry in tensor_entries:
            name = str(entry["name"])
            nbytes = int(entry["nbytes"])
            if is_source:
                source_value = fragments[name]["value"]
                source_bytes = source_value.view(dtype=torch.uint8).reshape(-1)
                if source_bytes.device != transport_device:
                    raise RuntimeConfigError(
                        "bounded model-parallel broadcast source tensor is not on "
                        f"the {transport_device.type} transport device: {name!r}"
                    )
                if int(source_bytes.numel()) != nbytes:
                    raise RuntimeConfigError(
                        f"model-parallel source byte count mismatch for {name!r}"
                    )
                destination_bytes = None
            else:
                assert wire_buffer is not None
                source_bytes = None
                destination_bytes = fragments[name]["value"].view(
                    dtype=torch.uint8
                ).reshape(-1)
            for offset in range(0, nbytes, chunk_bytes):
                count = min(chunk_bytes, nbytes - offset)
                wire_tensor = (
                    source_bytes[offset:offset + count]
                    if is_source
                    else wire_buffer[:count]
                )
                dist.broadcast(wire_tensor, src=int(source_rank), group=group)
                if destination_bytes is not None:
                    destination_bytes[offset:offset + count].copy_(
                        wire_tensor,
                        non_blocking=False,
                    )
        if wire_buffer is not None:
            del wire_buffer
    else:
        for stream_group in _tensor_stream_groups(entries, bucket_bytes):
            group_nbytes = sum(int(entry["nbytes"]) for entry in stream_group)
            if is_source:
                packed = _pack_tensor_stream_group(stream_group, fragments)
                wire_tensor = packed.view(dtype=torch.uint8).reshape(-1)
                if wire_tensor.device != transport_device:
                    wire_tensor = wire_tensor.to(device=transport_device, non_blocking=False)
            elif len(stream_group) == 1:
                wire_tensor = fragments[stream_group[0]["name"]]["value"].view(
                    dtype=torch.uint8
                ).reshape(-1)
            else:
                wire_tensor = torch.empty(
                    group_nbytes,
                    dtype=torch.uint8,
                    device=transport_device,
                )
            if int(wire_tensor.numel()) != group_nbytes:
                raise RuntimeConfigError("model-parallel activation bucket size mismatch")
            dist.broadcast(wire_tensor, src=int(source_rank), group=group)
            if not is_source and len(stream_group) > 1:
                offset = 0
                for entry in stream_group:
                    destination = fragments[entry["name"]]["value"]
                    nbytes = int(entry["nbytes"])
                    destination.view(dtype=torch.uint8).reshape(-1).copy_(
                        wire_tensor[offset:offset + nbytes]
                    )
                    offset += nbytes
    if transport_device.type == "cuda":
        torch.cuda.synchronize(transport_device)
    output = dict(metadata)
    output["fragments"] = fragments
    return output, int(tensor_bytes) + len(manifest_bytes)


def _torch_dtype_from_wire(name: str):
    import torch

    dtype = getattr(torch, str(name), None)
    if dtype not in {
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    }:
        raise RuntimeConfigError(f"unsupported tensor-native weight dtype {name!r}")
    return dtype


def _torch_tensor_ucx_buffer(tensor: Any) -> Any:
    if not tensor.is_contiguous():
        raise RuntimeConfigError("UCX tensor buffers must be contiguous")
    if tensor.device.type == "cuda":
        return _CudaByteBuffer(tensor)
    if tensor.device.type != "cpu":
        raise RuntimeConfigError(f"UCX tensor buffers do not support device {tensor.device}")
    return memoryview(tensor.view(dtype=__import__("torch").uint8).numpy())


def _ucx_tensor_bucket_bytes() -> int:
    raw = os.environ.get("RPEG_UCX_TENSOR_BUCKET_BYTES")
    if raw is None:
        return _UCX_TENSOR_BUCKET_BYTES
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeConfigError("RPEG_UCX_TENSOR_BUCKET_BYTES must be an integer") from None
    if value < 1024 * 1024:
        raise RuntimeConfigError("RPEG_UCX_TENSOR_BUCKET_BYTES must be at least 1 MiB")
    return value


def _tensor_stream_groups(entries: list[dict[str, Any]], bucket_bytes: int) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    for entry in entries:
        if entry.get("kind") != "tensor" or int(entry.get("nbytes", 0)) == 0:
            continue
        nbytes = int(entry["nbytes"])
        if current and current_bytes + nbytes > bucket_bytes:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(entry)
        current_bytes += nbytes
        if current_bytes >= bucket_bytes:
            groups.append(current)
            current = []
            current_bytes = 0
    if current:
        groups.append(current)
    return groups


def _pack_tensor_stream_group(
    group: list[dict[str, Any]],
    fragments: dict[str, dict[str, Any]],
):
    import torch

    tensors = [fragments[entry["name"]]["value"] for entry in group]
    if len(tensors) == 1:
        return tensors[0]
    device = tensors[0].device
    if any(tensor.device != device for tensor in tensors):
        raise RuntimeConfigError("tensor-native UCX bucket spans multiple devices")
    packed = torch.cat([tensor.view(dtype=torch.uint8).reshape(-1) for tensor in tensors])
    if packed.device.type == "cuda":
        torch.cuda.synchronize(packed.device)
    return packed


async def _ucx_direct_send_tensor_record(
    *,
    endpoint: dict[str, Any],
    record: dict[str, Any],
    timeout_s: float,
    ready_path: Path | None = None,
    ready_wait_s: float = 0.0,
    connect_grace_s: float = 0.0,
    schema_version: str = _UCX_TENSOR_STREAM_SCHEMA,
) -> bool:
    ucx = _ucx_direct_module()
    _ucx_direct_init_once(ucx)
    await _wait_direct_ready_marker_async(ready_path, timeout_s=ready_wait_s)
    ep = _ucx_direct_cached_endpoint("send", endpoint)
    cached = ep is not None
    if ep is None:
        ep = await _ucx_connect_with_retry(
            ucx,
            str(endpoint["host"]),
            int(endpoint["port"]),
            timeout_s,
            initial_delay_s=connect_grace_s,
        )
    stored = False
    try:
        manifest, _tensor_bytes = _tensor_stream_manifest(
            record,
            schema_version=schema_version,
        )
        manifest_bytes = _tensor_stream_manifest_bytes(manifest)
        await _ucx_send(ep, struct.pack("!Q", len(manifest_bytes)))
        await _ucx_send(ep, manifest_bytes)
        fragments = record["fragments"]
        groups = _tensor_stream_groups(manifest["entries"], int(manifest["bucket_bytes"]))
        for group in groups:
            packed = _pack_tensor_stream_group(group, fragments)
            await _ucx_send(ep, _torch_tensor_ucx_buffer(packed))
        if not cached:
            stored = _ucx_direct_store_cached_endpoint("send", endpoint, ep)
        return cached
    except Exception:
        if cached:
            _ucx_direct_discard_cached_endpoint("send", endpoint)
        raise
    finally:
        if not cached and not stored:
            await _ucx_close(ep)


async def _ucx_direct_recv_tensor_record(
    *,
    endpoint: dict[str, Any],
    timeout_s: float,
    ready_path: Path | None = None,
    device: Any = None,
    expected_schema: str = _UCX_TENSOR_STREAM_SCHEMA,
) -> tuple[dict[str, Any], int, str, bool]:
    import torch

    device = torch.device("cpu") if device is None else torch.device(device)
    ucx = _ucx_direct_module()
    _ucx_direct_init_once(ucx)
    cached_ep = _ucx_direct_cached_endpoint("recv", endpoint)
    if cached_ep is not None:
        try:
            record, bytes_, sha256 = await _ucx_direct_recv_tensor_record_payload(
                cached_ep,
                device=device,
                expected_schema=expected_schema,
            )
            return record, bytes_, sha256, True
        except Exception:
            _ucx_direct_discard_cached_endpoint("recv", endpoint)
            raise
    endpoint_future = asyncio.get_running_loop().create_future()

    async def _listener_callback(ep):
        if not endpoint_future.done():
            endpoint_future.set_result(ep)
        else:
            await _ucx_close(ep)

    def _sync_listener_callback(ep):
        if not endpoint_future.done():
            endpoint_future.set_result(ep)

    listener = await _ucx_create_listener_with_retry(
        ucx,
        _listener_callback,
        _sync_listener_callback,
        port=int(endpoint["port"]),
        timeout_s=timeout_s,
    )
    _write_direct_ready_marker(ready_path, endpoint)
    try:
        ep = await asyncio.wait_for(endpoint_future, timeout=max(1.0, float(timeout_s)))
        stored = False
        try:
            record, bytes_, sha256 = await _ucx_direct_recv_tensor_record_payload(
                ep,
                device=device,
                expected_schema=expected_schema,
            )
            stored = _ucx_direct_store_cached_endpoint("recv", endpoint, ep)
            return record, bytes_, sha256, False
        finally:
            if not stored:
                await _ucx_close(ep)
    finally:
        await _ucx_close(listener)


async def _ucx_direct_recv_tensor_record_payload(
    ep: Any,
    *,
    device: Any,
    expected_schema: str = _UCX_TENSOR_STREAM_SCHEMA,
) -> tuple[dict[str, Any], int, str]:
    import torch

    manifest_len = struct.unpack("!Q", await _ucx_recv_exact(ep, 8))[0]
    if manifest_len > 16 * 1024 * 1024:
        raise RuntimeConfigError(f"invalid tensor-native weight manifest length {manifest_len}")
    manifest_bytes = await _ucx_recv_exact(ep, int(manifest_len))
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeConfigError("invalid tensor-native weight manifest") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != expected_schema:
        raise RuntimeConfigError(
            "unsupported tensor-native stream schema: "
            f"expected {expected_schema!r}, got "
            f"{manifest.get('schema_version') if isinstance(manifest, dict) else None!r}"
        )
    metadata = manifest.get("record")
    entries = manifest.get("entries")
    if not isinstance(metadata, dict) or not isinstance(entries, list):
        raise RuntimeConfigError("malformed tensor-native weight manifest")
    fragments: dict[str, dict[str, Any]] = {}
    tensor_bytes = 0
    tensors_by_name: dict[str, Any] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise RuntimeConfigError("malformed tensor-native weight entry")
        name = entry["name"]
        if name in fragments:
            raise RuntimeConfigError(f"duplicate tensor-native weight entry {name!r}")
        kind = str(entry.get("kind"))
        if kind == "none":
            value = None
        elif kind == "tensor":
            shape = entry.get("shape")
            if not isinstance(shape, list) or any(not isinstance(dim, int) or dim < 0 for dim in shape):
                raise RuntimeConfigError(f"invalid tensor-native weight shape for {name!r}")
            value = torch.empty(
                tuple(shape),
                dtype=_torch_dtype_from_wire(str(entry.get("dtype"))),
                device=device,
            )
            expected_nbytes = int(value.numel()) * int(value.element_size())
            nbytes = int(entry.get("nbytes", -1))
            if nbytes != expected_nbytes:
                raise RuntimeConfigError(
                    f"tensor-native weight byte count mismatch for {name!r}: {nbytes} != {expected_nbytes}"
                )
            tensors_by_name[name] = value
            tensor_bytes += nbytes
        else:
            raise RuntimeConfigError(f"unsupported tensor-native weight entry kind {kind!r}")
        fragments[name] = {
            "value": value,
            "mode": str(entry.get("mode")),
            "order": int(entry.get("order", 0)),
            "stride": int(entry.get("stride", 1)),
            "dim": None if entry.get("dim") is None else int(entry["dim"]),
        }
    bucket_bytes = int(manifest.get("bucket_bytes", 0))
    if bucket_bytes < 1024 * 1024:
        raise RuntimeConfigError(f"invalid tensor-native UCX bucket size {bucket_bytes}")
    copied_on_device = False
    for group in _tensor_stream_groups(entries, bucket_bytes):
        group_nbytes = sum(int(entry["nbytes"]) for entry in group)
        if len(group) == 1:
            destination = tensors_by_name[group[0]["name"]]
            await _ucx_recv_into(ep, _torch_tensor_ucx_buffer(destination), group_nbytes)
            continue
        packed = torch.empty(group_nbytes, dtype=torch.uint8, device=device)
        await _ucx_recv_into(ep, _torch_tensor_ucx_buffer(packed), group_nbytes)
        offset = 0
        for entry in group:
            destination = tensors_by_name[entry["name"]]
            nbytes = int(entry["nbytes"])
            destination.view(dtype=torch.uint8).reshape(-1).copy_(packed[offset:offset + nbytes])
            offset += nbytes
        copied_on_device = copied_on_device or device.type == "cuda"
    if copied_on_device:
        torch.cuda.synchronize(device)
    record = dict(metadata)
    record["fragments"] = fragments
    return record, tensor_bytes + len(manifest_bytes), hashlib.sha256(manifest_bytes).hexdigest()


async def _ucx_direct_send_file_with_metadata(
    *,
    endpoint: dict[str, Any],
    path: Path,
    bytes_: int,
    sha256: str,
    metadata: dict[str, Any],
    timeout_s: float,
    ready_path: Path | None = None,
    ready_wait_s: float = 0.0,
    connect_grace_s: float = 0.0,
) -> bool:
    ucx = _ucx_direct_module()
    _ucx_direct_init_once(ucx)
    await _wait_direct_ready_marker_async(ready_path, timeout_s=ready_wait_s)
    ep = _ucx_direct_cached_endpoint("send", endpoint)
    cached = ep is not None
    if ep is None:
        ep = await _ucx_connect_with_retry(
            ucx,
            str(endpoint["host"]),
            int(endpoint["port"]),
            timeout_s,
            initial_delay_s=connect_grace_s,
        )
    try:
        metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        await _ucx_send(ep, struct.pack("!Q", len(metadata_bytes)))
        await _ucx_send(ep, metadata_bytes)
        header = struct.pack("!Q", int(bytes_)) + bytes.fromhex(sha256)
        await _ucx_send(ep, header)
        remaining = int(bytes_)
        with path.open("rb") as src:
            while remaining > 0:
                chunk = src.read(min(16 * 1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeConfigError(f"ucx_direct activation send ended early for {path}")
                await _ucx_send(ep, chunk)
                remaining -= len(chunk)
        return cached
    except Exception:
        if cached:
            _ucx_direct_discard_cached_endpoint("send", endpoint)
        raise
    finally:
        if not cached:
            await _ucx_close(ep)


async def _ucx_direct_recv_file(
    *,
    endpoint: dict[str, Any],
    destination: Path,
    bytes_: int,
    timeout_s: float,
    ready_path: Path | None = None,
) -> tuple[int, str, bool]:
    ucx = _ucx_direct_module()
    _ucx_direct_init_once(ucx)
    cached_ep = _ucx_direct_cached_endpoint("recv", endpoint)
    if cached_ep is not None:
        try:
            total, sha256 = await _ucx_direct_recv_file_payload(cached_ep, destination, bytes_)
            return total, sha256, True
        except Exception:
            _ucx_direct_discard_cached_endpoint("recv", endpoint)
            raise
    endpoint_future = asyncio.get_running_loop().create_future()

    async def _listener_callback(ep):
        if not endpoint_future.done():
            endpoint_future.set_result(ep)
        else:
            await _ucx_close(ep)

    def _sync_listener_callback(ep):
        if not endpoint_future.done():
            endpoint_future.set_result(ep)

    listener = await _ucx_create_listener_with_retry(
        ucx,
        _listener_callback,
        _sync_listener_callback,
        port=int(endpoint["port"]),
        timeout_s=timeout_s,
    )
    _write_direct_ready_marker(ready_path, endpoint)
    try:
        ep = await asyncio.wait_for(endpoint_future, timeout=max(1.0, float(timeout_s)))
        try:
            total, sha256 = await _ucx_direct_recv_file_payload(ep, destination, bytes_)
            return total, sha256, False
        finally:
            await _ucx_close(ep)
    finally:
        await _ucx_close(listener)


async def _ucx_direct_recv_blob(
    *,
    endpoint: dict[str, Any],
    timeout_s: float,
    ready_path: Path | None = None,
) -> tuple[bytes, str, bool]:
    ucx = _ucx_direct_module()
    _ucx_direct_init_once(ucx)
    cached_ep = _ucx_direct_cached_endpoint("recv", endpoint)
    if cached_ep is not None:
        try:
            payload, sha256 = await _ucx_direct_recv_blob_payload(cached_ep)
            return payload, sha256, True
        except Exception:
            _ucx_direct_discard_cached_endpoint("recv", endpoint)
            raise
    endpoint_future = asyncio.get_running_loop().create_future()

    async def _listener_callback(ep):
        if not endpoint_future.done():
            endpoint_future.set_result(ep)
        else:
            await _ucx_close(ep)

    def _sync_listener_callback(ep):
        if not endpoint_future.done():
            endpoint_future.set_result(ep)

    listener = await _ucx_create_listener_with_retry(
        ucx,
        _listener_callback,
        _sync_listener_callback,
        port=int(endpoint["port"]),
        timeout_s=timeout_s,
    )
    _write_direct_ready_marker(ready_path, endpoint)
    try:
        ep = await asyncio.wait_for(endpoint_future, timeout=max(1.0, float(timeout_s)))
        stored = False
        try:
            payload, sha256 = await _ucx_direct_recv_blob_payload(ep)
            stored = _ucx_direct_store_cached_endpoint("recv", endpoint, ep)
            return payload, sha256, False
        finally:
            if not stored:
                await _ucx_close(ep)
    finally:
        await _ucx_close(listener)


async def _ucx_direct_recv_blob_payload(ep: Any) -> tuple[bytes, str]:
    header = await _ucx_recv_exact(ep, 40)
    bytes_ = struct.unpack("!Q", header[:8])[0]
    expected_sha = header[8:].hex()
    chunks = []
    remaining = int(bytes_)
    while remaining > 0:
        chunk = await _ucx_recv_exact(ep, min(16 * 1024 * 1024, remaining))
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(chunks)
    sha256 = hashlib.sha256(payload).hexdigest()
    if sha256 != expected_sha:
        raise RuntimeConfigError(
            f"ucx_direct in-memory activation checksum mismatch: {sha256} != {expected_sha}"
        )
    return payload, sha256


async def _ucx_direct_recv_endpoint_blobs(
    *,
    endpoint_sources: list[tuple[dict[str, Any], int, int, int]],
    timeout_s: float,
    state_root: Path,
) -> list[tuple[int, int, int, bytes, str, bool]]:
    async def receive_one(
        item: tuple[dict[str, Any], int, int, int],
    ) -> tuple[int, int, int, bytes, str, bool]:
        endpoint, source_dp_rank, overlap_start, overlap_end = item
        payload, sha256, cached = await _ucx_direct_recv_blob(
            endpoint=endpoint,
            timeout_s=timeout_s,
            ready_path=_direct_ready_marker_path(state_root, endpoint),
        )
        return (
            source_dp_rank,
            overlap_start,
            overlap_end,
            payload,
            sha256,
            cached,
        )

    return list(await asyncio.gather(*(receive_one(item) for item in endpoint_sources)))


async def _ucx_direct_recv_endpoint_tensor_records(
    *,
    endpoint_sources: list[tuple[dict[str, Any], int, int, int]],
    timeout_s: float,
    state_root: Path,
    device: Any,
    expected_schema: str,
) -> list[tuple[int, int, int, dict[str, Any], int, str, bool]]:
    """
    Receive independent DP-overlap tensor records concurrently.
    """
    async def receive_one(
        item: tuple[dict[str, Any], int, int, int],
    ) -> tuple[int, int, int, dict[str, Any], int, str, bool]:
        endpoint, source_dp_rank, overlap_start, overlap_end = item
        record, bytes_, sha256, cached = await _ucx_direct_recv_tensor_record(
            endpoint=endpoint,
            timeout_s=timeout_s,
            ready_path=_direct_ready_marker_path(state_root, endpoint),
            device=device,
            expected_schema=expected_schema,
        )
        return (
            source_dp_rank,
            overlap_start,
            overlap_end,
            record,
            bytes_,
            sha256,
            cached,
        )

    return list(await asyncio.gather(*(receive_one(item) for item in endpoint_sources)))


async def _ucx_direct_recv_files_with_metadata(
    *,
    endpoint: dict[str, Any],
    destinations: list[Path],
    expected_count: int,
    timeout_s: float,
    ready_path: Path | None = None,
) -> tuple[list[tuple[Path, int, str, dict[str, Any]]], int]:
    ucx = _ucx_direct_module()
    _ucx_direct_init_once(ucx)
    if expected_count != len(destinations):
        raise RuntimeConfigError("ucx_direct metadata receive expected_count must match destination count")
    queue: asyncio.Queue[Any] = asyncio.Queue()

    async def _listener_callback(ep):
        await queue.put(ep)

    def _sync_listener_callback(ep):
        queue.put_nowait(ep)

    listener = await _ucx_create_listener_with_retry(
        ucx,
        _listener_callback,
        _sync_listener_callback,
        port=int(endpoint["port"]),
        timeout_s=timeout_s,
    )
    _write_direct_ready_marker(ready_path, endpoint)
    received: list[tuple[Path, int, str, dict[str, Any]]] = []
    cached_count = 0
    try:
        for idx in range(int(expected_count)):
            ep = await asyncio.wait_for(queue.get(), timeout=max(1.0, float(timeout_s)))
            try:
                bytes_, sha256, metadata = await _ucx_direct_recv_file_metadata_payload(ep, destinations[idx])
                received.append((destinations[idx], bytes_, sha256, metadata))
            finally:
                await _ucx_close(ep)
        return received, cached_count
    finally:
        await _ucx_close(listener)


async def _ucx_direct_recv_endpoint_files_with_metadata(
    *,
    endpoint_destinations: list[tuple[dict[str, Any], Path]],
    timeout_s: float,
    ready_paths: list[Path | None] | None = None,
) -> tuple[list[tuple[Path, int, str, dict[str, Any]]], int]:
    ucx = _ucx_direct_module()
    _ucx_direct_init_once(ucx)
    loop = asyncio.get_running_loop()
    listeners = []
    futures = []

    async def _make_listener(endpoint: dict[str, Any]):
        future = loop.create_future()

        async def _listener_callback(ep):
            if not future.done():
                future.set_result(ep)
            else:
                await _ucx_close(ep)

        def _sync_listener_callback(ep):
            if not future.done():
                future.set_result(ep)

        listener = await _ucx_create_listener_with_retry(
            ucx,
            _listener_callback,
            _sync_listener_callback,
            port=int(endpoint["port"]),
            timeout_s=timeout_s,
        )
        return listener, future

    try:
        if ready_paths is not None and len(ready_paths) != len(endpoint_destinations):
            raise RuntimeConfigError("ucx_direct ready_paths must match endpoint_destinations")
        for idx, (endpoint, _destination) in enumerate(endpoint_destinations):
            cached_ep = _ucx_direct_cached_endpoint("recv", endpoint)
            if cached_ep is not None:
                future = loop.create_future()
                future.set_result(cached_ep)
                futures.append(future)
                _write_direct_ready_marker(None if ready_paths is None else ready_paths[idx], endpoint)
                continue
            listener, future = await _make_listener(endpoint)
            listeners.append(listener)
            futures.append(future)
            _write_direct_ready_marker(None if ready_paths is None else ready_paths[idx], endpoint)
        received: list[tuple[Path, int, str, dict[str, Any]]] = []
        cached_count = 0
        for future, (_endpoint, destination) in zip(futures, endpoint_destinations, strict=True):
            ep = await asyncio.wait_for(future, timeout=max(1.0, float(timeout_s)))
            cached = _ucx_direct_cached_endpoint("recv", _endpoint) is ep
            try:
                bytes_, sha256, metadata = await _ucx_direct_recv_file_metadata_payload(ep, destination)
                received.append((destination, bytes_, sha256, metadata))
                cached_count += int(cached)
            except Exception:
                if cached:
                    _ucx_direct_discard_cached_endpoint("recv", _endpoint)
                raise
            finally:
                if not cached:
                    await _ucx_close(ep)
        return received, cached_count
    finally:
        for listener in listeners:
            await _ucx_close(listener)


async def _ucx_direct_recv_file_metadata_payload(
    ep: Any,
    destination: Path,
) -> tuple[int, str, dict[str, Any]]:
    metadata_len = struct.unpack("!Q", await _ucx_recv_exact(ep, 8))[0]
    if metadata_len <= 0 or metadata_len > 1024 * 1024:
        raise RuntimeConfigError(f"invalid ucx_direct activation slice metadata length {metadata_len}")
    metadata = json.loads((await _ucx_recv_exact(ep, int(metadata_len))).decode("utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != "rpeg.activation_direct_slice.v1":
        raise RuntimeConfigError("unsupported ucx_direct activation slice metadata")
    header = await _ucx_recv_exact(ep, 40)
    bytes_ = struct.unpack("!Q", header[:8])[0]
    expected_sha = header[8:].hex()
    digest = hashlib.sha256()
    remaining = int(bytes_)
    total = 0
    with destination.open("wb") as dst:
        while remaining > 0:
            payload = await _ucx_recv_exact(ep, min(16 * 1024 * 1024, remaining))
            digest.update(payload)
            dst.write(payload)
            total += len(payload)
            remaining -= len(payload)
    sha256 = digest.hexdigest()
    if sha256 != expected_sha:
        raise RuntimeConfigError(
            f"ucx_direct activation checksum mismatch while receiving {destination}: {sha256} != {expected_sha}"
        )
    return total, sha256, metadata


async def _ucx_direct_recv_file_payload(
    ep: Any,
    destination: Path,
    bytes_: int,
) -> tuple[int, str]:
    header = await _ucx_recv_exact(ep, 40)
    declared_bytes = struct.unpack("!Q", header[:8])[0]
    expected_sha = header[8:].hex()
    if int(declared_bytes) != int(bytes_):
        raise RuntimeConfigError(
            f"ucx_direct activation byte-count mismatch: header={declared_bytes}, metadata={bytes_}"
        )
    digest = hashlib.sha256()
    remaining = int(bytes_)
    total = 0
    with destination.open("wb") as dst:
        while remaining > 0:
            payload = await _ucx_recv_exact(ep, min(16 * 1024 * 1024, remaining))
            digest.update(payload)
            dst.write(payload)
            total += len(payload)
            remaining -= len(payload)
    sha256 = digest.hexdigest()
    if sha256 != expected_sha:
        raise RuntimeConfigError(
            f"ucx_direct activation checksum mismatch while receiving {destination}: {sha256} != {expected_sha}"
        )
    return total, sha256


async def _ucx_direct_send_prewarm(
    *,
    endpoint: dict[str, Any],
    timeout_s: float,
    connect_grace_s: float = 0.0,
) -> bool:
    ucx = _ucx_direct_module()
    _ucx_direct_init_once(ucx)
    ep = await _ucx_connect_with_retry(
        ucx,
        str(endpoint["host"]),
        int(endpoint["port"]),
        timeout_s,
        initial_delay_s=connect_grace_s,
    )
    cached = False
    try:
        await _ucx_send(ep, _UCX_DIRECT_PREWARM_MAGIC)
        cached = _ucx_direct_store_cached_endpoint("send", endpoint, ep)
        if cached:
            return True
        return False
    finally:
        if not cached:
            await _ucx_close(ep)


async def _ucx_direct_recv_prewarm(
    *,
    endpoint: dict[str, Any],
    timeout_s: float,
    ready_path: Path | None = None,
) -> bool:
    ucx = _ucx_direct_module()
    _ucx_direct_init_once(ucx)
    loop = asyncio.get_running_loop()
    received: asyncio.Future[None] = loop.create_future()

    async def _handle(ep):
        cached = False
        try:
            payload = await _ucx_recv_exact(ep, len(_UCX_DIRECT_PREWARM_MAGIC))
            if payload != _UCX_DIRECT_PREWARM_MAGIC:
                raise RuntimeConfigError("ucx_direct prewarm token mismatch")
            cached = _ucx_direct_store_cached_endpoint("recv", endpoint, ep)
            if not received.done():
                received.set_result(cached)
        except Exception as exc:
            if not received.done():
                received.set_exception(exc)
        finally:
            if not cached:
                await _ucx_close(ep)

    def _handler(ep):
        asyncio.create_task(_handle(ep))

    listener = await _ucx_create_listener_with_retry(
        ucx,
        _handler,
        None,
        port=int(endpoint["port"]),
        timeout_s=timeout_s,
    )
    _write_direct_ready_marker(ready_path, endpoint)
    try:
        return await asyncio.wait_for(received, timeout=max(0.1, float(timeout_s)))
    finally:
        await _ucx_close(listener)


async def _ucx_connect_with_retry(
    ucx: Any,
    host: str,
    port: int,
    timeout_s: float,
    *,
    initial_delay_s: float = 0.0,
):
    deadline = time.monotonic() + float(timeout_s)
    last_error: Exception | None = None
    delay = max(0.0, min(float(initial_delay_s), max(0.0, float(timeout_s))))
    if delay > 0.0:
        await asyncio.sleep(delay)
    while time.monotonic() < deadline:
        try:
            return await _maybe_await(ucx.create_endpoint(host, int(port)))
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(0.05)
    raise RuntimeConfigError(f"timed out connecting ucx_direct sender to {host}:{port}: {last_error}")


async def _ucx_create_listener_with_retry(
    ucx: Any,
    callback: Any,
    sync_callback: Any | None,
    *,
    port: int,
    timeout_s: float,
):
    deadline = time.monotonic() + min(max(0.1, float(timeout_s)), 10.0)
    last_error: Exception | None = None
    while True:
        try:
            return await _maybe_await(ucx.create_listener(callback, port=int(port)))
        except TypeError:
            if sync_callback is None:
                raise
            try:
                return await _maybe_await(ucx.create_listener(sync_callback, port=int(port)))
            except Exception as exc:
                last_error = exc
        except Exception as exc:
            last_error = exc
        if time.monotonic() >= deadline:
            break
        await asyncio.sleep(0.05)
    raise RuntimeConfigError(f"timed out creating ucx_direct listener on port {port}: {last_error}") from last_error


def _ucx_direct_connect_grace_s(payload: dict[str, Any]) -> float:
    raw = payload.get("direct_connect_grace_s", 0.05)
    try:
        return max(0.0, min(float(raw), 1.0))
    except (TypeError, ValueError):
        raise RuntimeConfigError("direct_connect_grace_s must be a non-negative float") from None


def _ucx_direct_ready_wait_s(payload: dict[str, Any]) -> float:
    raw = payload.get("direct_ready_wait_s", 1.0)
    try:
        return max(0.0, min(float(raw), 1.0))
    except (TypeError, ValueError):
        raise RuntimeConfigError("direct_ready_wait_s must be a non-negative float") from None


def _direct_ready_marker_path(state_root: Path, endpoint: dict[str, Any]) -> Path:
    host = str(endpoint.get("host") or "")
    port = int(endpoint.get("port", 0))
    key = hashlib.sha1(f"{host}:{port}".encode("utf-8")).hexdigest()
    return state_root / "runtime" / "direct_ready" / f"{key}.json"


def _write_direct_ready_marker(path: Path | None, endpoint: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "rpeg.direct_ready.v1",
        "host": str(endpoint.get("host") or ""),
        "port": int(endpoint.get("port", 0)),
        "pid": os.getpid(),
        "timestamp": time.time(),
    }
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    tmp.replace(path)


async def _wait_direct_ready_marker_async(path: Path | None, *, timeout_s: float) -> None:
    if path is None:
        return
    deadline = time.monotonic() + min(max(0.0, float(timeout_s)), 5.0)
    while time.monotonic() < deadline:
        if path.exists():
            return
        await asyncio.sleep(0.01)


async def _ucx_send(ep: Any, payload: bytes) -> None:
    await _maybe_await(ep.send(payload))


async def _ucx_recv_exact(ep: Any, nbytes: int) -> bytes:
    buffer = bytearray(int(nbytes))
    result = await _maybe_await(ep.recv(buffer))
    if isinstance(result, (bytes, bytearray, memoryview)):
        data = bytes(result)
        if len(data) != int(nbytes):
            raise RuntimeConfigError(f"ucx_direct short receive: {len(data)} != {nbytes}")
        return data
    return bytes(buffer)


async def _ucx_recv_into(ep: Any, buffer: Any, nbytes: int) -> None:
    result = await _maybe_await(ep.recv(buffer))
    if isinstance(result, (bytes, bytearray, memoryview)):
        data = bytes(result)
        if len(data) != int(nbytes):
            raise RuntimeConfigError(f"ucx_direct short tensor receive: {len(data)} != {nbytes}")
        if isinstance(buffer, _CudaByteBuffer):
            import torch

            source = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            buffer.tensor.view(dtype=torch.uint8).copy_(source.to(device=buffer.tensor.device))
        else:
            memoryview(buffer)[:] = data


async def _ucx_close(obj: Any) -> None:
    close = getattr(obj, "close", None)
    if not callable(close):
        return
    result = close()
    await _maybe_await(result)


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _ucx_direct_init_once(ucx: Any) -> None:
    global _UCX_DIRECT_INITIALIZED
    init = getattr(ucx, "init", None)
    if not callable(init):
        return
    with _UCX_DIRECT_INIT_LOCK:
        if _UCX_DIRECT_INITIALIZED:
            return
        try:
            init()
        except Exception as exc:
            message = str(exc).lower()
            if "already initiated" not in message and "already initialized" not in message:
                raise
        _UCX_DIRECT_INITIALIZED = True


def _torch_direct_backend(payload: dict[str, Any]) -> str:
    backend = str(payload.get("direct_backend", "auto")).strip().lower().replace("-", "_")
    if backend in {"", "auto"}:
        try:
            import torch

            return "nccl" if torch.cuda.is_available() else "gloo"
        except ModuleNotFoundError as exc:
            raise RuntimeDependencyError("torch_dist activation transfer requires PyTorch") from exc
    if backend in {"gloo", "nccl"}:
        return backend
    raise RuntimeConfigError("direct_backend must be one of: auto, gloo, nccl")


def _direct_endpoint_backend(endpoint: dict[str, Any], default_backend: str | None) -> str:
    backend = str(endpoint.get("endpoint_backend") or default_backend or "").strip().lower()
    if backend not in {"gloo", "nccl", "cuda_ipc"}:
        raise RuntimeConfigError(f"unsupported direct endpoint backend {backend!r}")
    return backend


def _torch_direct_process_group_cache_key(
    endpoint: dict[str, Any],
    backend: str,
    pg_rank: int,
) -> tuple[str, int, str] | None:
    channel_id = endpoint.get("direct_channel_id")
    if not endpoint.get("direct_channel_reuse") or not channel_id:
        return None
    return str(backend), int(pg_rank), str(channel_id)


def _torch_direct_cached_process_group(
    key: tuple[str, int, str] | None,
) -> Any | None:
    if key is None:
        return None
    with _TORCH_DIRECT_PROCESS_GROUP_CACHE_LOCK:
        entry = _TORCH_DIRECT_PROCESS_GROUP_CACHE.get(key)
        if entry is not None:
            # Plain dictionaries preserve insertion order. Refresh the entry for stable diagnostics;
            # live NCCL groups are not evicted here, because teardown must be coordinated by every
            # participant.
            _TORCH_DIRECT_PROCESS_GROUP_CACHE.pop(key)
            _TORCH_DIRECT_PROCESS_GROUP_CACHE[key] = entry
        return None if entry is None else entry[0]


def _torch_direct_release_process_group(pg: Any) -> None:
    try:
        import torch.distributed as dist
    except ModuleNotFoundError:
        return
    backend_name = ""
    name = getattr(pg, "name", None)
    if callable(name):
        try:
            backend_name = str(name()).lower()
        except Exception:
            backend_name = ""
    if "nccl" in backend_name:
        abort = getattr(pg, "abort", None)
        if callable(abort):
            try:
                abort()
            except Exception:
                pass
        return
    try:
        dist.destroy_process_group(pg)
    except Exception:
        abort = getattr(pg, "abort", None)
        if callable(abort):
            try:
                abort()
            except Exception:
                pass


def _torch_direct_store_process_group(
    key: tuple[str, int, str] | None,
    pg: Any,
    store: Any,
) -> Any:
    if key is None:
        return pg
    with _TORCH_DIRECT_PROCESS_GROUP_CACHE_LOCK:
        # ProcessGroupNCCL initializes communicators lazily. Retain its TCPStore for the full
        # cached-group lifetime so bootstrap remains valid.
        _TORCH_DIRECT_PROCESS_GROUP_CACHE[key] = (pg, store)
        _TORCH_DIRECT_PERSISTENT_PROCESS_GROUP_IDS.add(id(pg))
    return pg


def _torch_direct_shutdown_cached_process_groups() -> None:
    with _TORCH_DIRECT_PROCESS_GROUP_CACHE_LOCK:
        entries = list(_TORCH_DIRECT_PROCESS_GROUP_CACHE.values())
        _TORCH_DIRECT_PROCESS_GROUP_CACHE.clear()
        _TORCH_DIRECT_PERSISTENT_PROCESS_GROUP_IDS.clear()
    if not entries:
        return
    for pg, _store in entries:
        _torch_direct_release_process_group(pg)


def _torch_direct_process_group(
    *,
    endpoint: dict[str, Any],
    backend: str,
    pg_rank: int,
    timeout_s: float,
):
    try:
        import torch
        import torch.distributed as dist
    except ModuleNotFoundError as exc:
        raise RuntimeDependencyError("torch_dist activation transfer requires PyTorch") from exc

    if not dist.is_available():
        raise RuntimeDependencyError("torch.distributed is unavailable; cannot use torch_dist activation transfer")
    if backend == "gloo" and not dist.is_gloo_available():
        raise RuntimeDependencyError("torch.distributed Gloo backend is unavailable")
    if backend == "nccl":
        if not dist.is_nccl_available():
            raise RuntimeDependencyError("torch.distributed NCCL backend is unavailable")
        if not torch.cuda.is_available():
            raise RuntimeDependencyError("nccl_direct activation transfer requires CUDA")
    cache_key = _torch_direct_process_group_cache_key(endpoint, backend, pg_rank)
    cached = _torch_direct_cached_process_group(cache_key)
    if cached is not None:
        return cached

    timeout = datetime.timedelta(seconds=max(1.0, float(timeout_s)))
    world_size = int(endpoint.get("world_size", 2))
    master_rank = int(endpoint.get("master_rank", 1))
    if world_size < 2 or not 0 <= int(pg_rank) < world_size or not 0 <= master_rank < world_size:
        raise RuntimeConfigError(
            f"invalid torch_dist group rank={pg_rank}, world_size={world_size}, master_rank={master_rank}"
        )
    is_master = int(pg_rank) == master_rank
    try:
        store = dist.TCPStore(
            str(endpoint["host"]),
            int(endpoint["port"]),
            world_size,
            is_master,
            timeout,
        )
    except Exception as exc:
        role = "master" if is_master else "participant"
        raise RuntimeConfigError(
            f"failed to create torch_dist TCPStore for {role} rank {pg_rank}/{world_size} "
            f"at {endpoint['host']}:{endpoint['port']}: {type(exc).__name__}: {exc}"
        ) from exc
    if backend == "gloo":
        return _torch_direct_store_process_group(
            cache_key, dist.ProcessGroupGloo(store, int(pg_rank), world_size, timeout), store
        )
    try:
        options = dist.ProcessGroupNCCL.Options()
        if hasattr(options, "_timeout"):
            options._timeout = timeout
        pg = dist.ProcessGroupNCCL(store, int(pg_rank), world_size, options)
        return _torch_direct_store_process_group(cache_key, pg, store)
    except Exception as exc:
        raise RuntimeDependencyError(
            "failed to create NCCL ProcessGroup for nccl_direct activation transfer: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _torch_direct_send_file(
    pg: Any,
    backend: str,
    path: Path,
    bytes_: int,
    *,
    ack_endpoint: dict[str, Any] | None = None,
    timeout_s: float = 120.0,
) -> None:
    import torch

    device = torch.device("cuda") if backend == "nccl" else torch.device("cpu")
    remaining = int(bytes_)
    tag = 0
    ack_sock = _torch_direct_connect_ack_socket(ack_endpoint, timeout_s=timeout_s)
    try:
        with path.open("rb") as src:
            while remaining > 0:
                chunk = src.read(min(16 * 1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeConfigError(f"torch_dist activation send ended early for {path}")
                tensor = torch.frombuffer(bytearray(chunk), dtype=torch.uint8)
                if device.type == "cuda":
                    tensor = tensor.to(device=device, non_blocking=False)
                _torch_direct_send_tensor(pg, backend, tensor, 1, tag)
                if ack_sock is not None:
                    _torch_direct_recv_socket_ack(ack_sock)
                elif backend == "nccl":
                    _torch_direct_recv_chunk_ack(pg, backend, tag)
                remaining -= len(chunk)
                tag += 1
    finally:
        if ack_sock is not None:
            ack_sock.close()


def _torch_direct_recv_file(
    pg: Any,
    backend: str,
    destination: Path,
    bytes_: int,
    *,
    ack_endpoint: dict[str, Any] | None = None,
    timeout_s: float = 120.0,
) -> tuple[int, str]:
    import torch

    device = torch.device("cuda") if backend == "nccl" else torch.device("cpu")
    digest = hashlib.sha256()
    remaining = int(bytes_)
    total = 0
    tag = 0
    ack_conn = _torch_direct_accept_ack_socket(ack_endpoint, timeout_s=timeout_s)
    try:
        with destination.open("wb") as dst:
            while remaining > 0:
                chunk_size = min(16 * 1024 * 1024, remaining)
                tensor = torch.empty(chunk_size, dtype=torch.uint8, device=device)
                _torch_direct_recv_tensor(pg, backend, tensor, 0, tag)
                if device.type == "cuda":
                    tensor = tensor.cpu()
                payload = tensor.numpy().tobytes()
                digest.update(payload)
                dst.write(payload)
                if ack_conn is not None:
                    _torch_direct_send_socket_ack(ack_conn)
                elif backend == "nccl":
                    _torch_direct_send_chunk_ack(pg, backend, tag)
                total += len(payload)
                remaining -= len(payload)
                tag += 1
    finally:
        if ack_conn is not None:
            ack_conn.close()
    return total, digest.hexdigest()


def _torch_direct_send_blob(
    pg: Any,
    backend: str,
    payload: bytes,
    sha256: str,
    *,
    ack_endpoint: dict[str, Any] | None = None,
    timeout_s: float = 120.0,
) -> None:
    import torch

    if hashlib.sha256(payload).hexdigest() != sha256:
        raise RuntimeConfigError("torch_dist in-memory activation checksum changed before send")
    header = struct.pack("!Q", len(payload)) + sha256.encode("ascii")
    _torch_direct_send_bytes(pg, backend, header, tag=0)
    device = torch.device("cuda") if backend == "nccl" else torch.device("cpu")
    ack_sock = _torch_direct_connect_ack_socket(ack_endpoint, timeout_s=timeout_s)
    try:
        for tag, offset in enumerate(range(0, len(payload), 16 * 1024 * 1024), start=1):
            chunk = payload[offset : offset + 16 * 1024 * 1024]
            tensor = torch.frombuffer(bytearray(chunk), dtype=torch.uint8)
            if device.type == "cuda":
                tensor = tensor.to(device=device, non_blocking=False)
            _torch_direct_send_tensor(pg, backend, tensor, 1, tag)
            if ack_sock is not None:
                _torch_direct_recv_socket_ack(ack_sock)
            elif backend == "nccl":
                _torch_direct_recv_chunk_ack(pg, backend, tag)
    finally:
        if ack_sock is not None:
            ack_sock.close()


def _torch_direct_recv_blob(
    pg: Any,
    backend: str,
    *,
    ack_endpoint: dict[str, Any] | None = None,
    timeout_s: float = 120.0,
) -> tuple[bytes, str]:
    import torch

    header = _torch_direct_recv_bytes(pg, backend, 72, tag=0)
    bytes_ = struct.unpack("!Q", header[:8])[0]
    try:
        expected_sha256 = header[8:].decode("ascii")
    except UnicodeDecodeError as exc:
        raise RuntimeConfigError("invalid torch_dist in-memory activation checksum header") from exc
    if len(expected_sha256) != 64:
        raise RuntimeConfigError("invalid torch_dist in-memory activation checksum length")
    device = torch.device("cuda") if backend == "nccl" else torch.device("cpu")
    remaining = int(bytes_)
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    ack_conn = _torch_direct_accept_ack_socket(ack_endpoint, timeout_s=timeout_s)
    try:
        tag = 1
        while remaining > 0:
            chunk_size = min(16 * 1024 * 1024, remaining)
            tensor = torch.empty(chunk_size, dtype=torch.uint8, device=device)
            _torch_direct_recv_tensor(pg, backend, tensor, 0, tag)
            if device.type == "cuda":
                tensor = tensor.cpu()
            chunk = tensor.numpy().tobytes()
            chunks.append(chunk)
            digest.update(chunk)
            if ack_conn is not None:
                _torch_direct_send_socket_ack(ack_conn)
            elif backend == "nccl":
                _torch_direct_send_chunk_ack(pg, backend, tag)
            remaining -= len(chunk)
            tag += 1
    finally:
        if ack_conn is not None:
            ack_conn.close()
    sha256 = digest.hexdigest()
    if sha256 != expected_sha256:
        raise RuntimeConfigError(
            f"torch_dist in-memory activation checksum mismatch: {sha256} != {expected_sha256}"
        )
    return b"".join(chunks), sha256


def _torch_direct_send_file_with_metadata(
    pg: Any,
    backend: str,
    path: Path,
    bytes_: int,
    sha256: str,
    metadata: dict[str, Any],
    *,
    ack_endpoint: dict[str, Any] | None = None,
    timeout_s: float = 120.0,
) -> None:
    metadata_payload = json.dumps(metadata, sort_keys=True).encode("utf-8")
    if len(metadata_payload) > 1024 * 1024:
        raise RuntimeConfigError("torch_dist activation slice metadata exceeds 1 MiB")
    header = struct.pack("!QQ", int(bytes_), len(metadata_payload))
    _torch_direct_send_bytes(pg, backend, header, tag=0)
    _torch_direct_send_bytes(pg, backend, metadata_payload, tag=1)
    _torch_direct_send_file_chunks(
        pg,
        backend,
        path,
        int(bytes_),
        start_tag=2,
        ack_endpoint=ack_endpoint,
        timeout_s=timeout_s,
    )


def _torch_direct_recv_file_with_metadata(
    pg: Any,
    backend: str,
    destination: Path,
    *,
    ack_endpoint: dict[str, Any] | None = None,
    timeout_s: float = 120.0,
) -> tuple[int, str, dict[str, Any]]:
    header = _torch_direct_recv_bytes(pg, backend, 16, tag=0)
    bytes_, metadata_len = struct.unpack("!QQ", header)
    if metadata_len > 1024 * 1024:
        raise RuntimeConfigError(f"invalid torch_dist activation slice metadata length {metadata_len}")
    metadata_payload = _torch_direct_recv_bytes(pg, backend, int(metadata_len), tag=1)
    metadata = json.loads(metadata_payload.decode("utf-8")) if metadata_payload else {}
    if not isinstance(metadata, dict) or metadata.get("schema_version") != "rpeg.activation_direct_slice.v1":
        raise RuntimeConfigError("unsupported torch_dist activation slice metadata")
    received_bytes, sha256 = _torch_direct_recv_file_chunks(
        pg,
        backend,
        destination,
        int(bytes_),
        start_tag=2,
        ack_endpoint=ack_endpoint,
        timeout_s=timeout_s,
    )
    expected_sha = str(metadata.get("sha256") or metadata.get("payload_sha256") or "")
    if expected_sha and sha256 != expected_sha:
        raise RuntimeConfigError(
            f"torch_dist activation checksum mismatch while receiving {destination}: {sha256} != {expected_sha}"
        )
    return received_bytes, sha256, metadata


def _torch_direct_collective_broadcast_tensor(
    pg: Any,
    tensor: Any,
    *,
    root_rank: int,
) -> None:
    import torch.distributed as dist

    options = dist.BroadcastOptions()
    options.rootRank = int(root_rank)
    options.rootTensor = 0
    pg.broadcast([tensor], options).wait()


def _torch_direct_collective_broadcast_blob(
    pg: Any,
    backend: str,
    *,
    pg_rank: int,
    root_rank: int,
    payload: bytes | None = None,
    sha256: str | None = None,
) -> tuple[bytes, str]:
    import torch

    pg_rank = int(pg_rank)
    root_rank = int(root_rank)
    is_root = pg_rank == root_rank
    if is_root and payload is None:
        raise RuntimeConfigError(f"collective root rank {root_rank} is missing its activation payload")
    if not is_root and payload is not None:
        raise RuntimeConfigError(
            f"collective non-root rank {pg_rank} provided payload for root rank {root_rank}"
        )
    if is_root:
        assert payload is not None
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if sha256 is not None and str(sha256) != actual_sha256:
            raise RuntimeConfigError("collective activation checksum changed before broadcast")
        expected_sha256 = actual_sha256
        header_payload = struct.pack("!Q", len(payload)) + expected_sha256.encode("ascii")
    else:
        expected_sha256 = ""
        header_payload = bytes(72)
    device = torch.device("cuda") if backend == "nccl" else torch.device("cpu")
    header = torch.frombuffer(bytearray(header_payload), dtype=torch.uint8)
    if device.type == "cuda":
        header = header.to(device=device, non_blocking=False)
    _torch_direct_collective_broadcast_tensor(pg, header, root_rank=root_rank)
    if device.type == "cuda":
        header = header.cpu()
    received_header = header.numpy().tobytes()
    bytes_ = struct.unpack("!Q", received_header[:8])[0]
    try:
        announced_sha256 = received_header[8:].decode("ascii")
    except UnicodeDecodeError as exc:
        raise RuntimeConfigError("invalid collective activation checksum header") from exc
    if len(announced_sha256) != 64:
        raise RuntimeConfigError("invalid collective activation checksum length")
    if is_root and (bytes_ != len(payload) or announced_sha256 != expected_sha256):
        raise RuntimeConfigError("collective activation header changed during broadcast")
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    for offset in range(0, int(bytes_), 16 * 1024 * 1024):
        chunk_size = min(16 * 1024 * 1024, int(bytes_) - offset)
        if is_root:
            assert payload is not None
            tensor = torch.frombuffer(bytearray(payload[offset : offset + chunk_size]), dtype=torch.uint8)
        else:
            tensor = torch.empty(chunk_size, dtype=torch.uint8)
        if device.type == "cuda":
            tensor = tensor.to(device=device, non_blocking=False)
        _torch_direct_collective_broadcast_tensor(pg, tensor, root_rank=root_rank)
        if device.type == "cuda":
            tensor = tensor.cpu()
        chunk = tensor.numpy().tobytes()
        chunks.append(chunk)
        digest.update(chunk)
    received_payload = b"".join(chunks)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != announced_sha256:
        raise RuntimeConfigError(
            f"collective activation checksum mismatch: {actual_sha256} != {announced_sha256}"
        )
    return received_payload, actual_sha256

def _torch_direct_send_tensor(
    pg: Any,
    backend: str,
    tensor: Any,
    destination_rank: int,
    tag: int,
) -> None:
    if backend != "nccl":
        pg.send([tensor], int(destination_rank), int(tag)).wait()
        return
    import torch.distributed as dist

    if int(destination_rank) not in {0, 1}:
        raise RuntimeConfigError("two-rank NCCL direct send requires destination rank 0 or 1")
    options = dist.BroadcastOptions()
    options.rootRank = 1 - int(destination_rank)
    options.rootTensor = 0
    pg.broadcast([tensor], options).wait()


def _torch_direct_recv_tensor(
    pg: Any,
    backend: str,
    tensor: Any,
    source_rank: int,
    tag: int,
) -> None:
    if backend != "nccl":
        pg.recv([tensor], int(source_rank), int(tag)).wait()
        return
    import torch.distributed as dist

    options = dist.BroadcastOptions()
    options.rootRank = int(source_rank)
    options.rootTensor = 0
    pg.broadcast([tensor], options).wait()


def _torch_direct_send_bytes(
    pg: Any,
    backend: str,
    payload: bytes,
    *,
    tag: int,
    destination_rank: int = 1,
) -> None:
    import torch

    device = torch.device("cuda") if backend == "nccl" else torch.device("cpu")
    tensor = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    if device.type == "cuda":
        tensor = tensor.to(device=device, non_blocking=False)
    _torch_direct_send_tensor(pg, backend, tensor, destination_rank, tag)


def _torch_direct_recv_bytes(
    pg: Any,
    backend: str,
    nbytes: int,
    *,
    tag: int,
    source_rank: int = 0,
) -> bytes:
    import torch

    device = torch.device("cuda") if backend == "nccl" else torch.device("cpu")
    tensor = torch.empty(int(nbytes), dtype=torch.uint8, device=device)
    _torch_direct_recv_tensor(pg, backend, tensor, source_rank, tag)
    if device.type == "cuda":
        tensor = tensor.cpu()
    return tensor.numpy().tobytes()


def _torch_direct_coalesced_p2p(
    pg: Any,
    operations: list[tuple[str, Any, int, int]],
) -> None:
    """
    Run one NCCL P2P phase inside ncclGroupStart/End.
    """
    if not operations:
        return
    # The Python ProcessGroup wrapper used by dist.group.WORLD in PyTorch 2.9 requires an explicit
    # device, while a directly constructed ProcessGroupNCCL on older releases exposes the
    # no-argument overload. Remember which overload succeeded so start/end always stay paired.
    device = operations[0][1].device
    device_aware = True
    try:
        pg._start_coalescing(device)
    except TypeError:
        device_aware = False
        pg._start_coalescing()
    try:
        for kind, tensor, peer_rank, tag in operations:
            if kind == "send":
                pg.send([tensor], int(peer_rank), int(tag))
            elif kind == "recv":
                pg.recv([tensor], int(peer_rank), int(tag))
            else:
                raise RuntimeConfigError(f"unsupported coalesced P2P operation {kind!r}")
    finally:
        work = pg._end_coalescing(device) if device_aware else pg._end_coalescing()
    work.wait()


def _torch_direct_exchange_tensor_records_coalesced(
    pg: Any,
    outgoing: list[tuple[dict[str, Any], dict[str, Any]]],
    incoming: list[dict[str, Any]],
    *,
    device: Any,
    schema_version: str = _UCX_TENSOR_STREAM_SCHEMA,
    expected_schema: str = _UCX_TENSOR_STREAM_SCHEMA,
) -> tuple[list[tuple[dict[str, Any], int, str]], int]:
    """
    All-peer tensor-record exchange on one persistent NCCL communicator.

    Every phase posts all sends and receives inside one NCCL group. This is required for an
    in-place all-rank layout transition: running the existing one-way sender and receiver helpers
    sequentially would deadlock when every rank is both a producer and a consumer.
    """
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeConfigError("coalesced NCCL tensor exchange requires CUDA")

    send_items = []
    sent_wire_bytes = 0
    for endpoint, record in outgoing:
        manifest, tensor_bytes = _tensor_stream_manifest(
            record,
            schema_version=schema_version,
        )
        manifest_bytes = _tensor_stream_manifest_bytes(manifest)
        groups = _tensor_stream_groups(manifest["entries"], int(manifest["bucket_bytes"]))
        sent_wire_bytes += int(tensor_bytes) + len(manifest_bytes)
        send_items.append({
            "endpoint": endpoint,
            "record": record,
            "manifest_bytes": manifest_bytes,
            "groups": groups,
        })

    send_headers = [
        torch.frombuffer(
            bytearray(struct.pack("!Q", len(item["manifest_bytes"]))),
            dtype=torch.uint8,
        ).to(device=device, non_blocking=False)
        for item in send_items
    ]
    recv_headers = [torch.empty(8, dtype=torch.uint8, device=device) for _item in incoming]
    _torch_direct_coalesced_p2p(
        pg,
        [
            ("send", tensor, int(item["endpoint"]["destination_group_rank"]), 0)
            for tensor, item in zip(send_headers, send_items)
        ]
        + [
            ("recv", tensor, int(endpoint["source_group_rank"]), 0)
            for tensor, endpoint in zip(recv_headers, incoming)
        ],
    )
    manifest_lengths = [
        struct.unpack("!Q", tensor.cpu().numpy().tobytes())[0]
        for tensor in recv_headers
    ]
    if any(length > 16 * 1024 * 1024 for length in manifest_lengths):
        raise RuntimeConfigError(
            f"invalid coalesced tensor-native manifest lengths {manifest_lengths}"
        )

    send_manifests = [
        torch.frombuffer(bytearray(item["manifest_bytes"]), dtype=torch.uint8).to(
            device=device,
            non_blocking=False,
        )
        for item in send_items
    ]
    recv_manifests = [
        torch.empty(int(length), dtype=torch.uint8, device=device)
        for length in manifest_lengths
    ]
    _torch_direct_coalesced_p2p(
        pg,
        [
            ("send", tensor, int(item["endpoint"]["destination_group_rank"]), 1)
            for tensor, item in zip(send_manifests, send_items)
        ]
        + [
            ("recv", tensor, int(endpoint["source_group_rank"]), 1)
            for tensor, endpoint in zip(recv_manifests, incoming)
        ],
    )

    recv_items = []
    for manifest_tensor in recv_manifests:
        manifest_bytes = manifest_tensor.cpu().numpy().tobytes()
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeConfigError("invalid coalesced tensor-native manifest") from exc
        if not isinstance(manifest, dict) or manifest.get("schema_version") != expected_schema:
            raise RuntimeConfigError(
                "unsupported coalesced tensor-native schema: "
                f"expected {expected_schema!r}, got "
                f"{manifest.get('schema_version') if isinstance(manifest, dict) else None!r}"
            )
        metadata = manifest.get("record")
        entries = manifest.get("entries")
        if not isinstance(metadata, dict) or not isinstance(entries, list):
            raise RuntimeConfigError("malformed coalesced tensor-native weight manifest")
        fragments: dict[str, dict[str, Any]] = {}
        tensors_by_name: dict[str, Any] = {}
        tensor_bytes = 0
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                raise RuntimeConfigError("malformed coalesced tensor-native weight entry")
            name = entry["name"]
            if name in fragments:
                raise RuntimeConfigError(f"duplicate coalesced tensor-native entry {name!r}")
            kind = str(entry.get("kind"))
            if kind == "none":
                value = None
            elif kind == "tensor":
                shape = entry.get("shape")
                if not isinstance(shape, list) or any(
                    not isinstance(dim, int) or dim < 0 for dim in shape
                ):
                    raise RuntimeConfigError(
                        f"invalid coalesced tensor-native shape for {name!r}"
                    )
                value = torch.empty(
                    tuple(shape),
                    dtype=_torch_dtype_from_wire(str(entry.get("dtype"))),
                    device=device,
                )
                nbytes = int(entry.get("nbytes", -1))
                expected_nbytes = int(value.numel()) * int(value.element_size())
                if nbytes != expected_nbytes:
                    raise RuntimeConfigError(
                        f"coalesced tensor byte count mismatch for {name!r}: "
                        f"{nbytes} != {expected_nbytes}"
                    )
                tensors_by_name[name] = value
                tensor_bytes += nbytes
            else:
                raise RuntimeConfigError(
                    f"unsupported coalesced tensor-native entry kind {kind!r}"
                )
            fragments[name] = {
                "value": value,
                "mode": str(entry.get("mode")),
                "order": int(entry.get("order", 0)),
                "stride": int(entry.get("stride", 1)),
                "dim": None if entry.get("dim") is None else int(entry["dim"]),
            }
        bucket_bytes = int(manifest.get("bucket_bytes", 0))
        if bucket_bytes < 1024 * 1024:
            raise RuntimeConfigError(
                f"invalid coalesced tensor-native bucket size {bucket_bytes}"
            )
        recv_items.append({
            "manifest_bytes": manifest_bytes,
            "metadata": metadata,
            "fragments": fragments,
            "tensors_by_name": tensors_by_name,
            "tensor_bytes": tensor_bytes,
            "groups": _tensor_stream_groups(entries, bucket_bytes),
        })

    send_markers = [torch.zeros(1, dtype=torch.uint8, device=device) for _item in send_items]
    recv_markers = [torch.empty(1, dtype=torch.uint8, device=device) for _item in recv_items]
    _torch_direct_coalesced_p2p(
        pg,
        [
            ("send", marker, int(item["endpoint"]["destination_group_rank"]), 2)
            for marker, item in zip(send_markers, send_items)
        ]
        + [
            ("recv", marker, int(endpoint["source_group_rank"]), 2)
            for marker, endpoint in zip(recv_markers, incoming)
        ],
    )
    if any(int(marker.item()) != 0 for marker in recv_markers):
        raise RuntimeConfigError("invalid coalesced tensor-native phase marker")

    max_groups = max(
        [len(item["groups"]) for item in send_items]
        + [len(item["groups"]) for item in recv_items],
        default=0,
    )
    for group_index in range(max_groups):
        operations = []
        keepalive = []
        packed_receives = []
        for item in send_items:
            if group_index >= len(item["groups"]):
                continue
            group = item["groups"][group_index]
            tensor = _pack_tensor_stream_group(group, item["record"]["fragments"])
            if tensor.device.type != "cuda":
                tensor = tensor.to(device=device, non_blocking=False)
            keepalive.append(tensor)
            operations.append((
                "send",
                tensor,
                int(item["endpoint"]["destination_group_rank"]),
                3 + group_index,
            ))
        for endpoint, item in zip(incoming, recv_items):
            if group_index >= len(item["groups"]):
                continue
            group = item["groups"][group_index]
            if len(group) == 1:
                tensor = item["tensors_by_name"][group[0]["name"]]
            else:
                group_nbytes = sum(int(entry["nbytes"]) for entry in group)
                tensor = torch.empty(group_nbytes, dtype=torch.uint8, device=device)
                packed_receives.append((item, group, tensor))
            operations.append((
                "recv",
                tensor,
                int(endpoint["source_group_rank"]),
                3 + group_index,
            ))
        _torch_direct_coalesced_p2p(pg, operations)
        for item, group, packed in packed_receives:
            offset = 0
            for entry in group:
                destination = item["tensors_by_name"][entry["name"]]
                nbytes = int(entry["nbytes"])
                destination.view(dtype=torch.uint8).reshape(-1).copy_(
                    packed[offset:offset + nbytes]
                )
                offset += nbytes
    torch.cuda.synchronize(device)
    results = []
    for item in recv_items:
        record = dict(item["metadata"])
        record["fragments"] = item["fragments"]
        manifest_bytes = item["manifest_bytes"]
        results.append((
            record,
            int(item["tensor_bytes"]) + len(manifest_bytes),
            hashlib.sha256(manifest_bytes).hexdigest(),
        ))
    return results, sent_wire_bytes


def _torch_direct_send_tensor_records_coalesced(
    pg: Any,
    records: list[tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    """
    Send routed records on one persistent all-rank NCCL group.
    """
    import torch

    prepared = []
    for endpoint, record in records:
        manifest, _tensor_bytes = _tensor_stream_manifest(record)
        manifest_bytes = _tensor_stream_manifest_bytes(manifest)
        groups = _tensor_stream_groups(manifest["entries"], int(manifest["bucket_bytes"]))
        prepared.append((endpoint, record, manifest_bytes, groups))

    headers = [
        torch.frombuffer(
            bytearray(struct.pack("!Q", len(manifest_bytes))),
            dtype=torch.uint8,
        ).to(device=torch.device("cuda"), non_blocking=False)
        for _endpoint, _record, manifest_bytes, _groups in prepared
    ]
    _torch_direct_coalesced_p2p(
        pg,
        [
            ("send", tensor, int(endpoint["destination_group_rank"]), 0)
            for tensor, (endpoint, _record, _manifest, _groups)
            in zip(headers, prepared)
        ],
    )
    manifests = [
        torch.frombuffer(bytearray(manifest_bytes), dtype=torch.uint8).to(
            device=torch.device("cuda"),
            non_blocking=False,
        )
        for _endpoint, _record, manifest_bytes, _groups in prepared
    ]
    _torch_direct_coalesced_p2p(
        pg,
        [
            ("send", tensor, int(endpoint["destination_group_rank"]), 1)
            for tensor, (endpoint, _record, _manifest, _groups)
            in zip(manifests, prepared)
        ],
    )
    markers = [
        torch.zeros(1, dtype=torch.uint8, device="cuda")
        for _item in prepared
    ]
    _torch_direct_coalesced_p2p(
        pg,
        [
            (
                "send",
                marker,
                int(item[0]["destination_group_rank"]),
                2,
            )
            for marker, item in zip(markers, prepared)
        ],
    )
    max_groups = max((len(item[3]) for item in prepared), default=0)
    for group_index in range(max_groups):
        operations = []
        keepalive = []
        for endpoint, record, _manifest_bytes, groups in prepared:
            if group_index >= len(groups):
                continue
            tensor = _pack_tensor_stream_group(groups[group_index], record["fragments"])
            if tensor.device.type != "cuda":
                tensor = tensor.to(device=torch.device("cuda"), non_blocking=False)
            keepalive.append(tensor)
            operations.append((
                "send",
                tensor,
                int(endpoint["destination_group_rank"]),
                3 + group_index,
            ))
        _torch_direct_coalesced_p2p(pg, operations)


def _torch_direct_recv_tensor_records_coalesced(
    pg: Any,
    endpoints: list[dict[str, Any]],
    *,
    device: Any,
) -> list[tuple[dict[str, Any], int, str]]:
    """
    Receive routed records on one persistent all-rank NCCL group.
    """
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeConfigError("coalesced NCCL weight transfer requires CUDA")
    headers = [torch.empty(8, dtype=torch.uint8, device=device) for _ in endpoints]
    _torch_direct_coalesced_p2p(
        pg,
        [
            ("recv", tensor, int(endpoint["source_group_rank"]), 0)
            for tensor, endpoint in zip(headers, endpoints)
        ],
    )
    manifest_lengths = [
        struct.unpack("!Q", tensor.cpu().numpy().tobytes())[0]
        for tensor in headers
    ]
    if any(length > 16 * 1024 * 1024 for length in manifest_lengths):
        raise RuntimeConfigError(
            f"invalid coalesced tensor-native manifest lengths {manifest_lengths}"
        )
    manifest_tensors = [
        torch.empty(int(length), dtype=torch.uint8, device=device)
        for length in manifest_lengths
    ]
    _torch_direct_coalesced_p2p(
        pg,
        [
            ("recv", tensor, int(endpoint["source_group_rank"]), 1)
            for tensor, endpoint in zip(manifest_tensors, endpoints)
        ],
    )

    prepared = []
    for manifest_tensor in manifest_tensors:
        manifest_bytes = manifest_tensor.cpu().numpy().tobytes()
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeConfigError("invalid coalesced tensor-native manifest") from exc
        if not isinstance(manifest, dict) or manifest.get("schema_version") != _UCX_TENSOR_STREAM_SCHEMA:
            raise RuntimeConfigError("unsupported coalesced tensor-native weight schema")
        metadata = manifest.get("record")
        entries = manifest.get("entries")
        if not isinstance(metadata, dict) or not isinstance(entries, list):
            raise RuntimeConfigError("malformed coalesced tensor-native weight manifest")
        fragments: dict[str, dict[str, Any]] = {}
        tensors_by_name: dict[str, Any] = {}
        tensor_bytes = 0
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                raise RuntimeConfigError("malformed coalesced tensor-native weight entry")
            name = entry["name"]
            if name in fragments:
                raise RuntimeConfigError(f"duplicate coalesced tensor-native entry {name!r}")
            kind = str(entry.get("kind"))
            if kind == "none":
                value = None
            elif kind == "tensor":
                shape = entry.get("shape")
                if not isinstance(shape, list) or any(
                    not isinstance(dim, int) or dim < 0 for dim in shape
                ):
                    raise RuntimeConfigError(
                        f"invalid coalesced tensor-native shape for {name!r}"
                    )
                value = torch.empty(
                    tuple(shape),
                    dtype=_torch_dtype_from_wire(str(entry.get("dtype"))),
                    device=device,
                )
                nbytes = int(entry.get("nbytes", -1))
                expected_nbytes = int(value.numel()) * int(value.element_size())
                if nbytes != expected_nbytes:
                    raise RuntimeConfigError(
                        f"coalesced tensor byte count mismatch for {name!r}: "
                        f"{nbytes} != {expected_nbytes}"
                    )
                tensors_by_name[name] = value
                tensor_bytes += nbytes
            else:
                raise RuntimeConfigError(
                    f"unsupported coalesced tensor-native entry kind {kind!r}"
                )
            fragments[name] = {
                "value": value,
                "mode": str(entry.get("mode")),
                "order": int(entry.get("order", 0)),
                "stride": int(entry.get("stride", 1)),
                "dim": None if entry.get("dim") is None else int(entry["dim"]),
            }
        bucket_bytes = int(manifest.get("bucket_bytes", 0))
        if bucket_bytes < 1024 * 1024:
            raise RuntimeConfigError(
                f"invalid coalesced tensor-native bucket size {bucket_bytes}"
            )
        prepared.append(
            {
                "manifest_bytes": manifest_bytes,
                "metadata": metadata,
                "fragments": fragments,
                "tensors_by_name": tensors_by_name,
                "tensor_bytes": tensor_bytes,
                "groups": _tensor_stream_groups(entries, bucket_bytes),
            }
        )

    markers = [torch.empty(1, dtype=torch.uint8, device=device) for _item in prepared]
    _torch_direct_coalesced_p2p(
        pg,
        [
            ("recv", marker, int(endpoint["source_group_rank"]), 2)
            for marker, endpoint in zip(markers, endpoints)
        ],
    )
    if any(int(marker.item()) != 0 for marker in markers):
        raise RuntimeConfigError("invalid coalesced tensor-native phase marker")
    max_groups = max((len(item["groups"]) for item in prepared), default=0)
    for group_index in range(max_groups):
        operations = []
        packed_groups = []
        for endpoint, item in zip(endpoints, prepared):
            if group_index >= len(item["groups"]):
                continue
            group = item["groups"][group_index]
            if len(group) == 1:
                tensor = item["tensors_by_name"][group[0]["name"]]
            else:
                group_nbytes = sum(int(entry["nbytes"]) for entry in group)
                tensor = torch.empty(group_nbytes, dtype=torch.uint8, device=device)
                packed_groups.append((item, group, tensor))
            operations.append((
                "recv",
                tensor,
                int(endpoint["source_group_rank"]),
                3 + group_index,
            ))
        _torch_direct_coalesced_p2p(pg, operations)
        for item, group, packed in packed_groups:
            offset = 0
            for entry in group:
                destination = item["tensors_by_name"][entry["name"]]
                nbytes = int(entry["nbytes"])
                destination.view(dtype=torch.uint8).reshape(-1).copy_(
                    packed[offset:offset + nbytes]
                )
                offset += nbytes
    torch.cuda.synchronize(device)
    results = []
    for item in prepared:
        record = dict(item["metadata"])
        record["fragments"] = item["fragments"]
        manifest_bytes = item["manifest_bytes"]
        results.append(
            (
                record,
                int(item["tensor_bytes"]) + len(manifest_bytes),
                hashlib.sha256(manifest_bytes).hexdigest(),
            )
        )
    return results


def _torch_direct_send_tensor_record(
    pg: Any,
    record: dict[str, Any],
    backend: str,
    *,
    destination_rank: int = 1,
    schema_version: str = _UCX_TENSOR_STREAM_SCHEMA,
) -> tuple[int, str]:
    import torch

    manifest, tensor_bytes = _tensor_stream_manifest(
        record,
        schema_version=schema_version,
    )
    manifest_bytes = _tensor_stream_manifest_bytes(manifest)
    _torch_direct_send_bytes(
        pg,
        backend,
        struct.pack("!Q", len(manifest_bytes)),
        tag=0,
        destination_rank=destination_rank,
    )
    _torch_direct_send_bytes(
        pg,
        backend,
        manifest_bytes,
        tag=1,
        destination_rank=destination_rank,
    )
    fragments = record["fragments"]
    for tag, group in enumerate(
        _tensor_stream_groups(manifest["entries"], int(manifest["bucket_bytes"])),
        start=2,
    ):
        packed = _pack_tensor_stream_group(group, fragments)
        if backend == "nccl" and packed.device.type != "cuda":
            packed = packed.to(device=torch.device("cuda"), non_blocking=False)
        elif backend == "gloo" and packed.device.type != "cpu":
            packed = packed.to(device=torch.device("cpu"), non_blocking=False)
        _torch_direct_send_tensor(pg, backend, packed, destination_rank, tag)
    return tensor_bytes + len(manifest_bytes), hashlib.sha256(manifest_bytes).hexdigest()


def _torch_direct_recv_tensor_record(
    pg: Any,
    backend: str,
    *,
    device: Any,
    source_rank: int = 0,
    expected_schema: str = _UCX_TENSOR_STREAM_SCHEMA,
) -> tuple[dict[str, Any], int, str]:
    import torch

    manifest_len = struct.unpack(
        "!Q",
        _torch_direct_recv_bytes(
            pg,
            backend,
            8,
            tag=0,
            source_rank=source_rank,
        ),
    )[0]
    if manifest_len > 16 * 1024 * 1024:
        raise RuntimeConfigError(f"invalid tensor-native weight manifest length {manifest_len}")
    manifest_bytes = _torch_direct_recv_bytes(
        pg,
        backend,
        int(manifest_len),
        tag=1,
        source_rank=source_rank,
    )
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeConfigError("invalid tensor-native weight manifest") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != expected_schema:
        raise RuntimeConfigError(
            "unsupported tensor-native stream schema: "
            f"expected {expected_schema!r}, got "
            f"{manifest.get('schema_version') if isinstance(manifest, dict) else None!r}"
        )
    metadata = manifest.get("record")
    entries = manifest.get("entries")
    if not isinstance(metadata, dict) or not isinstance(entries, list):
        raise RuntimeConfigError("malformed tensor-native weight manifest")
    device = torch.device(device)
    if backend == "nccl" and device.type != "cuda":
        raise RuntimeConfigError("nccl_direct tensor-native weights require a CUDA destination model")
    fragments: dict[str, dict[str, Any]] = {}
    tensors_by_name: dict[str, Any] = {}
    tensor_bytes = 0
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise RuntimeConfigError("malformed tensor-native weight entry")
        name = entry["name"]
        if name in fragments:
            raise RuntimeConfigError(f"duplicate tensor-native weight entry {name!r}")
        kind = str(entry.get("kind"))
        if kind == "none":
            value = None
        elif kind == "tensor":
            shape = entry.get("shape")
            if not isinstance(shape, list) or any(not isinstance(dim, int) or dim < 0 for dim in shape):
                raise RuntimeConfigError(f"invalid tensor-native weight shape for {name!r}")
            value = torch.empty(
                tuple(shape),
                dtype=_torch_dtype_from_wire(str(entry.get("dtype"))),
                device=device,
            )
            nbytes = int(entry.get("nbytes", -1))
            expected_nbytes = int(value.numel()) * int(value.element_size())
            if nbytes != expected_nbytes:
                raise RuntimeConfigError(
                    f"tensor-native weight byte count mismatch for {name!r}: {nbytes} != {expected_nbytes}"
                )
            tensors_by_name[name] = value
            tensor_bytes += nbytes
        else:
            raise RuntimeConfigError(f"unsupported tensor-native weight entry kind {kind!r}")
        fragments[name] = {
            "value": value,
            "mode": str(entry.get("mode")),
            "order": int(entry.get("order", 0)),
            "stride": int(entry.get("stride", 1)),
            "dim": None if entry.get("dim") is None else int(entry["dim"]),
        }
    bucket_bytes = int(manifest.get("bucket_bytes", 0))
    if bucket_bytes < 1024 * 1024:
        raise RuntimeConfigError(f"invalid tensor-native NCCL bucket size {bucket_bytes}")
    for tag, group in enumerate(_tensor_stream_groups(entries, bucket_bytes), start=2):
        group_nbytes = sum(int(entry["nbytes"]) for entry in group)
        if len(group) == 1:
            destination = tensors_by_name[group[0]["name"]]
            _torch_direct_recv_tensor(pg, backend, destination, source_rank, tag)
            continue
        packed = torch.empty(group_nbytes, dtype=torch.uint8, device=device)
        _torch_direct_recv_tensor(pg, backend, packed, source_rank, tag)
        offset = 0
        for entry in group:
            destination = tensors_by_name[entry["name"]]
            nbytes = int(entry["nbytes"])
            destination.view(dtype=torch.uint8).reshape(-1).copy_(packed[offset:offset + nbytes])
            offset += nbytes
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    record = dict(metadata)
    record["fragments"] = fragments
    return (
        record,
        tensor_bytes + len(manifest_bytes),
        hashlib.sha256(manifest_bytes).hexdigest(),
    )


def _torch_direct_send_file_chunks(
    pg: Any,
    backend: str,
    path: Path,
    bytes_: int,
    *,
    start_tag: int,
    ack_endpoint: dict[str, Any] | None = None,
    timeout_s: float = 120.0,
) -> None:
    import torch

    device = torch.device("cuda") if backend == "nccl" else torch.device("cpu")
    remaining = int(bytes_)
    tag = int(start_tag)
    ack_sock = _torch_direct_connect_ack_socket(ack_endpoint, timeout_s=timeout_s)
    try:
        with path.open("rb") as src:
            while remaining > 0:
                chunk = src.read(min(16 * 1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeConfigError(f"torch_dist activation send ended early for {path}")
                tensor = torch.frombuffer(bytearray(chunk), dtype=torch.uint8)
                if device.type == "cuda":
                    tensor = tensor.to(device=device, non_blocking=False)
                _torch_direct_send_tensor(pg, backend, tensor, 1, tag)
                if ack_sock is not None:
                    _torch_direct_recv_socket_ack(ack_sock)
                elif backend == "nccl":
                    _torch_direct_recv_chunk_ack(pg, backend, tag)
                remaining -= len(chunk)
                tag += 1
    finally:
        if ack_sock is not None:
            ack_sock.close()


def _torch_direct_recv_file_chunks(
    pg: Any,
    backend: str,
    destination: Path,
    bytes_: int,
    *,
    start_tag: int,
    ack_endpoint: dict[str, Any] | None = None,
    timeout_s: float = 120.0,
) -> tuple[int, str]:
    import torch

    device = torch.device("cuda") if backend == "nccl" else torch.device("cpu")
    digest = hashlib.sha256()
    remaining = int(bytes_)
    total = 0
    tag = int(start_tag)
    ack_conn = _torch_direct_accept_ack_socket(ack_endpoint, timeout_s=timeout_s)
    try:
        with destination.open("wb") as dst:
            while remaining > 0:
                chunk_size = min(16 * 1024 * 1024, remaining)
                tensor = torch.empty(chunk_size, dtype=torch.uint8, device=device)
                _torch_direct_recv_tensor(pg, backend, tensor, 0, tag)
                if device.type == "cuda":
                    tensor = tensor.cpu()
                payload = tensor.numpy().tobytes()
                digest.update(payload)
                dst.write(payload)
                if ack_conn is not None:
                    _torch_direct_send_socket_ack(ack_conn)
                elif backend == "nccl":
                    _torch_direct_send_chunk_ack(pg, backend, tag)
                total += len(payload)
                remaining -= len(payload)
                tag += 1
    finally:
        if ack_conn is not None:
            ack_conn.close()
    return total, digest.hexdigest()


_TORCH_DIRECT_ACK_TAG_BASE = 1_000_000
_TORCH_DIRECT_ACK_PAYLOAD = b"\x01"


def _torch_direct_ack_tag(data_tag: int) -> int:
    return _TORCH_DIRECT_ACK_TAG_BASE + int(data_tag)


def _torch_direct_send_chunk_ack(pg: Any, backend: str, data_tag: int) -> None:
    _torch_direct_send_bytes(
        pg,
        backend,
        _TORCH_DIRECT_ACK_PAYLOAD,
        tag=_torch_direct_ack_tag(data_tag),
        destination_rank=0,
    )


def _torch_direct_recv_chunk_ack(pg: Any, backend: str, data_tag: int) -> None:
    payload = _torch_direct_recv_bytes(
        pg,
        backend,
        len(_TORCH_DIRECT_ACK_PAYLOAD),
        tag=_torch_direct_ack_tag(data_tag),
        source_rank=1,
    )
    if payload != _TORCH_DIRECT_ACK_PAYLOAD:
        raise RuntimeConfigError("invalid torch_dist direct chunk acknowledgement")


def _torch_direct_ack_endpoint(endpoint: dict[str, Any] | None) -> tuple[str, int] | None:
    if not endpoint or endpoint.get("ack_port") is None:
        return None
    host = str(endpoint.get("host") or "")
    if not host:
        raise RuntimeConfigError("direct ack endpoint is missing host")
    return host, int(endpoint["ack_port"])


def _torch_direct_connect_ack_socket(
    endpoint: dict[str, Any] | None,
    *,
    timeout_s: float,
) -> socket.socket | None:
    address = _torch_direct_ack_endpoint(endpoint)
    if address is None:
        return None
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    last_exc: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            sock = socket.create_connection(address, timeout=min(5.0, max(0.1, deadline - time.monotonic())))
            sock.settimeout(max(1.0, float(timeout_s)))
            return sock
        except OSError as exc:
            last_exc = exc
            time.sleep(0.05)
    raise RuntimeConfigError(f"timed out connecting torch_dist direct ack socket at {address}: {last_exc}")


def _torch_direct_accept_ack_socket(
    endpoint: dict[str, Any] | None,
    *,
    timeout_s: float,
) -> socket.socket | None:
    address = _torch_direct_ack_endpoint(endpoint)
    if address is None:
        return None
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.settimeout(max(1.0, float(timeout_s)))
        server.bind(address)
        server.listen(1)
        conn, _peer = server.accept()
        conn.settimeout(max(1.0, float(timeout_s)))
        return conn
    except OSError as exc:
        raise RuntimeConfigError(f"failed to accept torch_dist direct ack socket at {address}: {exc}") from exc
    finally:
        server.close()


def _torch_direct_send_socket_ack(sock: socket.socket) -> None:
    sock.sendall(_TORCH_DIRECT_ACK_PAYLOAD)


def _torch_direct_recv_socket_ack(sock: socket.socket) -> None:
    payload = bytearray()
    while len(payload) < len(_TORCH_DIRECT_ACK_PAYLOAD):
        chunk = sock.recv(len(_TORCH_DIRECT_ACK_PAYLOAD) - len(payload))
        if not chunk:
            raise RuntimeConfigError("torch_dist direct ack socket closed early")
        payload.extend(chunk)
    if bytes(payload) != _TORCH_DIRECT_ACK_PAYLOAD:
        raise RuntimeConfigError("invalid torch_dist direct socket acknowledgement")


def _torch_direct_barrier(pg: Any) -> None:
    # P2P send/recv waits already prove byte delivery. A follow-up NCCL barrier can hang on
    # short-lived ad hoc process groups during repeated transfers.
    if id(pg) in _TORCH_DIRECT_PERSISTENT_PROCESS_GROUP_IDS:
        return
    del pg


def _torch_direct_close(pg: Any) -> None:
    if id(pg) in _TORCH_DIRECT_PERSISTENT_PROCESS_GROUP_IDS:
        return
    try:
        import torch.distributed as dist
    except ModuleNotFoundError:
        del pg
        return

    try:
        shutdown = getattr(pg, "shutdown", None)
        if callable(shutdown):
            shutdown()
        else:
            dist.destroy_process_group(pg)
    except Exception:
        abort = getattr(pg, "abort", None)
        if callable(abort):
            try:
                abort()
            except Exception:
                pass
    finally:
        del pg


def _socket_send_activation_file(sock: socket.socket, path: Path, bytes_: int, sha256: str) -> None:
    header = struct.pack("!Q", int(bytes_)) + bytes.fromhex(sha256)
    sock.sendall(header)
    with path.open("rb") as src:
        offset = 0
        while offset < bytes_:
            try:
                sent = sock.sendfile(src, offset, bytes_ - offset)
            except (AttributeError, OSError):
                src.seek(offset)
                chunk = src.read(min(16 * 1024 * 1024, bytes_ - offset))
                if not chunk:
                    raise RuntimeConfigError(f"direct activation send ended early for {path}")
                sock.sendall(chunk)
                sent = len(chunk)
            if sent is None:
                sent = bytes_ - offset
            if sent <= 0:
                raise RuntimeConfigError(f"direct activation send made no progress for {path}")
            offset += int(sent)


def _socket_recv_activation_file(sock: socket.socket, destination: Path) -> tuple[int, str]:
    header = _socket_recv_exact(sock, 40)
    bytes_ = struct.unpack("!Q", header[:8])[0]
    expected_sha = header[8:].hex()
    digest = hashlib.sha256()
    remaining = int(bytes_)
    total = 0
    with destination.open("wb") as dst:
        while remaining > 0:
            chunk = sock.recv(min(16 * 1024 * 1024, remaining))
            if not chunk:
                raise RuntimeConfigError(f"direct activation receive ended early for {destination}")
            digest.update(chunk)
            dst.write(chunk)
            total += len(chunk)
            remaining -= len(chunk)
    sha256 = digest.hexdigest()
    if sha256 != expected_sha:
        raise RuntimeConfigError(f"direct activation payload checksum mismatch: {sha256} != {expected_sha}")
    return total, sha256


def _socket_send_activation_file_with_metadata(
    sock: socket.socket,
    path: Path,
    bytes_: int,
    sha256: str,
    metadata: dict[str, Any],
) -> None:
    metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    sock.sendall(struct.pack("!Q", len(metadata_bytes)))
    sock.sendall(metadata_bytes)
    _socket_send_activation_file(sock, path, bytes_, sha256)


def _socket_recv_activation_file_with_metadata(sock: socket.socket, destination: Path) -> tuple[int, str, dict[str, Any]]:
    metadata_len = struct.unpack("!Q", _socket_recv_exact(sock, 8))[0]
    if metadata_len <= 0 or metadata_len > 1024 * 1024:
        raise RuntimeConfigError(f"invalid direct activation slice metadata length {metadata_len}")
    metadata = json.loads(_socket_recv_exact(sock, int(metadata_len)).decode("utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != "rpeg.activation_direct_slice.v1":
        raise RuntimeConfigError("unsupported direct activation slice metadata")
    bytes_, sha256 = _socket_recv_activation_file(sock, destination)
    return bytes_, sha256, metadata


def _socket_send_activation_blob(sock: socket.socket, payload: bytes, sha256: str) -> None:
    sock.sendall(struct.pack("!Q", len(payload)) + bytes.fromhex(sha256))
    sock.sendall(payload)


def _socket_recv_activation_blob(sock: socket.socket) -> tuple[bytes, str]:
    header = _socket_recv_exact(sock, 40)
    bytes_ = struct.unpack("!Q", header[:8])[0]
    expected_sha = header[8:].hex()
    payload = _socket_recv_exact(sock, int(bytes_))
    sha256 = hashlib.sha256(payload).hexdigest()
    if sha256 != expected_sha:
        raise RuntimeConfigError(
            f"direct activation payload checksum mismatch: {sha256} != {expected_sha}"
        )
    return payload, sha256


def _socket_send_activation_blob_with_metadata(
    sock: socket.socket,
    payload: bytes,
    sha256: str,
    metadata: dict[str, Any],
) -> None:
    metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    sock.sendall(struct.pack("!Q", len(metadata_bytes)))
    sock.sendall(metadata_bytes)
    _socket_send_activation_blob(sock, payload, sha256)


def _socket_recv_activation_blob_with_metadata(
    sock: socket.socket,
) -> tuple[bytes, str, dict[str, Any]]:
    metadata_len = struct.unpack("!Q", _socket_recv_exact(sock, 8))[0]
    if metadata_len <= 0 or metadata_len > 1024 * 1024:
        raise RuntimeConfigError(f"invalid direct activation slice metadata length {metadata_len}")
    metadata = json.loads(_socket_recv_exact(sock, int(metadata_len)).decode("utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != "rpeg.activation_direct_slice.v1":
        raise RuntimeConfigError("unsupported direct activation slice metadata")
    artifact_payload, sha256 = _socket_recv_activation_blob(sock)
    return artifact_payload, sha256, metadata


def _socket_recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(nbytes)
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RuntimeConfigError(f"socket closed while reading {nbytes} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _bytes_per_second(bytes_: int | float, duration_s: int | float) -> float:
    duration = float(duration_s)
    if duration <= 0.0:
        return 0.0
    return float(bytes_) / duration


def _reject_persisted_global_index_repartition(
    payload: dict[str, Any],
    transport: str,
) -> None:
    """
    Prevent sparse quantile records from taking a dense-range disk path.
    """
    if bool(payload.get("global_index_repartition", False)):
        raise RuntimeConfigError(
            "natural-volume quantile repartition requires an in-memory activation path; "
            f"{transport} persisted repartition only supports dense shard ranges"
        )


def _weight_state_payload_bytes(state: dict[str, Any]) -> int:
    recorded = state.get("_serialized_bytes")
    if recorded is not None:
        return int(recorded)
    state_dict = state.get("state_dict") or {}
    if not isinstance(state_dict, dict):
        return 0
    return sum(
        int(value.numel()) * int(value.element_size())
        for value in state_dict.values()
        if hasattr(value, "numel") and hasattr(value, "element_size")
    )


def _kernel_copy_file(source: Path, destination: Path) -> int:
    size = source.stat().st_size
    total = 0
    with source.open("rb") as src, destination.open("wb") as dst:
        src_fd = src.fileno()
        dst_fd = dst.fileno()
        while total < size:
            count = min(16 * 1024 * 1024, size - total)
            copied = _copy_file_range_once(src_fd, dst_fd, count)
            if copied is None:
                copied = _sendfile_once(src_fd, dst_fd, count)
            if copied is None:
                raise OSError(errno.ENOSYS, "kernel-assisted file copy is unavailable")
            if copied == 0:
                break
            total += int(copied)
    if total != size:
        raise OSError(errno.EIO, f"kernel-assisted file copy wrote {total} of {size} bytes")
    return total


def _copy_file_range_once(src_fd: int, dst_fd: int, count: int) -> int | None:
    copy_file_range = getattr(os, "copy_file_range", None)
    if copy_file_range is None:
        return None
    try:
        return int(copy_file_range(src_fd, dst_fd, count))
    except OSError as exc:
        if exc.errno in {errno.ENOSYS, errno.EXDEV, errno.EINVAL, errno.EOPNOTSUPP, errno.ENOTSUP}:
            return None
        raise


def _sendfile_once(src_fd: int, dst_fd: int, count: int) -> int | None:
    sendfile = getattr(os, "sendfile", None)
    if sendfile is None:
        return None
    try:
        return int(sendfile(dst_fd, src_fd, None, count))
    except OSError as exc:
        if exc.errno in {errno.ENOSYS, errno.EXDEV, errno.EINVAL, errno.EOPNOTSUPP, errno.ENOTSUP}:
            return None
        raise


def _cloned_file_sha256(copy_mode: str, source_metadata: dict[str, Any], destination: Path) -> str:
    if copy_mode in {"same_path", "hardlink", "reflink", "kernel", "stream"}:
        sha = str(source_metadata.get("sha256") or "")
        if len(sha) == 64:
            return sha
    return _file_sha256(destination)


def _reflink_file(source: Path, destination: Path) -> None:
    # Linux FICLONE: copy-on-write clone when the shared filesystem supports it.
    ficlone = 0x40049409
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("wb") as dst:
        try:
            fcntl.ioctl(dst.fileno(), ficlone, src.fileno())
        except OSError as exc:
            if exc.errno in {
                errno.EOPNOTSUPP,
                errno.ENOTTY,
                errno.EXDEV,
                errno.EINVAL,
                errno.ENOSYS,
                errno.EPERM,
                errno.EACCES,
            }:
                raise
            raise


if __name__ == "__main__":
    raise SystemExit(main())
