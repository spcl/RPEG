# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import hashlib
import json
import math

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from runtime.errors import RuntimeConfigError
from runtime.io import load_mapping


def _required(d: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise RuntimeConfigError(f"{ctx} missing required key {key!r}")
    return d[key]


def _as_int(value: Any, ctx: str) -> int:
    if isinstance(value, bool):
        raise RuntimeConfigError(f"{ctx} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError(f"{ctx} must be an integer") from exc


def _as_float(value: Any, ctx: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError(f"{ctx} must be a number") from exc


def _as_bool(value: Any, ctx: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in {0, 1}:
            return bool(value)
        raise RuntimeConfigError(f"{ctx} must be a boolean")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    raise RuntimeConfigError(f"{ctx} must be a boolean")


def _as_str_tuple(value: Any, ctx: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise RuntimeConfigError(f"{ctx} must be a string or list of strings")
    out = []
    for idx, item in enumerate(values):
        text = str(item).strip()
        if not text:
            raise RuntimeConfigError(f"{ctx}[{idx}] must be non-empty")
        out.append(text.removesuffix("."))
    return tuple(out)


def _distributed_optimizer(value: Any) -> str:
    normalized = str(value or "none").strip().lower().replace("-", "_")
    aliases = {
        "": "none",
        "none": "none",
        "adamw": "none",
        "zero1": "zero1",
        "zero_1": "zero1",
        "zero_stage_1": "zero1",
        "stage1": "zero1",
        "zero2": "zero2",
        "zero_2": "zero2",
        "zero_stage_2": "zero2",
        "stage2": "zero2",
    }
    if normalized not in aliases:
        raise RuntimeConfigError(
            f"optimizer_runtime.distributed_optimizer must be one of {sorted(set(aliases))}"
        )
    return aliases[normalized]


@dataclass(frozen=True)
class OptimizedPlanRef:
    path: str
    expected_hash: str | None = None
    require_exact_match: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OptimizedPlanRef":
        return cls(
            path=str(_required(d, "path", "optimized_plan")),
            expected_hash=d.get("expected_hash"),
            require_exact_match=bool(d.get("require_exact_match", True)),
        )


@dataclass(frozen=True)
class AlgorithmConfig:
    name: str
    beta: float = 0.1
    kl_coef: float = 0.0
    clip_range: float = 0.2
    gamma: float = 1.0
    gae_lambda: float = 0.95
    epochs: int = 1
    minibatch_size: int | None = None
    entropy_coef: float = 0.0
    normalize_advantages: bool = False
    seed: int = 1234
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    generation_backend: str = "full_sequence"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AlgorithmConfig":
        name = str(_required(d, "name", "algorithm")).lower()
        if name not in {"ppo", "grpo", "dpo"}:
            raise RuntimeConfigError(f"Unsupported algorithm {name!r}")
        temperature = _as_float(d.get("temperature", 1.0), "algorithm.temperature")
        top_p = _as_float(d.get("top_p", 1.0), "algorithm.top_p")
        top_k = _as_int(d.get("top_k", 0), "algorithm.top_k")
        if temperature <= 0.0:
            raise RuntimeConfigError("algorithm.temperature must be > 0")
        if not 0.0 < top_p <= 1.0:
            raise RuntimeConfigError("algorithm.top_p must be in (0, 1]")
        if top_k < 0:
            raise RuntimeConfigError("algorithm.top_k must be >= 0")
        beta = _as_float(d.get("beta", 0.1), "algorithm.beta")
        kl_coef = _as_float(d.get("kl_coef", 0.0), "algorithm.kl_coef")
        clip_range = _as_float(d.get("clip_range", 0.2), "algorithm.clip_range")
        gamma = _as_float(d.get("gamma", 1.0), "algorithm.gamma")
        gae_lambda = _as_float(d.get("gae_lambda", 0.95), "algorithm.gae_lambda")
        epochs = _as_int(d.get("epochs", 1), "algorithm.epochs")
        minibatch_raw = d.get("minibatch_size")
        minibatch_size = None if minibatch_raw is None else _as_int(minibatch_raw, "algorithm.minibatch_size")
        entropy_coef = _as_float(d.get("entropy_coef", 0.0), "algorithm.entropy_coef")
        if beta < 0.0 or kl_coef < 0.0 or entropy_coef < 0.0:
            raise RuntimeConfigError("algorithm beta, kl_coef, and entropy_coef must be >= 0")
        if clip_range <= 0.0:
            raise RuntimeConfigError("algorithm.clip_range must be > 0")
        if not 0.0 <= gamma <= 1.0:
            raise RuntimeConfigError("algorithm.gamma must be in [0, 1]")
        if not 0.0 <= gae_lambda <= 1.0:
            raise RuntimeConfigError("algorithm.gae_lambda must be in [0, 1]")
        if epochs <= 0:
            raise RuntimeConfigError("algorithm.epochs must be > 0")
        if minibatch_size is not None and minibatch_size <= 0:
            raise RuntimeConfigError("algorithm.minibatch_size must be > 0 when specified")
        generation_backend = str(d.get("generation_backend", "full_sequence")).strip().lower()
        if generation_backend not in {"full_sequence", "static_kv"}:
            raise RuntimeConfigError("algorithm.generation_backend must be one of: full_sequence, static_kv")
        return cls(
            name=name,
            beta=beta,
            kl_coef=kl_coef,
            clip_range=clip_range,
            gamma=gamma,
            gae_lambda=gae_lambda,
            epochs=epochs,
            minibatch_size=minibatch_size,
            entropy_coef=entropy_coef,
            normalize_advantages=bool(d.get("normalize_advantages", False)),
            seed=_as_int(d.get("seed", 1234), "algorithm.seed"),
            do_sample=bool(d.get("do_sample", True)),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            generation_backend=generation_backend,
        )


@dataclass(frozen=True)
class TrainingConfig:
    max_steps: int
    save_every_steps: int = 100
    eval_every_steps: int = 500
    log_every_steps: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingConfig":
        return cls(
            max_steps=_as_int(_required(d, "max_steps", "training"), "training.max_steps"),
            save_every_steps=_as_int(d.get("save_every_steps", 100), "training.save_every_steps"),
            eval_every_steps=_as_int(d.get("eval_every_steps", 500), "training.eval_every_steps"),
            log_every_steps=_as_int(d.get("log_every_steps", 1), "training.log_every_steps"),
        )


@dataclass(frozen=True)
class WorkloadConfig:
    batch_size: int
    samples_per_prompt: int
    context_length: int
    generation_length: int
    gen_quantiles: tuple[tuple[float, int], ...] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkloadConfig":
        batch_size = _as_int(_required(d, "batch_size", "workload"), "workload.batch_size")
        samples_per_prompt = _as_int(d.get("samples_per_prompt", 1), "workload.samples_per_prompt")
        context_length = _as_int(_required(d, "context_length", "workload"), "workload.context_length")
        generation_length = _as_int(_required(d, "generation_length", "workload"), "workload.generation_length")
        if min(batch_size, samples_per_prompt, context_length, generation_length) <= 0:
            raise RuntimeConfigError("workload batch_size, samples_per_prompt, context_length, and generation_length must be > 0")
        q = d.get("gen_quantiles")
        quantiles = None
        if q is not None:
            try:
                quantiles = tuple((float(pair[0]), int(pair[1])) for pair in q)
            except (TypeError, ValueError, IndexError) as exc:
                raise RuntimeConfigError("workload.gen_quantiles must contain [cumulative_fraction, token_count] pairs") from exc
            if not quantiles:
                raise RuntimeConfigError("workload.gen_quantiles cannot be empty")
            previous_fraction = 0.0
            previous_tokens = 0
            for fraction, tokens in quantiles:
                if not previous_fraction < fraction <= 1.0:
                    raise RuntimeConfigError("workload.gen_quantiles fractions must be strictly increasing in (0, 1]")
                if not previous_tokens < tokens <= generation_length:
                    raise RuntimeConfigError(
                        "workload.gen_quantiles token counts must be strictly increasing and <= generation_length"
                    )
                previous_fraction = fraction
                previous_tokens = tokens
            if quantiles[-1] != (1.0, generation_length):
                raise RuntimeConfigError(
                    "workload.gen_quantiles must end at [1.0, workload.generation_length]"
                )
        return cls(
            batch_size=batch_size,
            samples_per_prompt=samples_per_prompt,
            context_length=context_length,
            generation_length=generation_length,
            gen_quantiles=quantiles,
        )


@dataclass(frozen=True)
class RuntimeModelConfig:
    source: str
    role: str = "trainable"
    repo: str | None = None
    revision: str | None = None
    checkpoint_path: str | None = None
    dtype: str = "bf16"
    trust_remote_code: bool = False
    value_head_prefixes: tuple[str, ...] = ()
    reward_head_prefixes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RuntimeModelConfig":
        source = str(_required(d, "source", "models.<role>")).lower()
        return cls(
            source=source,
            role=str(d.get("role", "trainable")),
            repo=d.get("repo"),
            revision=d.get("revision"),
            checkpoint_path=d.get("checkpoint_path"),
            dtype=str(d.get("dtype", "bf16")),
            trust_remote_code=bool(d.get("trust_remote_code", False)),
            value_head_prefixes=_as_str_tuple(d.get("value_head_prefixes"), "models.<role>.value_head_prefixes"),
            reward_head_prefixes=_as_str_tuple(d.get("reward_head_prefixes"), "models.<role>.reward_head_prefixes"),
        )


@dataclass(frozen=True)
class TokenizerConfig:
    source: str
    repo: str | None = None
    revision: str | None = None
    checkpoint_path: str | None = None
    padding_side: str = "left"
    truncation_side: str = "left"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TokenizerConfig":
        return cls(
            source=str(_required(d, "source", "tokenizer")).lower(),
            repo=d.get("repo"),
            revision=d.get("revision"),
            checkpoint_path=d.get("checkpoint_path"),
            padding_side=str(d.get("padding_side", "left")),
            truncation_side=str(d.get("truncation_side", "left")),
        )


@dataclass(frozen=True)
class DeviceSpaceConfig:
    racks: int
    nodes_per_rack: int
    gpus_per_node: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DeviceSpaceConfig":
        return cls(
            racks=_as_int(_required(d, "racks", "device_space"), "device_space.racks"),
            nodes_per_rack=_as_int(_required(d, "nodes_per_rack", "device_space"), "device_space.nodes_per_rack"),
            gpus_per_node=_as_int(_required(d, "gpus_per_node", "device_space"), "device_space.gpus_per_node"),
        )


@dataclass(frozen=True)
class HardwareConfig:
    calibrated_hw_path: str | None = None
    gpu_type: str | None = None
    memory_per_gpu_bytes: int | None = None
    memory_reserve_fraction: float = 0.10
    require_calibration_match: bool = True
    worker_overhead_bytes: int = 10 * 1024**3
    semantic_activation_cache_bytes: int = 0
    max_worker_groups_per_gpu: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HardwareConfig":
        worker_overhead = _as_int(d.get("worker_overhead_bytes", 10 * 1024**3), "hardware.worker_overhead_bytes")
        semantic_activation_cache = _as_int(
            d.get("semantic_activation_cache_bytes", 0),
            "hardware.semantic_activation_cache_bytes",
        )
        max_worker_groups = _as_int(
            d.get("max_worker_groups_per_gpu", 0),
            "hardware.max_worker_groups_per_gpu",
        )
        mem = d.get("memory_per_gpu_bytes")
        reserve = _as_float(d.get("memory_reserve_fraction", 0.10), "hardware.memory_reserve_fraction")
        if worker_overhead < 0:
            raise RuntimeConfigError("hardware.worker_overhead_bytes must be >= 0")
        if semantic_activation_cache < 0:
            raise RuntimeConfigError("hardware.semantic_activation_cache_bytes must be >= 0")
        if max_worker_groups < 0:
            raise RuntimeConfigError("hardware.max_worker_groups_per_gpu must be >= 0")
        if not 0.0 <= reserve < 1.0:
            raise RuntimeConfigError("hardware.memory_reserve_fraction must be in [0, 1)")
        return cls(
            calibrated_hw_path=d.get("calibrated_hw_path"),
            worker_overhead_bytes=worker_overhead,
            semantic_activation_cache_bytes=semantic_activation_cache,
            max_worker_groups_per_gpu=max_worker_groups,
            gpu_type=d.get("gpu_type"),
            memory_per_gpu_bytes=None if mem is None else _as_int(mem, "hardware.memory_per_gpu_bytes"),
            memory_reserve_fraction=reserve,
            require_calibration_match=bool(d.get("require_calibration_match", True)),
        )


@dataclass(frozen=True)
class LauncherConfig:
    type: str = "local"
    partition: str | None = None
    account: str | None = None
    time_limit: str | None = None
    container: str | None = None
    mpi: str = "none"
    shared_step: bool = False
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "LauncherConfig":
        d = d or {}
        env = {str(k): str(v) for k, v in (d.get("env") or {}).items()}
        return cls(
            type=str(d.get("type", "local")).lower(),
            partition=d.get("partition"),
            account=d.get("account"),
            time_limit=d.get("time_limit"),
            container=d.get("container"),
            shared_step=bool(d.get("shared_step", False)),
            mpi=str(d.get("mpi", "none")).strip().lower(),
            env=env,
        )


@dataclass(frozen=True)
class ClusterConfig:
    scheduler: str = "local"
    inventory_path: str | None = None
    gpu_order: str = "pci_bus_id"
    launcher: LauncherConfig = field(default_factory=LauncherConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ClusterConfig":
        return cls(
            scheduler=str(d.get("scheduler", "local")).lower(),
            inventory_path=d.get("inventory_path"),
            gpu_order=str(d.get("gpu_order", "pci_bus_id")),
            launcher=LauncherConfig.from_dict(d.get("launcher")),
        )


@dataclass(frozen=True)
class RuntimeConfig:
    default_backend: str = "torch"
    rendezvous_port_base: int = 29500
    fail_on_backend_rewrite: bool = True
    dry_run: bool = False
    log_dir: str = "logs/runtime"
    store_dir: str | None = None
    trace_path: str = "traces/runtime.jsonl"
    use_transformer_engine: bool = True
    activation_transport: str = "auto"
    activation_store: str = "auto"
    activation_cache_device: str = "cpu"
    weight_transport: str = "auto"
    direct_channel_reuse: bool = True
    direct_source_overlap: bool = False
    semantic_control_transport: str = "tcp"
    semantic_weight_keep_latest: int = 8
    static_kv_cache_max_entries: int = 4
    static_kv_cache_batch_bucket: int = 0
    static_kv_cache_sequence_bucket: int = 0
    require_optimized_schedule: bool = False
    intra_stage_fusion: bool = False
    critic_pipeline_device_order: str = "auto"
    fusion_sync_backend: str = "cuda_ipc"
    fusion_pipeline_nccl_cta_policy: str = "default"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RuntimeConfig":
        activation_transport = str(d.get("activation_transport", "auto")).strip().lower().replace("-", "_")
        activation_aliases = {
            "direct": "tcp_direct",
            "torch": "torch_dist",
            "torch_direct": "torch_dist",
            "torch_dist_direct": "torch_dist",
            "gloo": "gloo_direct",
            "gloo_direct": "gloo_direct",
            "nccl": "nccl_direct",
            "nccl_direct": "nccl_direct",
            "ucx": "ucx_direct",
            "ucx_direct": "ucx_direct",
            "rdma": "ucx_direct",
            "rdma_direct": "ucx_direct",
        }
        if activation_transport in activation_aliases:
            activation_transport = activation_aliases[activation_transport]
        if activation_transport == "direct":
            activation_transport = "tcp_direct"
        if activation_transport not in {
            "auto",
            "tcp_direct",
            "torch_dist",
            "gloo_direct",
            "nccl_direct",
            "ucx_direct",
            "kernel",
            "stream",
            "copy",
        }:
            raise RuntimeConfigError(
                "runtime.activation_transport must be one of: auto, tcp_direct, torch_dist, "
                "gloo_direct, nccl_direct, ucx_direct, kernel, stream, copy"
            )
        activation_store = str(d.get("activation_store", "auto")).strip().lower().replace("-", "_")
        if activation_store not in {"auto", "memory", "filesystem", "hybrid"}:
            raise RuntimeConfigError(
                "runtime.activation_store must be one of: auto, memory, filesystem, hybrid"
            )
        activation_cache_device = (
            str(d.get("activation_cache_device", "cpu"))
            .strip()
            .lower()
            .replace("-", "_")
        )
        if activation_cache_device not in {"auto", "cpu", "cuda"}:
            raise RuntimeConfigError(
                "runtime.activation_cache_device must be one of: auto, cpu, cuda"
            )
        weight_transport = str(d.get("weight_transport", "auto")).strip().lower().replace("-", "_")
        weight_aliases = {
            "torch": "torch_dist",
            "torch_direct": "torch_dist",
            "torch_dist_direct": "torch_dist",
            "gloo": "gloo_direct",
            "gloo_direct": "gloo_direct",
            "nccl": "nccl_direct",
            "nccl_direct": "nccl_direct",
            "ucx": "ucx_direct",
            "ucx_direct": "ucx_direct",
            "rdma": "ucx_direct",
            "rdma_direct": "ucx_direct",
            "file": "auto",
            "filesystem": "auto",
            "shared_filesystem": "auto",
        }
        if weight_transport in weight_aliases:
            weight_transport = weight_aliases[weight_transport]
        if weight_transport not in {"auto", "torch_dist", "gloo_direct", "nccl_direct", "ucx_direct"}:
            raise RuntimeConfigError(
                "runtime.weight_transport must be one of: auto, torch_dist, gloo_direct, nccl_direct, ucx_direct"
            )
        semantic_weight_keep_latest = _as_int(
            d.get("semantic_weight_keep_latest", 8),
            "runtime.semantic_weight_keep_latest",
        )
        if semantic_weight_keep_latest < 0:
            raise RuntimeConfigError("runtime.semantic_weight_keep_latest must be >= 0")
        semantic_control_transport = str(d.get("semantic_control_transport", "tcp")).strip().lower()
        if semantic_control_transport not in {"tcp", "file"}:
            raise RuntimeConfigError("runtime.semantic_control_transport must be one of: tcp, file")
        static_kv_cache_max_entries = _as_int(
            d.get("static_kv_cache_max_entries", 4),
            "runtime.static_kv_cache_max_entries",
        )
        if static_kv_cache_max_entries < 0:
            raise RuntimeConfigError("runtime.static_kv_cache_max_entries must be >= 0")
        static_kv_cache_batch_bucket = _as_int(
            d.get("static_kv_cache_batch_bucket", 0),
            "runtime.static_kv_cache_batch_bucket",
        )
        if static_kv_cache_batch_bucket < 0:
            raise RuntimeConfigError("runtime.static_kv_cache_batch_bucket must be >= 0")
        static_kv_cache_sequence_bucket = _as_int(
            d.get("static_kv_cache_sequence_bucket", 0),
            "runtime.static_kv_cache_sequence_bucket",
        )
        if static_kv_cache_sequence_bucket < 0:
            raise RuntimeConfigError("runtime.static_kv_cache_sequence_bucket must be >= 0")
        fusion_sync_backend = str(d.get("fusion_sync_backend", "cuda_ipc")).strip().lower().replace("-", "_")
        fusion_sync_aliases = {
            "cuda": "cuda_ipc",
            "event": "cuda_ipc",
            "cuda_event": "cuda_ipc",
            "file": "filesystem",
        }
        fusion_sync_backend = fusion_sync_aliases.get(fusion_sync_backend, fusion_sync_backend)
        if fusion_sync_backend not in {"cuda_ipc", "filesystem"}:
            raise RuntimeConfigError(
                "runtime.fusion_sync_backend must be one of: cuda_ipc, filesystem"
            )
        fusion_pipeline_nccl_cta_policy = (
            str(d.get("fusion_pipeline_nccl_cta_policy", "default"))
            .strip()
            .lower()
        )
        if fusion_pipeline_nccl_cta_policy not in {
            "default",
            "efficiency",
            "zero",
        }:
            raise RuntimeConfigError(
                "runtime.fusion_pipeline_nccl_cta_policy must be one of: "
                "default, efficiency, zero"
            )
        critic_pipeline_device_order = (
            str(d.get("critic_pipeline_device_order", "auto"))
            .strip()
            .lower()
            .replace("-", "_")
        )
        critic_pipeline_device_order = {
            "forward": "canonical",
            "backward": "reverse",
            "reversed": "reverse",
        }.get(critic_pipeline_device_order, critic_pipeline_device_order)
        if critic_pipeline_device_order not in {"auto", "canonical", "reverse"}:
            raise RuntimeConfigError(
                "runtime.critic_pipeline_device_order must be one of: auto, canonical, reverse"
            )
        return cls(
            default_backend=str(d.get("default_backend", "torch")).lower(),
            rendezvous_port_base=_as_int(d.get("rendezvous_port_base", 29500), "runtime.rendezvous_port_base"),
            fail_on_backend_rewrite=bool(d.get("fail_on_backend_rewrite", True)),
            dry_run=bool(d.get("dry_run", False)),
            log_dir=str(d.get("log_dir", "logs/runtime")),
            store_dir=None if d.get("store_dir") is None else str(d.get("store_dir")),
            trace_path=str(d.get("trace_path", "traces/runtime.jsonl")),
            use_transformer_engine=_as_bool(d.get("use_transformer_engine", True), "runtime.use_transformer_engine"),
            activation_transport=activation_transport,
            activation_store=activation_store,
            activation_cache_device=activation_cache_device,
            weight_transport=weight_transport,
            direct_channel_reuse=_as_bool(d.get("direct_channel_reuse", True), "runtime.direct_channel_reuse"),
            direct_source_overlap=_as_bool(
                d.get("direct_source_overlap", False),
                "runtime.direct_source_overlap",
            ),
            semantic_control_transport=semantic_control_transport,
            semantic_weight_keep_latest=semantic_weight_keep_latest,
            static_kv_cache_max_entries=static_kv_cache_max_entries,
            static_kv_cache_batch_bucket=static_kv_cache_batch_bucket,
            static_kv_cache_sequence_bucket=static_kv_cache_sequence_bucket,
            require_optimized_schedule=_as_bool(
                d.get("require_optimized_schedule", False),
                "runtime.require_optimized_schedule",
            ),
            intra_stage_fusion=_as_bool(
                d.get("intra_stage_fusion", False),
                "runtime.intra_stage_fusion",
            ),
            critic_pipeline_device_order=critic_pipeline_device_order,
            fusion_sync_backend=fusion_sync_backend,
            fusion_pipeline_nccl_cta_policy=fusion_pipeline_nccl_cta_policy,
        )


@dataclass(frozen=True)
class DataConfig:
    train_prompts: str | None = None
    eval_prompts: str | None = None
    preferences: str | None = None
    response_format: str = "plain"
    max_prompt_length: int | None = None
    max_generation_length: int | None = None
    dataloader_workers: int = 0
    seed: int = 0
    shuffle: bool = False
    prompt_offset: int = 0
    preference_offset: int = 0
    shard_index: int = 0
    num_shards: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DataConfig":
        prompt_offset = _as_int(d.get("prompt_offset", 0), "data.prompt_offset")
        preference_offset = _as_int(d.get("preference_offset", 0), "data.preference_offset")
        shard_index = _as_int(d.get("shard_index", 0), "data.shard_index")
        num_shards = _as_int(d.get("num_shards", 1), "data.num_shards")
        if prompt_offset < 0:
            raise RuntimeConfigError("data.prompt_offset must be >= 0")
        if preference_offset < 0:
            raise RuntimeConfigError("data.preference_offset must be >= 0")
        if num_shards <= 0:
            raise RuntimeConfigError("data.num_shards must be > 0")
        if shard_index < 0:
            raise RuntimeConfigError("data.shard_index must be >= 0")
        if shard_index >= num_shards:
            raise RuntimeConfigError("data.shard_index must be < data.num_shards")
        return cls(
            train_prompts=d.get("train_prompts"),
            eval_prompts=d.get("eval_prompts"),
            preferences=d.get("preferences"),
            response_format=str(d.get("response_format", "plain")),
            max_prompt_length=d.get("max_prompt_length"),
            max_generation_length=d.get("max_generation_length"),
            dataloader_workers=_as_int(d.get("dataloader_workers", 0), "data.dataloader_workers"),
            seed=_as_int(d.get("seed", 0), "data.seed"),
            shuffle=_as_bool(d.get("shuffle", False), "data.shuffle"),
            prompt_offset=prompt_offset,
            preference_offset=preference_offset,
            shard_index=shard_index,
            num_shards=num_shards,
        )


@dataclass(frozen=True)
class OptimizerRuntimeConfig:
    learning_rate_actor: float = 1e-6
    learning_rate_critic: float = 5e-6
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    activation_checkpointing: bool = False
    distributed_optimizer: str = "none"
    zero2_bucket_bytes: int = 64 * 1024 * 1024
    zero2_max_inflight_buckets: int = 2

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OptimizerRuntimeConfig":
        zero2_bucket_bytes = _as_int(
            d.get("zero2_bucket_bytes", 64 * 1024 * 1024),
            "optimizer_runtime.zero2_bucket_bytes",
        )
        if zero2_bucket_bytes <= 0:
            raise RuntimeConfigError("optimizer_runtime.zero2_bucket_bytes must be > 0")
        zero2_max_inflight_buckets = _as_int(
            d.get("zero2_max_inflight_buckets", 2),
            "optimizer_runtime.zero2_max_inflight_buckets",
        )
        if zero2_max_inflight_buckets <= 0:
            raise RuntimeConfigError("optimizer_runtime.zero2_max_inflight_buckets must be > 0")
        return cls(
            learning_rate_actor=_as_float(d.get("learning_rate_actor", 1e-6), "optimizer_runtime.learning_rate_actor"),
            learning_rate_critic=_as_float(d.get("learning_rate_critic", 5e-6), "optimizer_runtime.learning_rate_critic"),
            weight_decay=_as_float(d.get("weight_decay", 0.0), "optimizer_runtime.weight_decay"),
            grad_clip=_as_float(d.get("grad_clip", 1.0), "optimizer_runtime.grad_clip"),
            activation_checkpointing=bool(d.get("activation_checkpointing", False)),
            distributed_optimizer=_distributed_optimizer(d.get("distributed_optimizer", "none")),
            zero2_bucket_bytes=zero2_bucket_bytes,
            zero2_max_inflight_buckets=zero2_max_inflight_buckets,
        )

@dataclass(frozen=True)
class CheckpointConfig:
    output_dir: str
    interval_steps: int = 100
    format: str = "torch"
    resume_from: str | None = None
    keep_last_interval_steps: int | None = None
    save_final: bool = True
    portable_weight_versions: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckpointConfig":
        keep_last = d.get("keep_last_interval_steps")
        if keep_last is None:
            keep_last = d.get("keep_last")
        keep_last_interval_steps = None if keep_last is None else _as_int(keep_last, "checkpoints.keep_last_interval_steps")
        if keep_last_interval_steps is not None and keep_last_interval_steps < 0:
            raise RuntimeConfigError("checkpoints.keep_last_interval_steps must be >= 0")
        return cls(
            output_dir=str(_required(d, "output_dir", "checkpoints")),
            interval_steps=_as_int(d.get("interval_steps", 100), "checkpoints.interval_steps"),
            format=str(d.get("format", "torch")),
            resume_from=d.get("resume_from"),
            keep_last_interval_steps=keep_last_interval_steps,
            save_final=_as_bool(d.get("save_final", True), "checkpoints.save_final"),
            portable_weight_versions=_as_bool(
                d.get("portable_weight_versions", False),
                "checkpoints.portable_weight_versions",
            ),
        )


@dataclass(frozen=True)
class RuntimeRunConfig:
    schema_version: str
    run_id: str
    optimized_plan: OptimizedPlanRef
    algorithm: AlgorithmConfig
    training: TrainingConfig
    workload: WorkloadConfig
    models: dict[str, RuntimeModelConfig]
    tokenizer: TokenizerConfig
    device_space: DeviceSpaceConfig
    hardware: HardwareConfig
    cluster: ClusterConfig
    runtime: RuntimeConfig
    data: DataConfig
    optimizer_runtime: OptimizerRuntimeConfig
    checkpoints: CheckpointConfig
    monitoring: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RuntimeRunConfig":
        version = str(_required(d, "schema_version", "run config"))
        if version != "rpeg.run.v1":
            raise RuntimeConfigError(f"Unsupported run schema_version {version!r}")
        models_d = _required(d, "models", "run config")
        if not isinstance(models_d, dict):
            raise RuntimeConfigError("models must be a mapping")
        models = {str(k): RuntimeModelConfig.from_dict(v) for k, v in models_d.items()}
        hardware = HardwareConfig.from_dict(d.get("hardware") or {})
        runtime = RuntimeConfig.from_dict(d.get("runtime") or {})
        if (
            runtime.activation_cache_device == "cuda"
            and hardware.semantic_activation_cache_bytes <= 0
        ):
            raise RuntimeConfigError(
                "runtime.activation_cache_device=cuda requires a positive "
                "hardware.semantic_activation_cache_bytes runtime limit"
            )
        return cls(
            schema_version=version,
            run_id=str(_required(d, "run_id", "run config")),
            optimized_plan=OptimizedPlanRef.from_dict(_required(d, "optimized_plan", "run config")),
            algorithm=AlgorithmConfig.from_dict(_required(d, "algorithm", "run config")),
            training=TrainingConfig.from_dict(_required(d, "training", "run config")),
            workload=WorkloadConfig.from_dict(_required(d, "workload", "run config")),
            models=models,
            tokenizer=TokenizerConfig.from_dict(_required(d, "tokenizer", "run config")),
            device_space=DeviceSpaceConfig.from_dict(_required(d, "device_space", "run config")),
            hardware=hardware,
            cluster=ClusterConfig.from_dict(d.get("cluster") or {}),
            runtime=runtime,
            data=DataConfig.from_dict(d.get("data") or {}),
            optimizer_runtime=OptimizerRuntimeConfig.from_dict(d.get("optimizer_runtime") or {}),
            checkpoints=CheckpointConfig.from_dict(_required(d, "checkpoints", "run config")),
            monitoring=d.get("monitoring") or {},
        )

    @classmethod
    def load(cls, path: str | Path) -> "RuntimeRunConfig":
        return cls.from_dict(load_mapping(path))


@dataclass(frozen=True)
class PlanVertex:
    id: str
    model: str
    role: str
    origin: tuple[int, int, int]
    extent: tuple[int, int, int]
    parallelism: tuple[int, int, int]
    microbatch: int

    @classmethod
    def from_dict(cls, vid: str, d: dict[str, Any]) -> "PlanVertex":
        placement = _required(d, "placement", f"plan.{vid}")
        parallelism = _required(d, "parallelism", f"plan.{vid}")
        return cls(
            id=vid,
            model=str(_required(d, "model", f"plan.{vid}")),
            role=str(_required(d, "role", f"plan.{vid}")),
            origin=tuple(_as_int(x, f"plan.{vid}.placement.origin") for x in placement["origin"]),  # type: ignore[arg-type]
            extent=tuple(_as_int(x, f"plan.{vid}.placement.extent") for x in placement["extent"]),  # type: ignore[arg-type]
            parallelism=(
                _as_int(parallelism["dp"], f"plan.{vid}.parallelism.dp"),
                _as_int(parallelism["pp"], f"plan.{vid}.parallelism.pp"),
                _as_int(parallelism["tp"], f"plan.{vid}.parallelism.tp"),
            ),
            microbatch=_as_int(_required(d, "microbatch", f"plan.{vid}"), f"plan.{vid}.microbatch"),
        )


@dataclass(frozen=True)
class ExecutionScheduleNode:
    node_id: str
    dispatch_index: int
    predicted_start_s: float
    predicted_finish_s: float


@dataclass(frozen=True)
class OptimizedExecutionSchedule:
    policy: str
    horizon: int
    predicted_makespan_s: float
    nodes: tuple[ExecutionScheduleNode, ...]

    @property
    def order(self) -> tuple[str, ...]:
        return tuple(node.node_id for node in self.nodes)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OptimizedExecutionSchedule":
        policy = str(_required(d, "policy", "execution_schedule"))
        if policy != "rpeg_critical_path_list_v1":
            raise RuntimeConfigError(f"unsupported execution_schedule.policy {policy!r}")
        horizon = _as_int(_required(d, "horizon", "execution_schedule"), "execution_schedule.horizon")
        if horizon < 1:
            raise RuntimeConfigError("execution_schedule.horizon must be >= 1")
        makespan = _as_float(
            _required(d, "predicted_makespan_s", "execution_schedule"),
            "execution_schedule.predicted_makespan_s",
        )
        if not math.isfinite(makespan) or makespan < 0.0:
            raise RuntimeConfigError("execution_schedule.predicted_makespan_s must be finite and >= 0")
        raw_nodes = _required(d, "nodes", "execution_schedule")
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise RuntimeConfigError("execution_schedule.nodes must be a non-empty list")
        parsed = []
        for idx, raw in enumerate(raw_nodes):
            if not isinstance(raw, dict):
                raise RuntimeConfigError(f"execution_schedule.nodes[{idx}] must be a mapping")
            dispatch_index = _as_int(
                _required(raw, "dispatch_index", f"execution_schedule.nodes[{idx}]"),
                f"execution_schedule.nodes[{idx}].dispatch_index",
            )
            start = _as_float(
                _required(raw, "predicted_start_s", f"execution_schedule.nodes[{idx}]"),
                f"execution_schedule.nodes[{idx}].predicted_start_s",
            )
            finish = _as_float(
                _required(raw, "predicted_finish_s", f"execution_schedule.nodes[{idx}]"),
                f"execution_schedule.nodes[{idx}].predicted_finish_s",
            )
            if not math.isfinite(start) or not math.isfinite(finish) or start < 0.0 or finish < start:
                raise RuntimeConfigError(
                    f"execution_schedule.nodes[{idx}] must have finite 0 <= predicted_start_s <= predicted_finish_s"
                )
            node_id = str(_required(raw, "node_id", f"execution_schedule.nodes[{idx}]")).strip()
            if not node_id:
                raise RuntimeConfigError(f"execution_schedule.nodes[{idx}].node_id must be non-empty")
            parsed.append(ExecutionScheduleNode(node_id, dispatch_index, start, finish))
        parsed.sort(key=lambda node: node.dispatch_index)
        if [node.dispatch_index for node in parsed] != list(range(len(parsed))):
            raise RuntimeConfigError("execution_schedule dispatch_index values must be contiguous from 0")
        ids = [node.node_id for node in parsed]
        if len(ids) != len(set(ids)):
            raise RuntimeConfigError("execution_schedule contains duplicate node_id values")
        max_finish = max(node.predicted_finish_s for node in parsed)
        if not math.isclose(max_finish, makespan, rel_tol=1e-9, abs_tol=1e-12):
            raise RuntimeConfigError(
                "execution_schedule.predicted_makespan_s does not match the maximum predicted_finish_s"
            )
        return cls(policy=policy, horizon=horizon, predicted_makespan_s=makespan, nodes=tuple(parsed))


@dataclass(frozen=True)
class OptimizedPlan:
    schema_version: str
    plan_hash: str | None
    source_config_hash: str | None
    algorithm: str
    workload: WorkloadConfig
    device_space: DeviceSpaceConfig
    structure: dict[str, Any]
    vertices: dict[str, PlanVertex]
    execution_schedule: OptimizedExecutionSchedule | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OptimizedPlan":
        version = str(d.get("schema_version", "rpeg.optimized_plan.v0"))
        if version not in {"rpeg.optimized_plan.v1", "rpeg.optimized_plan.v0"}:
            raise RuntimeConfigError(f"Unsupported optimized plan schema_version {version!r}")
        if version != "rpeg.optimized_plan.v1":
            raise RuntimeConfigError(
                "legacy plan files are inspectable with ir.serialize.load_plan, "
                "but runtime execution requires schema_version rpeg.optimized_plan.v1"
            )
        declared_hash = d.get("plan_hash")
        if declared_hash is not None:
            canonical = json.dumps(
                {key: value for key, value in d.items() if key != "plan_hash"},
                sort_keys=True,
                separators=(",", ":"),
            )
            actual_hash = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
            if str(declared_hash) != actual_hash:
                raise RuntimeConfigError(
                    f"optimized plan content hash mismatch: declared {declared_hash}, computed {actual_hash}"
                )
        plan_d = _required(d, "plan", "optimized plan")
        if not isinstance(plan_d, dict):
            raise RuntimeConfigError("optimized plan 'plan' must be a mapping")
        return cls(
            schema_version=version,
            plan_hash=d.get("plan_hash"),
            source_config_hash=d.get("source_config_hash"),
            algorithm=str(_required(d, "algorithm", "optimized plan")).lower(),
            workload=WorkloadConfig.from_dict(_required(d, "workload", "optimized plan")),
            device_space=DeviceSpaceConfig.from_dict(_required(d, "device_space", "optimized plan")),
            structure=d.get("structure") or {},
            vertices={str(vid): PlanVertex.from_dict(str(vid), info) for vid, info in plan_d.items()},
            execution_schedule=(
                None
                if d.get("execution_schedule") is None
                else OptimizedExecutionSchedule.from_dict(d["execution_schedule"])
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> "OptimizedPlan":
        return cls.from_dict(load_mapping(path))
