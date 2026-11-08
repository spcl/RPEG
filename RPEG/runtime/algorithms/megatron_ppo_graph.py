# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import threading
import zlib

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from core.types import Model
from model.memory import training_activation_bytes_per_microbatch
from passes.lower import ExecNodeType
from runtime.compiler import CompiledRuntimePlan, WorkerSpec
from runtime.containers import ContainerRef, ContainerStore, ShardingSpec, WeightVersion
from runtime.errors import RuntimeConfigError, RuntimeExecutionError
from runtime.executor import NodeExecutionResult, RuntimeExecutionContext, RuntimeExecutor
from runtime.graph import RuntimeGraph, RuntimeNodeSpec, build_runtime_graph
from runtime.fused_pipeline import FusedPipelineModel, build_fused_pipeline_schedule
from runtime.hf import resolve_model_path
from runtime.launcher import SlurmWorkerGroupLauncher
from runtime.metrics import write_runtime_summary_for_run
from runtime.schema import LauncherConfig, RuntimeRunConfig
from runtime.semantic import SemanticOperationNames, SemanticResult
from runtime.semantic_controller import SemanticWorkerGroupRuntime
from runtime.semantic_worker import MegatronSemanticBackend
from runtime.trace import RuntimeTraceWriter
from runtime.weight_versions import consumer_weight_version, uses_shadow_copies
from runtime.reshard import dp_pp_tp_to_rank, rank_to_dp_pp_tp, routed_weight_source_tp_ranks


@dataclass(frozen=True)
class MegatronPPOCapability:
    vertex_id: str
    required_operation: str
    supported: bool
    reason: str

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "vertex_id": self.vertex_id,
            "required_operation": self.required_operation,
            "supported": self.supported,
            "reason": self.reason,
        }


_PPO_REQUIRED_OPS: dict[str, str] = {
    "g_pi": SemanticOperationNames.PPO_ACTOR_GENERATE,
    "g_ac": SemanticOperationNames.PPO_ACTOR_GENERATE,
    "i_ref": SemanticOperationNames.PPO_REFERENCE_LOGPROBS,
    "i_R": SemanticOperationNames.PPO_REWARD_SCORE,
    "i_V": SemanticOperationNames.PPO_CRITIC_VALUES,
    "i_ac": SemanticOperationNames.PPO_CRITIC_VALUES,
    "t_pi": SemanticOperationNames.PPO_ACTOR_TRAIN,
    "t_V": SemanticOperationNames.PPO_CRITIC_TRAIN,
    "t_ac": SemanticOperationNames.PPO_AC_TRAIN,
}

_GRPO_REQUIRED_OPS: dict[str, str] = {
    "g_pi": SemanticOperationNames.GRPO_ACTOR_GENERATE,
    "i_ref": SemanticOperationNames.GRPO_REFERENCE_LOGPROBS,
    "i_R": SemanticOperationNames.GRPO_REWARD_SCORE,
    "t_pi": SemanticOperationNames.GRPO_ACTOR_TRAIN,
}

_DPO_REQUIRED_OPS: dict[str, str] = {
    "i_pi": SemanticOperationNames.DPO_ACTOR_LOGPROBS,
    "i_ref": SemanticOperationNames.DPO_REFERENCE_LOGPROBS,
    "t_pi": SemanticOperationNames.DPO_ACTOR_TRAIN,
}

_REQUIRED_OPS_BY_ALGORITHM = {
    "ppo": _PPO_REQUIRED_OPS,
    "grpo": _GRPO_REQUIRED_OPS,
    "dpo": _DPO_REQUIRED_OPS,
}

_RPEG_MODEL_TO_PLAN_NAME = {
    "Pi": "actor",
    "V": "critic",
    "Ref": "reference",
    "R": "reward",
    "AC": "ac_share",
}

_ACTIVATION_LABEL_TO_ARTIFACT = {
    "completions": "rollout",
    "log_pi": "rollout",
    "log_ref": "ref_logps",
    "rewards": "rewards",
    "values": "values",
}


class MegatronPPOGraphHandler:
    def __init__(
        self,
        compiled: CompiledRuntimePlan,
        *,
        step: int,
        attempt: int = 0,
        semantic_runtime: SemanticWorkerGroupRuntime | None = None,
    ) -> None:
        self.compiled = compiled
        self.run = compiled.run_config
        self.step = step
        self.attempt = attempt
        self.semantic_runtime = semantic_runtime
        self.worker_by_vertex = {
            vertex_id: worker
            for worker in compiled.workers
            for vertex_id in worker.owned_vertex_ids
        }
        self._train_nodes_by_step = _train_nodes_by_step(compiled)
        self._checkpointed_steps: set[int] = set()
        all_required_ops = _REQUIRED_OPS_BY_ALGORITHM.get(self.run.algorithm.name, _PPO_REQUIRED_OPS)
        self.required_ops = {
            vertex_id: operation
            for vertex_id, operation in all_required_ops.items()
            if vertex_id in self.worker_by_vertex
        }
        self.capabilities = self._capabilities()
        self._direct_port_lock = threading.Lock()
        self._direct_port_claims: set[tuple[str, int]] = set()
        self._direct_channel_ports: dict[tuple[str, str], int] = {}
        self._direct_weight_load_lock = threading.Lock()
        self._direct_weight_loads: set[tuple[str, str, int, bool]] = set()
        self._runtime_nodes = compiled.runtime_graph().nodes
        self._activation_transfer_lock = threading.Lock()
        self._completed_activation_transfers: dict[
            tuple[
                str,
                str,
                str,
                int,
                int,
                tuple[int, int, int],
                tuple[int, int, int],
            ],
            dict[str, object],
        ] = {}

    def validate(self) -> None:
        if self.run.algorithm.name not in _REQUIRED_OPS_BY_ALGORITHM:
            raise RuntimeConfigError(
                f"MegatronPPOGraphHandler only supports {sorted(_REQUIRED_OPS_BY_ALGORITHM)}"
            )
        if self.run.runtime.default_backend != "megatron":
            raise RuntimeConfigError("MegatronPPOGraphHandler requires runtime.default_backend=megatron")
        if self.run.cluster.launcher.type != "slurm":
            raise RuntimeConfigError("Megatron PPO requires launcher.type=slurm for distributed worker groups")
        unknown_vertices = sorted(set(self.worker_by_vertex) - set(self.required_ops))
        if unknown_vertices:
            raise RuntimeConfigError(
                f"Megatron {self.run.algorithm.name.upper()} has no semantic operation mapping for vertices: "
                f"{unknown_vertices}"
            )
        for worker in self.compiled.workers:
            _validate_worker_exactness(worker)
            if (
                worker.has_layout_transitions
                and self.run.optimizer_runtime.distributed_optimizer != "zero2"
            ):
                raise RuntimeConfigError(
                    f"{worker.worker_id}: colocated RPEG layout transitions require "
                    "optimizer_runtime.distributed_optimizer=zero2"
                )
        _validate_publication_layouts(self.compiled, self.worker_by_vertex)
        _validate_checkpoint_portability(self.compiled)
        unsupported = [cap for cap in self.capabilities if not cap.supported]
        if unsupported:
            details = "; ".join(
                f"{cap.vertex_id}:{cap.required_operation} ({cap.reason})"
                for cap in unsupported
            )
            raise RuntimeConfigError(f"Megatron PPO worker API incomplete: {details}")

    def prewarm_direct_transports(self, graph: RuntimeGraph) -> dict[str, object]:
        if self.semantic_runtime is None:
            return {}
        # Endpoint creation is cached and happens lazily on first use. Eager UCX prewarm is useful
        # for tightly controlled benchmarks, but on multi-node deployments it can dominate short
        # validation runs.
        if not _monitoring_bool(self.run.monitoring.get("ucx_direct_prewarm", False)):
            return {"ucx_direct_prewarm_enabled": False}
        if self.run.runtime.activation_transport != "ucx_direct" and self.run.runtime.weight_transport != "ucx_direct":
            return {}
        request_many = getattr(self.semantic_runtime, "request_many", None)
        if request_many is None:
            raise RuntimeConfigError("ucx_direct prewarm requires semantic_runtime.request_many")
        pairs = self._ucx_direct_prewarm_pairs(graph)
        if not pairs:
            return {"ucx_direct_prewarm_enabled": True, "ucx_direct_prewarm_pairs": 0}
        total_connections = 0
        total_rank_duration_s = 0.0
        warmed_pairs = 0
        for source_worker_id, destination_worker_id in sorted(pairs):
            source_worker = self._worker_by_id(source_worker_id)
            destination_worker = self._worker_by_id(destination_worker_id)
            endpoints = self._ucx_direct_prewarm_endpoints(source_worker, destination_worker)
            if not endpoints:
                continue
            payload = {
                "direct_transport": "ucx",
                "direct_endpoints": endpoints,
                "direct_timeout_s": float(
                    self.run.monitoring.get(
                        "ucx_direct_prewarm_timeout_s",
                        self.run.monitoring.get("operation_timeout_s", 120.0),
                    )
                ),
            }
            results = request_many(
                (
                    (
                        destination_worker.worker_id,
                        SemanticOperationNames.RECV_DIRECT_PREWARM,
                        f"runtime.ucx_prewarm.recv.{source_worker.worker_id}.{destination_worker.worker_id}",
                        0,
                        dict(payload),
                        None,
                    ),
                    (
                        source_worker.worker_id,
                        SemanticOperationNames.SEND_DIRECT_PREWARM,
                        f"runtime.ucx_prewarm.send.{source_worker.worker_id}.{destination_worker.worker_id}",
                        0,
                        dict(payload),
                        None,
                    ),
                )
            )
            warmed_pairs += 1
            for group_results in results.values():
                for result in group_results:
                    counters = result.counters or {}
                    total_connections += int(counters.get("prewarm_count") or 0)
                    total_rank_duration_s += float(counters.get("prewarm_rank_duration_s") or 0.0)
        return {
            "ucx_direct_prewarm_enabled": True,
            "ucx_direct_prewarm_pairs": warmed_pairs,
            "ucx_direct_prewarm_connections": total_connections,
            "ucx_direct_prewarm_rank_duration_s": total_rank_duration_s,
        }

    def _ucx_direct_prewarm_pairs(self, graph: RuntimeGraph) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for node in graph.nodes.values():
            if node.kind not in {ExecNodeType.COMM, ExecNodeType.RESHARD}:
                continue
            if _is_weight_label(node.label):
                model = _model_from_weight_label(node.label)
                source_vertex, destination_vertex = _weight_transfer_vertices(
                    node,
                    self.worker_by_vertex,
                    model,
                )
            else:
                source_vertex = _adjacent_vertex(node.dependencies)
                destination_vertex = _adjacent_vertex(node.successors)
            if source_vertex is None or destination_vertex is None:
                continue
            source_worker = self.worker_by_vertex.get(source_vertex)
            destination_worker = self.worker_by_vertex.get(destination_vertex)
            if source_worker is None or destination_worker is None:
                continue
            if _is_weight_label(node.label):
                if (
                    self.run.runtime.weight_transport == "ucx_direct"
                    and source_worker.worker_id != destination_worker.worker_id
                ):
                    pairs.add((source_worker.worker_id, destination_worker.worker_id))
                continue
            if self.run.runtime.activation_transport != "ucx_direct":
                continue
            source_artifact = _artifact_from_activation_label(
                node.label,
                source_vertex=source_vertex,
                algorithm=self.run.algorithm.name,
            )
            if source_artifact is None:
                continue
            if source_worker.worker_id != destination_worker.worker_id:
                pairs.add((source_worker.worker_id, destination_worker.worker_id))
            rollout_source_worker = self.worker_by_vertex.get("g_pi") or self.worker_by_vertex.get("g_ac")
            if (
                destination_vertex.startswith("t_")
                and source_artifact != "rollout"
                and self.run.algorithm.name in {"ppo", "grpo"}
                and rollout_source_worker is not None
                and rollout_source_worker.worker_id != destination_worker.worker_id
            ):
                pairs.add((rollout_source_worker.worker_id, destination_worker.worker_id))
        return pairs

    def _ucx_direct_prewarm_endpoints(
        self,
        source_worker: WorkerSpec,
        destination_worker: WorkerSpec,
    ) -> list[dict[str, object]]:
        if self.semantic_runtime is None:
            return []
        source_hosts = _semantic_rank_hosts(self.semantic_runtime, source_worker.worker_id, len(source_worker.devices))
        destination_hosts = _semantic_rank_hosts(
            self.semantic_runtime,
            destination_worker.worker_id,
            len(destination_worker.devices),
        )
        source_rank_by_host = _first_rank_by_host(source_hosts)
        destination_rank_by_host = _first_rank_by_host(destination_hosts)
        endpoints: list[dict[str, object]] = []
        if self._ucx_direct_channel_reuse_enabled():
            for source_rank, source_host in enumerate(source_hosts):
                for destination_rank, destination_host in enumerate(destination_hosts):
                    endpoints.append(
                        self._ucx_direct_channel_endpoint(
                            source_worker=source_worker,
                            destination_worker=destination_worker,
                            source_rank=source_rank,
                            destination_rank=destination_rank,
                            source_host=source_host,
                            destination_host=destination_host,
                        )
                    )
            return endpoints
        for source_host, source_rank in sorted(source_rank_by_host.items()):
            for destination_host, destination_rank in sorted(destination_rank_by_host.items()):
                endpoints.append(
                    {
                        "source_rank": source_rank,
                        "rank": destination_rank,
                        "source_host": source_host,
                        "host": destination_host,
                        "port": self._claim_direct_transfer_port(
                            destination_host,
                            f"ucx_prewarm:{source_worker.worker_id}:{destination_worker.worker_id}:"
                            f"{source_host}:{destination_host}",
                            destination_rank,
                        ),
                    }
                )
        return endpoints

    def _ucx_direct_channel_reuse_enabled(self) -> bool:
        return bool(getattr(self.run.runtime, "direct_channel_reuse", False))

    def _ucx_direct_channel_endpoint(
        self,
        *,
        source_worker: WorkerSpec,
        destination_worker: WorkerSpec,
        source_rank: int,
        destination_rank: int,
        source_host: str | None,
        destination_host: str,
        include_ack_port: bool = False,
    ) -> dict[str, object]:
        channel_id = (
            f"{self.run.run_id}:ucx_direct:{source_worker.worker_id}->{destination_worker.worker_id}:"
            f"src{int(source_rank)}:dst{int(destination_rank)}"
        )
        attempt = int(getattr(self, "attempt", 0))
        if attempt:
            channel_id = f"{channel_id}:attempt{attempt}"
        endpoint = {
            "source_rank": int(source_rank),
            "rank": int(destination_rank),
            "source_host": "" if source_host is None else str(source_host),
            "host": str(destination_host),
            "port": self._claim_direct_channel_port(str(destination_host), channel_id, int(destination_rank)),
            "direct_channel_id": channel_id,
            "direct_channel_reuse": True,
        }
        if include_ack_port:
            endpoint["ack_port"] = self._claim_direct_channel_port(
                str(destination_host), f"{channel_id}:ack", int(destination_rank)
            )
        return endpoint

    def _worker_by_id(self, worker_id: str) -> WorkerSpec:
        for worker in self.compiled.workers:
            if worker.worker_id == worker_id:
                return worker
        raise RuntimeConfigError(f"unknown semantic worker id {worker_id}")

    async def execute(self, node: RuntimeNodeSpec, context: RuntimeExecutionContext) -> NodeExecutionResult:
        if node.kind in {ExecNodeType.COMM, ExecNodeType.RESHARD}:
            return await asyncio.to_thread(self._transfer_metadata, node, context)
        if node.kind != ExecNodeType.OP:
            raise RuntimeConfigError(f"Unsupported Megatron PPO runtime node kind {node.kind}")
        vertex_id = node.vertex_id or "<unknown>"
        op = self.required_ops.get(vertex_id, "<unknown>")
        if self.semantic_runtime is None:
            raise RuntimeConfigError(
                f"Megatron PPO operation {op!r} for node {node.node_id} requires a semantic worker runtime"
            )
        if node.worker_id is None:
            raise RuntimeConfigError(f"Megatron PPO node {node.node_id} has no worker assignment")
        loaded_weight, results, produced_weight, operational_weights, implicit_transfers = await asyncio.to_thread(
            self._execute_operation_sync,
            node,
            op,
            context,
        )
        counters = _merge_rank_counters(results)
        if loaded_weight is not None:
            counters["loaded_weight"] = loaded_weight
        if produced_weight is not None:
            counters["produced_weight"] = {
                "model": produced_weight.model,
                "version": produced_weight.version,
                "shadow": produced_weight.shadow,
            }
        if operational_weights:
            counters["operational_weights"] = operational_weights
        if implicit_transfers:
            counters["implicit_activation_transfers"] = implicit_transfers
        weights = () if produced_weight is None else (produced_weight,)
        return NodeExecutionResult(node_id=node.node_id, counters=counters, weights=weights)

    def _execute_operation_sync(
        self,
        node: RuntimeNodeSpec,
        operation: str,
        context: RuntimeExecutionContext,
    ) -> tuple[
        dict[str, object] | None,
        tuple[SemanticResult, ...],
        WeightVersion | None,
        dict[str, object],
        tuple[dict[str, object], ...],
    ]:
        assert self.semantic_runtime is not None
        assert node.worker_id is not None
        implicit_transfers = self._ensure_implicit_activation_inputs(node)
        loaded_weight = self._load_required_weight_version(node, operation)
        ensured = self.semantic_runtime.request(
            node.worker_id,
            SemanticOperationNames.ENSURE_OPERATIONAL_WEIGHTS,
            node_id=f"{node.node_id}.ensure_operational_weights",
            step=_graph_node_step(node) or self.step,
            payload={"model": str(node.model)},
        )
        results = self.semantic_runtime.request(
            node.worker_id,
            operation,
            node_id=node.node_id,
            step=_graph_node_step(node) or self.step,
            payload=self._payload_for_node(node),
        )
        if node.role == "TRAIN":
            self._prune_publisher_weight_versions_before_snapshot(node, context)
        produced_weight = self._publish_produced_weight_version(node, operation)
        released: tuple[SemanticResult, ...] = ()
        if self._releases_operational_weights(node):
            released = self.semantic_runtime.request(
                node.worker_id,
                SemanticOperationNames.RELEASE_OPERATIONAL_WEIGHTS,
                node_id=f"{node.node_id}.release_operational_weights",
                step=_graph_node_step(node) or self.step,
                payload={"model": str(node.model)},
            )
        return (
            loaded_weight,
            results,
            produced_weight,
            _operational_weight_lifecycle_summary(ensured, released),
            implicit_transfers,
        )

    def _prune_publisher_weight_versions_before_snapshot(
        self,
        node: RuntimeNodeSpec,
        context: RuntimeExecutionContext,
    ) -> None:
        """
        Release dead device snapshots before cloning a newly trained version.
        """
        if self.semantic_runtime is None or node.worker_id is None or node.model is None:
            return
        required = _semantic_weight_versions_required_by_incomplete_consumers(
            self.compiled,
            context,
        )
        versions = sorted(required.get((node.worker_id, node.model), set()))
        self.semantic_runtime.request(
            node.worker_id,
            SemanticOperationNames.PRUNE_WEIGHT_VERSIONS,
            node_id=f"{node.node_id}.prune_before_snapshot",
            step=_graph_node_step(node) or self.step,
            payload={
                "keep_versions_by_model": {node.model: versions},
                # Device snapshots are transfer buffers, not a historical HBM cache. Liveness pins
                # these buffers and retains every version still needed by an incomplete
                # transfer/consumer. Keeping additional dead snapshots here grows training HBM by
                # one model shard per step.
                "preserve_latest": False,
            },
        )

    def _ensure_implicit_activation_inputs(
        self,
        node: RuntimeNodeSpec,
    ) -> tuple[dict[str, object], ...]:
        """
        Materialize colocated cross-worker edges omitted by RPEG lowering.
        """
        if self.semantic_runtime is None or node.vertex_id is None:
            return ()
        destination_worker = self.worker_by_vertex[node.vertex_id]
        destination_step = _graph_node_step(node) or self.step
        summaries: list[dict[str, object]] = []
        # PPO/GRPO training always consumes the rollout from this step. It is intentionally implicit
        # when generation and training share a persistent worker, so it need not appear as a direct
        # lowered OP predecessor. Enforce availability as a training-input invariant, including an
        # in-place layout transition when the worker reshards.
        rollout_source_worker = self.worker_by_vertex.get("g_pi") or self.worker_by_vertex.get("g_ac")
        if (
            node.vertex_id.startswith("t_")
            and self.run.algorithm.name in {"ppo", "grpo"}
            and rollout_source_worker is not None
        ):
            rollout_vertex = (
                "g_pi" if "g_pi" in rollout_source_worker.owned_vertex_ids else "g_ac"
            )
            rollout_source_parallelism = rollout_source_worker.parallelism_for(rollout_vertex)
            training_parallelism = destination_worker.parallelism_for(node.vertex_id)
            if (
                rollout_source_worker.worker_id != destination_worker.worker_id
                or rollout_source_parallelism != training_parallelism
            ):
                summaries.append(
                    self._ensure_activation_artifact(
                        node=node,
                        source_worker=rollout_source_worker,
                        destination_worker=destination_worker,
                        source_artifact="rollout",
                        source_step=destination_step,
                        destination_step=destination_step,
                        label="completions",
                        source_parallelism=rollout_source_parallelism,
                        destination_parallelism=training_parallelism,
                        colocated=True,
                    )
                )
        for dependency_id in node.dependencies:
            dependency = self._runtime_nodes.get(dependency_id)
            if dependency is None or dependency.kind != ExecNodeType.OP or dependency.vertex_id is None:
                continue
            source_worker = self.worker_by_vertex[dependency.vertex_id]
            source_parallelism = source_worker.parallelism_for(dependency.vertex_id)
            destination_parallelism = destination_worker.parallelism_for(node.vertex_id)
            if (
                source_worker.worker_id == destination_worker.worker_id
                and source_parallelism == destination_parallelism
            ):
                continue
            source_artifact = _artifact_for_direct_dependency(
                dependency.vertex_id,
                node.vertex_id,
                algorithm=self.run.algorithm.name,
            )
            if source_artifact is None:
                continue
            source_part = _split_part_index(dependency.node_id)
            destination_part = _split_part_index(node.node_id)
            if (
                source_part is not None
                and destination_part is not None
                and dependency.vertex_id in {"g_pi", "g_ac"}
            ):
                source_artifact = f"{source_artifact}.part{source_part}"
            source_step = _graph_node_step(dependency) or destination_step
            summary = self._ensure_activation_artifact(
                node=node,
                source_worker=source_worker,
                destination_worker=destination_worker,
                source_artifact=source_artifact,
                source_step=source_step,
                destination_step=destination_step,
                label=_activation_label_for_artifact(
                    source_artifact,
                    source_vertex=dependency.vertex_id,
                    algorithm=self.run.algorithm.name,
                ),
                source_parallelism=source_parallelism,
                destination_parallelism=destination_parallelism,
                colocated=True,
            )
            summaries.append(summary)

        return tuple(summaries)

    def _ensure_activation_artifact(
        self,
        *,
        node: RuntimeNodeSpec,
        source_worker: WorkerSpec,
        destination_worker: WorkerSpec,
        source_artifact: str,
        source_step: int,
        destination_step: int,
        label: str,
        source_parallelism: tuple[int, int, int],
        destination_parallelism: tuple[int, int, int],
        colocated: bool = False,
    ) -> dict[str, object]:
        key = (
            source_worker.worker_id,
            destination_worker.worker_id,
            source_artifact,
            int(source_step),
            int(destination_step),
            tuple(source_parallelism),
            tuple(destination_parallelism),
        )
        with self._activation_transfer_lock:
            cached = self._completed_activation_transfers.get(key)
        if cached is not None:
            return {**cached, "cached": True}
        if (
            source_worker.worker_id == destination_worker.worker_id
            and source_parallelism != destination_parallelism
        ):
            assert self.semantic_runtime is not None
            results = self.semantic_runtime.request(
                destination_worker.worker_id,
                SemanticOperationNames.REPARTITION_LOCAL_ACTIVATION,
                node_id=f"{node.node_id}.repartition_{source_artifact}",
                step=destination_step,
                payload={
                    "source_worker_id": source_worker.worker_id,
                    "source_name": source_artifact,
                    "destination_name": source_artifact,
                    "source_step": source_step,
                    "destination_step": destination_step,
                    "label": label,
                    "source_parallelism": list(source_parallelism),
                    "destination_parallelism": list(destination_parallelism),
                    "global_batch_size": (
                        self.run.workload.batch_size
                        * self.run.workload.samples_per_prompt
                        if ".part" in source_artifact
                        else _artifact_global_batch_size(self.run, source_artifact)
                    ),
                    "global_index_repartition": ".part" in source_artifact,
                },
            )
        elif colocated and _resolved_activation_store(self.run) != "memory":
            assert self.semantic_runtime is not None
            results = self.semantic_runtime.request(
                destination_worker.worker_id,
                SemanticOperationNames.TRANSFER_ACTIVATION,
                node_id=f"{node.node_id}.transfer_{source_artifact}",
                step=destination_step,
                payload={
                    "source_worker_id": source_worker.worker_id,
                    "source_name": source_artifact,
                    "destination_name": source_artifact,
                    "source_step": source_step,
                    "destination_step": destination_step,
                    "label": label,
                    # Let the semantic backend choose hardlink, reflink, or copy according to the
                    # shared filesystem's capabilities.
                    "activation_transport": "auto",
                },
            )
        else:
            results = self._transfer_activation_artifact(
                node=node,
                step=destination_step,
                source_worker=source_worker,
                destination_worker=destination_worker,
                source_artifact=source_artifact,
                source_step=source_step,
                label=label,
                source_parallelism=source_parallelism,
                destination_parallelism=destination_parallelism,
            )
        summary = {
            "source_worker_id": source_worker.worker_id,
            "destination_worker_id": destination_worker.worker_id,
            "source_artifact": source_artifact,
            "source_step": int(source_step),
            "destination_step": int(destination_step),
            "colocated": colocated,
            **_semantic_transfer_summary(results),
        }
        with self._activation_transfer_lock:
            self._completed_activation_transfers[key] = summary
        return summary

    def _releases_operational_weights(self, node: RuntimeNodeSpec) -> bool:
        if node.model not in {"actor", "critic", "ac_share"} or node.worker_id is None:
            return False
        worker = self._worker_by_id(node.worker_id)
        if worker.has_layout_transitions:
            return False
        return not uses_shadow_copies(self.compiled, str(node.model))

    def select_batch(
        self,
        node: RuntimeNodeSpec,
        ready_nodes: tuple[RuntimeNodeSpec, ...],
        context: RuntimeExecutionContext,
    ) -> tuple[RuntimeNodeSpec, ...]:
        del context
        if node.exec_node.fusion_group is not None:
            fused = tuple(
                candidate
                for candidate in ready_nodes
                if candidate.exec_node.fusion_group == node.exec_node.fusion_group
            )
            if {candidate.vertex_id for candidate in fused} == {"t_pi", "t_V"}:
                return (node,) + tuple(candidate for candidate in fused if candidate.node_id != node.node_id)
        if self.semantic_runtime is None or not hasattr(self.semantic_runtime, "request_batch"):
            return (node,)
        if not _env_flag("RPEG_DYNAMIC_GENERATION_BATCHING", True):
            return (node,)
        max_requests = max(1, int(os.getenv("RPEG_DYNAMIC_GENERATION_BATCH_MAX_REQUESTS", "8")))
        if max_requests <= 1:
            return (node,)
        key = self._generation_batch_key_for_node(node)
        if key is None:
            return (node,)
        selected = [node]
        for candidate in ready_nodes:
            if len(selected) >= max_requests:
                break
            if candidate.node_id == node.node_id:
                continue
            if self._generation_batch_key_for_node(candidate) == key:
                selected.append(candidate)
        return tuple(selected)

    async def execute_batch(
        self,
        nodes: tuple[RuntimeNodeSpec, ...],
        context: RuntimeExecutionContext,
    ) -> tuple[NodeExecutionResult, ...]:
        if (
            len(nodes) == 2
            and nodes[0].exec_node.fusion_group is not None
            and all(node.exec_node.fusion_group == nodes[0].exec_node.fusion_group for node in nodes)
            and {node.vertex_id for node in nodes} == {"t_pi", "t_V"}
        ):
            if self.run.runtime.intra_stage_fusion:
                return await asyncio.to_thread(self._execute_fused_training_sync, nodes, context)
            results = await asyncio.gather(*(self.execute(node, context) for node in nodes))
            for result in results:
                result.counters["intra_stage_fusion"] = False
                result.counters["actor_critic_concurrent_dispatch"] = True
                result.counters["intra_stage_fusion_group"] = nodes[0].exec_node.fusion_group
                result.counters["intra_stage_fusion_schedule"] = "uncoordinated_concurrent_dispatch"
                result.counters["coordinated_intra_stage_1f1b"] = False
                microbatch_count = int(nodes[0].microbatch or 1)
                global_sequences = (
                    int(self.run.workload.batch_size)
                    * int(self.run.workload.samples_per_prompt)
                )
                data_parallel_size = int((nodes[0].parallelism or (1, 1, 1))[0])
                local_sequences = global_sequences // data_parallel_size
                result.counters["intra_stage_fusion_microbatch_count"] = microbatch_count
                result.counters["intra_stage_fusion_microbatch_size"] = (
                    local_sequences // microbatch_count
                )
            return tuple(results)
        if len(nodes) <= 1 or self._generation_batch_key_for_node(nodes[0]) is None:
            return tuple([await self.execute(nodes[0], context)])
        if self.semantic_runtime is None or not hasattr(self.semantic_runtime, "request_batch"):
            results = []
            for node in nodes:
                results.append(await self.execute(node, context))
            return tuple(results)
        first_key = self._generation_batch_key_for_node(nodes[0])
        if any(self._generation_batch_key_for_node(node) != first_key for node in nodes[1:]):
            results = []
            for node in nodes:
                results.append(await self.execute(node, context))
            return tuple(results)
        return await asyncio.to_thread(self._execute_generation_batch, nodes)

    def _fused_training_schedule(self, nodes: tuple[RuntimeNodeSpec, ...]):
        by_vertex = {node.vertex_id: node for node in nodes}
        actor = by_vertex.get("t_pi")
        critic = by_vertex.get("t_V")
        if actor is None or critic is None:
            raise RuntimeConfigError("fused PPO training requires t_pi and t_V")
        if actor.parallelism != critic.parallelism or actor.microbatch != critic.microbatch:
            raise RuntimeConfigError("fused PPO training nodes must have identical layouts and microbatches")
        if actor.parallelism is None or actor.microbatch is None:
            raise RuntimeConfigError("fused PPO training nodes are missing layout metadata")
        _dp, pp, _tp = actor.parallelism
        actor_work = float(self.compiled.model_suite.n_params(Model.ACTOR))
        critic_work = float(self.compiled.model_suite.n_params(Model.CRITIC))
        work_scale = max(actor_work, critic_work, 1.0)
        actor_activation = int(training_activation_bytes_per_microbatch(
            actor.exec_node,
            self.compiled.model_suite,
            self.compiled.workload,
            activation_checkpointing=self.run.optimizer_runtime.activation_checkpointing,
        ))
        critic_activation = int(training_activation_bytes_per_microbatch(
            critic.exec_node,
            self.compiled.model_suite,
            self.compiled.workload,
            activation_checkpointing=self.run.optimizer_runtime.activation_checkpointing,
        ))
        return build_fused_pipeline_schedule((
            FusedPipelineModel.uniform(
                "actor",
                pipeline_stages=int(pp),
                microbatches=int(actor.microbatch),
                forward_s=actor_work / work_scale,
                backward_s=2.0 * actor_work / work_scale,
                activation_bytes=actor_activation,
            ),
            FusedPipelineModel.uniform(
                "critic",
                pipeline_stages=int(pp),
                microbatches=int(critic.microbatch),
                forward_s=critic_work / work_scale,
                backward_s=2.0 * critic_work / work_scale,
                activation_bytes=critic_activation,
                reverse=True,
            ),
        ))

    def _execute_fused_training_sync(
        self,
        nodes: tuple[RuntimeNodeSpec, ...],
        context: RuntimeExecutionContext,
    ) -> tuple[NodeExecutionResult, ...]:
        if self.semantic_runtime is None:
            raise RuntimeConfigError("fused PPO training requires a semantic worker runtime")
        schedule = self._fused_training_schedule(nodes)
        fusion_group = str(nodes[0].exec_node.fusion_group)
        sync_root = os.getenv(
            "RPEG_FUSION_SYNC_ROOT",
            f"/dev/shm/rpeg-fusion-{os.getenv('SLURM_JOB_ID', self.run.run_id)}",
        )
        implicit_by_node = {
            node.node_id: self._ensure_implicit_activation_inputs(node)
            for node in nodes
        }
        ensure_requests = tuple(
            (
                str(node.worker_id),
                SemanticOperationNames.ENSURE_OPERATIONAL_WEIGHTS,
                f"{node.node_id}.ensure_operational_weights",
                _graph_node_step(node) or self.step,
                {"model": str(node.model)},
                None,
            )
            for node in nodes
        )
        ensured = self.semantic_runtime.request_many(ensure_requests)
        fused_requests = []
        for node in nodes:
            if node.worker_id is None:
                raise RuntimeConfigError(f"fused PPO node {node.node_id} has no worker assignment")
            payload = self._payload_for_node(node)
            payload.update({
                "fused_model": "actor" if node.vertex_id == "t_pi" else "critic",
                "fusion_group": fusion_group,
                "fusion_attempt": int(self.attempt),
                "fused_pipeline_schedule": schedule.to_dict(include_timings=True),
                "fused_pipeline_schedule_hash": schedule.schedule_hash,
                "fusion_sync_root": sync_root,
                "fusion_sync_backend": self.run.runtime.fusion_sync_backend,
            })
            fused_requests.append((
                node.worker_id,
                SemanticOperationNames.PPO_FUSED_ACTOR_CRITIC_TRAIN,
                node.node_id,
                _graph_node_step(node) or self.step,
                payload,
                None,
            ))
        results_by_worker = self.semantic_runtime.request_many(tuple(fused_requests))

        published: dict[str, WeightVersion | None] = {}
        for node in nodes:
            original_op = self.required_ops[str(node.vertex_id)]
            self._prune_publisher_weight_versions_before_snapshot(node, context)
            published[node.node_id] = self._publish_produced_weight_version(node, original_op)
        release_requests = tuple(
            (
                str(node.worker_id),
                SemanticOperationNames.RELEASE_OPERATIONAL_WEIGHTS,
                f"{node.node_id}.release_operational_weights",
                _graph_node_step(node) or self.step,
                {"model": str(node.model)},
                None,
            )
            for node in nodes
            if self._releases_operational_weights(node)
        )
        released = self.semantic_runtime.request_many(release_requests) if release_requests else {}

        out = []
        for node in nodes:
            assert node.worker_id is not None
            counters = _merge_rank_counters(results_by_worker[node.worker_id])
            counters.update({
                "intra_stage_fusion": True,
                "actor_critic_concurrent_dispatch": False,
                "intra_stage_fusion_group": fusion_group,
                "intra_stage_fusion_schedule": "coordinated_bidirectional_microtasks",
                "coordinated_intra_stage_1f1b": True,
                "fused_pipeline_schedule_hash": schedule.schedule_hash,
                "fused_pipeline_makespan_units": schedule.makespan_s,
                "fused_pipeline_peak_activation_bytes": list(schedule.peak_activation_bytes),
                "fused_pipeline_optimizer": schedule.optimizer,
                "fused_pipeline_optimizer_candidates": schedule.optimizer_candidates,
                "fused_pipeline_baseline_makespan_units": schedule.baseline_makespan_s,
            })
            if implicit_by_node[node.node_id]:
                counters["implicit_activation_transfers"] = implicit_by_node[node.node_id]
            lifecycle = _operational_weight_lifecycle_summary(
                ensured.get(node.worker_id, ()),
                released.get(node.worker_id, ()),
            )
            if lifecycle:
                counters["operational_weights"] = lifecycle
            weight = published[node.node_id]
            if weight is not None:
                counters["produced_weight"] = {
                    "model": weight.model,
                    "version": weight.version,
                    "shadow": weight.shadow,
                }
            out.append(NodeExecutionResult(
                node_id=node.node_id,
                counters=counters,
                weights=() if weight is None else (weight,),
            ))
        return tuple(out)

    def _generation_batch_key_for_node(self, node: RuntimeNodeSpec) -> tuple[object, ...] | None:
        if node.kind != ExecNodeType.OP or node.worker_id is None:
            return None
        vertex_id = node.vertex_id or "<unknown>"
        op = self.required_ops.get(vertex_id)
        if op not in {SemanticOperationNames.PPO_ACTOR_GENERATE, SemanticOperationNames.GRPO_ACTOR_GENERATE}:
            return None
        payload = self._payload_for_node(node)
        if payload.get("artifact_part") is not None:
            return None
        do_sample = bool(payload.get("do_sample", True))
        seed = (int(payload.get("seed", 1234)) + int(_graph_node_step(node) or self.step) * 100_000) if do_sample else None
        model = "ac_share" if node.model == "ac_share" else "actor"
        step = _graph_node_step(node) or self.step
        return (
            node.worker_id,
            op,
            int(payload["generation_length"]),
            str(payload.get("generation_backend", "full_sequence")),
            do_sample,
            float(payload.get("temperature", 1.0)),
            float(payload.get("top_p", 1.0)),
            int(payload.get("top_k", 0)),
            seed,
            int(payload.get("samples_per_prompt", 1)),
            consumer_weight_version(self.compiled, model, step),
            uses_shadow_copies(self.compiled, model),
        )

    def _execute_generation_batch(self, nodes: tuple[RuntimeNodeSpec, ...]) -> tuple[NodeExecutionResult, ...]:
        if self.semantic_runtime is None:
            raise RuntimeConfigError("batched Megatron generation requires a semantic worker runtime")
        first = nodes[0]
        if first.worker_id is None:
            raise RuntimeConfigError(f"Megatron PPO node {first.node_id} has no worker assignment")
        op = self.required_ops.get(first.vertex_id or "<unknown>", "<unknown>")
        loaded_weight = self._load_required_weight_version(first, op)
        ensured = self.semantic_runtime.request(
            first.worker_id,
            SemanticOperationNames.ENSURE_OPERATIONAL_WEIGHTS,
            node_id=f"{first.node_id}.ensure_operational_weights",
            step=_graph_node_step(first) or self.step,
            payload={"model": str(first.model)},
        )
        request_batch = getattr(self.semantic_runtime, "request_batch", None)
        if request_batch is None:
            raise RuntimeConfigError("batched Megatron generation requires semantic_runtime.request_batch")
        requests = tuple(
            (
                op,
                node.node_id,
                _graph_node_step(node) or self.step,
                self._payload_for_node(node),
                None,
            )
            for node in nodes
        )
        results_by_node = request_batch(first.worker_id, requests)
        released: tuple[SemanticResult, ...] = ()
        if self._releases_operational_weights(first):
            released = self.semantic_runtime.request(
                first.worker_id,
                SemanticOperationNames.RELEASE_OPERATIONAL_WEIGHTS,
                node_id=f"{first.node_id}.release_operational_weights",
                step=_graph_node_step(first) or self.step,
                payload={"model": str(first.model)},
            )
        lifecycle = _operational_weight_lifecycle_summary(ensured, released)
        out: list[NodeExecutionResult] = []
        for node in nodes:
            results = results_by_node[node.node_id]
            counters = _merge_rank_counters(results)
            if loaded_weight is not None:
                counters["loaded_weight"] = dict(loaded_weight)
            if lifecycle:
                counters["operational_weights"] = lifecycle
            out.append(NodeExecutionResult(node_id=node.node_id, counters=counters))
        return tuple(out)

    def on_node_complete(
        self,
        node: RuntimeNodeSpec,
        context: RuntimeExecutionContext,
        result: NodeExecutionResult,
    ) -> None:
        del result
        if self.semantic_runtime is None:
            return
        if node.role == "TRAIN":
            step = _graph_node_step(node)
            if step is not None and step not in self._checkpointed_steps:
                train_nodes = self._train_nodes_by_step.get(step)
                if train_nodes and train_nodes <= set(context.completed):
                    _save_semantic_checkpoints_if_needed(self.compiled, self.semantic_runtime, step)
                    self._release_semantic_step_activations(step)
                    self._checkpointed_steps.add(step)
                    if self._should_inject_failure_after_checkpoint(step):
                        raise RuntimeExecutionError(
                            f"injected Megatron graph failure after checkpoint step {step} "
                            f"on attempt {self.attempt}"
                        )
        if (
            node.kind == ExecNodeType.OP
            and (node.role == "TRAIN" or node.role in _WEIGHT_CONSUMER_ROLES)
        ) or (
            node.kind in {ExecNodeType.COMM, ExecNodeType.RESHARD}
            and _is_weight_label(node.label)
        ):
            _prune_semantic_weight_versions(self.compiled, self.semantic_runtime, context)

    def _release_semantic_step_activations(self, step: int) -> None:
        """
        Release a step only after every graph-level training consumer ends.
        """
        if self.semantic_runtime is None:
            return
        workers = tuple(sorted({worker.worker_id for worker in self.compiled.workers}))
        request_many = getattr(self.semantic_runtime, "request_many", None)
        if request_many is not None:
            request_many(tuple(
                (
                    worker_id,
                    SemanticOperationNames.RELEASE_ACTIVATIONS,
                    f"runtime.release_activations.step{step}.{worker_id}",
                    step,
                    {"step": step, "remove_persisted": True},
                    None,
                )
                for worker_id in workers
            ))
        else:
            for worker_id in workers:
                self.semantic_runtime.request(
                    worker_id,
                    SemanticOperationNames.RELEASE_ACTIVATIONS,
                    node_id=f"runtime.release_activations.step{step}.{worker_id}",
                    step=step,
                    payload={"step": step, "remove_persisted": True},
                )
        with self._activation_transfer_lock:
            for key in tuple(self._completed_activation_transfers):
                if key[3] == step or key[4] == step:
                    self._completed_activation_transfers.pop(key, None)

    def _should_inject_failure_after_checkpoint(self, step: int) -> bool:
        raw_step = self.run.monitoring.get("test_inject_failure_after_checkpoint_step")
        if raw_step is None:
            return False
        raw_attempt = self.run.monitoring.get("test_inject_failure_attempt", 0)
        return int(raw_step) == step and int(raw_attempt) == self.attempt

    def _load_required_weight_version(self, node: RuntimeNodeSpec, operation: str) -> dict[str, object] | None:
        if self.semantic_runtime is None or node.worker_id is None:
            return None
        if operation in {SemanticOperationNames.PPO_ACTOR_GENERATE, SemanticOperationNames.GRPO_ACTOR_GENERATE}:
            model = "ac_share" if node.model == "ac_share" else "actor"
        elif operation == SemanticOperationNames.DPO_ACTOR_LOGPROBS:
            model = "actor"
        elif operation == SemanticOperationNames.PPO_CRITIC_VALUES:
            model = "ac_share" if node.model == "ac_share" else "critic"
        else:
            return None
        step = _graph_node_step(node) or self.step
        version = consumer_weight_version(self.compiled, model, step)
        shadow = uses_shadow_copies(self.compiled, model)
        loaded: dict[str, object] = {"model": model, "version": version, "shadow": shadow}
        direct_transfer = self._maybe_transfer_weight_for_load(
            node=node,
            step=step,
            model=model,
            version=version,
            shadow=shadow,
        )
        if direct_transfer is not None:
            loaded["transfer"] = direct_transfer
        self.semantic_runtime.request(
            node.worker_id,
            SemanticOperationNames.LOAD_WEIGHT_VERSION,
            node_id=f"{node.node_id}.load_{model}_v{version}",
            step=step,
            payload={"model": model, "version": version, "strict": True, "shadow": shadow},
        )
        return loaded

    def _maybe_transfer_weight_for_load(
        self,
        *,
        node: RuntimeNodeSpec,
        step: int,
        model: str,
        version: int,
        shadow: bool,
    ) -> dict[str, object] | None:
        if (
            self.semantic_runtime is None
            or node.worker_id is None
            or self.run.runtime.weight_transport not in _PAIRED_DIRECT_WEIGHT_TRANSPORTS
            or version <= 0
        ):
            return None
        destination_worker = self.worker_by_vertex.get(node.vertex_id or "")
        source_worker = self._train_worker_for_model(model)
        if destination_worker is None or source_worker is None:
            return None
        if source_worker.worker_id == destination_worker.worker_id:
            return None
        if source_worker.parallelism != destination_worker.parallelism:
            return None
        if len(source_worker.devices) != len(destination_worker.devices):
            return None
        cache_key = (destination_worker.worker_id, model, int(version), bool(shadow))
        with self._direct_weight_load_lock:
            if cache_key in self._direct_weight_loads:
                return {"transport": self.run.runtime.weight_transport, "cached": True}
            self._direct_weight_loads.add(cache_key)
        try:
            results = self._transfer_weight_version_direct(
                node=node,
                step=step,
                model=model,
                source_worker=source_worker,
                destination_worker=destination_worker,
                source_version=version,
                shadow=shadow,
            )
        except Exception:
            with self._direct_weight_load_lock:
                self._direct_weight_loads.discard(cache_key)
            raise
        return _merge_rank_counters(results)

    def _train_worker_for_model(self, model: str) -> WorkerSpec | None:
        for worker in self.compiled.workers:
            if worker.role == "TRAIN" and worker.model == model:
                return worker
        return None

    def _publish_produced_weight_version(self, node: RuntimeNodeSpec, operation: str) -> WeightVersion | None:
        if self.semantic_runtime is None or node.worker_id is None:
            return None
        if operation in {
            SemanticOperationNames.PPO_ACTOR_TRAIN,
            SemanticOperationNames.GRPO_ACTOR_TRAIN,
            SemanticOperationNames.DPO_ACTOR_TRAIN,
        }:
            model = "actor"
        elif operation == SemanticOperationNames.PPO_CRITIC_TRAIN:
            model = "critic"
        elif operation == SemanticOperationNames.PPO_AC_TRAIN:
            model = "ac_share"
        else:
            return None
        step = _graph_node_step(node) or self.step
        shadow = uses_shadow_copies(self.compiled, model)
        cross_worker_publication = len({
            worker.worker_id
            for worker in self.compiled.workers
            if worker.model == model
        }) > 1
        self.semantic_runtime.request(
            node.worker_id,
            SemanticOperationNames.SAVE_WEIGHT_VERSION,
            node_id=f"{node.node_id}.save_{model}_v{step}",
            step=step,
            payload={
                "model": model,
                "version": step,
                "step": step,
                "shadow": shadow,
                "include_optimizer_state": False,
                "active_only": not cross_worker_publication,
                "persist": cross_worker_publication
                and self.run.runtime.weight_transport not in {"ucx_direct", "nccl_direct"},
                "device_snapshot": cross_worker_publication
                and self.run.runtime.weight_transport in {"ucx_direct", "nccl_direct"},
            },
        )
        return WeightVersion(
            model=model,
            version=step,
            produced_by=node.node_id,
            step=step,
            layout=ShardingSpec("worker", (node.worker_id,)),
            optimizer_state_owner=node.worker_id,
            shadow=shadow,
            metadata={"backend": "megatron", "publication": "file"},
        )

    def _payload_for_node(self, node: RuntimeNodeSpec) -> dict[str, object]:
        step = _graph_node_step(node) or self.step
        batch_size = self.run.workload.batch_size * self.run.workload.samples_per_prompt
        payload: dict[str, object] = {
            "vertex_id": node.vertex_id,
            "model": node.model,
            "role": node.role,
            "parallelism": node.parallelism,
            "microbatch": node.microbatch,
            "step": step,
            "context_length": self.run.workload.context_length,
            "generation_length": self.run.workload.generation_length,
            "max_generation_length": self.run.workload.generation_length,
            "batch_size": batch_size,
            "samples_per_prompt": self.run.workload.samples_per_prompt,
            "clip_range": self.run.algorithm.clip_range,
            "kl_coef": self.run.algorithm.kl_coef,
            "beta": self.run.algorithm.beta,
            "gamma": self.run.algorithm.gamma,
            "gae_lambda": self.run.algorithm.gae_lambda,
            "epochs": self.run.algorithm.epochs,
            "minibatch_size": self.run.algorithm.minibatch_size,
            "entropy_coef": self.run.algorithm.entropy_coef,
            "normalize_advantages": self.run.algorithm.normalize_advantages,
            "do_sample": self.run.algorithm.do_sample,
            "temperature": self.run.algorithm.temperature,
            "top_p": self.run.algorithm.top_p,
            "top_k": self.run.algorithm.top_k,
            "generation_backend": self.run.algorithm.generation_backend,
            "learning_rate": self.run.optimizer_runtime.learning_rate_critic
            if node.model == "critic"
            else self.run.optimizer_runtime.learning_rate_actor,
            "seed": self.run.algorithm.seed,
        }
        if node.exec_node.n_seqs is not None:
            payload["partial_n_seqs"] = node.exec_node.n_seqs
            payload["partial_n_active"] = node.exec_node.n_active
            payload["partial_phase_steps"] = node.exec_node.phase_steps
            payload["partial_phase_t_start"] = node.exec_node.phase_t_start
            payload["partial_is_prefill_phase"] = node.exec_node.is_prefill_phase
            part = _split_part_index(node.node_id)
            if part is not None:
                payload["artifact_part"] = part
                payload["artifact_parts"] = len(self.run.workload.gen_quantiles or ())
                payload["partial_batch_offset"] = _quantile_batch_offset(self.run, part)
                if node.vertex_id in {"g_pi", "g_ac"}:
                    payload["generation_length"] = _quantile_generation_length(self.run, part)
        if self.run.data.train_prompts:
            payload["train_prompts"] = self.run.data.train_prompts
        if self.run.data.preferences:
            payload["preferences"] = self.run.data.preferences
        if self.run.data.max_prompt_length is not None:
            payload["max_prompt_length"] = self.run.data.max_prompt_length
        if self.run.data.max_generation_length is not None:
            payload["max_generation_length"] = self.run.data.max_generation_length
        payload["data_seed"] = self.run.data.seed
        payload["data_shuffle"] = self.run.data.shuffle
        payload["prompt_offset"] = self.run.data.prompt_offset
        payload["preference_offset"] = self.run.data.preference_offset
        payload["data_shard_index"] = self.run.data.shard_index
        payload["data_num_shards"] = self.run.data.num_shards

        tokenizer_checkpoint = self.run.tokenizer.checkpoint_path
        if tokenizer_checkpoint is None and self.run.tokenizer.source == "local":
            actor_cfg = self.run.models.get("actor")
            if actor_cfg is not None:
                actor_dir = resolve_model_path(
                    actor_cfg.source,
                    actor_cfg.repo,
                    actor_cfg.checkpoint_path,
                    revision=actor_cfg.revision,
                )
                tokenizer_checkpoint = str(actor_dir)
        if tokenizer_checkpoint is not None or self.run.tokenizer.source != "local":
            tokenizer_path = resolve_model_path(
                self.run.tokenizer.source,
                self.run.tokenizer.repo,
                tokenizer_checkpoint,
                revision=self.run.tokenizer.revision,
            )
            payload["tokenizer_path"] = str(tokenizer_path)
        payload["tokenizer_padding_side"] = self.run.tokenizer.padding_side
        payload["tokenizer_truncation_side"] = self.run.tokenizer.truncation_side
        payload["tokenizer_trust_remote_code"] = bool(self.run.models.get("actor") and self.run.models["actor"].trust_remote_code)
        return payload

    def _capabilities(self) -> tuple[MegatronPPOCapability, ...]:
        with tempfile.TemporaryDirectory(prefix="rpeg_megatron_cap_") as tmp:
            backend_caps = MegatronSemanticBackend(
                worker_id="capability_probe",
                rank=0,
                world_size=1,
                state_root=tmp,
            ).capabilities()
        return tuple(
            MegatronPPOCapability(
                vertex_id=vertex_id,
                required_operation=operation,
                supported=backend_caps.get(operation).supported if operation in backend_caps else False,
                reason=backend_caps.get(operation).reason if operation in backend_caps else "operation is not registered",
            )
            for vertex_id, operation in self.required_ops.items()
        )

    def _transfer_metadata(self, node: RuntimeNodeSpec, context: RuntimeExecutionContext) -> NodeExecutionResult:
        step = _graph_node_step(node) or self.step
        weight, weight_transfer = self._transfer_weight_version(node, step)
        activation = self._transfer_activation(node, step) if weight is None else None
        counters = {"kind": node.kind.value, "payload_in_bytes": node.payload_in_bytes}
        if weight is not None:
            counters["weight_transfer"] = {
                "model": weight.model,
                "version": weight.version,
                "producer": weight.produced_by,
                "shadow": weight.shadow,
            }
            if weight_transfer is not None:
                counters["weight_transfer"].update(weight_transfer)
        if activation is not None:
            counters["activation_transfer"] = activation
        ref = context.store.put_container(
            ContainerRef(
                name=f"{node.node_id}:{node.label or node.kind.value}:{step}",
                step=step,
                producer_node_id=node.node_id,
                sharding=ShardingSpec(node.kind.value),
                metadata={"payload_in_bytes": node.payload_in_bytes, "label": node.label},
            )
        )
        weights = () if weight is None else (weight,)
        return NodeExecutionResult(
            node_id=node.node_id,
            counters=counters,
            containers=(ref,),
            weights=weights,
        )

    def _transfer_weight_version(self, node: RuntimeNodeSpec, step: int) -> tuple[WeightVersion | None, dict[str, object] | None]:
        if self.semantic_runtime is None or not _is_weight_label(node.label):
            return None, None
        model = _model_from_weight_label(node.label)
        source_vertex, destination_vertex = _weight_transfer_vertices(
            node,
            self.worker_by_vertex,
            model,
        )
        source_worker = self.worker_by_vertex[source_vertex]
        destination_worker = self.worker_by_vertex[destination_vertex]
        source_parallelism = source_worker.parallelism_for(source_vertex)
        destination_parallelism = destination_worker.parallelism_for(destination_vertex)
        source_step = _adjacent_step(node.dependencies) or step
        if source_worker.role_for(source_vertex) == "TRAIN" and source_worker.model == model:
            source_version = source_step
        else:
            source_version = consumer_weight_version(self.compiled, model, source_step)
        if source_worker.worker_id == destination_worker.worker_id:
            if source_parallelism == destination_parallelism:
                return None, None
            results = self.semantic_runtime.request(
                source_worker.worker_id,
                SemanticOperationNames.SWITCH_MODEL_LAYOUT,
                node_id=f"{node.node_id}.switch_{model}_v{source_version}",
                step=step,
                payload={
                    "model": model,
                    "source_parallelism": source_parallelism,
                    "destination_parallelism": destination_parallelism,
                    "microbatch": destination_worker.microbatch_for(destination_vertex),
                    "source_version": source_version,
                    "step": step,
                },
            )
            merged = _merge_rank_counters(results)
            return (
                WeightVersion(
                    model=model,
                    version=source_version,
                    produced_by=node.node_id,
                    step=step,
                    layout=ShardingSpec(
                        "in_place_worker",
                        (destination_worker.worker_id,),
                        metadata={
                            "source_parallelism": source_parallelism,
                            "destination_parallelism": destination_parallelism,
                        },
                    ),
                    optimizer_state_owner=destination_worker.worker_id,
                    shadow=False,
                    metadata={"backend": "megatron", "publication": "in_place_reshard"},
                ),
                merged,
            )
        shadow = uses_shadow_copies(self.compiled, model)
        if (
            self.run.runtime.weight_transport in _PAIRED_DIRECT_WEIGHT_TRANSPORTS
            and (
                (
                    source_parallelism == destination_parallelism
                    and len(source_worker.devices) == len(destination_worker.devices)
                )
                or self.run.runtime.weight_transport in {"ucx_direct", "nccl_direct"}
            )
        ):
            results = self._transfer_weight_version_direct(
                node=node,
                step=step,
                model=model,
                source_worker=source_worker,
                destination_worker=destination_worker,
                source_version=source_version,
                shadow=shadow,
                source_parallelism=source_parallelism,
                destination_parallelism=destination_parallelism,
            )
            # The explicit RPEG weight edge has populated this destination. Record the same cache
            # identity used by the load-time fallback, so the successor inference operation does not
            # launch a redundant second transfer of the old version. Besides wasting bandwidth, that
            # duplicate can race the next training publication, whose liveness pruning is then
            # entitled to release the old ephemeral device snapshot.
            with self._direct_weight_load_lock:
                self._direct_weight_loads.add(
                    (
                        destination_worker.worker_id,
                        model,
                        int(source_version),
                        bool(shadow),
                    )
                )
            merged = _merge_rank_counters(results)
            return (
                WeightVersion(
                    model=model,
                    version=source_version,
                    produced_by=node.node_id,
                    step=step,
                    layout=ShardingSpec(
                        "ucx_direct_worker" if self.run.runtime.weight_transport == "ucx_direct" else "direct_worker",
                        (destination_worker.worker_id,),
                        metadata={
                            "source_worker_id": source_worker.worker_id,
                            "parallelism": source_worker.parallelism,
                            "transport": self.run.runtime.weight_transport,
                        },
                    ),
                    optimizer_state_owner=destination_worker.worker_id,
                    shadow=shadow,
                    metadata={"backend": "megatron", "publication": self.run.runtime.weight_transport},
                ),
                merged,
            )
        results = self.semantic_runtime.request(
            destination_worker.worker_id,
            SemanticOperationNames.RESHARD_WEIGHT_VERSION,
            node_id=f"{node.node_id}.reshard_{model}_v{source_version}",
            step=step,
            payload={
                "model": model,
                "source_worker_id": source_worker.worker_id,
                "source_version": source_version,
                "destination_version": source_version,
                "source_world_size": len(source_worker.devices),
                "source_parallelism": source_parallelism,
                "destination_parallelism": destination_parallelism,
                "shadow": shadow,
                "step": step,
            },
        )
        merged = _merge_rank_counters(results)
        return WeightVersion(
            model=model,
            version=source_version,
            produced_by=node.node_id,
            step=step,
            layout=ShardingSpec(
                "resharded_worker",
                (destination_worker.worker_id,),
                metadata={
                    "source_worker_id": source_worker.worker_id,
                    "source_parallelism": source_parallelism,
                    "destination_parallelism": destination_parallelism,
                },
            ),
            optimizer_state_owner=destination_worker.worker_id,
            shadow=shadow,
            metadata={"backend": "megatron", "publication": "file_reshard"},
        ), merged

    def _transfer_weight_version_direct(
        self,
        *,
        node: RuntimeNodeSpec,
        step: int,
        model: str,
        source_worker: WorkerSpec,
        destination_worker: WorkerSpec,
        source_version: int,
        shadow: bool,
        source_parallelism: tuple[int, int, int] | None = None,
        destination_parallelism: tuple[int, int, int] | None = None,
    ) -> tuple[SemanticResult, ...]:
        if self.semantic_runtime is None:
            raise RuntimeConfigError("direct weight transfer requires a semantic worker runtime")
        direct_transport, direct_backend = _direct_weight_transport(self.run.runtime.weight_transport)
        source_parallelism = source_parallelism or source_worker.parallelism
        destination_parallelism = destination_parallelism or destination_worker.parallelism
        hosts = _semantic_rank_hosts(self.semantic_runtime, destination_worker.worker_id, len(destination_worker.devices))
        source_hosts = _semantic_rank_hosts(self.semantic_runtime, source_worker.worker_id, len(source_worker.devices))
        endpoints = []
        layout_change = (
            source_parallelism != destination_parallelism
            or len(source_worker.devices) != len(destination_worker.devices)
        )
        model_suite = getattr(getattr(self, "compiled", None), "model_suite", None)
        model_specs = getattr(model_suite, "specs", {}) if model_suite is not None else {}
        model_spec = model_specs.get(model)
        if model_spec is None:
            model_enum = {
                "actor": Model.ACTOR,
                "critic": Model.CRITIC,
                "reference": Model.REFERENCE,
                "reward": Model.REWARD,
                "ac_share": Model.AC_SHARE,
            }.get(model)
            if model_enum is not None:
                model_spec = model_specs.get(model_enum)
        num_layers = None if model_spec is None else int(model_spec.L)
        source_dp_rank = 0
        if layout_change and direct_transport == "torch_dist" and direct_backend == "nccl":
            source_dp_rank = _most_local_source_dp_rank(
                source_worker,
                destination_worker,
                source_parallelism,
                destination_parallelism,
                num_layers=num_layers,
            )
        logical_endpoint_pairs = (
            _routed_weight_endpoint_pairs(
                source_parallelism,
                destination_parallelism,
                source_dp_rank,
                num_layers=num_layers,
            )
            if layout_change
            else tuple((rank, rank) for rank in range(len(destination_worker.devices)))
        )
        destination_dp_relay = (
            layout_change
            and direct_transport == "torch_dist"
            and direct_backend == "nccl"
            and destination_parallelism[0] > 1
        )
        if destination_dp_relay:
            endpoint_pairs, relay_routes = _routed_weight_relay_endpoint_pairs(
                logical_endpoint_pairs,
                source_worker=source_worker,
                destination_worker=destination_worker,
                destination_parallelism=destination_parallelism,
            )
        else:
            endpoint_pairs = logical_endpoint_pairs
            destination_mp_width = destination_parallelism[1] * destination_parallelism[2]
            relay_routes = tuple(
                {
                    "source_rank": int(source_rank),
                    "destination_mp_rank": int(destination_rank) % destination_mp_width,
                    "relay_destination_rank": int(destination_rank),
                }
                for source_rank, destination_rank in endpoint_pairs
            )
        source_endpoint_degrees = Counter(source_rank for source_rank, _ in endpoint_pairs)
        destination_endpoint_degrees = Counter(destination_rank for _, destination_rank in endpoint_pairs)
        source_transfer_ranks = tuple(sorted({source_rank for source_rank, _ in endpoint_pairs}))
        coalesced_group_by_pair: dict[tuple[int, int], dict[str, object]] = {}
        can_coalesce = (
            layout_change
            and direct_transport == "torch_dist"
            and direct_backend == "nccl"
            and self._ucx_direct_channel_reuse_enabled()
        )
        if can_coalesce:
            network_pairs = []
            for source_rank, destination_rank in endpoint_pairs:
                source_device = source_worker.devices[source_rank]
                destination_device = destination_worker.devices[destination_rank]
                if (
                    source_device.hostname == destination_device.hostname
                    and source_device.local_cuda_index == destination_device.local_cuda_index
                ):
                    continue
                network_pairs.append((source_rank, destination_rank))

            destination_mp_width = destination_parallelism[1] * destination_parallelism[2]
            candidate_groups: list[tuple[str, tuple[tuple[int, int], ...]]] = []
            by_destination_mp: dict[int, list[tuple[int, int]]] = {}
            for pair in network_pairs:
                by_destination_mp.setdefault(pair[1] % destination_mp_width, []).append(pair)
            for destination_mp_rank, candidate_pairs in sorted(by_destination_mp.items()):
                source_ranks = tuple(sorted({source_rank for source_rank, _ in candidate_pairs}))
                destination_ranks = tuple(sorted({destination_rank for _, destination_rank in candidate_pairs}))
                source_devices = {
                    (
                        source_worker.devices[source_rank].hostname,
                        source_worker.devices[source_rank].local_cuda_index,
                    )
                    for source_rank in source_ranks
                }
                destination_devices = {
                    (
                        destination_worker.devices[destination_rank].hostname,
                        destination_worker.devices[destination_rank].local_cuda_index,
                    )
                    for destination_rank in destination_ranks
                }
                if source_devices.isdisjoint(destination_devices):
                    candidate_groups.append(
                        (f"mp{destination_mp_rank}", tuple(sorted(candidate_pairs)))
                    )
                    continue
                by_destination: dict[int, list[tuple[int, int]]] = {}
                for pair in candidate_pairs:
                    by_destination.setdefault(pair[1], []).append(pair)
                candidate_groups.extend(
                    (
                        f"mp{destination_mp_rank}.dst{destination_rank}",
                        tuple(sorted(group_pairs)),
                    )
                    for destination_rank, group_pairs in sorted(by_destination.items())
                )

            for group_name, group_pairs in candidate_groups:
                source_ranks = tuple(sorted({source_rank for source_rank, _ in group_pairs}))
                destination_ranks = tuple(sorted({destination_rank for _, destination_rank in group_pairs}))
                source_devices = [
                    (
                        source_worker.devices[source_rank].hostname,
                        source_worker.devices[source_rank].local_cuda_index,
                    )
                    for source_rank in source_ranks
                ]
                destination_devices = [
                    (
                        destination_worker.devices[destination_rank].hostname,
                        destination_worker.devices[destination_rank].local_cuda_index,
                    )
                    for destination_rank in destination_ranks
                ]
                if (
                    len(set(source_devices)) != len(source_devices)
                    or len(set(destination_devices)) != len(destination_devices)
                    or not set(source_devices).isdisjoint(destination_devices)
                ):
                    raise RuntimeConfigError(
                        f"cannot create duplicate-device-free NCCL weight group {group_name}: "
                        f"sources={source_ranks}, destinations={destination_ranks}"
                    )
                source_group_ranks = {
                    source_rank: group_rank
                    for group_rank, source_rank in enumerate(source_ranks)
                }
                destination_group_ranks = {
                    destination_rank: len(source_ranks) + group_rank
                    for group_rank, destination_rank in enumerate(destination_ranks)
                }
                channel_id = (
                    f"weight:{source_worker.worker_id}->{destination_worker.worker_id}:"
                    f"src{source_parallelism}:dp{source_dp_rank}:dst{destination_parallelism}:"
                    f"{group_name}"
                )
                master_destination_rank = destination_ranks[0]
                group_fields: dict[str, object] = {
                    "host": hosts[master_destination_rank],
                    "port": self._claim_direct_channel_port(
                        hosts[master_destination_rank],
                        channel_id,
                        master_destination_rank,
                    ),
                    "world_size": len(source_ranks) + len(destination_ranks),
                    "master_rank": destination_group_ranks[master_destination_rank],
                    "direct_channel_id": channel_id,
                    "direct_channel_reuse": True,
                    "coalesced_weight_group": True,
                }
                for source_rank, destination_rank in group_pairs:
                    coalesced_group_by_pair[(source_rank, destination_rank)] = {
                        **group_fields,
                        "source_group_rank": source_group_ranks[source_rank],
                        "destination_group_rank": destination_group_ranks[destination_rank],
                    }
        coalesced_weight_group = bool(coalesced_group_by_pair)
        for source_rank, destination_rank in endpoint_pairs:
            source_device = source_worker.devices[source_rank]
            destination_device = destination_worker.devices[destination_rank]
            same_physical_device = (
                source_device.hostname == destination_device.hostname
                and source_device.local_cuda_index == destination_device.local_cuda_index
            )
            reuse_endpoint = (
                self._ucx_direct_channel_reuse_enabled()
                and source_endpoint_degrees[source_rank] == 1
                and destination_endpoint_degrees[destination_rank] == 1
            )
            group_fields = coalesced_group_by_pair.get((source_rank, destination_rank))
            if group_fields is not None:
                endpoint = {
                    **group_fields,
                    "source_rank": source_rank,
                    "rank": destination_rank,
                    "source_host": source_hosts[source_rank],
                }
            elif reuse_endpoint:
                endpoint = self._ucx_direct_channel_endpoint(
                    source_worker=source_worker,
                    destination_worker=destination_worker,
                    source_rank=source_rank,
                    destination_rank=destination_rank,
                    source_host=source_hosts[source_rank],
                    destination_host=hosts[destination_rank],
                    include_ack_port=direct_transport == "torch_dist",
                )
            else:
                endpoint = {
                    "source_rank": source_rank,
                    "rank": destination_rank,
                    "source_host": source_hosts[source_rank],
                    "host": hosts[destination_rank],
                    "port": self._claim_direct_transfer_port(
                        hosts[destination_rank],
                        f"{node.node_id}:weight:{model}:v{source_version}:"
                        f"src{source_rank}:dst{destination_rank}",
                        destination_rank,
                    ),
                    "ack_port": self._claim_direct_transfer_port(
                        hosts[destination_rank],
                        f"{node.node_id}:weight:{model}:v{source_version}:"
                        f"src{source_rank}:dst{destination_rank}:ack",
                        destination_rank,
                    ),
                    "direct_channel_reuse": False,
                }
            endpoint["same_physical_device"] = same_physical_device
            if same_physical_device and direct_transport == "torch_dist" and direct_backend == "nccl":
                endpoint["endpoint_backend"] = (
                    "gloo"
                    if os.environ.get("RPEG_DISABLE_CUDA_IPC", "0") == "1"
                    else "cuda_ipc"
                )
            endpoints.append(endpoint)
        has_local_device_endpoints = any(
            bool(endpoint.get("same_physical_device"))
            for endpoint in endpoints
        )
        payload = {
            "model": model,
            "source_worker_id": source_worker.worker_id,
            "source_version": source_version,
            "destination_version": source_version,
            "source_world_size": len(source_worker.devices),
            "source_transfer_ranks": source_transfer_ranks,
            "source_dp_rank": source_dp_rank,
            "source_parallelism": source_parallelism,
            "destination_parallelism": destination_parallelism,
            "shadow": shadow,
            "step": step,
            "direct_endpoints": endpoints,
            "direct_transport": direct_transport,
            "direct_backend": direct_backend,
            "has_local_device_endpoints": has_local_device_endpoints,
            "coalesced_weight_group": coalesced_weight_group,
            "coalesced_weight_group_count": len({
                str(endpoint["direct_channel_id"])
                for endpoint in endpoints
                if bool(endpoint.get("coalesced_weight_group", False))
            }),
            "coalesced_weight_group_world_sizes": tuple(sorted({
                int(endpoint["world_size"])
                for endpoint in endpoints
                if bool(endpoint.get("coalesced_weight_group", False))
            })),
            "direct_channel_reuse": any(
                bool(endpoint.get("direct_channel_reuse", False))
                for endpoint in endpoints
            ),
            "source_overlap": direct_transport in {"ucx", "tcp"}
            and getattr(self.run.runtime, "direct_source_overlap", False),
            "destination_dp_relay": destination_dp_relay,
            "destination_dp_relay_routes": relay_routes,
            "in_memory": direct_transport == "ucx" or direct_backend == "nccl",
            "in_memory_reshard": (direct_transport == "ucx" or direct_backend == "nccl")
            and layout_change,
            "direct_timeout_s": float(self.run.monitoring.get("operation_timeout_s", 900.0)),
        }
        request_many = getattr(self.semantic_runtime, "request_many", None)
        if request_many is None:
            raise RuntimeConfigError("direct weight transfer requires semantic_runtime.request_many")
        results = request_many(
            (
                (
                    destination_worker.worker_id,
                    SemanticOperationNames.RECV_WEIGHT_VERSION,
                    f"{node.node_id}.recv_{model}_v{source_version}",
                    step,
                    dict(payload),
                    None,
                ),
                (
                    source_worker.worker_id,
                    SemanticOperationNames.SEND_WEIGHT_VERSION,
                    f"{node.node_id}.send_{model}_v{source_version}",
                    step,
                    dict(payload),
                    None,
                ),
            )
        )
        return results[destination_worker.worker_id]

    def _transfer_activation(self, node: RuntimeNodeSpec, step: int) -> dict[str, object] | None:
        if self.semantic_runtime is None or node.label is None or _is_weight_label(node.label):
            return None
        source_vertex = node.source_vertex_id or _adjacent_vertex(node.dependencies)
        destination_vertex = node.destination_vertex_id or _adjacent_vertex(node.successors)
        if source_vertex is None or destination_vertex is None:
            raise RuntimeConfigError(f"cannot infer activation transfer endpoints for {node.node_id}")
        source_artifact = _artifact_from_activation_label(
            node.label,
            source_vertex=source_vertex,
            algorithm=self.run.algorithm.name,
        )
        if source_artifact is None:
            return None
        source_part = _split_part_index(node.dependencies[0]) if node.dependencies else None
        if source_part is not None and source_vertex in {"g_pi", "g_ac"}:
            source_artifact = f"{source_artifact}.part{source_part}"
        source_worker = self.worker_by_vertex[source_vertex]
        destination_worker = self.worker_by_vertex[destination_vertex]
        source_parallelism = source_worker.parallelism_for(source_vertex)
        destination_parallelism = destination_worker.parallelism_for(destination_vertex)
        source_step = _adjacent_step(node.dependencies) or step
        if source_worker.worker_id == destination_worker.worker_id:
            return None
        primary_summary = self._ensure_activation_artifact(
            node=node,
            source_worker=source_worker,
            destination_worker=destination_worker,
            source_artifact=source_artifact,
            source_step=source_step,
            destination_step=step,
            label=str(node.label),
            source_parallelism=source_parallelism,
            destination_parallelism=destination_parallelism,
        )
        copied = [source_artifact]
        semantic_transfers = {
            source_artifact: primary_summary,
        }
        rollout_source_worker = self.worker_by_vertex.get("g_pi") or self.worker_by_vertex.get("g_ac")
        if (
            destination_vertex.startswith("t_")
            and source_artifact != "rollout"
            and self.run.algorithm.name in {"ppo", "grpo"}
            and rollout_source_worker is not None
            and rollout_source_worker.worker_id != destination_worker.worker_id
        ):
            rollout_summary = self._ensure_activation_artifact(
                node=node,
                source_worker=rollout_source_worker,
                destination_worker=destination_worker,
                source_artifact="rollout",
                source_step=source_step,
                destination_step=step,
                label="completions",
                source_parallelism=rollout_source_worker.parallelism_for(
                    "g_pi" if "g_pi" in rollout_source_worker.owned_vertex_ids else "g_ac"
                ),
                destination_parallelism=destination_parallelism,
            )
            copied.append("rollout")
            semantic_transfers["rollout"] = rollout_summary
        return {
            "label": node.label,
            "source_worker_id": source_worker.worker_id,
            "destination_worker_id": destination_worker.worker_id,
            "rollout_source_worker_id": rollout_source_worker.worker_id if "rollout" in copied else None,
            "source_step": source_step,
            "destination_step": step,
            "artifacts": tuple(copied),
            "semantic_transfers": semantic_transfers,
        }

    def _transfer_activation_artifact(
        self,
        *,
        node: RuntimeNodeSpec,
        step: int,
        source_worker: WorkerSpec,
        destination_worker: WorkerSpec,
        source_artifact: str,
        source_step: int,
        label: str,
        source_parallelism: tuple[int, int, int],
        destination_parallelism: tuple[int, int, int],
    ) -> tuple[SemanticResult, ...]:
        if self.run.runtime.activation_transport in _PAIRED_DIRECT_ACTIVATION_TRANSPORTS:
            return self._transfer_activation_paired_direct(
                node=node,
                step=step,
                source_worker=source_worker,
                destination_worker=destination_worker,
                source_artifact=source_artifact,
                source_step=source_step,
                label=label,
                source_parallelism=source_parallelism,
                destination_parallelism=destination_parallelism,
            )
        return self.semantic_runtime.request(
            destination_worker.worker_id,
            SemanticOperationNames.TRANSFER_ACTIVATION,
            node_id=f"{node.node_id}.transfer_{source_artifact}",
            step=step,
            payload={
                "source_worker_id": source_worker.worker_id,
                "source_name": source_artifact,
                "destination_name": source_artifact,
                "source_step": source_step,
                "destination_step": step,
                "label": label,
                "activation_transport": self.run.runtime.activation_transport,
            },
        )

    def _transfer_activation_paired_direct(
        self,
        *,
        node: RuntimeNodeSpec,
        step: int,
        source_worker: WorkerSpec,
        destination_worker: WorkerSpec,
        source_artifact: str,
        source_step: int,
        label: str,
        source_parallelism: tuple[int, int, int] | None = None,
        destination_parallelism: tuple[int, int, int] | None = None,
    ) -> tuple[SemanticResult, ...]:
        direct_transport, direct_backend = _direct_activation_transport(self.run.runtime.activation_transport)
        source_parallelism = source_parallelism or source_worker.parallelism
        destination_parallelism = destination_parallelism or destination_worker.parallelism
        has_same_physical_device = _activation_direct_has_same_physical_device(
            source_worker,
            destination_worker,
            source_parallelism,
            destination_parallelism,
        )
        if (
            direct_transport == "torch_dist"
            and direct_backend in {"nccl", "auto"}
            and _resolved_activation_store(self.run) in {"filesystem", "hybrid"}
            and has_same_physical_device
        ):
            return self.semantic_runtime.request(
                destination_worker.worker_id,
                SemanticOperationNames.TRANSFER_ACTIVATION,
                node_id=f"{node.node_id}.transfer_{source_artifact}",
                step=step,
                payload={
                    "source_worker_id": source_worker.worker_id,
                    "source_name": source_artifact,
                    "destination_name": source_artifact,
                    "source_step": source_step,
                    "destination_step": step,
                    "label": label,
                    "activation_transport": "auto",
                },
            )
        if (
            direct_transport == "torch_dist"
            and direct_backend in {"nccl", "auto"}
            and _resolved_activation_store(self.run) == "memory"
            and has_same_physical_device
        ):
            # Colocated graph vertices still execute in distinct worker processes. A process-local
            # artifact cache therefore needs a real transfer, but NCCL cannot form a two-rank
            # communicator whose ranks both bind the same CUDA device. Activation records are
            # serialized before transport, so Gloo provides a correct host-local path without
            # staging them in the shared filesystem.
            direct_backend = "gloo"
        if source_parallelism != destination_parallelism:
            if _activation_layout_has_same_dp(source_parallelism, destination_parallelism):
                return self._transfer_activation_direct_dp_writer_copy(
                    node=node,
                    step=step,
                    source_worker=source_worker,
                    destination_worker=destination_worker,
                    source_artifact=source_artifact,
                    source_step=source_step,
                    label=label,
                    source_parallelism=source_parallelism,
                    destination_parallelism=destination_parallelism,
                    direct_backend_override=direct_backend,
                )
            if (
                self.run.runtime.activation_transport not in _PAIRED_DIRECT_REPARTITION_TRANSPORTS
                or not _activation_layout_changes_dp(
                    source_parallelism,
                    destination_parallelism,
                )
            ):
                return self.semantic_runtime.request(
                    destination_worker.worker_id,
                    SemanticOperationNames.TRANSFER_ACTIVATION,
                    node_id=f"{node.node_id}.transfer_{source_artifact}",
                    step=step,
                    payload={
                        "source_worker_id": source_worker.worker_id,
                        "source_name": source_artifact,
                        "destination_name": source_artifact,
                        "source_step": source_step,
                        "destination_step": step,
                        "label": label,
                        "activation_transport": "stream",
                    },
                )
            return self._transfer_activation_direct_repartition(
                node=node,
                step=step,
                source_worker=source_worker,
                destination_worker=destination_worker,
                source_artifact=source_artifact,
                source_step=source_step,
                label=label,
                source_parallelism=source_parallelism,
                destination_parallelism=destination_parallelism,
                direct_backend_override=direct_backend,
            )
        if len(source_worker.devices) != len(destination_worker.devices):
            raise RuntimeConfigError(
                f"{self.run.runtime.activation_transport} activation transfer requires identical worker world sizes"
            )
        hosts = _semantic_rank_hosts(self.semantic_runtime, destination_worker.worker_id, len(destination_worker.devices))
        source_hosts = _semantic_rank_hosts(self.semantic_runtime, source_worker.worker_id, len(source_worker.devices))
        endpoints = []
        for rank in range(len(destination_worker.devices)):
            if self._ucx_direct_channel_reuse_enabled():
                endpoint = self._ucx_direct_channel_endpoint(
                    source_worker=source_worker,
                    destination_worker=destination_worker,
                    source_rank=rank,
                    destination_rank=rank,
                    source_host=source_hosts[rank],
                    destination_host=hosts[rank],
                    include_ack_port=direct_transport == "torch_dist",
                )
            else:
                endpoint = {
                    "rank": rank,
                    "host": hosts[rank],
                    "port": self._claim_direct_transfer_port(
                        hosts[rank],
                        f"{node.node_id}:{source_artifact}",
                        rank,
                    ),
                    "ack_port": self._claim_direct_transfer_port(
                        hosts[rank],
                        f"{node.node_id}:{source_artifact}:ack",
                        rank,
                    ),
                }
            endpoints.append(endpoint)
        payload = {
            "source_worker_id": source_worker.worker_id,
            "source_name": source_artifact,
            "destination_name": source_artifact,
            "source_step": source_step,
            "destination_step": step,
            "label": label,
            "direct_endpoints": endpoints,
            "direct_transport": direct_transport,
            "direct_backend": direct_backend,
            "direct_channel_reuse": self._ucx_direct_channel_reuse_enabled(),
            "source_overlap": direct_transport in {"ucx", "tcp"}
            and getattr(self.run.runtime, "direct_source_overlap", False),
            "in_memory": _resolved_activation_store(self.run) == "memory"
            or direct_transport == "ucx"
            or (
                direct_backend == "nccl"
                and _resolved_activation_store(self.run) in {"memory", "hybrid"}
            ),
            # A persistent worker may have changed layout after producing the artifact. Rank mapping
            # belongs to the artifact/consumer layouts, not either worker's currently active model
            # layout.
            "source_parallelism": list(source_parallelism),
            "destination_parallelism": list(destination_parallelism),
            "direct_timeout_s": float(self.run.monitoring.get("operation_timeout_s", 900.0)),
        }
        request_many = getattr(self.semantic_runtime, "request_many", None)
        if request_many is None:
            raise RuntimeConfigError(
                f"{self.run.runtime.activation_transport} activation transfer requires semantic_runtime.request_many"
            )
        if payload["source_overlap"]:
            # Arm the nonblocking source transfer before the destination wait. The graph reserves
            # only the destination in this mode, so the source worker can accept its next inference
            # operation as soon as send has launched the background data-plane operation.
            self.semantic_runtime.request(
                source_worker.worker_id,
                SemanticOperationNames.SEND_ACTIVATION,
                node_id=f"{node.node_id}.send_{source_artifact}",
                step=step,
                payload=dict(payload),
            )
            return self.semantic_runtime.request(
                destination_worker.worker_id,
                SemanticOperationNames.RECV_ACTIVATION,
                node_id=f"{node.node_id}.recv_{source_artifact}",
                step=step,
                payload=dict(payload),
            )
        results = request_many(
            (
                (
                    destination_worker.worker_id,
                    SemanticOperationNames.RECV_ACTIVATION,
                    f"{node.node_id}.recv_{source_artifact}",
                    step,
                    dict(payload),
                    None,
                ),
                (
                    source_worker.worker_id,
                    SemanticOperationNames.SEND_ACTIVATION,
                    f"{node.node_id}.send_{source_artifact}",
                    step,
                    dict(payload),
                    None,
                ),
            )
        )
        return results[destination_worker.worker_id]

    def _transfer_activation_direct_dp_writer_copy(
        self,
        *,
        node: RuntimeNodeSpec,
        step: int,
        source_worker: WorkerSpec,
        destination_worker: WorkerSpec,
        source_artifact: str,
        source_step: int,
        label: str,
        source_parallelism: tuple[int, int, int],
        destination_parallelism: tuple[int, int, int],
        direct_backend_override: str | None = None,
    ) -> tuple[SemanticResult, ...]:
        direct_transport, direct_backend = _direct_activation_transport(self.run.runtime.activation_transport)
        if direct_backend_override is not None:
            direct_backend = direct_backend_override
        src_dp, src_pp, src_tp = source_parallelism
        dst_dp, dst_pp, dst_tp = destination_parallelism
        if src_dp != dst_dp:
            raise RuntimeConfigError(
                "DP-writer direct activation copy requires identical source and destination DP sizes"
            )
        hosts = _semantic_rank_hosts(self.semantic_runtime, destination_worker.worker_id, len(destination_worker.devices))
        source_hosts = (
            _semantic_rank_hosts(self.semantic_runtime, source_worker.worker_id, len(source_worker.devices))
            if self._ucx_direct_channel_reuse_enabled()
            else ()
        )
        endpoints = []
        for dp_rank in range(dst_dp):
            destination_rank = dp_pp_tp_to_rank(dp_rank, 0, 0, dst_pp, dst_tp)
            source_rank = dp_pp_tp_to_rank(dp_rank, 0, 0, src_pp, src_tp)
            if self._ucx_direct_channel_reuse_enabled():
                endpoint = self._ucx_direct_channel_endpoint(
                    source_worker=source_worker,
                    destination_worker=destination_worker,
                    source_rank=source_rank,
                    destination_rank=destination_rank,
                    source_host=source_hosts[source_rank],
                    destination_host=hosts[destination_rank],
                    include_ack_port=direct_transport == "torch_dist",
                )
                endpoint["dp_rank"] = dp_rank
            else:
                endpoint = {
                    "rank": destination_rank,
                    "source_rank": source_rank,
                    "dp_rank": dp_rank,
                    "host": hosts[destination_rank],
                    "port": self._claim_direct_transfer_port(
                        hosts[destination_rank],
                        f"{node.node_id}:{source_artifact}:dp{dp_rank}:src{source_rank}:dst{destination_rank}",
                        destination_rank,
                    ),
                    "ack_port": self._claim_direct_transfer_port(
                        hosts[destination_rank],
                        f"{node.node_id}:{source_artifact}:dp{dp_rank}:src{source_rank}:dst{destination_rank}:ack",
                        destination_rank,
                    ),
                }
            endpoints.append(endpoint)
        payload = {
            "source_worker_id": source_worker.worker_id,
            "source_name": source_artifact,
            "destination_name": source_artifact,
            "source_step": source_step,
            "destination_step": step,
            "label": label,
            "direct_endpoints": endpoints,
            "direct_transport": direct_transport,
            "direct_backend": direct_backend,
            "direct_dp_writer_copy": True,
            "direct_channel_reuse": self._ucx_direct_channel_reuse_enabled(),
            "source_overlap": direct_transport == "ucx"
            and getattr(self.run.runtime, "direct_source_overlap", False),
            "in_memory": _resolved_activation_store(self.run) == "memory"
            or direct_transport == "ucx"
            or (
                direct_backend == "nccl"
                and _resolved_activation_store(self.run) in {"memory", "hybrid"}
            ),
            "source_dp_size": src_dp,
            "destination_dp_size": dst_dp,
            "source_parallelism": list(source_parallelism),
            "destination_parallelism": list(destination_parallelism),
            "direct_timeout_s": float(self.run.monitoring.get("operation_timeout_s", 900.0)),
        }
        request_many = getattr(self.semantic_runtime, "request_many", None)
        if request_many is None:
            raise RuntimeConfigError(
                f"{self.run.runtime.activation_transport} DP-writer activation copy requires "
                "semantic_runtime.request_many"
            )
        if payload["source_overlap"]:
            self.semantic_runtime.request(
                source_worker.worker_id,
                SemanticOperationNames.SEND_ACTIVATION,
                node_id=f"{node.node_id}.send_{source_artifact}",
                step=step,
                payload=dict(payload),
            )
            return self.semantic_runtime.request(
                destination_worker.worker_id,
                SemanticOperationNames.RECV_ACTIVATION,
                node_id=f"{node.node_id}.recv_{source_artifact}",
                step=step,
                payload=dict(payload),
            )
        results = request_many(
            (
                (
                    destination_worker.worker_id,
                    SemanticOperationNames.RECV_ACTIVATION,
                    f"{node.node_id}.recv_{source_artifact}",
                    step,
                    dict(payload),
                    None,
                ),
                (
                    source_worker.worker_id,
                    SemanticOperationNames.SEND_ACTIVATION,
                    f"{node.node_id}.send_{source_artifact}",
                    step,
                    dict(payload),
                    None,
                ),
            )
        )
        return results[destination_worker.worker_id]

    def _transfer_activation_direct_repartition(
        self,
        *,
        node: RuntimeNodeSpec,
        step: int,
        source_worker: WorkerSpec,
        destination_worker: WorkerSpec,
        source_artifact: str,
        source_step: int,
        label: str,
        source_parallelism: tuple[int, int, int],
        destination_parallelism: tuple[int, int, int],
        direct_backend_override: str | None = None,
    ) -> tuple[SemanticResult, ...]:
        direct_transport, direct_backend = _direct_activation_transport(self.run.runtime.activation_transport)
        if direct_backend_override is not None:
            direct_backend = direct_backend_override
        src_dp, _src_pp, _src_tp = source_parallelism
        dst_dp, dst_pp, dst_tp = destination_parallelism
        hosts = _semantic_rank_hosts(self.semantic_runtime, destination_worker.worker_id, len(destination_worker.devices))
        source_hosts = (
            _semantic_rank_hosts(self.semantic_runtime, source_worker.worker_id, len(source_worker.devices))
            if self._ucx_direct_channel_reuse_enabled()
            else ()
        )
        in_memory = (
            _resolved_activation_store(self.run) == "memory"
            or direct_transport == "ucx"
            or (
                direct_backend == "nccl"
                and _resolved_activation_store(self.run) in {"memory", "hybrid"}
            )
        )
        use_torch_dist_collective = direct_transport == "torch_dist"
        endpoints = []
        use_pair_endpoints = self.run.runtime.activation_transport in {"gloo_direct", "nccl_direct", "torch_dist", "ucx_direct"}
        for dp_rank in (() if use_torch_dist_collective else range(dst_dp)):
            rank = dp_pp_tp_to_rank(dp_rank, 0, 0, dst_pp, dst_tp)
            if use_pair_endpoints:
                for source_dp_rank in range(src_dp):
                    source_rank = dp_pp_tp_to_rank(
                        source_dp_rank,
                        0,
                        0,
                        source_parallelism[1],
                        source_parallelism[2],
                    )
                    if self._ucx_direct_channel_reuse_enabled():
                        endpoint = self._ucx_direct_channel_endpoint(
                            source_worker=source_worker,
                            destination_worker=destination_worker,
                            source_rank=source_rank,
                            destination_rank=rank,
                            source_host=source_hosts[source_rank],
                            destination_host=hosts[rank],
                            include_ack_port=direct_transport == "torch_dist",
                        )
                        endpoint.update({
                            "rank": rank,
                            "dp_rank": dp_rank,
                            "source_dp_rank": source_dp_rank,
                        })
                    else:
                        endpoint = {
                            "rank": rank,
                            "dp_rank": dp_rank,
                            "source_dp_rank": source_dp_rank,
                            "host": hosts[rank],
                            "port": self._claim_direct_transfer_port(
                                hosts[rank],
                                f"{node.node_id}:{source_artifact}:dst{dp_rank}:src{source_dp_rank}",
                                rank,
                            ),
                            "ack_port": self._claim_direct_transfer_port(
                                hosts[rank],
                                f"{node.node_id}:{source_artifact}:dst{dp_rank}:src{source_dp_rank}:ack",
                                rank,
                            ),
                        }
                    endpoints.append(endpoint)
            else:
                endpoints.append(
                    {
                        "rank": rank,
                        "dp_rank": dp_rank,
                        "host": hosts[rank],
                        "port": self._claim_direct_transfer_port(
                            hosts[rank],
                            f"{node.node_id}:{source_artifact}:dst{dp_rank}",
                            rank,
                        ),
                        "ack_port": self._claim_direct_transfer_port(
                            hosts[rank],
                            f"{node.node_id}:{source_artifact}:dst{dp_rank}:ack",
                            rank,
                        ),
                    }
                )
        payload = {
            "source_worker_id": source_worker.worker_id,
            "source_name": source_artifact,
            "destination_name": source_artifact,
            "source_step": source_step,
            "destination_step": step,
            "label": label,
            "direct_endpoints": endpoints,
            "direct_transport": direct_transport,
            "direct_backend": direct_backend,
            "direct_repartition": True,
            "direct_repartition_aggregate": False,
            "in_memory": in_memory,
            "direct_channel_reuse": self._ucx_direct_channel_reuse_enabled(),
            "source_dp_size": src_dp,
            "destination_dp_size": dst_dp,
            "source_parallelism": list(source_parallelism),
            "destination_parallelism": list(destination_parallelism),
            "direct_timeout_s": float(self.run.monitoring.get("operation_timeout_s", 900.0)),
            "global_index_repartition": ".part" in source_artifact,
        }
        if payload["in_memory"] or use_torch_dist_collective:
            payload["global_batch_size"] = (
                self.run.workload.batch_size * self.run.workload.samples_per_prompt
                if payload["global_index_repartition"]
                else _artifact_global_batch_size(self.run, source_artifact)
            )
        if use_torch_dist_collective:
            master_worker_rank = dp_pp_tp_to_rank(0, 0, 0, dst_pp, dst_tp)
            collective_id = (
                f"{self.run.run_id}:torch_dist_collective_repartition:"
                f"{source_worker.worker_id}->{destination_worker.worker_id}:dp{src_dp}->{dst_dp}"
            )
            attempt = int(getattr(self, "attempt", 0))
            if attempt:
                collective_id = f"{collective_id}:attempt{attempt}"
            # Repartition topology is stable across training steps. Reuse one communicator per
            # stable source/destination topology. Sharing it across artifacts avoids one CXI
            # endpoint set per artifact; graph resource ordering serializes transfers on this worker
            # pair. Recreating the same NCCL/OFI endpoints eventually exhausts CXI resources even
            # after an ordered ProcessGroupNCCL shutdown.
            reuse_collective_channel = self._ucx_direct_channel_reuse_enabled()
            if reuse_collective_channel:
                collective_port = self._claim_direct_channel_port(
                    hosts[master_worker_rank],
                    collective_id,
                    master_worker_rank,
                )
            else:
                collective_port = self._claim_direct_transfer_port(
                    hosts[master_worker_rank],
                    f"{node.node_id}:{source_artifact}:collective",
                    master_worker_rank,
                )
            payload["direct_collective_endpoint"] = {
                "host": hosts[master_worker_rank],
                "port": collective_port,
                "world_size": src_dp + dst_dp,
                "master_rank": src_dp,
                "direct_channel_id": collective_id,
                "direct_channel_reuse": reuse_collective_channel,
            }
        request_many = getattr(self.semantic_runtime, "request_many", None)
        if request_many is None:
            raise RuntimeConfigError(
                f"{self.run.runtime.activation_transport} repartition activation transfer requires "
                "semantic_runtime.request_many"
            )
        results = request_many(
            (
                (
                    destination_worker.worker_id,
                    SemanticOperationNames.RECV_ACTIVATION,
                    f"{node.node_id}.recv_{source_artifact}",
                    step,
                    dict(payload),
                    None,
                ),
                (
                    source_worker.worker_id,
                    SemanticOperationNames.SEND_ACTIVATION,
                    f"{node.node_id}.send_{source_artifact}",
                    step,
                    dict(payload),
                    None,
                ),
            )
        )
        return results[destination_worker.worker_id]

    def _claim_direct_transfer_port(self, host: str, node_id: str, rank: int) -> int:
        seed_node_id = f"{self.run.run_id}:{node_id}"
        attempt = int(getattr(self, "attempt", 0))
        if attempt:
            seed_node_id = f"{seed_node_id}:attempt{attempt}"
        first_port = _direct_transfer_port(self.run.runtime.rendezvous_port_base, seed_node_id, rank)
        with self._direct_port_lock:
            port = first_port
            for _ in range(_DIRECT_TRANSFER_PORT_SPAN):
                claim = (host, port)
                if claim not in self._direct_port_claims:
                    self._direct_port_claims.add(claim)
                    return port
                port += 1
                if port >= _DIRECT_TRANSFER_PORT_BASE + _DIRECT_TRANSFER_PORT_SPAN:
                    port = _DIRECT_TRANSFER_PORT_BASE
        raise RuntimeConfigError(f"exhausted direct activation transfer ports for host {host}")

    def _claim_direct_channel_port(self, host: str, channel_id: str, rank: int) -> int:
        key = (str(host), str(channel_id))
        with self._direct_port_lock:
            if key in self._direct_channel_ports:
                return self._direct_channel_ports[key]
        port = self._claim_direct_transfer_port(host, f"channel:{channel_id}", rank)
        with self._direct_port_lock:
            self._direct_channel_ports[key] = port
        return port


_PAIRED_DIRECT_ACTIVATION_TRANSPORTS = {"tcp_direct", "torch_dist", "gloo_direct", "nccl_direct", "ucx_direct"}
_PAIRED_DIRECT_REPARTITION_TRANSPORTS = {"tcp_direct", "torch_dist", "gloo_direct", "nccl_direct", "ucx_direct"}
_PAIRED_DIRECT_WEIGHT_TRANSPORTS = {"torch_dist", "gloo_direct", "nccl_direct", "ucx_direct"}


def _activation_layout_has_same_dp(
    source_parallelism: tuple[int, int, int],
    destination_parallelism: tuple[int, int, int],
) -> bool:
    return source_parallelism[0] == destination_parallelism[0]


def _activation_layout_changes_dp(
    source_parallelism: tuple[int, int, int],
    destination_parallelism: tuple[int, int, int],
) -> bool:
    return source_parallelism[0] != destination_parallelism[0]


def _activation_direct_has_same_physical_device(
    source_worker: WorkerSpec,
    destination_worker: WorkerSpec,
    source_parallelism: tuple[int, int, int],
    destination_parallelism: tuple[int, int, int],
) -> bool:
    src_dp, src_pp, src_tp = source_parallelism
    dst_dp, dst_pp, dst_tp = destination_parallelism
    if src_dp == dst_dp:
        routed_dp_pairs = ((dp_rank, dp_rank) for dp_rank in range(src_dp))
    else:
        routed_dp_pairs = (
            (source_dp_rank, destination_dp_rank)
            for source_dp_rank in range(src_dp)
            for destination_dp_rank in range(dst_dp)
            if (
                source_dp_rank * dst_dp < (destination_dp_rank + 1) * src_dp
                and destination_dp_rank * src_dp < (source_dp_rank + 1) * dst_dp
            )
        )
    for source_dp_rank, destination_dp_rank in routed_dp_pairs:
        source_rank = dp_pp_tp_to_rank(source_dp_rank, 0, 0, src_pp, src_tp)
        destination_rank = dp_pp_tp_to_rank(destination_dp_rank, 0, 0, dst_pp, dst_tp)
        source_device = source_worker.devices[source_rank]
        destination_device = destination_worker.devices[destination_rank]
        if (
            source_device.hostname == destination_device.hostname
            and source_device.local_cuda_index == destination_device.local_cuda_index
        ):
            return True
    return False


def _direct_activation_transport(mode: str) -> tuple[str, str | None]:
    if mode == "tcp_direct":
        return "tcp", None
    if mode == "gloo_direct":
        return "torch_dist", "gloo"
    if mode == "nccl_direct":
        return "torch_dist", "nccl"
    if mode == "torch_dist":
        return "torch_dist", "auto"
    if mode == "ucx_direct":
        return "ucx", None
    raise RuntimeConfigError(f"unsupported paired activation transport {mode!r}")


def _direct_weight_transport(mode: str) -> tuple[str, str | None]:
    if mode == "gloo_direct":
        return "torch_dist", "gloo"
    if mode == "nccl_direct":
        return "torch_dist", "nccl"
    if mode == "torch_dist":
        return "torch_dist", "auto"
    if mode == "ucx_direct":
        return "ucx", None
    raise RuntimeConfigError(f"unsupported paired weight transport {mode!r}")


def _validate_worker_exactness(worker: WorkerSpec) -> None:
    dp, pp, tp = worker.parallelism
    if len(worker.devices) != dp * pp * tp:
        raise RuntimeConfigError(
            f"{worker.worker_id}: assigned devices={len(worker.devices)} but dp*pp*tp={dp * pp * tp}"
        )
    if worker.backend != "megatron":
        raise RuntimeConfigError(f"{worker.worker_id}: expected backend=megatron, got {worker.backend}")


def _semantic_rank_hosts(semantic_runtime, worker_id: str, world_size: int) -> tuple[str, ...]:
    rank_hosts = getattr(semantic_runtime, "rank_hosts", None)
    if rank_hosts is None:
        return tuple("127.0.0.1" for _ in range(world_size))
    hosts = tuple(str(host) for host in rank_hosts(worker_id))
    if len(hosts) != world_size:
        raise RuntimeConfigError(f"{worker_id}: rank_hosts returned {len(hosts)} hosts, expected {world_size}")
    return hosts


def _first_rank_by_host(hosts: tuple[str, ...]) -> dict[str, int]:
    out: dict[str, int] = {}
    for rank, host in enumerate(hosts):
        out.setdefault(str(host), int(rank))
    return out


def _operational_weight_lifecycle_summary(
    ensured: tuple[SemanticResult, ...],
    released: tuple[SemanticResult, ...],
) -> dict[str, object]:
    if not ensured and not released:
        return {}
    ensure_counters = [dict(result.counters or {}) for result in ensured]
    release_counters = [dict(result.counters or {}) for result in released]
    return {
        "rank_count": len(ensured) or len(released),
        "materialized_rank_count": sum(
            int(bool(counters.get("materialized")))
            for counters in ensure_counters
        ),
        "materialized_bytes": sum(
            int(counters.get("materialized_bytes") or 0)
            for counters in ensure_counters
        ),
        "released_rank_count": sum(
            int(bool(counters.get("released")))
            for counters in release_counters
        ),
        "released_bytes": sum(
            int(counters.get("released_bytes") or 0)
            for counters in release_counters
        ),
        "optimizer_persistent_rank_count": sum(
            int(bool(counters.get("optimizer_persistent")))
            for counters in release_counters
        ),
    }


def _monitoring_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    raise RuntimeConfigError(f"invalid boolean monitoring value {value!r}")


# Keep deterministic listeners below Linux's default ephemeral range (32768-60999). Otherwise an
# unrelated outgoing NCCL/control connection can transiently occupy the port between planning and
# bind.
_DIRECT_TRANSFER_PORT_BASE = 15000
_DIRECT_TRANSFER_PORT_SPAN = 14000


def _direct_transfer_port(base_port: int, node_id: str, rank: int) -> int:
    offset = zlib.crc32(f"{int(base_port)}:{node_id}:{rank}".encode("utf-8")) % _DIRECT_TRANSFER_PORT_SPAN
    return _DIRECT_TRANSFER_PORT_BASE + offset


def _validate_publication_layouts(compiled: CompiledRuntimePlan, worker_by_vertex: dict[str, WorkerSpec]) -> None:
    """
    Publication layout changes are handled by RESHARD_WEIGHT_VERSION, which remaps PP layer
    ownership and TP tensor shards from per-rank RPEG manifests. Keep this hook as a named
    validation boundary for future restrictions, but do not reject valid RPEG parallelism changes
    here.
    """
    del compiled, worker_by_vertex


def _train_nodes_by_step(compiled: CompiledRuntimePlan) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for node in compiled.lowered.nodes:
        if node.kind != ExecNodeType.OP or node.op is None:
            continue
        if node.op.template_vertex.kind.role.value != "TRAIN":
            continue
        step = int(node.op.k) + 1
        out.setdefault(step, set()).add(node.id)
    return out


def _graph_node_step(node: RuntimeNodeSpec) -> int | None:
    if node.logical_step is not None:
        return int(node.logical_step)
    if node.exec_node.op is not None:
        return int(node.exec_node.op.k) + 1
    for node_id in (*node.dependencies, *node.successors):
        if "@" not in node_id:
            continue
        try:
            return int(node_id.rsplit("@", 1)[1].split(":", 1)[0]) + 1
        except ValueError:
            continue
    return None


def _adjacent_vertex(node_ids: tuple[str, ...]) -> str | None:
    for node_id in node_ids:
        if "@" in node_id:
            return node_id.split("@", 1)[0]
    return None


def _adjacent_step(node_ids: tuple[str, ...]) -> int | None:
    for node_id in node_ids:
        if "@" not in node_id:
            continue
        try:
            return int(node_id.rsplit("@", 1)[1].split(":", 1)[0]) + 1
        except ValueError:
            continue
    return None


def _is_weight_label(label: str | None) -> bool:
    return isinstance(label, str) and label.startswith("W_")


def _model_from_weight_label(label: str | None) -> str:
    if not _is_weight_label(label):
        raise RuntimeConfigError(f"not a weight transfer label: {label!r}")
    raw = str(label)[2:]
    try:
        return _RPEG_MODEL_TO_PLAN_NAME[raw]
    except KeyError as exc:
        raise RuntimeConfigError(f"unsupported RPEG weight transfer label {label!r}") from exc


def _weight_transfer_vertices(
    node: RuntimeNodeSpec,
    worker_by_vertex: dict[str, WorkerSpec],
    model: str,
) -> tuple[str, str]:
    def resolve(node_ids: tuple[str, ...], *, endpoint: str) -> str:
        vertices = tuple(dict.fromkeys(
            vertex
            for node_id in node_ids
            if (vertex := _adjacent_vertex((node_id,))) is not None
        ))
        matches = tuple(
            vertex
            for vertex in vertices
            if (worker := worker_by_vertex.get(vertex)) is not None and worker.model == model
        )
        if len(matches) != 1:
            raise RuntimeConfigError(
                f"cannot infer unique {endpoint} endpoint for {model} weight transfer "
                f"{node.node_id}: candidates={vertices}, model_matches={matches}"
            )
        return matches[0]

    source = node.source_vertex_id
    destination = node.destination_vertex_id
    if (
        source is not None
        and destination is not None
        and (source_worker := worker_by_vertex.get(source)) is not None
        and (destination_worker := worker_by_vertex.get(destination)) is not None
        and source_worker.model == model
        and destination_worker.model == model
    ):
        return source, destination
    return (
        resolve(node.dependencies, endpoint="source"),
        resolve(node.successors, endpoint="destination"),
    )


def _semantic_transfer_summary(results: tuple[SemanticResult, ...]) -> dict[str, object]:
    ranks = []
    total_bytes = 0
    total_rank_duration_s = 0.0
    copied = False
    source_shards = None
    transport = None
    copy_modes: set[str] = set()
    wire_formats: set[str] = set()
    sha256_scopes: set[str] = set()
    saw_dp_repartitioned = False
    dp_repartitioned = False
    saw_direct_dp_writer_copy = False
    direct_dp_writer_copy = False
    local_layout_transition = False
    source_shards_read_by_rank: dict[int, list[int]] = {}
    for result in results:
        counters = dict(result.counters or {})
        bytes_value = counters.get("bytes")
        if bytes_value is not None:
            total_bytes += int(bytes_value)
        duration_value = counters.get("transfer_rank_duration_s")
        if duration_value is not None:
            total_rank_duration_s += float(duration_value)
        copied = copied or bool(counters.get("copied", False))
        rank_entry = {
            "rank": result.rank,
            "copied": bool(counters.get("copied", False)),
            "bytes": int(bytes_value) if bytes_value is not None else 0,
        }
        if duration_value is not None:
            rank_entry["transfer_rank_duration_s"] = float(duration_value)
            rank_entry["transfer_rank_bytes_per_s"] = (
                float(rank_entry["bytes"]) / float(duration_value) if float(duration_value) > 0.0 else 0.0
            )
        if counters.get("sha256") is not None:
            rank_entry["sha256"] = str(counters["sha256"])
        if counters.get("source_sha256") is not None:
            rank_entry["source_sha256"] = str(counters["source_sha256"])
        if counters.get("source_rank_count") is not None:
            rank_entry["source_rank_count"] = int(counters["source_rank_count"])
        if counters.get("source_shards") is not None:
            source_shards = counters["source_shards"]
        if counters.get("transport") is not None:
            transport = counters["transport"]
        if counters.get("copy_mode") is not None:
            copy_mode = str(counters["copy_mode"])
            copy_modes.add(copy_mode)
            rank_entry["copy_mode"] = copy_mode
        if counters.get("wire_format") is not None:
            wire_format = str(counters["wire_format"])
            wire_formats.add(wire_format)
            rank_entry["wire_format"] = wire_format
        if counters.get("sha256_scope") is not None:
            sha256_scope = str(counters["sha256_scope"])
            sha256_scopes.add(sha256_scope)
            rank_entry["sha256_scope"] = sha256_scope
        for key in (
            "sent_wire_bytes",
            "logical_wire_bytes",
            "logical_network_bytes",
            "useful_source_bytes",
            "collective_received_bytes",
        ):
            if counters.get(key) is not None:
                rank_entry[key] = int(counters[key])
        if counters.get("dp_repartitioned") is not None:
            saw_dp_repartitioned = True
            rank_dp_repartitioned = bool(counters["dp_repartitioned"])
            dp_repartitioned = dp_repartitioned or rank_dp_repartitioned
            rank_entry["dp_repartitioned"] = rank_dp_repartitioned
        if counters.get("direct_dp_writer_copy") is not None:
            saw_direct_dp_writer_copy = True
            rank_direct_dp_writer_copy = bool(counters["direct_dp_writer_copy"])
            direct_dp_writer_copy = direct_dp_writer_copy or rank_direct_dp_writer_copy
            rank_entry["direct_dp_writer_copy"] = rank_direct_dp_writer_copy
        if counters.get("local_layout_transition") is not None:
            rank_local_layout_transition = bool(counters["local_layout_transition"])
            local_layout_transition = local_layout_transition or rank_local_layout_transition
            rank_entry["local_layout_transition"] = rank_local_layout_transition
        if counters.get("source_shards_read") is not None:
            source_shards_read = [int(rank) for rank in counters["source_shards_read"]]
            source_shards_read_by_rank[int(result.rank)] = source_shards_read
            rank_entry["source_shards_read"] = source_shards_read
        if counters.get("ucx_direct_cached_endpoint") is not None:
            rank_entry["ucx_direct_cached_endpoint"] = bool(counters["ucx_direct_cached_endpoint"])
        if counters.get("ucx_direct_cached_endpoint_count") is not None:
            rank_entry["ucx_direct_cached_endpoint_count"] = int(counters["ucx_direct_cached_endpoint_count"])
        if counters.get("direct_channel_reuse") is not None:
            rank_entry["direct_channel_reuse"] = bool(counters["direct_channel_reuse"])
        ranks.append(rank_entry)
    out: dict[str, object] = {
        "rank_count": len(results),
        "copied": copied,
        "bytes": total_bytes,
        "ranks": tuple(ranks),
    }
    if total_rank_duration_s > 0.0:
        out["transfer_rank_duration_s"] = total_rank_duration_s
        out["transfer_rank_bytes_per_s"] = float(total_bytes) / total_rank_duration_s
    if source_shards is not None:
        out["source_shards"] = source_shards
    if transport is not None:
        out["transport"] = transport
    if copy_modes:
        out["copy_mode"] = next(iter(copy_modes)) if len(copy_modes) == 1 else tuple(sorted(copy_modes))
    if wire_formats:
        out["wire_format"] = (
            next(iter(wire_formats))
            if len(wire_formats) == 1
            else tuple(sorted(wire_formats))
        )
    if sha256_scopes:
        out["sha256_scope"] = (
            next(iter(sha256_scopes))
            if len(sha256_scopes) == 1
            else tuple(sorted(sha256_scopes))
        )
    if saw_dp_repartitioned:
        out["dp_repartitioned"] = dp_repartitioned
    if saw_direct_dp_writer_copy:
        out["direct_dp_writer_copy"] = direct_dp_writer_copy
    if local_layout_transition:
        out["local_layout_transition"] = True
    if source_shards_read_by_rank:
        out["source_shards_read_by_rank"] = source_shards_read_by_rank
        if len(source_shards_read_by_rank) == 1:
            out["source_shards_read"] = next(iter(source_shards_read_by_rank.values()))
    cached_count = sum(int(rank.get("ucx_direct_cached_endpoint_count") or 0) for rank in ranks)
    if cached_count:
        out["ucx_direct_cached_endpoint_count"] = cached_count
        out["ucx_direct_cached_endpoint"] = True
        out["direct_channel_reuse"] = True
    return out


def _artifact_from_activation_label(
    label: str | None,
    *,
    source_vertex: str | None = None,
    algorithm: str | None = None,
) -> str | None:
    if label is None:
        return None
    if str(label) == "log_ref" and source_vertex == "i_ref" and algorithm == "dpo":
        return "log_ref"
    if str(label) == "log_pi" and source_vertex == "i_pi" and algorithm == "dpo":
        return "log_pi"
    return _ACTIVATION_LABEL_TO_ARTIFACT.get(str(label))


def _artifact_for_direct_dependency(
    source_vertex: str,
    destination_vertex: str,
    *,
    algorithm: str,
) -> str | None:
    if algorithm == "dpo" and destination_vertex == "t_pi":
        return {"i_pi": "log_pi", "i_ref": "log_ref"}.get(source_vertex)
    if algorithm not in {"ppo", "grpo"}:
        return None
    if source_vertex in {"g_pi", "g_ac"}:
        return "rollout"
    if not destination_vertex.startswith("t_"):
        return None
    return {
        "i_ref": "ref_logps",
        "i_R": "rewards",
        "i_V": "values",
        "i_ac": "values",
    }.get(source_vertex)


def _activation_label_for_artifact(
    artifact: str,
    *,
    source_vertex: str,
    algorithm: str,
) -> str:
    base = artifact.split(".part", 1)[0]
    if algorithm == "dpo":
        return {"log_pi": "log_pi", "log_ref": "log_ref"}[base]
    return {
        "rollout": "completions",
        "ref_logps": "log_ref",
        "rewards": "rewards",
        "values": "values",
    }[base]


def _split_part_index(node_id: str) -> int | None:
    marker = ":g"
    if marker not in node_id:
        marker = ":i"
    if marker not in node_id:
        return None
    raw = node_id.rsplit(marker, 1)[1]
    try:
        return int(raw)
    except ValueError:
        return None


def _quantile_batch_offset(run, part: int) -> int:
    quantiles = run.workload.gen_quantiles or ()
    if part <= 0:
        return 0
    total = run.workload.batch_size * run.workload.samples_per_prompt
    return sum(round(float(quantiles[idx][0]) * total) - (round(float(quantiles[idx - 1][0]) * total) if idx else 0) for idx in range(part))


def _quantile_generation_length(run, part: int) -> int:
    quantiles = run.workload.gen_quantiles or ()
    if not quantiles:
        return run.workload.generation_length
    return int(quantiles[part][1])


def _artifact_global_batch_size(run, artifact_name: str) -> int:
    total = run.workload.batch_size * run.workload.samples_per_prompt
    marker = ".part"
    if marker not in artifact_name:
        return total
    try:
        part = int(artifact_name.rsplit(marker, 1)[1])
    except ValueError as exc:
        raise RuntimeConfigError(f"invalid quantile artifact name {artifact_name!r}") from exc
    quantiles = run.workload.gen_quantiles or ()
    if part < 0 or part >= len(quantiles):
        raise RuntimeConfigError(
            f"quantile artifact part {part} is outside {len(quantiles)} configured phases"
        )
    upper = round(float(quantiles[part][0]) * total)
    lower = round(float(quantiles[part - 1][0]) * total) if part else 0
    return upper - lower


def megatron_ppo_capability_report(compiled: CompiledRuntimePlan) -> dict[str, object]:
    graph = compiled.runtime_graph()
    handler = MegatronPPOGraphHandler(compiled, step=1)
    return {
        "run_id": compiled.run_config.run_id,
        "node_count": len(graph.nodes),
        "workers": [
            {
                "worker_id": worker.worker_id,
                "vertex_id": worker.vertex_id,
                "vertex_ids": list(worker.owned_vertex_ids),
                "parallelism": worker.parallelism,
                "vertex_parallelisms": dict(worker.vertex_parallelisms),
                "devices": [f"{d.hostname}:cuda:{d.local_cuda_index}" for d in worker.devices],
            }
            for worker in compiled.workers
        ],
        "capabilities": [cap.to_dict() for cap in handler.capabilities],
    }


def run_megatron_graph_ppo(compiled: CompiledRuntimePlan):
    graph: RuntimeGraph = (
        compiled.runtime_graph()
        if hasattr(compiled, "runtime_graph")
        else build_runtime_graph(compiled.lowered, compiled.workers)
    )
    MegatronPPOGraphHandler(compiled, step=1).validate()
    replay_retries = int(compiled.run_config.monitoring.get("graph_replay_retries", 0))
    previous_error: RuntimeExecutionError | None = None
    for attempt in range(max(0, replay_retries) + 1):
        try:
            return _run_megatron_graph_attempt(compiled, graph, attempt=attempt, previous_error=previous_error)
        except RuntimeExecutionError as exc:
            previous_error = exc
            if attempt >= replay_retries:
                raise
    raise RuntimeExecutionError("unreachable Megatron graph replay state")


def _run_megatron_graph_attempt(
    compiled: CompiledRuntimePlan,
    graph: RuntimeGraph,
    *,
    attempt: int,
    previous_error: RuntimeExecutionError | None = None,
):
    recovery = _checkpoint_recovery_for_attempt(compiled, attempt)
    active_compiled = compiled
    active_graph = graph
    if recovery is not None:
        active_compiled = _compiled_with_resume_from(compiled, recovery.path)
        active_graph = _suffix_runtime_graph(graph, recovery.step)
    semantic_runtime = _build_semantic_runtime(active_compiled, attempt=attempt)
    restored_weight_files = 0
    restored_weight_bytes = 0
    checkpoint_weight_stats: dict[str, int] = {}
    if recovery is not None:
        restored_weight_files = _restore_checkpoint_weight_versions(
            recovery.path,
            _semantic_worker_state_root(semantic_runtime.state_root),
        )
        checkpoint_weight_stats = _checkpoint_weight_manifest_stats(recovery.path)
        restored_weight_bytes = checkpoint_weight_stats.get("checkpoint_weight_bytes", 0)
    handler = MegatronPPOGraphHandler(active_compiled, step=1, attempt=attempt, semantic_runtime=semantic_runtime)
    trace = RuntimeTraceWriter(compiled.run_config.runtime.trace_path, compiled.run_config.run_id)
    if attempt > 0:
        counters = {"attempt": attempt}
        if recovery is not None:
            counters.update(
                {
                    "checkpoint_step": recovery.step,
                    "checkpoint_path": str(recovery.path),
                    "checkpoint_local": True,
                }
            )
        trace.append(
            "replay",
            "runtime.graph_replay",
            status="retry",
            counters=counters,
            message=None if previous_error is None else str(previous_error),
        )
    if recovery is not None:
        trace.append(
            "recover",
            "runtime.checkpoint_recovery",
            status="ok",
            step=recovery.step,
            counters={
                "attempt": attempt,
                "step": recovery.step,
                "path": str(recovery.path),
                "skipped_nodes": len(graph.nodes) - len(active_graph.nodes),
                "restored_weight_files": restored_weight_files,
                "restored_weight_bytes": restored_weight_bytes,
                **checkpoint_weight_stats,
            },
        )
    store = ContainerStore(_runtime_store_root(compiled, attempt=attempt) / "containers")
    last_context = None
    try:
        with trace.node(
            "runtime.workers.start",
            counters={"worker_count": len(active_compiled.workers)},
        ):
            semantic_runtime.start_all(active_compiled.workers)
        with trace.node(
            "runtime.workers.initialize",
            counters={"worker_count": len(active_compiled.workers)},
        ):
            _initialize_semantic_workers(active_compiled, semantic_runtime)
        with trace.node(
            "runtime.weights.publish_initial",
            counters={"worker_count": len(active_compiled.workers)},
        ):
            _publish_initial_weight_versions(
                active_compiled,
                semantic_runtime,
                version_base=0 if recovery is None else recovery.step,
            )
        with trace.node("runtime.transports.prewarm") as prewarm_trace_counters:
            prewarm_counters = handler.prewarm_direct_transports(active_graph)
            prewarm_trace_counters.update(prewarm_counters)
        executor = RuntimeExecutor(
            active_graph,
            run_id=compiled.run_config.run_id,
            handler=handler,
            trace=trace,
            store=store,
        )
        last_context = executor.run()
        if compiled.run_config.checkpoints.save_final:
            output_dir = _save_semantic_final(compiled, semantic_runtime)
        else:
            output_dir = str(Path(compiled.run_config.checkpoints.output_dir))
        summary_checkpoint = Path(output_dir)
        if _load_weight_version_checkpoint_manifest(summary_checkpoint) is None:
            latest_checkpoint = _checkpoint_recovery_for_attempt(compiled, 1)
            if latest_checkpoint is not None:
                summary_checkpoint = latest_checkpoint.path
        current_checkpoint_stats = (
            _checkpoint_storage_stats(summary_checkpoint)
            if recovery is None
            else {}
        )
        last_actor_loss = None
        last_value_loss = None
        if last_context is not None:
            last_actor_loss = _last_loss_from_context(last_context, "t_ac")
            if last_actor_loss is None:
                last_actor_loss = _last_loss_from_context(last_context, "t_pi")
            if "t_V" in handler.worker_by_vertex:
                last_value_loss = _last_loss_from_context(last_context, "t_V")
        write_runtime_summary_for_run(
            compiled.run_config,
            extra={
                "algorithm": compiled.run_config.algorithm.name,
                "steps": compiled.run_config.training.max_steps,
                "output_dir": output_dir,
                "last_actor_loss": last_actor_loss,
                "last_value_loss": last_value_loss,
                "attempt": attempt,
                "checkpoint_recovery_step": None if recovery is None else recovery.step,
                **current_checkpoint_stats,
                **prewarm_counters,
            },
        )
        return _MegatronPPOTrainResult(
            steps=compiled.run_config.training.max_steps,
            last_actor_loss=last_actor_loss,
            last_value_loss=last_value_loss,
            output_dir=output_dir,
        )
    finally:
        semantic_runtime.shutdown(timeout_s=float(compiled.run_config.monitoring.get("worker_shutdown_timeout_s", 60.0)))


def _build_semantic_runtime(compiled: CompiledRuntimePlan, *, attempt: int = 0) -> SemanticWorkerGroupRuntime:
    run = compiled.run_config
    launcher = SlurmWorkerGroupLauncher(
        run.run_id,
        run.runtime.log_dir,
        LauncherConfig(
            type="slurm",
            partition=run.cluster.launcher.partition,
            account=run.cluster.launcher.account,
            time_limit=run.cluster.launcher.time_limit,
            container=run.cluster.launcher.container,
            mpi=run.cluster.launcher.mpi,
            shared_step=run.cluster.launcher.shared_step,
            env=run.cluster.launcher.env,
        ),
    )
    return SemanticWorkerGroupRuntime(
        run_id=run.run_id,
        launcher=launcher,
        command_root=_semantic_command_root(run.runtime.log_dir, attempt),
        state_root=_runtime_store_root(compiled, attempt=attempt) / "semantic",
        backend="megatron",
        control_transport=run.runtime.semantic_control_transport,
        ready_timeout_s=float(run.monitoring.get("worker_ready_timeout_s", 120.0)),
        operation_timeout_s=float(run.monitoring.get("operation_timeout_s", 600.0)),
        restart_retries=int(run.monitoring.get("worker_restart_retries", 0)),
    )


def _semantic_command_root(log_dir: str, attempt: int) -> Path:
    root = Path(log_dir)
    if attempt <= 0:
        return root / "semantic"
    return root / f"semantic_attempt_{attempt}"


def _runtime_store_root(compiled: CompiledRuntimePlan, *, attempt: int = 0) -> Path:
    store_dir = compiled.run_config.runtime.store_dir
    root = Path(store_dir) if store_dir else Path(compiled.run_config.checkpoints.output_dir) / "runtime_store"
    if attempt <= 0:
        return root
    return root / f"attempt_{attempt}"


@dataclass(frozen=True)
class _CheckpointRecovery:
    step: int
    path: Path


def _checkpoint_recovery_for_attempt(compiled: CompiledRuntimePlan, attempt: int) -> _CheckpointRecovery | None:
    if attempt <= 0:
        return None
    index_path = _checkpoint_index_path(compiled)
    if not index_path.exists():
        return None
    try:
        index = json.loads(index_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeConfigError(f"invalid checkpoint index: {index_path}") from exc
    if index.get("schema_version") != "rpeg.checkpoint_index.v1":
        raise RuntimeConfigError(f"unsupported checkpoint index schema in {index_path}")
    latest = index.get("latest_interval")
    if latest is None:
        return None
    candidates = [
        item
        for item in index.get("checkpoints", [])
        if item.get("kind") == "interval" and str(item.get("path")) == str(latest)
    ]
    if not candidates:
        candidates = [item for item in index.get("checkpoints", []) if item.get("kind") == "interval"]
    if not candidates:
        return None
    candidates.sort(key=_checkpoint_index_sort_key)
    item = candidates[-1]
    step = int(item.get("step", 0))
    if step <= 0:
        return None
    path = Path(str(item.get("path", latest)))
    if not path.is_absolute():
        path = index_path.parent / path
    if not path.exists():
        return None
    requires_historical_versions = any(
        version < step
        for _model, version in _required_checkpoint_weight_versions(compiled, step)
    )
    if requires_historical_versions and not _checkpoint_has_weight_manifest(path):
        return None
    return _CheckpointRecovery(step=min(step, compiled.run_config.training.max_steps), path=path)


def _max_trainable_tau(compiled: CompiledRuntimePlan) -> int:
    tau = compiled.optimized_plan.structure.get("tau") or {}
    trainable = {"actor", "critic", "ac_share", "Pi", "V", "AC"}
    values = [int(value) for key, value in tau.items() if str(key) in trainable]
    return max(values, default=0)


def _compiled_with_resume_from(compiled: CompiledRuntimePlan, path: Path) -> CompiledRuntimePlan:
    checkpoints = replace(compiled.run_config.checkpoints, resume_from=str(path))
    run_config = replace(compiled.run_config, checkpoints=checkpoints)
    return replace(compiled, run_config=run_config)


def _suffix_runtime_graph(graph: RuntimeGraph, recovered_step: int) -> RuntimeGraph:
    """
    An interval checkpoint makes train state durable, but worker-group consumers are restarted with
    volatile replicas. Weight publications downstream of the checkpointed train nodes must therefore
    run again before the next-step consumers. Replaying them is idempotent even when another train
    model finished and published before the checkpoint barrier.
    """
    keep = {
        node_id
        for node_id, node in graph.nodes.items()
        if (
            _graph_node_step(node) is None
            or int(_graph_node_step(node) or 0) > recovered_step
            or (
                int(_graph_node_step(node) or 0) == recovered_step
                and node.kind in {ExecNodeType.COMM, ExecNodeType.RESHARD}
                and str(node.label or "").startswith("W_")
            )
        )
    }
    nodes = {
        node_id: replace(
            graph.nodes[node_id],
            dependencies=tuple(dep for dep in graph.nodes[node_id].dependencies if dep in keep),
            successors=tuple(succ for succ in graph.nodes[node_id].successors if succ in keep),
        )
        for node_id in keep
    }
    return RuntimeGraph(
        nodes=nodes,
        topological_order=tuple(node_id for node_id in graph.topological_order if node_id in keep),
        composite_operators=tuple(
            composite
            for composite in graph.composite_operators
            if set(composite.get("member_node_ids") or ()) <= keep
        ),
    )


def _resolved_activation_store(run: RuntimeRunConfig) -> str:
    activation_store = str(getattr(run.runtime, "activation_store", "auto"))
    if activation_store == "auto":
        if run.runtime.activation_transport == "ucx_direct":
            return "hybrid" if int(run.monitoring.get("graph_replay_retries", 0)) > 0 else "memory"
        return "filesystem"
    return activation_store


def _resolved_activation_cache_device(run: RuntimeRunConfig) -> str:
    device = str(getattr(run.runtime, "activation_cache_device", "cpu"))
    if device == "auto":
        return (
            "cuda"
            if _resolved_activation_store(run) in {"memory", "hybrid"}
            and int(run.hardware.semantic_activation_cache_bytes) > 0
            else "cpu"
        )
    return device


def _fused_training_pipeline_nccl_cta_policy(
    run: RuntimeRunConfig,
    worker,
) -> int | None:
    """
    Return a PP-only policy for the two colocated fused trainers.
    """
    if (
        not run.runtime.intra_stage_fusion
        or not worker.owns_training
        or worker.model not in {"actor", "critic"}
    ):
        return None
    return {
        "default": None,
        "efficiency": 1,
        "zero": 2,
    }[run.runtime.fusion_pipeline_nccl_cta_policy]


def _initialize_semantic_workers(
    compiled: CompiledRuntimePlan,
    semantic_runtime: SemanticWorkerGroupRuntime,
) -> None:
    run = compiled.run_config
    activation_store = _resolved_activation_store(run)
    activation_cache_device = _resolved_activation_cache_device(run)
    if activation_store == "memory" and int(run.monitoring.get("graph_replay_retries", 0)) > 0:
        raise RuntimeConfigError(
            "runtime.activation_store=memory is incompatible with graph replay; use hybrid for durable recovery"
        )
    if activation_cache_device == "cuda" and activation_store not in {"memory", "hybrid"}:
        raise RuntimeConfigError(
            "runtime.activation_cache_device=cuda requires activation_store=memory or hybrid"
        )
    resume_weight_sources: dict[str, _CheckpointWeightVersionSource] = {}
    if run.checkpoints.resume_from:
        _restore_resume_weight_versions_if_available(run.checkpoints.resume_from, semantic_runtime)
        resume_weight_sources = _resume_weight_version_sources_by_model(run.checkpoints.resume_from)
    for worker in compiled.workers:
        model_cfg = run.models.get(worker.model) or run.models.get("actor")
        if model_cfg is None:
            raise RuntimeConfigError(f"no model config for worker {worker.worker_id} model={worker.model}")
        if model_cfg.source == "actor":
            model_cfg = run.models.get("actor")
            if model_cfg is None:
                raise RuntimeConfigError(f"{worker.worker_id}: source=actor but models.actor is missing")
        model_dir = resolve_model_path(
            model_cfg.source,
            model_cfg.repo,
            model_cfg.checkpoint_path,
            revision=model_cfg.revision,
        )
        dp, pp, tp = worker.parallelism
        if worker.model in {"critic", "ac_share"}:
            auxiliary_heads = ("value",)
        elif worker.model == "reward":
            auxiliary_heads = ("reward",)
        else:
            auxiliary_heads = ()
        semantic_runtime.request(
            worker.worker_id,
            SemanticOperationNames.BUILD_MODEL,
            node_id=f"{worker.worker_id}.build_model",
            step=0,
            payload={
                "model": worker.model,
                "model_path": str(model_dir),
                "tokenizer_path": str(model_dir),
                "dp": dp,
                "pp": pp,
                "tp": tp,
                "dtype": model_cfg.dtype,
                "use_transformer_engine": run.runtime.use_transformer_engine,
                "perform_initialization": False,
                "seed": run.algorithm.seed,
                "max_sequence_length": run.workload.context_length + run.workload.generation_length,
                "activation_checkpointing": run.optimizer_runtime.activation_checkpointing,
                "activation_store": activation_store,
                "activation_cache_device": activation_cache_device,
                "activation_cache_max_bytes": int(
                    run.hardware.semantic_activation_cache_bytes
                ),
                "weight_cache_keep_latest": run.runtime.semantic_weight_keep_latest,
                "microbatch": worker.microbatch,
                "auxiliary_heads": auxiliary_heads,
                "value_head_prefixes": model_cfg.value_head_prefixes,
                "reward_head_prefixes": model_cfg.reward_head_prefixes,
                "learning_rate": run.optimizer_runtime.learning_rate_critic
                if worker.model == "critic"
                else run.optimizer_runtime.learning_rate_actor,
                "weight_decay": run.optimizer_runtime.weight_decay,
                "grad_clip": run.optimizer_runtime.grad_clip,
                "distributed_optimizer": run.optimizer_runtime.distributed_optimizer,
                "zero2_bucket_bytes": run.optimizer_runtime.zero2_bucket_bytes,
                "static_kv_cache_max_entries": run.runtime.static_kv_cache_max_entries,
                "zero2_max_inflight_buckets": run.optimizer_runtime.zero2_max_inflight_buckets,
                "static_kv_cache_batch_bucket": run.runtime.static_kv_cache_batch_bucket,
                "static_kv_cache_sequence_bucket": run.runtime.static_kv_cache_sequence_bucket,
                "pipeline_nccl_cta_policy": _fused_training_pipeline_nccl_cta_policy(
                    run,
                    worker,
                ),
            },
        )
        if run.checkpoints.resume_from and worker.owns_training and worker.model in {"actor", "critic", "ac_share"}:
            resume_source = resume_weight_sources.get(worker.model)
            if resume_source is not None:
                semantic_runtime.request(
                    worker.worker_id,
                    SemanticOperationNames.RESHARD_WEIGHT_VERSION,
                    node_id=f"{worker.worker_id}.reshard_resume_{worker.model}_v{resume_source.version}",
                    step=0,
                    payload={
                        "model": worker.model,
                        "source_worker_id": resume_source.worker_id,
                        "source_version": resume_source.version,
                        "destination_version": resume_source.version,
                        "source_world_size": resume_source.world_size,
                        "source_parallelism": resume_source.parallelism,
                        "destination_parallelism": worker.parallelism,
                        "shadow": resume_source.shadow,
                        "step": resume_source.version,
                    },
                )
                semantic_runtime.request(
                    worker.worker_id,
                    SemanticOperationNames.LOAD_WEIGHT_VERSION,
                    node_id=f"{worker.worker_id}.load_resume_{worker.model}_v{resume_source.version}",
                    step=0,
                    payload={
                        "model": worker.model,
                        "version": resume_source.version,
                        "strict": True,
                        "shadow": resume_source.shadow,
                        "load_optimizer_state": True,
                    },
                )
            else:
                semantic_runtime.request(
                    worker.worker_id,
                    SemanticOperationNames.LOAD_DISTRIBUTED_CHECKPOINT,
                    node_id=f"{worker.worker_id}.load_resume",
                    step=0,
                    payload={
                        "path": _resume_checkpoint_path(run.checkpoints.resume_from, worker),
                        "strict": True,
                    },
                )
        elif not run.checkpoints.resume_from or worker.model in {"reference", "reward"}:
            semantic_runtime.request(
                worker.worker_id,
                SemanticOperationNames.LOAD_HF_CHECKPOINT,
                node_id=f"{worker.worker_id}.load_hf",
                step=0,
                payload={"model_path": str(model_dir), "strict": False},
            )


def _publish_initial_weight_versions(
    compiled: CompiledRuntimePlan,
    semantic_runtime: SemanticWorkerGroupRuntime,
    *,
    version_base: int = 0,
) -> None:
    run = compiled.run_config
    worker_ids_by_model = {
        model: {
            worker.worker_id
            for worker in compiled.workers
            if worker.model == model
        }
        for model in {"actor", "critic", "ac_share"}
    }
    if run.checkpoints.resume_from:
        source_by_model = _initial_weight_source_by_model(compiled.workers)
        for source in source_by_model.values():
            cross_worker_publication = len(worker_ids_by_model[source.model]) > 1
            semantic_runtime.request(
                source.worker_id,
                SemanticOperationNames.SAVE_WEIGHT_VERSION,
                node_id=f"{source.worker_id}.save_{source.model}_v0",
                step=0,
                payload={
                    "model": source.model,
                    "version": version_base,
                    "step": version_base,
                    "shadow": uses_shadow_copies(compiled, source.model),
                    "include_optimizer_state": False,
                    "active_only": not cross_worker_publication,
                },
            )
        for worker in compiled.workers:
            if worker.model not in source_by_model:
                continue
            source = source_by_model[worker.model]
            if worker.worker_id == source.worker_id:
                continue
            semantic_runtime.request(
                worker.worker_id,
                SemanticOperationNames.RESHARD_WEIGHT_VERSION,
                node_id=f"{worker.worker_id}.resume_{worker.model}_v0",
                step=0,
                payload={
                    "model": worker.model,
                    "source_worker_id": source.worker_id,
                    "source_version": version_base,
                    "destination_version": version_base,
                    "source_world_size": len(source.devices),
                    "source_parallelism": source.parallelism,
                    "destination_parallelism": worker.parallelism,
                    "shadow": uses_shadow_copies(compiled, worker.model),
                    "step": version_base,
                },
            )
        return
    for worker in compiled.workers:
        if worker.model not in {"actor", "critic", "ac_share"}:
            continue
        # Every non-resume worker loaded the same immutable HF version directly into its own runtime
        # layout during initialization. Mark that local module active instead of cloning a redundant
        # GPU snapshot. Direct snapshots are needed only for versions produced after training.
        semantic_runtime.request(
            worker.worker_id,
            SemanticOperationNames.SAVE_WEIGHT_VERSION,
            node_id=f"{worker.worker_id}.save_{worker.model}_v0",
            step=0,
            payload={
                "model": worker.model,
                "version": 0,
                "step": 0,
                "shadow": uses_shadow_copies(compiled, worker.model),
                "include_optimizer_state": False,
                "active_only": True,
                "persist": False,
                "device_snapshot": False,
            },
        )


def _initial_weight_source_by_model(workers: tuple[WorkerSpec, ...]) -> dict[str, WorkerSpec]:
    out = {}
    for worker in workers:
        if worker.owns_training and worker.model in {"actor", "critic", "ac_share"}:
            out[worker.model] = worker
    return out


def _resume_checkpoint_path(resume_from: str, worker: WorkerSpec) -> str:
    root = _resolve_resume_checkpoint_root(Path(resume_from))
    candidates = [
        root / worker.vertex_id,
        root / worker.model,
        root,
    ]
    for candidate in candidates:
        checkpoint = _checkpoint_leaf(candidate)
        if checkpoint is not None:
            return str(checkpoint)
    return str(candidates[0])


@dataclass(frozen=True)
class _CheckpointWeightVersionSource:
    worker_id: str
    model: str
    version: int
    world_size: int
    parallelism: tuple[int, int, int]
    shadow: bool


def _restore_resume_weight_versions_if_available(
    resume_from: str,
    semantic_runtime: SemanticWorkerGroupRuntime,
) -> int:
    state_root = getattr(semantic_runtime, "state_root", None)
    if state_root is None:
        return 0
    root = _resolve_resume_checkpoint_root(Path(resume_from))
    if _load_weight_version_checkpoint_manifest(root) is None:
        return 0
    return _restore_checkpoint_weight_versions(root, _semantic_worker_state_root(state_root))


def _resume_weight_version_sources_by_model(resume_from: str) -> dict[str, _CheckpointWeightVersionSource]:
    root = _resolve_resume_checkpoint_root(Path(resume_from))
    manifest = _load_weight_version_checkpoint_manifest(root)
    if manifest is None:
        return {}
    try:
        checkpoint_step = int(manifest.get("step", -1))
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError(f"invalid weight-version checkpoint step in {root}") from exc
    if checkpoint_step < 0:
        return {}
    source_root = _checkpoint_weight_manifest_source_root(manifest, root)
    candidates: dict[tuple[str, str, int], dict[str, object]] = {}
    for entry in manifest.get("weights") or []:
        relative = Path(str(entry.get("relative_path") or ""))
        if relative.suffix != ".json":
            continue
        if relative.is_absolute() or not relative.parts or relative.parts[0] != "weights":
            raise RuntimeConfigError(f"invalid weight-version manifest path: {relative}")
        path = source_root / relative
        _validate_weight_version_artifact_sidecar(source_root, path)
        metadata = json.loads(path.read_text())
        version = int(metadata.get("version", -1))
        model = str(metadata.get("model") or "")
        if version != checkpoint_step or model not in _TRAINABLE_WEIGHT_MODELS:
            continue
        if not bool(metadata.get("optimizer_state", False)):
            continue
        parallelism = _metadata_parallelism(metadata, path)
        world_size = int(metadata.get("world_size", -1))
        if world_size != parallelism[0] * parallelism[1] * parallelism[2]:
            raise RuntimeConfigError(
                f"weight-version parallelism {parallelism} does not match world_size={world_size} in {path}"
            )
        key = (str(metadata["worker_id"]), model, version)
        record = candidates.setdefault(
            key,
            {
                "ranks": set(),
                "world_size": world_size,
                "parallelism": parallelism,
                "shadow": bool(metadata.get("shadow", False)),
            },
        )
        if int(record["world_size"]) != world_size or tuple(record["parallelism"]) != parallelism:
            raise RuntimeConfigError(f"inconsistent weight-version source metadata for {key} in {root}")
        ranks = record["ranks"]
        if not isinstance(ranks, set):
            raise RuntimeConfigError(f"invalid source-rank accumulator for {key}")
        ranks.add(int(metadata["rank"]))

    out: dict[str, _CheckpointWeightVersionSource] = {}
    score_by_model: dict[str, tuple[int, str]] = {}
    for (worker_id, model, version), record in candidates.items():
        ranks = record["ranks"]
        if not isinstance(ranks, set):
            continue
        world_size = int(record["world_size"])
        if ranks != set(range(world_size)):
            continue
        score = (
            1 if worker_id.startswith("wg_t_") or worker_id.startswith("t_") else 0,
            worker_id,
        )
        if model in score_by_model and score <= score_by_model[model]:
            continue
        score_by_model[model] = score
        out[model] = _CheckpointWeightVersionSource(
            worker_id=worker_id,
            model=model,
            version=version,
            world_size=world_size,
            parallelism=tuple(record["parallelism"]),
            shadow=bool(record.get("shadow", False)),
        )
    return out


def _metadata_parallelism(metadata: dict[str, object], path: Path) -> tuple[int, int, int]:
    raw = metadata.get("parallelism")
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise RuntimeConfigError(
            f"portable weight-version checkpoint resume requires dp/pp/tp parallelism metadata in {path}"
        )
    try:
        parallelism = tuple(int(x) for x in raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError(f"invalid weight-version parallelism metadata in {path}") from exc
    if any(x <= 0 for x in parallelism):
        raise RuntimeConfigError(f"invalid weight-version parallelism metadata in {path}")
    return parallelism


def _resolve_resume_checkpoint_root(path: Path) -> Path:
    index_path = path if path.name == "checkpoint_index.json" else path / "checkpoint_index.json"
    if not index_path.exists():
        return path
    try:
        index = json.loads(index_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeConfigError(f"invalid checkpoint index: {index_path}") from exc
    if index.get("schema_version") != "rpeg.checkpoint_index.v1":
        raise RuntimeConfigError(f"unsupported checkpoint index schema in {index_path}")

    selected = index.get("latest_interval") or index.get("latest_final")
    if selected is None:
        checkpoints = list(index.get("checkpoints") or [])
        checkpoints.sort(key=_checkpoint_index_sort_key)
        if checkpoints:
            selected = checkpoints[-1].get("path")
    if selected is None:
        raise RuntimeConfigError(f"checkpoint index has no checkpoints: {index_path}")
    selected_path = Path(str(selected))
    if not selected_path.is_absolute():
        cwd_relative = selected_path
        index_relative = index_path.parent / selected_path
        selected_path = cwd_relative if cwd_relative.exists() else index_relative
    return selected_path


def _checkpoint_leaf(path: Path) -> Path | None:
    if not path.exists():
        return None
    if path.name.startswith("iter_"):
        return path
    iterations = sorted(path.glob("iter_*"))
    if iterations:
        return iterations[-1]
    return path


@dataclass(frozen=True)
class _MegatronPPOTrainResult:
    steps: int
    last_actor_loss: float | None
    last_value_loss: float | None
    output_dir: str


def _save_semantic_checkpoints_if_needed(
    compiled: CompiledRuntimePlan,
    semantic_runtime: SemanticWorkerGroupRuntime,
    step: int,
) -> None:
    interval = compiled.run_config.checkpoints.interval_steps
    if interval <= 0 or step % interval != 0:
        return
    _save_semantic_checkpoint(compiled, semantic_runtime, f"step_{step}", step)
    _prune_semantic_interval_checkpoints(compiled)


def _save_semantic_final(
    compiled: CompiledRuntimePlan,
    semantic_runtime: SemanticWorkerGroupRuntime,
) -> str:
    return _save_semantic_checkpoint(
        compiled,
        semantic_runtime,
        "final",
        compiled.run_config.training.max_steps,
    )


def _save_semantic_checkpoint(
    compiled: CompiledRuntimePlan,
    semantic_runtime: SemanticWorkerGroupRuntime,
    name: str,
    step: int,
) -> str:
    root = Path(compiled.run_config.checkpoints.output_dir) / name
    portable = compiled.run_config.checkpoints.portable_weight_versions
    saved_workers = []
    for worker in compiled.workers:
        if worker.role != "TRAIN":
            continue
        worker_path = root / worker.vertex_id
        semantic_runtime.request(
            worker.worker_id,
            SemanticOperationNames.SAVE_DISTRIBUTED_CHECKPOINT,
            node_id=f"{worker.worker_id}.save_checkpoint.{name}",
            step=step,
            payload={
                "path": str(worker_path),
                "step": step,
                "include_optimizer_state": not portable,
            },
        )
        if portable and worker.model in _TRAINABLE_WEIGHT_MODELS:
            semantic_runtime.request(
                worker.worker_id,
                SemanticOperationNames.SAVE_WEIGHT_VERSION,
                node_id=f"{worker.worker_id}.save_checkpoint_weight.{name}",
                step=step,
                payload={
                    "model": worker.model,
                    "version": step,
                    "step": step,
                    "shadow": uses_shadow_copies(compiled, worker.model),
                    "include_optimizer_state": True,
                },
            )
        saved_workers.append(
            {
                "worker_id": worker.worker_id,
                "vertex_id": worker.vertex_id,
                "model": worker.model,
                "path": str(worker_path),
                "portable_weight_version": (
                    step if portable and worker.model in _TRAINABLE_WEIGHT_MODELS else None
                ),
            }
        )
    if portable:
        _write_weight_version_checkpoint_manifest(
            semantic_runtime,
            root,
            step=step,
            required_versions=_required_checkpoint_weight_versions(compiled, step),
        )
    _update_checkpoint_index(compiled, name=name, step=step, workers=saved_workers)
    return str(root)


def _checkpoint_weight_manifest_path(checkpoint_root: str | Path) -> Path:
    return Path(checkpoint_root) / "rpeg_weight_versions.json"


def _checkpoint_has_weight_manifest(checkpoint_root: str | Path) -> bool:
    return _checkpoint_weight_manifest_stats(checkpoint_root).get("checkpoint_weight_files", 0) > 0


def _semantic_worker_state_root(state_root: str | Path) -> Path:
    root = Path(state_root)
    if (root / "weights").exists():
        return root
    return root / "state"


def _write_weight_version_checkpoint_manifest(
    semantic_runtime: SemanticWorkerGroupRuntime,
    checkpoint_root: str | Path,
    *,
    step: int,
    required_versions: set[tuple[str, int]] | None = None,
) -> None:
    state_root = getattr(semantic_runtime, "state_root", None)
    if state_root is None:
        return
    state_root = _semantic_worker_state_root(state_root)
    weights_root = state_root / "weights"
    if not weights_root.exists():
        return
    root = Path(checkpoint_root)
    artifact_root = root / "rpeg_weight_versions"
    entries = []
    for path in sorted(weights_root.glob("*/*/version_*/*")):
        if not path.is_file() or path.suffix not in {".pt", ".json"}:
            continue
        parsed = _parse_weight_version_artifact_path(state_root, path)
        if required_versions is not None and (parsed[1], parsed[2]) not in required_versions:
            continue
        _validate_weight_version_artifact_sidecar(state_root, path)
        relative = path.relative_to(state_root)
        destination = artifact_root / relative
        _link_or_copy_file(path, destination)
        entries.append(
            {
                "relative_path": str(relative),
                "bytes": destination.stat().st_size,
                "sha256": _file_sha256(destination),
            }
        )
    if not entries:
        return
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = _checkpoint_weight_manifest_path(root)
    manifest = {
        "schema_version": "rpeg.weight_version_checkpoint.v1",
        "step": int(step),
        "source_state_root": str(artifact_root),
        "file_count": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "weights": entries,
    }
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(manifest_path)


def _restore_checkpoint_weight_versions(checkpoint_root: str | Path, state_root: str | Path | None) -> int:
    if state_root is None:
        raise RuntimeConfigError("checkpoint recovery requires a semantic state_root")
    manifest = _load_weight_version_checkpoint_manifest(checkpoint_root)
    if manifest is None:
        return 0
    source_root = _checkpoint_weight_manifest_source_root(manifest, checkpoint_root)
    destination_root = Path(state_root)
    restored = 0
    for entry in manifest.get("weights") or []:
        relative = Path(str(entry.get("relative_path") or ""))
        if relative.is_absolute() or not relative.parts or relative.parts[0] != "weights":
            raise RuntimeConfigError(f"invalid weight-version manifest path: {relative}")
        source = source_root / relative
        if not source.exists():
            raise RuntimeConfigError(f"missing checkpoint weight-version artifact: {source}")
        expected_bytes = int(entry.get("bytes", -1))
        if source.stat().st_size != expected_bytes:
            raise RuntimeConfigError(f"weight-version artifact byte count mismatch for {source}")
        expected_sha = str(entry.get("sha256") or "")
        if _file_sha256(source) != expected_sha:
            raise RuntimeConfigError(f"weight-version artifact checksum mismatch for {source}")
        _validate_weight_version_artifact_sidecar(source_root, source)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.stat().st_size == expected_bytes and _file_sha256(destination) == expected_sha:
                restored += 1
                continue
            destination.unlink()
        tmp = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
        if tmp.exists():
            tmp.unlink()
        try:
            try:
                os.link(source, tmp)
            except OSError:
                shutil.copyfile(source, tmp)
            tmp.replace(destination)
        finally:
            if tmp.exists():
                tmp.unlink()
        restored += 1
    return restored


def _checkpoint_weight_manifest_stats(checkpoint_root: str | Path) -> dict[str, int]:
    manifest = _load_weight_version_checkpoint_manifest(checkpoint_root)
    if manifest is None:
        return {}
    weights = manifest.get("weights") or []
    source_root = _checkpoint_weight_manifest_source_root(manifest, checkpoint_root)
    total_bytes = sum(_manifest_entry_bytes(entry, checkpoint_root) for entry in weights)
    data_files = 0
    metadata_files = 0
    optimizer_state_rank_files = 0
    portable_candidates: dict[tuple[str, str, int], dict[str, object]] = {}
    for entry in weights:
        relative = Path(str(entry.get("relative_path") or ""))
        if source_root and source_root.exists():
            path = source_root / relative
            _validate_weight_version_artifact_sidecar(source_root, path)
            if path.suffix == ".pt":
                data_files += 1
                continue
            if path.suffix != ".json":
                continue
            metadata_files += 1
            metadata = json.loads(path.read_text())
            model = str(metadata.get("model") or "")
            version = int(metadata.get("version", -1))
            if model not in _TRAINABLE_WEIGHT_MODELS or version != int(manifest.get("step", -1)):
                continue
            if not bool(metadata.get("optimizer_state", False)):
                continue
            parallelism = _metadata_parallelism(metadata, path)
            world_size = int(metadata.get("world_size", -1))
            if world_size != parallelism[0] * parallelism[1] * parallelism[2]:
                raise RuntimeConfigError(
                    f"weight-version parallelism {parallelism} does not match world_size={world_size} in {path}"
                )
            optimizer_state_rank_files += 1
            key = (str(metadata["worker_id"]), model, version)
            record = portable_candidates.setdefault(
                key,
                {
                    "ranks": set(),
                    "world_size": world_size,
                    "parallelism": parallelism,
                },
            )
            if int(record["world_size"]) != world_size or tuple(record["parallelism"]) != parallelism:
                raise RuntimeConfigError(f"inconsistent portable checkpoint metadata for {key} in {checkpoint_root}")
            ranks = record["ranks"]
            if not isinstance(ranks, set):
                raise RuntimeConfigError(f"invalid portable checkpoint rank accumulator for {key}")
            ranks.add(int(metadata["rank"]))
        else:
            if relative.suffix == ".pt":
                data_files += 1
            elif relative.suffix == ".json":
                metadata_files += 1
    declared_count = manifest.get("file_count")
    if declared_count is not None and int(declared_count) != len(weights):
        raise RuntimeConfigError(f"weight-version manifest file_count mismatch in {checkpoint_root}")
    declared_bytes = manifest.get("total_bytes")
    if declared_bytes is not None and int(declared_bytes) != total_bytes:
        raise RuntimeConfigError(f"weight-version manifest total_bytes mismatch in {checkpoint_root}")
    complete_portable_versions = 0
    complete_portable_rank_files = 0
    for record in portable_candidates.values():
        ranks = record["ranks"]
        if not isinstance(ranks, set):
            continue
        world_size = int(record["world_size"])
        if ranks == set(range(world_size)):
            complete_portable_versions += 1
            complete_portable_rank_files += world_size
    return {
        "checkpoint_weight_files": len(weights),
        "checkpoint_weight_bytes": total_bytes,
        "checkpoint_weight_data_files": data_files,
        "checkpoint_weight_metadata_files": metadata_files,
        "checkpoint_optimizer_state_rank_files": optimizer_state_rank_files,
        "checkpoint_portable_train_versions": complete_portable_versions,
        "checkpoint_portable_train_rank_files": complete_portable_rank_files,
    }


def _checkpoint_storage_stats(checkpoint_root: str | Path) -> dict[str, int]:
    """
    Report both portable-version and actual retained checkpoint storage.
    """
    root = Path(checkpoint_root)
    stats = _checkpoint_weight_manifest_stats(root)
    files = [path for path in root.rglob("*") if path.is_file()] if root.exists() else []
    stats["checkpoint_files_on_disk"] = len(files)
    stats["checkpoint_bytes_on_disk"] = sum(path.stat().st_size for path in files)
    stats["checkpoint_optimizer_state_rank_files_on_disk"] = sum(
        path.name.startswith("rpeg_optimizer_rank") and path.suffix == ".pt"
        for path in files
    )
    return stats


def _load_weight_version_checkpoint_manifest(checkpoint_root: str | Path) -> dict[str, object] | None:
    manifest_path = _checkpoint_weight_manifest_path(checkpoint_root)
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeConfigError(f"invalid weight-version checkpoint manifest: {manifest_path}") from exc
    if manifest.get("schema_version") != "rpeg.weight_version_checkpoint.v1":
        raise RuntimeConfigError(f"unsupported weight-version checkpoint manifest schema in {manifest_path}")
    weights = manifest.get("weights") or []
    if not isinstance(weights, list):
        raise RuntimeConfigError(f"invalid weight-version manifest weights in {manifest_path}")
    for entry in weights:
        if not isinstance(entry, dict):
            raise RuntimeConfigError(f"invalid weight-version manifest entry in {manifest_path}")
    return manifest


def _checkpoint_weight_manifest_source_root(
    manifest: dict[str, object],
    checkpoint_root: str | Path,
) -> Path:
    source_root = Path(str(manifest.get("source_state_root") or ""))
    if source_root.exists():
        return source_root
    checkpoint_local = Path(checkpoint_root) / "rpeg_weight_versions"
    if checkpoint_local.exists():
        return checkpoint_local
    return source_root


def _manifest_entry_bytes(entry: dict[str, object], checkpoint_root: str | Path) -> int:
    try:
        byte_count = int(entry.get("bytes", -1))
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError(f"invalid weight-version manifest byte count in {checkpoint_root}") from exc
    if byte_count < 0:
        raise RuntimeConfigError(f"invalid weight-version manifest byte count in {checkpoint_root}")
    return byte_count


def _required_checkpoint_weight_versions(compiled: CompiledRuntimePlan, checkpoint_step: int) -> set[tuple[str, int]]:
    graph = compiled.runtime_graph()
    required: set[tuple[str, int]] = {
        (worker.model, int(checkpoint_step))
        for worker in compiled.workers
        if worker.role == "TRAIN" and worker.model in _TRAINABLE_WEIGHT_MODELS
    }
    for node in graph.nodes.values():
        step = _graph_node_step(node)
        if step is None or step <= checkpoint_step:
            continue
        model = _checkpoint_weight_consumer_model(node)
        if model is None:
            continue
        version = consumer_weight_version(compiled, model, step)
        if version < checkpoint_step:
            required.add((model, version))
    return required


def _validate_checkpoint_portability(compiled: CompiledRuntimePlan) -> None:
    checkpoints = compiled.run_config.checkpoints
    replay_retries = int(compiled.run_config.monitoring.get("graph_replay_retries", 0))
    if (
        replay_retries <= 0
        or checkpoints.portable_weight_versions
        or checkpoints.interval_steps <= 0
    ):
        return
    for step in range(
        checkpoints.interval_steps,
        compiled.run_config.training.max_steps + 1,
        checkpoints.interval_steps,
    ):
        historical = sorted(
            (model, version)
            for model, version in _required_checkpoint_weight_versions(compiled, step)
            if version < step
        )
        if historical:
            raise RuntimeConfigError(
                "native-only checkpoint recovery cannot reconstruct historical weight versions "
                f"required after step {step}: {historical}; set "
                "checkpoints.portable_weight_versions=true"
            )


def _checkpoint_weight_consumer_model(node: RuntimeNodeSpec) -> str | None:
    if node.kind != ExecNodeType.OP:
        return None
    if node.vertex_id in {"g_pi", "i_pi"}:
        return "actor"
    if node.vertex_id in {"g_ac", "i_ac"}:
        return "ac_share"
    if node.vertex_id == "i_V":
        return "critic"
    return None


def _parse_weight_version_artifact_path(state_root: Path, path: Path) -> tuple[str, str, int]:
    relative = path.relative_to(state_root)
    parts = relative.parts
    if len(parts) < 5 or parts[0] != "weights" or not parts[3].startswith("version_"):
        raise RuntimeConfigError(f"invalid weight-version artifact path: {path}")
    worker_id = parts[1]
    model = parts[2]
    try:
        version = int(parts[3].removeprefix("version_"))
    except ValueError as exc:
        raise RuntimeConfigError(f"invalid weight-version artifact version in {path}") from exc
    return worker_id, model, version


def _validate_weight_version_artifact_sidecar(state_root: Path, path: Path) -> None:
    if path.suffix not in {".pt", ".json"}:
        return
    worker_id, model, version = _parse_weight_version_artifact_path(state_root, path)
    try:
        rank = int(path.stem.removeprefix("rank"))
    except ValueError as exc:
        raise RuntimeConfigError(f"invalid weight-version artifact rank in {path}") from exc
    data_path = path.with_suffix(".pt")
    metadata_path = path.with_suffix(".json")
    if not data_path.exists():
        raise RuntimeConfigError(f"missing RPEG weight version data for sidecar: {data_path}")
    if not metadata_path.exists():
        raise RuntimeConfigError(f"missing RPEG weight version metadata: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeConfigError(f"invalid RPEG weight version metadata: {metadata_path}") from exc
    if metadata.get("schema_version") != "rpeg.weight_version_meta.v1":
        raise RuntimeConfigError(f"unsupported RPEG weight version metadata schema in {metadata_path}")
    expected = {
        "worker_id": worker_id,
        "rank": rank,
        "model": model,
        "version": version,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeConfigError(
                f"weight version metadata mismatch for {metadata_path}: {key}={metadata.get(key)!r}, expected {value!r}"
            )
    try:
        world_size = int(metadata.get("world_size", -1))
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError(f"invalid weight version world_size in {metadata_path}") from exc
    if world_size <= rank:
        raise RuntimeConfigError(f"invalid weight version world_size in {metadata_path}")
    if int(metadata.get("bytes", -1)) != data_path.stat().st_size:
        raise RuntimeConfigError(f"weight version byte count mismatch for {data_path}")
    expected_sha = str(metadata.get("sha256") or "")
    if len(expected_sha) != 64 or _file_sha256(data_path) != expected_sha:
        raise RuntimeConfigError(f"weight version checksum mismatch for {data_path}")


def _link_or_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        destination.exists()
        and destination.stat().st_size == source.stat().st_size
        and _file_sha256(destination) == _file_sha256(source)
    ):
        return
    if destination.exists():
        destination.unlink()
    tmp = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        try:
            os.link(source, tmp)
        except OSError:
            shutil.copyfile(source, tmp)
        tmp.replace(destination)
    finally:
        if tmp.exists():
            tmp.unlink()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_index_path(compiled: CompiledRuntimePlan) -> Path:
    return Path(compiled.run_config.checkpoints.output_dir) / "checkpoint_index.json"


def _update_checkpoint_index(
    compiled: CompiledRuntimePlan,
    *,
    name: str,
    step: int,
    workers: list[dict[str, object]],
) -> None:
    index_path = _checkpoint_index_path(compiled)
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeConfigError(f"invalid checkpoint index: {index_path}") from exc
    else:
        index = {
            "schema_version": "rpeg.checkpoint_index.v1",
            "run_id": compiled.run_config.run_id,
            "checkpoints": [],
        }
    if index.get("schema_version") != "rpeg.checkpoint_index.v1":
        raise RuntimeConfigError(f"unsupported checkpoint index schema in {index_path}")
    existing = [
        item
        for item in index.get("checkpoints", [])
        if not (item.get("name") == name and int(item.get("step", -1)) == int(step))
    ]
    kind = "final" if name == "final" else "interval"
    existing.append(
        {
            "name": name,
            "kind": kind,
            "step": int(step),
            "path": str(Path(compiled.run_config.checkpoints.output_dir) / name),
            "workers": workers,
        }
    )
    existing.sort(key=_checkpoint_index_sort_key)
    index["run_id"] = compiled.run_config.run_id
    index["checkpoints"] = existing
    index["latest_final"] = next(
        (item["path"] for item in reversed(existing) if item.get("kind") == "final"),
        None,
    )
    interval_paths = [item["path"] for item in existing if item.get("kind") == "interval"]
    index["latest_interval"] = interval_paths[-1] if interval_paths else None
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_suffix(index_path.suffix + ".tmp")
    tmp.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    tmp.replace(index_path)


def _prune_semantic_interval_checkpoints(compiled: CompiledRuntimePlan) -> None:
    keep = compiled.run_config.checkpoints.keep_last_interval_steps
    if keep is None:
        return
    index_path = _checkpoint_index_path(compiled)
    if not index_path.exists():
        return
    index = json.loads(index_path.read_text())
    checkpoints = list(index.get("checkpoints", []))
    intervals = [item for item in checkpoints if item.get("kind") == "interval"]
    intervals.sort(key=lambda item: (int(item.get("step", 0)), str(item.get("name", ""))))
    prune = intervals[: max(0, len(intervals) - keep)]
    if not prune:
        return
    prune_names = {item.get("name") for item in prune}
    for item in prune:
        path = Path(str(item.get("path", "")))
        if path.name.startswith("step_") and path.exists():
            shutil.rmtree(path)
    index["checkpoints"] = [item for item in checkpoints if item.get("name") not in prune_names]
    index["checkpoints"].sort(key=_checkpoint_index_sort_key)
    interval_paths = [item["path"] for item in index["checkpoints"] if item.get("kind") == "interval"]
    index["latest_interval"] = interval_paths[-1] if interval_paths else None
    index["latest_final"] = next(
        (item["path"] for item in reversed(index["checkpoints"]) if item.get("kind") == "final"),
        None,
    )
    tmp = index_path.with_suffix(index_path.suffix + ".tmp")
    tmp.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    tmp.replace(index_path)


def _prune_semantic_weight_versions(
    compiled: CompiledRuntimePlan,
    semantic_runtime: SemanticWorkerGroupRuntime,
    context: RuntimeExecutionContext | None = None,
) -> None:
    keep = _semantic_weight_keep_latest(compiled)
    if keep is None:
        return
    required = _semantic_weight_versions_required_by_incomplete_consumers(compiled, context)
    publisher_roots = {
        (worker.worker_id, worker.model)
        for worker in compiled.workers
        if worker.role == "TRAIN" and worker.model in _TRAINABLE_WEIGHT_MODELS
    }
    request = getattr(semantic_runtime, "request", None)
    if request is not None and context is not None:
        for worker in compiled.workers:
            if worker.model not in _TRAINABLE_WEIGHT_MODELS:
                continue
            versions = sorted(required.get((worker.worker_id, worker.model), set()))
            request(
                worker.worker_id,
                SemanticOperationNames.PRUNE_WEIGHT_VERSIONS,
                node_id=f"runtime.prune_weights.{worker.worker_id}",
                step=max(versions, default=0),
                payload={
                    "keep_versions_by_model": {worker.model: versions},
                    # Keep-latest remains a durable-artifact policy below. In memory, only versions
                    # required by incomplete graph nodes are live; otherwise 70B publishers
                    # accumulate multi-GiB BF16 snapshots between optimizer steps.
                    "preserve_latest": False,
                    # Once the last consumer of an inference owner's active version completes,
                    # release its parameter storage before a later NCCL publication arrives. TRAIN
                    # owners retain their operational model/optimizer state.
                    "release_unrequired_active": worker.role != "TRAIN",
                },
            )
    state_root = getattr(semantic_runtime, "state_root", None)
    if state_root is None:
        return
    weights_root = _semantic_worker_state_root(state_root) / "weights"
    if not weights_root.exists():
        return
    for model_root in sorted(path for path in weights_root.glob("*/*") if path.is_dir()):
        worker_id = model_root.parent.name
        model = model_root.name
        versions = []
        for version_dir in model_root.glob("version_*"):
            if not version_dir.is_dir():
                continue
            try:
                version = int(version_dir.name.removeprefix("version_"))
            except ValueError:
                continue
            versions.append((version, version_dir))
        versions.sort(key=lambda item: item[0])
        required_versions = set(required.get((worker_id, model), set()))
        if (worker_id, model) in publisher_roots:
            required_versions.update(version for version, _path in versions[-keep:])
        stale = [(version, path) for version, path in versions if version not in required_versions]
        for _version, path in stale:
            shutil.rmtree(path)


def _semantic_weight_keep_latest(compiled: CompiledRuntimePlan) -> int | None:
    configured = int(compiled.run_config.runtime.semantic_weight_keep_latest)
    if configured <= 0:
        return None
    return configured


_TRAINABLE_WEIGHT_MODELS = frozenset({"actor", "critic", "ac_share"})
_WEIGHT_CONSUMER_ROLES = frozenset({"GEN", "INF", "GENERATION", "INFERENCE"})


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _semantic_weight_versions_required_by_incomplete_consumers(
    compiled: CompiledRuntimePlan,
    context: RuntimeExecutionContext | None,
) -> dict[tuple[str, str], set[int]]:
    if context is None:
        return {}
    completed = set(context.completed)
    graph = compiled.runtime_graph()
    worker_by_vertex = {
        vertex_id: worker
        for worker in compiled.workers
        for vertex_id in worker.owned_vertex_ids
    }
    required: dict[tuple[str, str], set[int]] = {}
    for node_id in graph.topological_order:
        if node_id in completed:
            continue
        node = graph.nodes[node_id]
        if node.kind in {ExecNodeType.COMM, ExecNodeType.RESHARD} and _is_weight_label(node.label):
            model = _model_from_weight_label(node.label)
            source_vertex, destination_vertex = _weight_transfer_vertices(
                node,
                worker_by_vertex,
                model,
            )
            source_worker = worker_by_vertex[source_vertex]
            destination_worker = worker_by_vertex[destination_vertex]
            source_step = _adjacent_step(node.dependencies) or _graph_node_step(node)
            if source_step is None:
                continue
            version = (
                source_step
                if source_worker.role_for(source_vertex) == "TRAIN" and source_worker.model == model
                else consumer_weight_version(compiled, model, source_step)
            )
            required.setdefault((source_worker.worker_id, model), set()).add(version)
            required.setdefault((destination_worker.worker_id, model), set()).add(version)
            continue
        if node.kind != ExecNodeType.OP:
            continue
        if node.worker_id is None or node.model not in _TRAINABLE_WEIGHT_MODELS:
            continue
        if node.role not in _WEIGHT_CONSUMER_ROLES:
            continue
        step = _graph_node_step(node)
        if step is None:
            continue
        version = consumer_weight_version(compiled, node.model, step)
        required.setdefault((node.worker_id, node.model), set()).add(version)
        # Once the explicit weight-transfer node has completed, the consumer's local cache is
        # sufficient. The transfer node itself keeps both the publisher and destination versions
        # live while it is incomplete; retaining the publisher snapshot for every later consumer
        # would unnecessarily pin a full model shard on colocated training GPUs.
    return required



def _most_local_source_dp_rank(
    source_worker: WorkerSpec,
    destination_worker: WorkerSpec,
    source_parallelism: tuple[int, int, int],
    destination_parallelism: tuple[int, int, int],
    *,
    num_layers: int | None = None,
) -> int:
    """
    Select the source DP replica with the most useful physical locality.

    Exact-device routes can use CUDA IPC and same-host routes avoid Slingshot. Network routes are
    decomposed below, so locality does not force duplicate CUDA devices into one NCCL communicator.
    """
    src_dp, src_pp, src_tp = source_parallelism
    expected_world_size = src_dp * src_pp * src_tp
    if len(source_worker.devices) != expected_world_size:
        raise RuntimeConfigError(
            f"source worker {source_worker.worker_id} has {len(source_worker.devices)} devices, "
            f"expected {expected_world_size} for parallelism {source_parallelism}"
        )
    candidates: list[tuple[int, int, int]] = []
    for dp_rank in range(src_dp):
        exact = 0
        same_host = 0
        for source_rank, destination_rank in _routed_weight_endpoint_pairs(
            source_parallelism,
            destination_parallelism,
            dp_rank,
            num_layers=num_layers,
        ):
            source_device = source_worker.devices[source_rank]
            destination_device = destination_worker.devices[destination_rank]
            if source_device.hostname == destination_device.hostname:
                same_host += 1
                if source_device.local_cuda_index == destination_device.local_cuda_index:
                    exact += 1
        candidates.append((-exact, -same_host, dp_rank))
    return min(candidates)[-1]

def _routed_weight_endpoint_pairs(
    source_parallelism: tuple[int, int, int],
    destination_parallelism: tuple[int, int, int],
    source_dp_rank: int = 0,
    *,
    num_layers: int | None = None,
) -> tuple[tuple[int, int], ...]:
    """
    Return only source/destination ranks that can share a parameter.

    One selected source DP replica is sufficient. Every destination uses one intersecting TP source
    for replicated tensors and all TP sources whose sharded intervals intersect the destination.
    """
    src_dp, src_pp, src_tp = source_parallelism
    dst_dp, dst_pp, dst_tp = destination_parallelism
    if min(src_dp, src_pp, src_tp, dst_dp, dst_pp, dst_tp) <= 0:
        raise RuntimeConfigError(
            f"invalid routed weight parallelism {source_parallelism}->{destination_parallelism}"
        )
    if not 0 <= int(source_dp_rank) < src_dp:
        raise RuntimeConfigError(
            f"source DP rank {source_dp_rank} is outside source DP size {src_dp}"
        )
    if num_layers is not None and (
        int(num_layers) <= 0
        or int(num_layers) % src_pp != 0
        or int(num_layers) % dst_pp != 0
    ):
        raise RuntimeConfigError(
            f"routed weight layer count {num_layers} must be positive and divisible by "
            f"source PP={src_pp} and destination PP={dst_pp}"
        )
    pairs: set[tuple[int, int]] = set()
    for destination_rank in range(dst_dp * dst_pp * dst_tp):
        if num_layers is None:
            source_pp_ranks = range(src_pp)
        else:
            destination_pp_rank = (destination_rank // dst_tp) % dst_pp
            destination_layer_start = destination_pp_rank * int(num_layers) // dst_pp
            destination_layer_end = (destination_pp_rank + 1) * int(num_layers) // dst_pp
            source_pp_ranks = tuple(
                source_pp_rank
                for source_pp_rank in range(src_pp)
                if max(
                    source_pp_rank * int(num_layers) // src_pp,
                    destination_layer_start,
                ) < min(
                    (source_pp_rank + 1) * int(num_layers) // src_pp,
                    destination_layer_end,
                )
            )
        source_tp_ranks = routed_weight_source_tp_ranks(
            source_parallelism,
            destination_parallelism,
            destination_rank,
        )
        for src_pp_rank in source_pp_ranks:
            for src_tp_rank in source_tp_ranks:
                source_rank = ((int(source_dp_rank) * src_pp + src_pp_rank) * src_tp) + src_tp_rank
                pairs.add((source_rank, destination_rank))
    return tuple(sorted(pairs))


def _routed_weight_relay_endpoint_pairs(
    logical_pairs: tuple[tuple[int, int], ...],
    *,
    source_worker: WorkerSpec,
    destination_worker: WorkerSpec,
    destination_parallelism: tuple[int, int, int],
) -> tuple[tuple[tuple[int, int], ...], tuple[dict[str, int], ...]]:
    """
    Route one unique source fragment to one destination-DP relay.

    Destination DP replicas have identical PP/TP parameter shards. Sending a fragment to every
    replica multiplies Slingshot traffic and opens one NCCL P2P connection per replica. Pick one
    locality-aware relay for each unique (source rank, destination model-parallel rank) fragment;
    the receiver fans it out on its already-initialized DP communicator.
    """
    destination_mp_width = destination_parallelism[1] * destination_parallelism[2]
    candidates: dict[tuple[int, int], list[int]] = {}
    for source_rank, destination_rank in logical_pairs:
        key = (int(source_rank), int(destination_rank) % destination_mp_width)
        candidates.setdefault(key, []).append(int(destination_rank))
    relay_load: Counter[int] = Counter()
    selected: list[tuple[int, int]] = []
    routes: list[dict[str, int]] = []
    for (source_rank, destination_mp_rank), destination_ranks in sorted(candidates.items()):
        source_device = source_worker.devices[source_rank]

        def relay_score(destination_rank: int) -> tuple[int, int, int, int]:
            destination_device = destination_worker.devices[destination_rank]
            exact = (
                source_device.hostname == destination_device.hostname
                and source_device.local_cuda_index == destination_device.local_cuda_index
            )
            same_host = source_device.hostname == destination_device.hostname
            return (
                0 if exact else 1,
                0 if same_host else 1,
                relay_load[destination_rank],
                destination_rank,
            )

        relay_rank = min(sorted(set(destination_ranks)), key=relay_score)
        relay_load[relay_rank] += 1
        selected.append((source_rank, relay_rank))
        routes.append({
            "source_rank": source_rank,
            "destination_mp_rank": destination_mp_rank,
            "relay_destination_rank": relay_rank,
        })
    return tuple(sorted(selected)), tuple(routes)


def _checkpoint_index_sort_key(item: dict[str, object]) -> tuple[int, int, str]:
    return (
        1 if item.get("kind") == "final" else 0,
        int(item.get("step", 0)),
        str(item.get("name", "")),
    )


def _last_loss_from_context(context, vertex_id: str) -> float | None:
    if context is None:
        return None
    for node_id in reversed(list(context.completed)):
        result = context.completed[node_id]
        if vertex_id not in node_id:
            continue
        ranks = result.counters.get("ranks") if isinstance(result.counters, dict) else None
        if not ranks:
            continue
        for rank_info in ranks:
            counters = rank_info.get("counters", {})
            if "loss" in counters:
                return float(counters["loss"])
    return None


def _merge_rank_counters(results: tuple[SemanticResult, ...]) -> dict[str, object]:
    ranks = []
    total_bytes = 0
    total_network_bytes = 0
    total_local_ipc_bytes = 0
    total_local_endpoint_count = 0
    total_rank_duration_s = 0.0
    saw_bytes = False
    saw_network_bytes = False
    saw_local_ipc_bytes = False
    saw_local_endpoint_count = False
    saw_rank_duration = False
    copy_modes: set[str] = set()
    transport = None
    for result in results:
        counters = dict(result.counters or {})
        rank_entry: dict[str, object] = {
            "rank": result.rank,
            "counters": result.counters,
        }
        if counters.get("bytes") is not None:
            saw_bytes = True
            bytes_value = int(counters["bytes"])
            total_bytes += bytes_value
            rank_entry["bytes"] = bytes_value
        if counters.get("network_bytes") is not None:
            saw_network_bytes = True
            network_bytes = int(counters["network_bytes"])
            total_network_bytes += network_bytes
            rank_entry["network_bytes"] = network_bytes
        if counters.get("local_ipc_bytes") is not None:
            saw_local_ipc_bytes = True
            local_ipc_bytes = int(counters["local_ipc_bytes"])
            total_local_ipc_bytes += local_ipc_bytes
            rank_entry["local_ipc_bytes"] = local_ipc_bytes
        if counters.get("local_endpoint_count") is not None:
            saw_local_endpoint_count = True
            local_endpoint_count = int(counters["local_endpoint_count"])
            total_local_endpoint_count += local_endpoint_count
            rank_entry["local_endpoint_count"] = local_endpoint_count
        if counters.get("transfer_rank_duration_s") is not None:
            saw_rank_duration = True
            duration = float(counters["transfer_rank_duration_s"])
            total_rank_duration_s += duration
            rank_entry["transfer_rank_duration_s"] = duration
            if rank_entry.get("bytes") is not None:
                rank_entry["transfer_rank_bytes_per_s"] = (
                    float(rank_entry["bytes"]) / duration if duration > 0.0 else 0.0
                )
        for key in ("sha256", "source_sha256"):
            if counters.get(key) is not None:
                rank_entry[key] = str(counters[key])
        for key in ("copy_mode", "transport"):
            if counters.get(key) is not None:
                rank_entry[key] = str(counters[key])
        if counters.get("copy_mode") is not None:
            copy_modes.add(str(counters["copy_mode"]))
        if counters.get("transport") is not None:
            transport = str(counters["transport"])
        ranks.append(rank_entry)
    out: dict[str, object] = {
        "rank_count": len(results),
        "ranks": ranks,
    }
    if saw_bytes:
        out["bytes"] = total_bytes
    if saw_network_bytes:
        out["network_bytes"] = total_network_bytes
    if saw_local_ipc_bytes:
        out["local_ipc_bytes"] = total_local_ipc_bytes
    if saw_local_endpoint_count:
        out["local_endpoint_count"] = total_local_endpoint_count
    if saw_rank_duration:
        out["transfer_rank_duration_s"] = total_rank_duration_s
        if saw_bytes:
            out["transfer_rank_bytes_per_s"] = (
                float(total_bytes) / total_rank_duration_s if total_rank_duration_s > 0.0 else 0.0
            )
    if copy_modes:
        out["copy_mode"] = next(iter(copy_modes)) if len(copy_modes) == 1 else tuple(sorted(copy_modes))
    if transport is not None:
        out["transport"] = transport
    for key in (
        "relay_direct_received_bytes",
        "relay_broadcast_bytes",
    ):
        values = [
            int(result.counters[key])
            for result in results
            if result.counters is not None and result.counters.get(key) is not None
        ]
        if values:
            out[key] = sum(values)
    for key in (
        "source_dp_rank",
        "coalesced_weight_group",
        "coalesced_weight_group_count",
        "coalesced_weight_group_world_sizes",
        "destination_dp_relay",
    ):
        values = [
            result.counters[key]
            for result in results
            if result.counters is not None and result.counters.get(key) is not None
        ]
        if values and len(values) == len(results) and all(value == values[0] for value in values[1:]):
            out[key] = values[0]
    split_keys = {
        "artifact_part",
        "artifact_parts",
        "planned_batch_offset",
        "planned_n_seqs",
        "partial_n_active",
        "partial_phase_steps",
        "partial_phase_t_start",
        "partial_is_prefill_phase",
        "phase_target_tokens",
        "phase_continuation",
        "generation_length",
    }
    for key in split_keys:
        values = [
            result.counters[key]
            for result in results
            if result.counters is not None and result.counters.get(key) is not None
        ]
        if values and len(values) == len(results) and all(value == values[0] for value in values[1:]):
            out[key] = values[0]
    # Actual phase cardinality is data-dependent. TP/PP ranks repeat the cardinality for their DP
    # shard, so sum once per DP rank.
    per_dp: dict[int, int] = {}
    for result in results:
        counters = result.counters or {}
        if counters.get("partial_n_seqs") is None:
            continue
        dp_rank = int(counters.get("dp_rank", 0))
        value = int(counters["partial_n_seqs"])
        previous = per_dp.setdefault(dp_rank, value)
        if previous != value:
            raise RuntimeExecutionError(
                f"inconsistent partial_n_seqs within DP rank {dp_rank}: {previous} != {value}"
            )
    if per_dp:
        out["partial_n_seqs"] = sum(per_dp.values())
    # These counters are also repeated by TP/PP ranks and variable by DP shard. Preserve one
    # observation per DP rank, so production traces expose natural completion volume rather than
    # only the planned quantile quota.
    for key in (
        "phase_active_at_start",
        "phase_active_at_end",
        "phase_completed_sequences",
        "phase_compacted_sequences",
        "phase_forced_completions",
        "phase_deferred_completions",
    ):
        per_dp = {}
        for result in results:
            counters = result.counters or {}
            if counters.get(key) is None:
                continue
            dp_rank = int(counters.get("dp_rank", 0))
            value = int(counters[key])
            previous = per_dp.setdefault(dp_rank, value)
            if previous != value:
                raise RuntimeExecutionError(
                    f"inconsistent {key} within DP rank {dp_rank}: {previous} != {value}"
                )
        if per_dp:
            out[key] = sum(per_dp.values())
    return out
