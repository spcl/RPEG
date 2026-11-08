# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import asyncio
import time

from dataclasses import dataclass, field
from typing import Any, Protocol

from passes.lower import ExecNodeType
from runtime.containers import ContainerRef, ContainerStore, ShardingSpec, WeightVersion
from runtime.errors import RuntimeExecutionError
from runtime.graph import RuntimeGraph, RuntimeNodeSpec
from runtime.trace import RuntimeTraceWriter


@dataclass
class NodeExecutionResult:
    node_id: str
    status: str = "ok"
    counters: dict[str, Any] = field(default_factory=dict)
    containers: tuple[ContainerRef, ...] = ()
    weights: tuple[WeightVersion, ...] = ()


@dataclass
class RuntimeExecutionContext:
    run_id: str
    store: ContainerStore
    completed: dict[str, NodeExecutionResult] = field(default_factory=dict)
    logical_step: int | None = None
    state: dict[str, Any] = field(default_factory=dict)


class RuntimeOperationHandler(Protocol):
    async def execute(self, node: RuntimeNodeSpec, context: RuntimeExecutionContext) -> NodeExecutionResult:
        ...


class MetadataOnlyOperationHandler:
    """
    Correctness-oriented handler used for graph-level validation.

    It materializes metadata objects for COMM/RESHARD/OP nodes without touching tensors. Real
    backends implement the same execute method and can be swapped into RuntimeExecutor without
    changing scheduling semantics.
    """
    def __init__(self, compiled: Any | None = None) -> None:
        self.compiled = compiled

    async def execute(self, node: RuntimeNodeSpec, context: RuntimeExecutionContext) -> NodeExecutionResult:
        await asyncio.sleep(0)
        counters = {
            "kind": node.kind.value,
            "dependency_count": len(node.dependencies),
            "resource_count": len(node.resource_keys),
        }
        containers: list[ContainerRef] = []
        weights: list[WeightVersion] = []
        if node.kind == ExecNodeType.OP:
            counters.update(self._split_counters(node))
            consumed_weight_version = self._consumed_weight_version(node)
            if consumed_weight_version is not None:
                counters["weight_version"] = consumed_weight_version
                counters["model"] = node.model
                counters["shadow"] = self._uses_shadow_copies(node.model)
            container_name = f"{node.node_id}:output"
            containers.append(
                context.store.put_container(
                    ContainerRef(
                        name=container_name,
                        step=_node_step(node),
                        producer_node_id=node.node_id,
                        sharding=ShardingSpec("worker", (node.worker_id or "",)),
                        owner_worker_id=node.worker_id,
                        metadata={"role": node.role, "vertex_id": node.vertex_id},
                    )
                )
            )
            if node.role == "TRAIN" and node.model is not None:
                previous = context.store.latest_weight(node.model)
                next_version = 1 if previous is None else previous.version + 1
                shadow = self._uses_shadow_copies(node.model)
                counters["produced_weight"] = {"model": node.model, "version": next_version, "shadow": shadow}
                weights.append(
                    context.store.publish_weight(
                        WeightVersion(
                            model=node.model,
                            version=next_version,
                            produced_by=node.node_id,
                            step=_node_step(node),
                            layout=ShardingSpec("worker", (node.worker_id or "",)),
                            optimizer_state_owner=node.worker_id,
                            shadow=shadow,
                        )
                    )
                )
        elif node.kind in {ExecNodeType.COMM, ExecNodeType.RESHARD}:
            container_name = f"{node.node_id}:{node.label or node.kind.value}"
            containers.append(
                context.store.put_container(
                    ContainerRef(
                        name=container_name,
                        step=None,
                        producer_node_id=node.node_id,
                        sharding=ShardingSpec(node.kind.value),
                        metadata={"payload_in_bytes": node.payload_in_bytes, "label": node.label},
                    )
                )
            )
            counters["payload_in_bytes"] = node.payload_in_bytes
        return NodeExecutionResult(
            node_id=node.node_id,
            counters=counters,
            containers=tuple(containers),
            weights=tuple(weights),
        )

    def _consumed_weight_version(self, node: RuntimeNodeSpec) -> int | None:
        if self.compiled is None or node.model not in {"actor", "critic", "ac_share"}:
            return None
        if node.role not in {"GEN", "INF", "GENERATION", "INFERENCE"}:
            return None
        from runtime.weight_versions import consumer_weight_version

        step = _node_step(node)
        if step is None:
            return None
        return consumer_weight_version(self.compiled, node.model, step)

    def _uses_shadow_copies(self, model: str | None) -> bool:
        if self.compiled is None or model is None:
            return False
        from runtime.weight_versions import uses_shadow_copies

        return uses_shadow_copies(self.compiled, model)

    def _split_counters(self, node: RuntimeNodeSpec) -> dict[str, Any]:
        if node.exec_node.n_seqs is None:
            return {}
        counters: dict[str, Any] = {"partial_n_seqs": int(node.exec_node.n_seqs)}
        if node.exec_node.n_active is not None:
            counters["partial_n_active"] = int(node.exec_node.n_active)
        if node.exec_node.phase_steps is not None:
            counters["partial_phase_steps"] = int(node.exec_node.phase_steps)
        if node.exec_node.phase_t_start is not None:
            counters["partial_phase_t_start"] = int(node.exec_node.phase_t_start)
        counters["partial_is_prefill_phase"] = bool(node.exec_node.is_prefill_phase)
        part = _split_part_index(node.node_id)
        if part is not None:
            workload = getattr(getattr(self.compiled, "run_config", None), "workload", None)
            counters["artifact_part"] = part
            counters["artifact_parts"] = len(getattr(workload, "gen_quantiles", None) or ())
            counters["partial_batch_offset"] = _quantile_batch_offset(self.compiled, part)
            if node.vertex_id in {"g_pi", "g_ac"}:
                generation_length = _quantile_generation_length(self.compiled, part)
                if generation_length is not None:
                    counters["generation_length"] = generation_length
        return counters


def _node_step(node: RuntimeNodeSpec) -> int | None:
    if node.logical_step is not None:
        return int(node.logical_step)
    if node.exec_node.op is None:
        for node_id in (*node.dependencies, *node.successors):
            if "@" not in node_id:
                continue
            try:
                return int(node_id.rsplit("@", 1)[1].split(":", 1)[0]) + 1
            except ValueError:
                continue
        return None
    return int(node.exec_node.op.k) + 1


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


def _quantile_batch_offset(compiled: Any | None, part: int) -> int:
    if compiled is None or part <= 0:
        return 0
    workload = getattr(getattr(compiled, "run_config", None), "workload", None)
    quantiles = getattr(workload, "gen_quantiles", None) or ()
    total = int(getattr(workload, "batch_size", 0) or 0) * int(getattr(workload, "samples_per_prompt", 1) or 1)
    return sum(
        round(float(quantiles[idx][0]) * total) - (round(float(quantiles[idx - 1][0]) * total) if idx else 0)
        for idx in range(min(part, len(quantiles)))
    )


def _quantile_generation_length(compiled: Any | None, part: int) -> int | None:
    if compiled is None:
        return None
    workload = getattr(getattr(compiled, "run_config", None), "workload", None)
    quantiles = getattr(workload, "gen_quantiles", None) or ()
    if not quantiles:
        return getattr(workload, "generation_length", None)
    return int(quantiles[part][1])


class RuntimeExecutor:
    """
    Dependency-driven executor for lowered RPEG graphs.

    Scheduling is exact with respect to lowered dependencies and planned resources. A node can start
    only when all predecessors are complete and all resource keys are free. This conservative
    default avoids resource-ordering deadlocks.
    """
    def __init__(
        self,
        graph: RuntimeGraph,
        *,
        run_id: str,
        handler: RuntimeOperationHandler | None = None,
        trace: RuntimeTraceWriter | None = None,
        store: ContainerStore | None = None,
        max_concurrency: int | None = None,
        node_timeout_s: float | None = None,
        logical_step: int | None = None,
    ) -> None:
        self.graph = graph
        self.run_id = run_id
        self.handler = handler or MetadataOnlyOperationHandler()
        self.trace = trace
        self.store = store or ContainerStore()
        self.max_concurrency = max_concurrency
        self.node_timeout_s = node_timeout_s
        self.logical_step = logical_step

    async def run_async(self) -> RuntimeExecutionContext:
        pending = set(self.graph.nodes)
        running: dict[asyncio.Task[tuple[NodeExecutionResult, ...]], tuple[RuntimeNodeSpec, ...]] = {}
        started_at: dict[asyncio.Task[tuple[NodeExecutionResult, ...]], float] = {}
        completed: dict[str, NodeExecutionResult] = {}
        occupied: set[str] = set()
        context = RuntimeExecutionContext(
            run_id=self.run_id,
            store=self.store,
            completed=completed,
            logical_step=self.logical_step,
        )

        while pending or running:
            launched = self._launch_ready(pending, running, started_at, completed, occupied, context)
            if running:
                timeout = self.node_timeout_s
                done, _ = await asyncio.wait(running.keys(), timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    active = [node.node_id for task in running for node in running[task]]
                    raise RuntimeExecutionError(
                        f"runtime node timeout after {timeout:.1f}s; running nodes={active}"
                    )
                for task in done:
                    nodes = running.pop(task)
                    finished_at = time.perf_counter()
                    duration_s = finished_at - started_at.pop(task, finished_at)
                    for node in nodes:
                        occupied.difference_update(node.resource_keys)
                    try:
                        results = task.result()
                    except Exception as exc:
                        if self.trace is not None:
                            for node in nodes:
                                self.trace.append(
                                    "finish",
                                    node.node_id,
                                    status="error",
                                    worker_id=node.worker_id,
                                    step=self._trace_step(node),
                                    counters={"kind": node.kind.value, "duration_s": duration_s},
                                    message=str(exc),
                                )
                        for remaining in running:
                            remaining.cancel()
                        node_ids = [node.node_id for node in nodes]
                        if len(node_ids) == 1:
                            raise RuntimeExecutionError(f"runtime node {node_ids[0]} failed: {exc}") from exc
                        raise RuntimeExecutionError(f"runtime nodes {node_ids} failed: {exc}") from exc
                    result_by_node_id = {result.node_id: result for result in results}
                    missing = [node.node_id for node in nodes if node.node_id not in result_by_node_id]
                    if missing:
                        raise RuntimeExecutionError(f"runtime batch missing results for nodes={missing}")
                    for node in nodes:
                        result = result_by_node_id[node.node_id]
                        completed[node.node_id] = result
                        result.counters.setdefault("duration_s", duration_s)
                        if self.trace is not None:
                            self.trace.append(
                                "finish",
                                node.node_id,
                                status=result.status,
                                worker_id=node.worker_id,
                                step=self._trace_step(node),
                                counters=result.counters,
                            )
                        on_complete = getattr(self.handler, "on_node_complete", None)
                        if on_complete is not None:
                            maybe_awaitable = on_complete(node, context, result)
                            if asyncio.iscoroutine(maybe_awaitable):
                                await maybe_awaitable
                continue

            if pending and not launched:
                ready = self._ready_nodes(pending, completed)
                if ready:
                    blocked = {
                        node.node_id: sorted(set(node.resource_keys) & occupied)
                        for node in ready
                    }
                    raise RuntimeExecutionError(f"ready nodes are blocked with no running work: {blocked}")
                waiting = {
                    node_id: sorted(set(self.graph.nodes[node_id].dependencies) - set(completed))
                    for node_id in sorted(pending)
                }
                raise RuntimeExecutionError(f"runtime graph made no progress; unresolved dependencies: {waiting}")

        return context

    def run(self) -> RuntimeExecutionContext:
        return asyncio.run(self.run_async())

    def _trace_step(self, node: RuntimeNodeSpec) -> int | None:
        return self.logical_step if self.logical_step is not None else _node_step(node)

    def _ready_nodes(
        self,
        pending: set[str],
        completed: dict[str, NodeExecutionResult],
    ) -> list[RuntimeNodeSpec]:
        completed_ids = set(completed)
        topological_position = {
            node_id: index for index, node_id in enumerate(self.graph.topological_order)
        }
        ready = [
            self.graph.nodes[node_id]
            for node_id in self.graph.topological_order
            if node_id in pending and set(self.graph.nodes[node_id].dependencies) <= completed_ids
        ]
        # Drain the oldest ready logical step first. This prevents a future inference op from
        # repeatedly taking a source worker immediately before its older activation send can be
        # launched, which otherwise serializes a nominal tau>0 inference/training pipeline.
        ready.sort(
            key=lambda node: (
                node.logical_step if node.logical_step is not None else 1 << 60,
                0 if node.kind in {ExecNodeType.COMM, ExecNodeType.RESHARD} else 1,
                topological_position[node.node_id],
            )
        )
        return ready

    def _launch_ready(
        self,
        pending: set[str],
        running: dict[asyncio.Task[tuple[NodeExecutionResult, ...]], tuple[RuntimeNodeSpec, ...]],
        started_at: dict[asyncio.Task[tuple[NodeExecutionResult, ...]], float],
        completed: dict[str, NodeExecutionResult],
        occupied: set[str],
        context: RuntimeExecutionContext,
    ) -> bool:
        launched = False
        ready_nodes = self._ready_nodes(pending, completed)
        for node in ready_nodes:
            if node.node_id not in pending:
                continue
            if self.max_concurrency is not None and len(running) >= self.max_concurrency:
                break
            if set(node.resource_keys) & occupied:
                continue
            batch_nodes = self._select_node_batch(node, ready_nodes, pending, occupied, context)
            batch_resources: set[str] = set()
            for batch_node in batch_nodes:
                batch_resources.update(batch_node.resource_keys)
            if batch_resources & occupied:
                continue
            for batch_node in batch_nodes:
                pending.remove(batch_node.node_id)
                if self.trace is not None:
                    trace_counters: dict[str, Any] = {
                        "kind": batch_node.kind.value,
                        "resources": list(batch_node.resource_keys),
                        "scheduled_at": time.time(),
                        "batch_scheduled_nodes": [item.node_id for item in batch_nodes],
                    }
                    for key, value in (
                        ("model", batch_node.model),
                        ("role", batch_node.role),
                        ("vertex_id", batch_node.vertex_id),
                        ("parallelism", batch_node.parallelism),
                        ("microbatch", batch_node.microbatch),
                        ("label", batch_node.label),
                    ):
                        if value is not None:
                            trace_counters[key] = value
                    self.trace.append(
                        "start",
                        batch_node.node_id,
                        worker_id=batch_node.worker_id,
                        step=self._trace_step(batch_node),
                        counters=trace_counters,
                    )
            occupied.update(batch_resources)
            task = asyncio.create_task(self._execute_node_batch(batch_nodes, context))
            running[task] = batch_nodes
            started_at[task] = time.perf_counter()
            launched = True
        return launched

    def _select_node_batch(
        self,
        node: RuntimeNodeSpec,
        ready_nodes: list[RuntimeNodeSpec],
        pending: set[str],
        occupied: set[str],
        context: RuntimeExecutionContext,
    ) -> tuple[RuntimeNodeSpec, ...]:
        selector = getattr(self.handler, "select_batch", None)
        if selector is None:
            return (node,)
        available = [
            candidate
            for candidate in ready_nodes
            if candidate.node_id in pending and not (set(candidate.resource_keys) & occupied)
        ]
        selected = tuple(selector(node, tuple(available), context))
        if not selected or selected[0].node_id != node.node_id:
            return (node,)
        seen: set[str] = set()
        out: list[RuntimeNodeSpec] = []
        available_ids = {candidate.node_id for candidate in available}
        for candidate in selected:
            if candidate.node_id in seen or candidate.node_id not in available_ids:
                continue
            seen.add(candidate.node_id)
            out.append(candidate)
        return tuple(out) or (node,)

    async def _execute_node_batch(
        self,
        nodes: tuple[RuntimeNodeSpec, ...],
        context: RuntimeExecutionContext,
    ) -> tuple[NodeExecutionResult, ...]:
        if len(nodes) > 1:
            execute_batch = getattr(self.handler, "execute_batch", None)
            if execute_batch is not None:
                result = execute_batch(nodes, context)
                if asyncio.iscoroutine(result):
                    result = await result
                return tuple(result)
        return (
            await self.handler.execute(nodes[0], context),
        )
