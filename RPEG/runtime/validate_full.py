# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import argparse
import json
import sys

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.types import Allocation, Model
from passes.lower import ExecNode, ExecNodeType
from passes.unroll import UnrolledNode
from runtime.algorithms.dpo_graph import run_graph_dpo
from runtime.algorithms.grpo_graph import run_graph_grpo
from runtime.algorithms.ppo_graph import run_graph_ppo
from runtime.compiler import CompiledRuntimePlan, WorkerSpec, compile_runtime_plan
from runtime.errors import RuntimeConfigError, RuntimeDependencyError, RuntimeExecutionError
from runtime.executor import MetadataOnlyOperationHandler, RuntimeExecutor
from runtime.graph import RuntimeGraph, RuntimeNodeSpec, build_runtime_graph
from runtime.trace import RuntimeTraceWriter
from runtime.verification import verify_runtime_trace


_PLAN_MODEL_NAMES = {"actor", "critic", "reference", "reward", "ac_share"}


@dataclass(frozen=True)
class PhaseReport:
    phase: str
    status: str
    details: dict[str, Any]


def _fail(message: str) -> None:
    raise RuntimeExecutionError(f"full validation failed: {message}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _node_key(node: UnrolledNode) -> tuple[str, int]:
    return (node.template_vertex.id, int(node.k))


def _exec_step(node: RuntimeNodeSpec) -> int | None:
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


def _allocation_resource_keys(allocation: Allocation) -> tuple[str, ...]:
    r0, n0, g0 = allocation.origin
    re, ne, ge = allocation.extent
    return tuple(
        f"gpu:{r}:{n}:{g}"
        for r in range(r0, r0 + re)
        for n in range(n0, n0 + ne)
        for g in range(g0, g0 + ge)
    )


def _path_exists(src_ids: tuple[str, ...], dst_ids: tuple[str, ...], graph: dict[str, tuple[str, ...]]) -> bool:
    targets = set(dst_ids)
    ready = deque(src_ids)
    seen = set(src_ids)
    while ready:
        node_id = ready.popleft()
        if node_id in targets:
            return True
        for succ in graph.get(node_id, ()):
            if succ not in seen:
                seen.add(succ)
                ready.append(succ)
    return False


def _validate_input_and_rpeg(compiled: CompiledRuntimePlan) -> PhaseReport:
    rpeg = compiled.config.rpeg
    run = compiled.run_config
    opt = compiled.optimized_plan

    try:
        rpeg.well_formed(verbose=False)
    except Exception as exc:  # noqa: BLE001 - report the exact well-formedness failure.
        _fail(f"RPEG is not well formed: {exc}")

    vertex_ids = [v.id for v in rpeg.vertices]
    _require(len(vertex_ids) == len(set(vertex_ids)), f"RPEG has duplicate vertex ids: {vertex_ids}")
    _require(set(vertex_ids) == set(opt.vertices), "RPEG vertices and optimized plan vertices diverge")
    _require(run.training.max_steps >= 1, "training.max_steps must be >= 1")

    tau_raw = opt.structure.get("tau") or {}
    unknown_tau = sorted(str(name) for name in tau_raw if str(name) not in _PLAN_MODEL_NAMES)
    _require(not unknown_tau, f"optimized plan contains unknown tau model names: {unknown_tau}")
    for model, tau in rpeg.tau.items():
        _require(isinstance(tau, int) and tau >= 0, f"tau[{model.value}] must be a non-negative integer")

    plan = compiled.config.plan
    for vertex in rpeg.vertices:
        _require(vertex in plan.sigma, f"execution plan missing allocation for {vertex.id}")
        _require(vertex in plan.pi, f"execution plan missing parallelism for {vertex.id}")
        _require(vertex in plan.mb, f"execution plan missing microbatch for {vertex.id}")

    return PhaseReport(
        "input",
        "ok",
        {
            "run_id": run.run_id,
            "algorithm": run.algorithm.name,
            "ac_share": bool(opt.structure.get("ac_share", False)),
            "training_steps": run.training.max_steps,
            "vertices": len(rpeg.vertices),
            "data_edges": len(rpeg.data_edges),
            "publication_edges": len(rpeg.pub_edges),
            "tau": {model.value: tau for model, tau in sorted(rpeg.tau.items(), key=lambda item: item[0].value)},
        },
    )


def _validate_unrolled_graph(compiled: CompiledRuntimePlan) -> PhaseReport:
    rpeg = compiled.config.rpeg
    unrolled = compiled.unrolled
    horizon = compiled.run_config.training.max_steps
    node_keys = [_node_key(node) for node in unrolled.nodes]

    _require(len(node_keys) == len(set(node_keys)), "unrolled graph contains duplicate operator nodes")
    _require(len(unrolled.nodes) == len(rpeg.vertices) * horizon, "unrolled graph node count does not match |V| * K")

    expected_keys = {(vertex.id, k) for vertex in rpeg.vertices for k in range(horizon)}
    _require(set(node_keys) == expected_keys, "unrolled graph is missing one or more vertex-step operators")
    _require(
        len(unrolled.data_edges) == len(rpeg.data_edges) * horizon,
        "unrolled data edge count does not match template edge count times K",
    )

    expected_pub_edges = 0
    for k in range(1, horizon):
        for edge in rpeg.pub_edges:
            if k > rpeg.tau.get(edge.src.kind.model, 0):
                expected_pub_edges += 1
    _require(
        len(unrolled.pub_edges) == expected_pub_edges,
        f"unrolled publication edge count {len(unrolled.pub_edges)} != expected {expected_pub_edges}",
    )

    unrolled_key_set = set(unrolled.nodes)
    for edge in (*unrolled.data_edges, *unrolled.pub_edges):
        _require(edge.src in unrolled_key_set, f"unrolled edge source is not a node: {edge.src}")
        _require(edge.dst in unrolled_key_set, f"unrolled edge destination is not a node: {edge.dst}")
    for edge in unrolled.data_edges:
        _require(edge.src.k == edge.dst.k, f"data edge crosses steps: {edge.src}->{edge.dst}")
    for edge in unrolled.pub_edges:
        _require(edge.src.k < edge.dst.k, f"publication edge is not forward in time: {edge.src}->{edge.dst}")

    return PhaseReport(
        "graph",
        "ok",
        {
            "unrolled_nodes": len(unrolled.nodes),
            "unrolled_data_edges": len(unrolled.data_edges),
            "unrolled_publication_edges": len(unrolled.pub_edges),
        },
    )


def _validate_lowered(compiled: CompiledRuntimePlan) -> PhaseReport:
    lowered = compiled.lowered
    unrolled = compiled.unrolled
    horizon = compiled.run_config.training.max_steps

    node_ids = [node.id for node in lowered.nodes]
    duplicates = sorted(node_id for node_id, count in Counter(node_ids).items() if count > 1)
    _require(not duplicates, f"lowered graph has duplicate node ids: {duplicates}")
    by_id = {node.id: node for node in lowered.nodes}

    successors: dict[str, list[str]] = defaultdict(list)
    for edge in lowered.edges:
        _require(edge.src.id in by_id, f"lowered edge references unknown source {edge.src.id}")
        _require(edge.dst.id in by_id, f"lowered edge references unknown destination {edge.dst.id}")
        successors[edge.src.id].append(edge.dst.id)

    op_nodes_by_unrolled: dict[UnrolledNode, list[ExecNode]] = defaultdict(list)
    for node in lowered.nodes:
        if node.kind == ExecNodeType.OP:
            _require(node.op is not None, f"OP node {node.id} is missing unrolled metadata")
            _require(0 <= node.op.k < horizon, f"OP node {node.id} has step outside validation horizon")
            _require(node.model == node.op.template_vertex.kind.model, f"OP node {node.id} model does not match template")
            _require(node.pi is not None, f"OP node {node.id} is missing parallelism")
            _require(node.mb is not None, f"OP node {node.id} is missing microbatch")
            op_nodes_by_unrolled[node.op].append(node)
        else:
            _require(len(node.allocs) >= 2, f"{node.kind.value} node {node.id} must carry source and destination allocs")
            _require(node.payload_in_bytes > 0, f"{node.kind.value} node {node.id} has non-positive payload")

    for unrolled_node in unrolled.nodes:
        _require(op_nodes_by_unrolled[unrolled_node], f"lowering lost unrolled node {_node_key(unrolled_node)}")

    graph = {node_id: tuple(dst_ids) for node_id, dst_ids in successors.items()}
    for edge in (*unrolled.data_edges, *unrolled.pub_edges):
        src_ids = tuple(node.id for node in op_nodes_by_unrolled[edge.src])
        dst_ids = tuple(node.id for node in op_nodes_by_unrolled[edge.dst])
        _require(
            _path_exists(src_ids, dst_ids, graph),
            f"lowering lost dependency path {edge.src.template_vertex.id}@{edge.src.k}"
            f" -> {edge.dst.template_vertex.id}@{edge.dst.k}",
        )

    counts = Counter(node.kind.value for node in lowered.nodes)
    return PhaseReport(
        "lower",
        "ok",
        {
            "lowered_nodes": len(lowered.nodes),
            "lowered_edges": len(lowered.edges),
            "counts": dict(sorted(counts.items())),
        },
    )


def _validate_worker_mapping(graph: RuntimeGraph, workers: tuple[WorkerSpec, ...]) -> None:
    worker_by_vertex = {
        vertex_id: worker
        for worker in workers
        for vertex_id in worker.owned_vertex_ids
    }
    for node in graph.nodes.values():
        if node.kind != ExecNodeType.OP:
            continue
        _require(node.vertex_id is not None, f"runtime OP node {node.node_id} is missing vertex_id")
        worker = worker_by_vertex.get(node.vertex_id)
        _require(worker is not None, f"runtime OP node {node.node_id} has no worker for vertex {node.vertex_id}")
        _require(node.worker_id == worker.worker_id, f"runtime OP node {node.node_id} is mapped to the wrong worker")
        _require(node.allocations == (worker.allocation,), f"runtime OP node {node.node_id} allocation diverges from worker")
        _require(node.parallelism == worker.parallelism, f"runtime OP node {node.node_id} parallelism diverges from worker")
        _require(node.microbatch is not None and node.microbatch > 0, f"runtime OP node {node.node_id} has invalid microbatch")


def _validate_runtime_graph(compiled: CompiledRuntimePlan, graph: RuntimeGraph) -> PhaseReport:
    lowered = compiled.lowered
    lowered_ids = {node.id for node in lowered.nodes}
    lowered_edges = {(edge.src.id, edge.dst.id) for edge in lowered.edges}

    _require(set(graph.nodes) == lowered_ids, "runtime graph does not preserve lowered node set")
    _require(set(graph.edges) == lowered_edges, "runtime graph does not preserve lowered edge set")
    _require(set(graph.topological_order) == set(graph.nodes), "runtime topological order does not cover every node")

    position = {node_id: idx for idx, node_id in enumerate(graph.topological_order)}
    for src, dst in graph.edges:
        _require(position[src] < position[dst], f"runtime topological order violates edge {src}->{dst}")

    from passes.transfer_resources import reserved_allocations

    lowered_by_id = {node.id: node for node in lowered.nodes}
    predecessor_ids = {node_id: set(node.dependencies) for node_id, node in graph.nodes.items()}
    successor_ids = {node_id: set(node.successors) for node_id, node in graph.nodes.items()}
    for node in graph.nodes.values():
        _require(all(dep in graph.nodes for dep in node.dependencies), f"{node.node_id} has unknown dependencies")
        _require(all(succ in graph.nodes for succ in node.successors), f"{node.node_id} has unknown successors")
        if node.kind in {ExecNodeType.COMM, ExecNodeType.RESHARD} and len(node.allocations) >= 2:
            reserved = reserved_allocations(
                node.exec_node,
                policy=compiled.config.transfer_resource_policy,
                predecessors=predecessor_ids,
                successors=successor_ids,
                by_id=lowered_by_id,
            )
            expected = {
                key
                for allocation in reserved
                for key in _allocation_resource_keys(allocation)
            }
            actual = {key for key in node.resource_keys if key.startswith("gpu:")}
            policy = "destination only" if reserved == node.allocations[-1:] else "both endpoints"
            _require(
                actual == expected,
                f"{node.node_id} transfer resources must reserve {policy}",
            )
        if node.kind == ExecNodeType.OP:
            _require(node.worker_id is not None, f"runtime OP node {node.node_id} has no worker")
            _require(any(key == f"worker:{node.worker_id}" for key in node.resource_keys), f"{node.node_id} lacks worker resource")

    _validate_worker_mapping(graph, compiled.workers)
    counts = graph.summary()["counts"]
    return PhaseReport(
        "runtime_graph",
        "ok",
        {
            "runtime_nodes": len(graph.nodes),
            "runtime_edges": len(graph.edges),
            "counts": counts,
        },
    )


def _read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _validate_execution(
    compiled: CompiledRuntimePlan,
    graph: RuntimeGraph,
    *,
    trace_path: Path,
    max_concurrency: int | None = None,
    node_timeout_s: float | None = None,
) -> PhaseReport:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("")
    trace = RuntimeTraceWriter(trace_path, compiled.run_config.run_id)
    context = RuntimeExecutor(
        graph,
        run_id=compiled.run_config.run_id,
        handler=MetadataOnlyOperationHandler(compiled),
        trace=trace,
        max_concurrency=max_concurrency,
        node_timeout_s=node_timeout_s,
    ).run()

    _require(set(context.completed) == set(graph.nodes), "metadata execution did not complete every runtime node")
    for node_id, result in context.completed.items():
        _require(result.status == "ok", f"metadata execution returned non-ok status for {node_id}: {result.status}")
        node = graph.nodes[node_id]
        if node.kind == ExecNodeType.OP:
            _require(result.containers, f"OP node {node_id} did not produce an output container")
            for ref in result.containers:
                _require(ref.name in context.store.containers, f"container {ref.name} missing from metadata store")
                _require(ref.step == _exec_step(node), f"container {ref.name} has incorrect logical step")
            if node.role == "TRAIN" and node.model is not None:
                _require(result.weights, f"TRAIN node {node_id} did not publish a weight version")
        elif node.kind in {ExecNodeType.COMM, ExecNodeType.RESHARD}:
            _require(result.containers, f"transfer node {node_id} did not produce transfer metadata")
            _require(
                result.counters.get("payload_in_bytes") == node.payload_in_bytes,
                f"transfer node {node_id} payload counter does not match lowered payload",
            )

    events = _read_trace(trace_path)
    _require(len(events) == 2 * len(graph.nodes), "trace does not contain exactly one start and finish per node")
    starts = Counter(event["node_id"] for event in events if event.get("event") == "start")
    finishes = Counter(event["node_id"] for event in events if event.get("event") == "finish")
    expected_events = Counter(graph.nodes.keys())
    _require(starts == expected_events, "trace start events do not match runtime nodes")
    _require(finishes == expected_events, "trace finish events do not match runtime nodes")
    trace_verification = verify_runtime_trace(compiled, graph, events)

    return PhaseReport(
        "execution",
        "ok",
        {
            "executed_nodes": len(context.completed),
            "containers": len(context.store.containers),
            "weight_versions": len(context.store.weights),
            "trace_path": str(trace_path),
            "trace_events": len(events),
            "trace_verification": trace_verification,
        },
    )


def _run_real_backend(compiled: CompiledRuntimePlan) -> PhaseReport:
    runtime_trace_path = Path(compiled.run_config.runtime.trace_path)
    runtime_trace_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_trace_path.write_text("")
    algorithm = compiled.run_config.algorithm.name
    if algorithm == "dpo":
        result = run_graph_dpo(compiled)
        details = asdict(result)
        details["trace_verification"] = _verify_real_backend_trace(compiled)
        return PhaseReport("real_backend", "ok", details)
    if algorithm == "ppo":
        result = run_graph_ppo(compiled)
        details = asdict(result)
        details["trace_verification"] = _verify_real_backend_trace(compiled)
        return PhaseReport("real_backend", "ok", details)
    if algorithm == "grpo":
        result = run_graph_grpo(compiled)
        details = asdict(result)
        details["trace_verification"] = _verify_real_backend_trace(compiled)
        return PhaseReport("real_backend", "ok", details)
    raise RuntimeConfigError(f"real backend execution is not implemented for {algorithm.upper()}")


def _verify_real_backend_trace(compiled: CompiledRuntimePlan) -> dict[str, Any]:
    graph = compiled.runtime_graph()
    events = _read_trace(Path(compiled.run_config.runtime.trace_path))
    _require(events, f"real backend did not write trace {compiled.run_config.runtime.trace_path}")
    return verify_runtime_trace(compiled, graph, events)


def validate_runtime_end_to_end(
    config_path: str | Path,
    *,
    trace_path: str | Path | None = None,
    execute_real_backend: bool = False,
    max_concurrency: int | None = None,
    node_timeout_s: float | None = None,
) -> dict[str, Any]:
    compiled = compile_runtime_plan(config_path, dry_run=None)
    graph = compiled.runtime_graph()
    default_trace = Path(compiled.run_config.runtime.trace_path).with_suffix(
        Path(compiled.run_config.runtime.trace_path).suffix + ".full_validate.jsonl"
    )
    validation_trace_path = Path(trace_path) if trace_path is not None else default_trace

    reports = [
        _validate_input_and_rpeg(compiled),
        _validate_unrolled_graph(compiled),
        _validate_lowered(compiled),
        _validate_runtime_graph(compiled, graph),
        _validate_execution(
            compiled,
            graph,
            trace_path=validation_trace_path,
            max_concurrency=max_concurrency,
            node_timeout_s=node_timeout_s,
        ),
    ]
    if execute_real_backend:
        reports.append(_run_real_backend(compiled))

    return {
        "status": "ok",
        "run_id": compiled.run_config.run_id,
        "algorithm": compiled.run_config.algorithm.name,
        "phases": [asdict(report) for report in reports],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate an RPEG runtime run end-to-end: input -> graph -> lower -> execution."
    )
    parser.add_argument("--config", required=True, help="Path to rpeg.run.v1 runtime config")
    parser.add_argument("--trace-path", help="Validation trace path. Defaults to runtime.trace_path + '.full_validate.jsonl'.")
    parser.add_argument("--execute-real-backend", action="store_true", help="Run the configured real backend after metadata validation")
    parser.add_argument("--max-concurrency", type=int, help="Optional executor concurrency limit for metadata validation")
    parser.add_argument("--node-timeout-s", type=float, help="Optional per-node timeout for metadata validation")
    args = parser.parse_args(argv)

    try:
        report = validate_runtime_end_to_end(
            args.config,
            trace_path=args.trace_path,
            execute_real_backend=args.execute_real_backend,
            max_concurrency=args.max_concurrency,
            node_timeout_s=args.node_timeout_s,
        )
    except (RuntimeConfigError, RuntimeDependencyError, RuntimeExecutionError) as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
