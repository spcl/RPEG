# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from core.types import Allocation, TransferResourcePolicy
from passes.lower import ExecNode, ExecNodeType, ExecutableRPEG
from passes.transfer_resources import reserved_allocations, transfer_reserves_destination_only
from runtime.errors import RuntimeConfigError


@dataclass(frozen=True)
class RuntimeNodeSpec:
    """
    Executable runtime view of one lowered RPEG node.
    """
    node_id: str
    kind: ExecNodeType
    exec_node: ExecNode
    dependencies: tuple[str, ...]
    successors: tuple[str, ...]
    worker_id: str | None
    vertex_id: str | None
    model: str | None
    role: str | None
    parallelism: tuple[int, int, int] | None
    microbatch: int | None
    allocations: tuple[Allocation, ...]
    resource_keys: tuple[str, ...]
    payload_in_bytes: int = 0
    label: str | None = None
    logical_step: int | None = None
    source_vertex_id: str | None = None
    destination_vertex_id: str | None = None

    def to_summary(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "dependencies": list(self.dependencies),
            "successors": list(self.successors),
            "worker_id": self.worker_id,
            "vertex_id": self.vertex_id,
            "model": self.model,
            "role": self.role,
            "parallelism": self.parallelism,
            "microbatch": self.microbatch,
            "resource_keys": list(self.resource_keys),
            "payload_in_bytes": self.payload_in_bytes,
            "label": self.label,
            "logical_step": self.logical_step,
            "source_vertex_id": self.source_vertex_id,
            "destination_vertex_id": self.destination_vertex_id,
            "fusion_group": self.exec_node.fusion_group,
        }


@dataclass(frozen=True)
class RuntimeGraph:
    nodes: dict[str, RuntimeNodeSpec]
    topological_order: tuple[str, ...]
    composite_operators: tuple[dict[str, Any], ...] = ()

    @property
    def edges(self) -> tuple[tuple[str, str], ...]:
        return tuple((src, dst) for src, node in self.nodes.items() for dst in node.successors)

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for node in self.nodes.values():
            counts[node.kind.value] = counts.get(node.kind.value, 0) + 1
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "counts": counts,
            "topological_order": list(self.topological_order),
            "nodes": [self.nodes[node_id].to_summary() for node_id in self.topological_order],
            "composite_operators": list(self.composite_operators),
        }


def _allocation_resource_keys(allocation: Allocation) -> tuple[str, ...]:
    r0, n0, g0 = allocation.origin
    re, ne, ge = allocation.extent
    keys = []
    for r in range(r0, r0 + re):
        for n in range(n0, n0 + ne):
            for g in range(g0, g0 + ge):
                keys.append(f"gpu:{r}:{n}:{g}")
    return tuple(keys)


def _logical_step(
    node: ExecNode,
    predecessors: set[str],
    successors: set[str],
) -> int | None:
    if node.op is not None:
        return int(node.op.k) + 1
    for adjacent_id in (*sorted(predecessors), *sorted(successors)):
        if "@" not in adjacent_id:
            continue
        try:
            return int(adjacent_id.rsplit("@", 1)[1].split(":", 1)[0]) + 1
        except ValueError:
            continue
    return None


def _adjacent_vertex_id(node_ids: set[str]) -> str | None:
    for node_id in sorted(node_ids):
        if "@" in node_id:
            return node_id.split("@", 1)[0]
    return None


def _transfer_worker_resource_keys(
    *,
    node_id: str,
    predecessors: dict[str, set[str]],
    successors: dict[str, set[str]],
    by_id: dict[str, ExecNode],
    worker_by_vertex: dict[str, Any],
) -> set[str]:
    keys: set[str] = set()
    adjacent = tuple(sorted(predecessors[node_id])) + tuple(sorted(successors[node_id]))
    for adjacent_id in adjacent:
        adjacent_node = by_id.get(adjacent_id)
        if adjacent_node is None or adjacent_node.op is None:
            continue
        vertex_id = adjacent_node.op.template_vertex.id
        worker = worker_by_vertex.get(vertex_id)
        worker_id = None if worker is None else getattr(worker, "worker_id", None)
        if worker_id is not None:
            keys.add(f"worker:{worker_id}")
    return keys


def _destination_worker_resource_key(
    *,
    node_id: str,
    successors: dict[str, set[str]],
    by_id: dict[str, ExecNode],
    worker_by_vertex: dict[str, Any],
) -> str | None:
    for adjacent_id in sorted(successors[node_id]):
        adjacent_node = by_id.get(adjacent_id)
        if adjacent_node is None or adjacent_node.op is None:
            continue
        worker = worker_by_vertex.get(adjacent_node.op.template_vertex.id)
        worker_id = None if worker is None else getattr(worker, "worker_id", None)
        if worker_id is not None:
            return f"worker:{worker_id}"
    return None


def _resource_keys(
    node: ExecNode,
    worker_id: str | None,
    *,
    predecessors: dict[str, set[str]],
    successors: dict[str, set[str]],
    by_id: dict[str, ExecNode],
    worker_by_vertex: dict[str, Any],
    transfer_resource_policy: TransferResourcePolicy,
) -> tuple[str, ...]:
    keys: set[str] = set()
    if worker_id is not None:
        keys.add(f"worker:{worker_id}")
    if node.kind in {ExecNodeType.COMM, ExecNodeType.RESHARD} and len(node.allocs) >= 2:
        # Semantic paired transports issue real commands to both persistent worker groups; reserve
        # those workers, so concurrent transfer nodes do not race through the same command channel.
        destination_only = transfer_reserves_destination_only(
            node,
            policy=transfer_resource_policy,
            predecessors=predecessors,
            successors=successors,
            by_id=by_id,
        )
        if destination_only:
            destination_worker_key = _destination_worker_resource_key(
                node_id=node.id,
                successors=successors,
                by_id=by_id,
                worker_by_vertex=worker_by_vertex,
            )
            if destination_worker_key is not None:
                keys.add(destination_worker_key)
        else:
            keys.update(
                _transfer_worker_resource_keys(
                    node_id=node.id,
                    predecessors=predecessors,
                    successors=successors,
                    by_id=by_id,
                    worker_by_vertex=worker_by_vertex,
                )
            )
        # A paired transfer consumes memory and transport workspaces on both endpoint allocations.
        # Reserve the source GPUs as well as the destination unless an explicitly supported
        # source-overlap mode made the transfer destination-only above. Worker keys alone are not
        # sufficient: another persistent worker can share the same physical GPU and otherwise
        # overlap a second transfer whose receive buffers exceed HBM.
        transfer_allocations = reserved_allocations(
            node,
            policy=transfer_resource_policy,
            predecessors=predecessors,
            successors=successors,
            by_id=by_id,
        )
        for allocation in transfer_allocations:
            keys.update(_allocation_resource_keys(allocation))
    else:
        for allocation in node.allocs:
            keys.update(_allocation_resource_keys(allocation))
    return tuple(sorted(keys))


def _topological_order(
    node_ids: set[str],
    predecessors: dict[str, set[str]],
    successors: dict[str, set[str]],
) -> tuple[str, ...]:
    indegree = {node_id: len(predecessors[node_id]) for node_id in node_ids}
    ready = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    order: list[str] = []
    while ready:
        node_id = ready.popleft()
        order.append(node_id)
        for succ in sorted(successors[node_id]):
            indegree[succ] -= 1
            if indegree[succ] == 0:
                ready.append(succ)
    if len(order) != len(node_ids):
        cyclic = sorted(node_id for node_id, degree in indegree.items() if degree > 0)
        raise RuntimeConfigError(f"lowered graph contains a dependency cycle involving: {cyclic[:20]}")
    return tuple(order)


def _validated_preferred_order(
    preferred_order: tuple[str, ...],
    node_ids: set[str],
    predecessors: dict[str, set[str]],
) -> tuple[str, ...]:
    if len(preferred_order) != len(set(preferred_order)):
        raise RuntimeConfigError("preferred runtime order contains duplicate node ids")
    actual = set(preferred_order)
    if actual != node_ids:
        raise RuntimeConfigError(
            "preferred runtime order node set mismatch: "
            f"missing={sorted(node_ids - actual)[:20]} extra={sorted(actual - node_ids)[:20]}"
        )
    position = {node_id: idx for idx, node_id in enumerate(preferred_order)}
    for node_id, deps in predecessors.items():
        for dependency in deps:
            if position[dependency] >= position[node_id]:
                raise RuntimeConfigError(
                    f"preferred runtime order violates dependency {dependency}->{node_id}"
                )
    return preferred_order


def build_runtime_graph(
    lowered: ExecutableRPEG,
    workers: tuple[Any, ...],
    *,
    preferred_order: tuple[str, ...] | None = None,
    transfer_resource_policy: TransferResourcePolicy | None = None,
    destination_only_weight_transfers: bool = False,
    destination_only_activation_transfers: bool = False,
    checkpoint_barrier_steps: tuple[int, ...] = (),
) -> RuntimeGraph:
    """
    Convert ExecutableRPEG into an executor-owned graph.

    This preserves every lowered node. OP nodes are mapped to their per-template worker group, while
    COMM/RESHARD nodes remain first-class executable nodes.
    """
    node_ids = [node.id for node in lowered.nodes]
    duplicate_ids = sorted({node_id for node_id in node_ids if node_ids.count(node_id) > 1})
    if duplicate_ids:
        raise RuntimeConfigError(f"lowered graph has duplicate node ids: {duplicate_ids}")

    if transfer_resource_policy is None:
        transfer_resource_policy = TransferResourcePolicy(
            weight=("destination_only" if destination_only_weight_transfers else "both_endpoints"),
            activation=(
                "destination_only" if destination_only_activation_transfers else "both_endpoints"
            ),
        )
    elif destination_only_weight_transfers or destination_only_activation_transfers:
        raise RuntimeConfigError(
            "transfer_resource_policy cannot be combined with legacy destination-only flags"
        )

    by_id = {node.id: node for node in lowered.nodes}
    predecessors = {node.id: set() for node in lowered.nodes}
    successors = {node.id: set() for node in lowered.nodes}
    for edge in lowered.edges:
        if edge.src.id not in by_id or edge.dst.id not in by_id:
            raise RuntimeConfigError(f"lowered graph edge references unknown node: {edge.src.id}->{edge.dst.id}")
        successors[edge.src.id].add(edge.dst.id)
        predecessors[edge.dst.id].add(edge.src.id)

    _add_checkpoint_publication_barriers(
        by_id,
        predecessors,
        successors,
        checkpoint_barrier_steps=set(checkpoint_barrier_steps),
    )

    order = (
        _topological_order(set(by_id), predecessors, successors)
        if preferred_order is None
        else _validated_preferred_order(preferred_order, set(by_id), predecessors)
    )
    worker_by_vertex = {
        vertex_id: worker
        for worker in workers
        for vertex_id in getattr(worker, "owned_vertex_ids", (worker.vertex_id,))
    }
    specs: dict[str, RuntimeNodeSpec] = {}

    for node_id in order:
        node = by_id[node_id]
        worker = None
        vertex_id = None
        role = None
        model = None
        if node.kind == ExecNodeType.OP:
            if node.op is None:
                raise RuntimeConfigError(f"OP node {node.id} is missing unrolled op metadata")
            vertex_id = node.op.template_vertex.id
            role = node.op.template_vertex.kind.role.value
            worker = worker_by_vertex.get(vertex_id)
            if worker is None:
                raise RuntimeConfigError(f"OP node {node.id} has no worker for vertex {vertex_id}")
            model = worker.model
        elif node.model is not None:
            model = node.model.value

        worker_id = None if worker is None else worker.worker_id
        specs[node_id] = RuntimeNodeSpec(
            node_id=node_id,
            kind=node.kind,
            exec_node=node,
            dependencies=tuple(sorted(predecessors[node_id])),
            successors=tuple(sorted(successors[node_id])),
            worker_id=worker_id,
            vertex_id=vertex_id,
            model=model,
            role=role,
            parallelism=node.pi,
            microbatch=node.mb,
            allocations=node.allocs,
            resource_keys=_resource_keys(
                node,
                worker_id,
                predecessors=predecessors,
                successors=successors,
                by_id=by_id,
                worker_by_vertex=worker_by_vertex,
                transfer_resource_policy=transfer_resource_policy,
            ),
            payload_in_bytes=node.payload_in_bytes,
            label=node.label,
            logical_step=_logical_step(node, predecessors[node_id], successors[node_id]),
            source_vertex_id=(
                _adjacent_vertex_id(predecessors[node_id])
                if node.kind in {ExecNodeType.COMM, ExecNodeType.RESHARD}
                else None
            ),
            destination_vertex_id=(
                _adjacent_vertex_id(successors[node_id])
                if node.kind in {ExecNodeType.COMM, ExecNodeType.RESHARD}
                else None
            ),
        )

    composites = tuple({
        "operator_id": composite.id,
        "member_node_ids": list(composite.member_ids),
        "invoked_models": [model.value for model in composite.invoked_models],
        "allocation": {
            "origin": list(composite.allocation.origin),
            "extent": list(composite.allocation.extent),
        },
        "parallelism": list(composite.parallelism),
        "microbatch": composite.microbatch,
        "incoming_dependencies": list(composite.incoming),
        "outgoing_publications": list(composite.outgoing),
        "runtime_projection": "coordinated_bidirectional_microtasks",
    } for composite in lowered.composite_operators)
    return RuntimeGraph(nodes=specs, topological_order=order, composite_operators=composites)


def _add_checkpoint_publication_barriers(
    by_id: dict[str, ExecNode],
    predecessors: dict[str, set[str]],
    successors: dict[str, set[str]],
    *,
    checkpoint_barrier_steps: set[int],
) -> None:
    if not checkpoint_barrier_steps:
        return
    train_by_step: dict[int, set[str]] = {}
    for node_id, node in by_id.items():
        if node.kind != ExecNodeType.OP or node.op is None:
            continue
        if node.op.template_vertex.kind.role.value != "TRAIN":
            continue
        step = _unrolled_step(node_id)
        if step is not None:
            train_by_step.setdefault(step, set()).add(node_id)
    for node_id, node in by_id.items():
        if node.kind not in {ExecNodeType.COMM, ExecNodeType.RESHARD}:
            continue
        if not str(node.label or "").startswith("W_"):
            continue
        source_steps = {
            step
            for predecessor in predecessors[node_id]
            if (step := _unrolled_step(predecessor)) is not None
            and predecessor in train_by_step.get(step, set())
        }
        for step in source_steps & checkpoint_barrier_steps:
            for train_node_id in train_by_step.get(step, set()):
                predecessors[node_id].add(train_node_id)
                successors[train_node_id].add(node_id)


def _unrolled_step(node_id: str) -> int | None:
    marker = node_id.rfind("@")
    if marker < 0:
        return None
    suffix = node_id[marker + 1 :].split(".", 1)[0]
    try:
        return int(suffix)
    except ValueError:
        return None
