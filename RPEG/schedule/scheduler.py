# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Authors: Leonard Schmidt
#          Pierre Pang

import heapq

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from core.types import (
    Allocation,
    DeviceSpace,
    RPEG_TRANSFER_RESOURCE_POLICY,
    TransferResourcePolicy,
)
from model.cost import NodeCostModel
from passes.lower import ExecEdge, ExecNode, ExecNodeType, ExecutableRPEG
from passes.transfer_resources import reserved_allocations
from runtime.fused_pipeline import FusedPipelineModel, build_fused_pipeline_schedule
from schedule.trace import Trace


def _topo_sort(nodes: Tuple[ExecNode, ...], edges: Tuple[ExecEdge, ...]) -> Tuple[List[ExecNode], Dict[ExecNode, List[ExecNode]]]:
    """
    Sort nodes in topological order with the standard Kahn's algorithm.

    precondition: G=(nodes, edges) is acyclic (holds by construction)
    runtime: O(V + E)

    :param nodes: Nodes of the graph to sort.
    :type nodes: Tuple[ExecNode, ...]
    :param edges: Edges of the graph.
    :type edges: Tuple[ExecEdge, ...]
    :return: Tuple of nodes sorted in topological order and the adjacency lists for outgoing edges.
    :rtype: Tuple[List[ExecNode], Dict[ExecNode, List[ExecNode]]]
    """
    in_degree: Dict[ExecNode, int] = {}
    adj: Dict[ExecNode, List[ExecNode]] = defaultdict(list)

    for v in nodes: 
        in_degree[v] = 0 

    for e in edges: 
        u, v = e.src, e.dst 
        in_degree[v] += 1 
        adj[u].append(v)

    topo: List[ExecNode] = []
    queue = deque() 
    for v in nodes: 
        if in_degree[v] == 0: 
            queue.append(v)

    while queue: 
        u = queue.popleft() 
        for v in adj[u]: 
            in_degree[v] -= 1 
            if in_degree[v] == 0: 
                queue.append(v)
        topo.append(u)
    
    return topo, adj


# reverse topo pass: rank(v) = T(v) + max(rank(w) for successors w)
def _compute_ranks(
    nodes: Tuple[ExecNode, ...],
    edges: Tuple[ExecEdge, ...],
    cost_model: Optional[NodeCostModel] = None,
    durations: Optional[Dict[ExecNode, float]] = None
) -> Dict[ExecNode, float]:
    """
    Reverse topology pass to compute the time to finish the current node and all its successors.

    rank(v) = T(v) + max(rank(w) for successors w)

    :param nodes: Nodes in the graph.
    :type nodes: Tuple[ExecNode, ...]
    :param edges: Edges in the graph.
    :type edges: Tuple[ExecEdge, ...]
    :param cost_model: Cost model to estimate the time for the graph nodes. Defaults to None.
    :type cost_model: Optional[NodeCostModel]
    :param  durations: Pre-existing time estimations for the nodes. Defaults to None.
    :type durations: Optional[Dict[ExecNode, float]]
    :return: Dictionary of the nodes and their respective times.
    :rtype: Dict[ExecNode, float]
    """
    topo, adj = _topo_sort(nodes, edges)
    rank = {}

    for v in topo[::-1]:
        dur = durations[v] if durations is not None else cost_model.time(v)
        rank[v] = dur + max((rank[w] for w in adj[v]), default=0) # if adj[v] is empty

    return rank


# Enumerate all (r, n, g) tuples in one allocation. They are cached by (origin, extent): Allocation
# is frozen, so the device set for a given (origin, extent) pair never changes. Across thousands of
# schedule() calls the same allocations recur, turning 600K set constructions into ~50.
_alloc_cache: Dict[Tuple, frozenset] = {}

def _alloc_devices(alloc: Allocation) -> frozenset:
    """
    Retrieve the devices set for an allocation from the cache. If there is no existing entry, the
    device set is computed and returned as well as added to the cache.

    :param alloc: Input allocation.
    :type alloc: Allocation
    :return: Device set.
    :rtype: frozenset
    """
    key = (alloc.origin, alloc.extent)
    result = _alloc_cache.get(key)
    if result is not None:
        return result
    o, e = alloc.origin, alloc.extent
    devices = set()
    dr, dn, dg = o[0] + e[0], o[1] + e[1], o[2] + e[2]
    for r in range(o[0], dr):
        for n in range(o[1], dn):
            for g in range(o[2], dg):
                devices.add((r, n, g))
    result = frozenset(devices)
    _alloc_cache[key] = result
    return result


def _node_devices(node: ExecNode) -> Set[Tuple[int,int,int]]:
    """
    Retrieve the device sets for a node in the graph.

    All allocated devices (r, n, g) by a node (all types, i.e. possibly two allocations).

    :param node: Input node.
    :type node: ExecNode
    :return: Device set.
    :rtype: Set[Tuple[int,int,int]]
    """
    result = set()
    for alloc in node.allocs:
        result |= _alloc_devices(alloc) # in-place set union
    return result


def _blocking_devices(
    node: ExecNode,
    allocations: Optional[Tuple[Allocation, ...]] = None,
) -> Set[Tuple[int,int,int]]:
    """
    Retrieve blocking devices for the given node.

    The analytical model's policy is destination-only after a send initiation. Runtime job planning
    can instead pass an endpoint-conservative policy for transports such as NCCL whose source work
    and buffers remain live.

    :param node: Input node.
    :type node: ExecNode
    :param allocations: Relevant allocations. Defaults to None.
    :type allocations: Optional[Tuple[Allocation, ...]]
    :return: Blocked device set.
    :rtype: Set[Tuple[int,int,int]]
    """
    if allocations is None and node.kind in (ExecNodeType.COMM, ExecNodeType.RESHARD):
        allocations = node.allocs[-1:]
    if allocations is None:
        allocations = node.allocs
    result: Set[Tuple[int, int, int]] = set()
    for alloc in allocations:
        result.update(_alloc_devices(alloc))
    return result


def _compute_est(node: ExecNode, avail: Dict[Tuple[int,int,int], float], pred_finish: List[float]) -> float:
    """
    Computes the earliest feasible start time of a node depending on when its blocking devices are
    ready and when all its predecessors are finished.

    :param node: Input node.
    :type node: ExecNode
    :param avail: Times, when device sets will be available.
    :type avail: Dict[Tuple[int,int,int]
    :param pred_finish: Times, when predecessors will be finished.
    :type pred_finish: List[float]
    :return: Earliest feasible start time of the node.
    :rtype: float
    """
    blocked = _blocking_devices(node)
    device_max = max([avail[d] for d in blocked])
    pred_max = max(pred_finish)
    return max(pred_max, device_max)


def _fused_pipeline_duration(
    members: List[ExecNode],
    node_duration: Dict[ExecNode, float],
) -> float:
    """
    Cost a compatible actor/critic pair with the runtime's microtask scheduler.

    A standalone uniform pipeline with m microbatches and p stages has m + p - 1 forward/backward
    slots. Scaling 1:2 forward/backward task costs by that depth makes each isolated pipeline
    reproduce its coarse node cost. The shared list scheduler then exposes only bubble overlap; it
    does not erase either model's work as a plain maximum approximation would.

    :param members: List of nodes to be fused.
    :type members: List[ExecNode]
    :param node_duration: Pre-existing time estimations for the nodes.
    :type node_duration: Dict[ExecNode, float]
    :return: Time estimation for fused execution.
    :rtype: float
    """
    by_vertex = {
        member.op.template_vertex.id: member
        for member in members
        if member.op is not None
    }
    if set(by_vertex) != {"t_pi", "t_V"}:
        return sum(node_duration[member] for member in members)
    actor = by_vertex["t_pi"]
    critic = by_vertex["t_V"]
    if actor.pi is None or actor.mb is None or critic.pi != actor.pi or critic.mb != actor.mb:
        return sum(node_duration[member] for member in members)
    _dp, pipeline_stages, _tp = actor.pi
    microbatches = int(actor.mb)
    depth_slots = microbatches + int(pipeline_stages) - 1
    if pipeline_stages <= 1 or microbatches < pipeline_stages or depth_slots <= 0:
        return sum(node_duration[member] for member in members)

    def model(name: str, node: ExecNode, *, reverse: bool) -> FusedPipelineModel:
        """
        Create a model that can be used inside a fused pipeline.

        :param name: Role name.
        :type name: str
        :param node: Vertex in the executable RPEG.
        :type node: ExecNode
        :param reverse: Flag, indicating the pipeline direction.
        :type reverse: bool
        :return: Model for fused pipeline.
        :rtype: FusedPipelineModel
        """
        forward = float(node_duration[node]) / (3.0 * depth_slots)
        return FusedPipelineModel.uniform(
            name,
            pipeline_stages=int(pipeline_stages),
            microbatches=microbatches,
            forward_s=forward,
            backward_s=2.0 * forward,
            activation_bytes=0,
            reverse=reverse,
        )

    fused = build_fused_pipeline_schedule((
        model("actor", actor, reverse=False),
        model("critic", critic, reverse=True),
    ))
    return float(fused.makespan_s)


# COMM/RESHARD nodes are prioritised over OP nodes when breaking EST ties, preventing device stalls
# where an OP blocks a pending transfer.
_COMM_PRIO = {ExecNodeType.COMM: 0, ExecNodeType.RESHARD: 0, ExecNodeType.OP: 1}

def schedule(
    executable: ExecutableRPEG,
    cost_model: NodeCostModel,
    device_space: DeviceSpace,
    *,
    transfer_resource_policy: TransferResourcePolicy = RPEG_TRANSFER_RESOURCE_POLICY,
) -> Trace:
    """
    Schedule algorithm.

    Uses a min-heap with lazy deletion for O(n log n) overall complexity.

    Heap entries are (est, comm_prio, -rank, topo_idx, tie, node).
        - est:       earliest feasible start time
        - comm_prio: COMM/RESHARD before OP at equal EST (prevents device stalls)
        - -rank:     longest remaining path first (critical-path heuristic)
        - topo_idx:  template vertex index — deterministic tie-breaker that ensures the same
                     execution order of operators across unrolled steps (steady-state regularity for
                     runtime engines)
        - tie:       unique counter so heapq never compares ExecNode objects
        - node:      payload (never compared)

    Because device availability (avail) grows monotonically, a node's EST can only increase after it
    is pushed. When a popped entry has a stale (too-early) EST, we recompute and push it back; the
    stale entry is discarded. This "recompute-on-pop" loop converges because each push uses a
    strictly larger EST, so a node can be re-pushed at most O(n) times in total.

    :param executable: Graph to execute.
    :type executable: ExecutableRPEG
    :param cost_model: Cost model to estimate the time for the graph nodes.
    :type cost_model: NodeCostModel
    :param device_space: Available devices.
    :type device_space: DeviceSpace
    :param transfer_resource_policy: Policy for how resources are transfered. Defaults to
                                     RPEG_TRANSFER_RESOURCE_POLICY.
    :type transfer_resource_policy: TransferResourcePolicy
    :return: Trace of the start and finish times of graph nodes.
    :rtype: Trace
    :raises ValueError: If a cycle is detected.
    """
    start: Dict[ExecNode, float] = {}
    finish: Dict[ExecNode, float] = {}

    avail: Dict[Tuple[int, int, int], float] = {
        (r, n, g): 0.0
        for r in range(device_space.R)
        for n in range(device_space.N)
        for g in range(device_space.G)
    }

    in_degree: Dict[ExecNode, int] = {v: 0 for v in executable.nodes}
    adj: Dict[ExecNode, List[ExecNode]] = defaultdict(list)
    for e in executable.edges:
        in_degree[e.dst] += 1
        adj[e.src].append(e.dst)

    by_id = {node.id: node for node in executable.nodes}
    predecessor_ids: Dict[str, Set[str]] = {node.id: set() for node in executable.nodes}
    successor_ids: Dict[str, Set[str]] = {node.id: set() for node in executable.nodes}
    for edge in executable.edges:
        predecessor_ids[edge.dst.id].add(edge.src.id)
        successor_ids[edge.src.id].add(edge.dst.id)

    reserved_allocs: Dict[ExecNode, Tuple[Allocation, ...]] = {
        node: reserved_allocations(
            node,
            policy=transfer_resource_policy,
            predecessors=predecessor_ids,
            successors=successor_ids,
            by_id=by_id,
        )
        for node in executable.nodes
    }

    # Pre-compute blocking device sets and node durations once per node.
    # ExecNode is frozen, so these never change within a schedule() call.
    blocked_devs: Dict[ExecNode, Set[Tuple[int,int,int]]] = {
        v: _blocking_devices(v, reserved_allocs[v]) for v in executable.nodes
    }
    node_duration: Dict[ExecNode, float] = {
        v: cost_model.time(v) for v in executable.nodes
    }

    ranks: Dict[ExecNode, float] = _compute_ranks(
        nodes=executable.nodes, edges=executable.edges, durations=node_duration
    )

    # Sub-node chain rank inheritance

    # Sub-nodes of the same generation/inference vertex (gen:g0 → gen:g1 → gen:g2) form a sequential
    # chain on the same devices. Rank naturally decreases along the chain (gen:g1 < gen:g0), which
    # can cause the scheduler to insert unrelated ops (e.g. training) between generation phases —
    # breaking KV-cache residency assumptions.
    #
    # Fix: all sub-nodes in a chain inherit the rank of the chain head (first sub-node), which has
    # the highest rank. This ensures chain continuations are always preferred over competing ops on
    # the same device.
    import re

    _SUB_RE = re.compile(r"^(.+@\d+):[gi](\d+)$")
    _chain_groups: Dict[str, List[ExecNode]] = defaultdict(list)
    for v in executable.nodes:
        m = _SUB_RE.match(v.id)
        if m:
            _chain_groups[m.group(1)].append(v)
    for base_id, members in _chain_groups.items():
        max_rank = max(ranks[v] for v in members)
        for v in members:
            ranks[v] = max_rank

    # Deterministic topo-index tie-breaker
    # Heap key: (EST, comm_prio, topo_idx, -rank, tie, node).
    # topo_idx before -rank ensures consistent operator ordering across unrolled steps (steady-state
    # regularity for async schedules). -rank breaks ties within the same template position.
    # COMM/RESHARD nodes inherit the topo_idx of their destination OP's template.
    _tpl_set = set()
    for v in executable.nodes:
        if v.kind == ExecNodeType.OP and v.op is not None:
            _tpl_set.add(v.op.template_vertex.id)
    _tpl_order = {tid: i for i, tid in enumerate(sorted(_tpl_set))}

    topo_idx: Dict[ExecNode, int] = {}
    for v in executable.nodes:
        if v.kind == ExecNodeType.OP and v.op is not None:
            topo_idx[v] = _tpl_order.get(v.op.template_vertex.id, 0)
        else:
            # COMM/RESHARD: use a high index so OPs are preferred at equal rank
            topo_idx[v] = len(_tpl_order)

    pred_finish: Dict[ExecNode, List[float]] = defaultdict(list)

    def _est(node: ExecNode) -> float:
        """
        Computes the earliest feasible start time of a node depending on when its blocking devices
        are ready and when all its predecessors finished.

        :param node: Input node.
        :type node: ExecNode
        :return: Earliest feasible start time of the node.
        :rtype: float
        """
        blocked = blocked_devs[node]
        device_max = max(avail[d] for d in blocked)
        pred_max = max(pred_finish[node])
        return max(pred_max, device_max)

    # ready: Set of nodes whose in-degree has dropped to zero and that have not yet been scheduled.
    # Membership is O(1); used for lazy-deletion checks.
    ready: Set[ExecNode] = set()
    parked_fusion: Dict[str, ExecNode] = {}
    heap: List = []
    tie = 0

    for v in executable.nodes:
        if in_degree[v] == 0:
            pred_finish[v].append(0.0)
            est = _est(v)
            heapq.heappush(heap, (est, _COMM_PRIO[v.kind], topo_idx[v], -ranks[v], tie, v))
            tie += 1
            ready.add(v)

    while heap:
        est_h, cp, _ti, nr, _, v_star = heapq.heappop(heap)

        if v_star not in ready:
            continue  # stale heap entry from a previous re-push; discard

        # Recompute EST: A device may have become busier since this entry was pushed.
        v_est = _est(v_star)
        if v_est > est_h:
            # EST increased — push back with the corrected value and retry later.
            heapq.heappush(heap, (v_est, cp, _ti, nr, tie, v_star))
            tie += 1
            continue

        # Schedule a colocated actor/critic composite atomically. Its duration is the makespan of
        # the same dependency-complete bidirectional microtask scheduler used by the runtime.
        fused_members = [v_star]
        if v_star.fusion_group is not None:
            fused_members = sorted(
                (
                    node
                    for node in ready
                    if node.fusion_group == v_star.fusion_group
                ),
                key=lambda node: node.id,
            )
            if len(fused_members) != 2:
                expected_members = [
                    node
                    for node in executable.nodes
                    if node.fusion_group == v_star.fusion_group
                ]
                if len(expected_members) == 2 and all(member not in start for member in expected_members):
                    # The actor and critic may have different input dependency chains. Park the
                    # first one until its peer becomes ready, so the pair is never accidentally
                    # serialized.
                    ready.discard(v_star)
                    parked_fusion[v_star.fusion_group] = v_star
                    continue
                fused_members = [v_star]
        if len(fused_members) == 2:
            group_est = max(_est(member) for member in fused_members)
            if group_est > est_h:
                heapq.heappush(heap, (group_est, cp, _ti, nr, tie, v_star))
                tie += 1
                continue
            group_finish = group_est + _fused_pipeline_duration(fused_members, node_duration)
            for member in fused_members:
                ready.discard(member)
                start[member] = group_est
                finish[member] = group_finish
            for d in set().union(*(blocked_devs[member] for member in fused_members)):
                avail[d] = group_finish
            for member in fused_members:
                for w in adj[member]:
                    in_degree[w] -= 1
                    pred_finish[w].append(group_finish)
                    if in_degree[w] == 0:
                        w_est = _est(w)
                        heapq.heappush(
                            heap,
                            (w_est, _COMM_PRIO[w.kind], topo_idx[w], -ranks[w], tie, w),
                        )
                        tie += 1
                        ready.add(w)
                        parked = parked_fusion.pop(w.fusion_group, None) if w.fusion_group else None
                        if parked is not None:
                            ready.add(parked)
                            parked_est = _est(parked)
                            heapq.heappush(
                                heap,
                                (
                                    parked_est,
                                    _COMM_PRIO[parked.kind],
                                    topo_idx[parked],
                                    -ranks[parked],
                                    tie,
                                    parked,
                                ),
                            )
                            tie += 1
            continue

        # Schedule v_star.
        ready.discard(v_star)
        dur = node_duration[v_star]
        start[v_star]  = v_est
        finish[v_star] = v_est + dur

        for d in blocked_devs[v_star]:
            avail[d] = finish[v_star]

        for w in adj[v_star]:
            in_degree[w] -= 1
            pred_finish[w].append(finish[v_star])
            if in_degree[w] == 0:
                w_est = _est(w)
                heapq.heappush(heap, (w_est, _COMM_PRIO[w.kind], topo_idx[w], -ranks[w], tie, w))
                tie += 1
                ready.add(w)
                parked = parked_fusion.pop(w.fusion_group, None) if w.fusion_group else None
                if parked is not None:
                    ready.add(parked)
                    parked_est = _est(parked)
                    heapq.heappush(
                        heap,
                        (
                            parked_est,
                            _COMM_PRIO[parked.kind],
                            topo_idx[parked],
                            -ranks[parked],
                            tie,
                            parked,
                        ),
                    )
                    tie += 1

    if len(start) != len(executable.nodes):
        missing = [n.id for n in executable.nodes if n not in start]
        raise ValueError(f"Cycle detected in executable graph; unscheduled: {missing}")

    return Trace(start=start, finish=finish, reserved_allocations=reserved_allocs)
