# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from core.types import Allocation, Model, Role, TemplateVertex
from ir.plan import ExecutionPlan
from ir.rpeg import RPEG
from model.models import ModelSuite
from model.workload import Workload
from passes.unroll import UnrolledRPEG, UnrolledNode


class ExecNodeType(Enum):
    """
    Enumeration class for the node types in the executable RPEG.
    """
    OP = "op"
    COMM = "comm"
    RESHARD = "reshard"


@dataclass(frozen=True, eq=False)
class ExecNode():
    """
    Data class for vertices in the executable RPEG.
    """
    id: str
    kind: ExecNodeType
    allocs: Tuple[Allocation, ...] # multiple allocations for comm/resharding nodes only

    # OP-only fields
    op: Optional[UnrolledNode] = None # reference to the unrolled operator vertex
    model: Optional[Model] = None # model being invoked
    pi: Optional[Tuple[int, int, int]] = None # 3D parallel strategy
    mb: Optional[int] = None # number of microbatches

    # Sub-node fields (gen/inf quantile splitting)
    n_seqs: Optional[int] = None # sequences produced/consumed by this sub-node
    n_active: Optional[int] = None # active sequences during this generation phase (for decode cost)
    phase_steps: Optional[int] = None # number of decode steps in this generation phase
    phase_t_start: Optional[int] = None # decode step offset (for average past length calculation)
    is_prefill_phase: bool = False # True only for the first generation phase (includes prefill)
    fusion_group: Optional[str] = None # compatible coarse operators dispatched as one atomic group

    # COMM/RESHARD-only fields
    payload_in_bytes: int = 0
    label: Optional[str] = None  # human-readable content label (e.g. container name)

    # Performance: ExecNode.id is unique within an ExecutableRPEG. The default frozen-dataclass
    # __hash__ recursively hashes all 14 fields (including nested Allocations and enums) at ~2.5µs
    # per call. Using id-based hash costs ~0.1µs — 25× faster. This is safe because lower() never
    # creates duplicate-id nodes.
    def __hash__(self) -> int:
        """
        Hash only vertex id, which is already unique.

        :return: Id-based hash.
        :rtype: int
        """
        return hash(self.id)

    def __eq__(self, other) -> bool:
        """
        Implement equality based on the vertex id.

        :param other: Second instance for comparison.
        :return: Whether both instances refer to the same node or NotImplemented, if other is not an
                 ExecNode.
        :rtype: bool
        """
        if not isinstance(other, ExecNode):
            return NotImplemented
        return self.id == other.id


@dataclass(frozen=True)
class ExecEdge():
    """
    Data class for edges in the executable RPEG.
    """
    src: ExecNode
    dst: ExecNode


@dataclass(frozen=True)
class CompositeExecOperator:
    """
    Data class for semantic RPEG operators represented by executable member projections.

    Intra-stage fusion replaces actor and critic training at the semantic IR level. The two member
    IDs remain in ExecutableRPEG.nodes solely as the runtime projections that address distinct
    persistent model workers.
    """
    id: str
    member_ids: Tuple[str, ...]
    invoked_models: Tuple[Model, ...]
    allocation: Allocation
    parallelism: Tuple[int, int, int]
    microbatch: int
    incoming: Tuple[str, ...]
    outgoing: Tuple[str, ...]


@dataclass(frozen=True)
class ExecutableRPEG():
    """
    Data class for the executable RPEG.
    """
    nodes: Tuple[ExecNode, ...]
    edges: Tuple[ExecEdge, ...]
    composite_operators: Tuple[CompositeExecOperator, ...] = ()


def node_id(x: UnrolledNode) -> str:
    """
    Return vertex ID for an unrolled node based on the ID of template vertex in the RPEG. Includes
    sub-node index i when present (generation/inference phase splitting).

    :param x: Unrolled vertex.
    :type x: UnrolledNode
    :return: Vertex ID of the unrolled vertex.
    :rtype: str
    """
    return f"{x.template_vertex.id}@{x.k}" if x.i is None else f"{x.template_vertex.id}@{x.k}:{x.i}"


def topo_order_template(rpeg: RPEG) -> List[TemplateVertex]:
    """
    Returns a topology order for the vertices in the RPEG.

    :param rpeg: RLM Pipeline Execution Graph.
    :type rpeg: RPEG
    :return: List of vertices in topological order.
    :rtype: List[TemplateVertex]
    :raises ValueError: If data edges in the RPEG form a circle.
    """
    vs = list(rpeg.vertices)
    vset = set(vs)
    indeg = {v: 0 for v in vs}
    succ = {v: [] for v in vs}

    for e in rpeg.data_edges: # Template data edges
        if e.src in vset and e.dst in vset:
            succ[e.src].append(e.dst)
            indeg[e.dst] += 1

    q = [v for v in vs if indeg[v] == 0]
    out = []
    while q:
        v = q.pop()
        out.append(v)
        for w in succ[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)

    if len(out) != len(vs):
        raise ValueError("Template data edge graph has a cycle.")
    return out


def _data_reachable(rpeg: RPEG) -> Set[Tuple[TemplateVertex, TemplateVertex]]:
    """
    Transitive closure of template data edges.

    :param rpeg: RLM Pipeline Execution Graph.
    :type rpeg: RPEG
    :return: Set of (u, v) pairs where u can reach v via data edges.
    :rtype: Set[Tuple[TemplateVertex, TemplateVertex]]
    """
    succ: Dict[TemplateVertex, List[TemplateVertex]] = {v: [] for v in rpeg.vertices}
    for e in rpeg.data_edges:
        succ[e.src].append(e.dst)
    reach: Set[Tuple[TemplateVertex, TemplateVertex]] = set()
    for src in rpeg.vertices:
        visited: Set[TemplateVertex] = set()
        stack = list(succ[src])
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            reach.add((src, v))
            stack.extend(succ[v])
    return reach


def within_step_priors(
    template_order: List[TemplateVertex],
    trainable: Set[Model],
) -> List[Tuple[TemplateVertex, TemplateVertex, Model]]:
    """
    List of dependencies for the trainable models and their vertices in the RPEG.

    :param template_order: Ordered list of the vertices in the RPEG.
    :type template_order: List[TemplateVertex]
    :param trainable: Set of trainable models.
    :type trainable: Set[Model]
    :return: List of dependencies for the vertices of trainable models.
    :rtype: List[Tuple[TemplateVertex, TemplateVertex, Model]]
    """
    last: Dict[Model, TemplateVertex] = {}
    pairs = []
    for v in template_order:
        m = v.kind.model
        if m not in trainable:
            continue
        prev = last.get(m)
        if prev is not None:
            pairs.append((prev, v, m))
        last[m] = v
    return pairs


def _allocs_disjoint(a: Allocation, b: Allocation) -> bool:
    """
    Check whether two allocations share any devices.

    :param a: First allocation.
    :type a: Allocation
    :param b: Second allocation.
    :type b: Allocation
    :return: True if two rectangular allocations share no devices. False otherwise.
    :rtype: bool
    """
    for dim in range(3):
        a_lo, a_hi = a.origin[dim], a.origin[dim] + a.extent[dim]
        b_lo, b_hi = b.origin[dim], b.origin[dim] + b.extent[dim]
        if a_hi <= b_lo or b_hi <= a_lo:
            return True
    return False


def _alloc_overlap_size(a: Allocation, b: Allocation) -> int:
    """
    Number of devices in the intersection of two rectangular allocations.

    :param a: First allocation.
    :type a: Allocation
    :param b: Second allocation.
    :type b: Allocation
    :return: Number of devices that both allocations share.
    :rtype: int
    """
    overlap = 1
    for dim in range(3):
        a_lo, a_hi = a.origin[dim], a.origin[dim] + a.extent[dim]
        b_lo, b_hi = b.origin[dim], b.origin[dim] + b.extent[dim]
        lo = max(a_lo, b_lo)
        hi = min(a_hi, b_hi)
        if hi <= lo:
            return 0
        overlap *= (hi - lo)
    return overlap


def _is_shadow(
    model: Model,
    tau: Dict[Model, int],
    alloc_u: Allocation,
    alloc_v: Allocation
) -> bool:
    """
    True if model m forms a shadow pair: tau(m) > 0 and disjoint placements.

    :param model: Model to check.
    :type model: Model
    :param tau: Dictionary of models and their tau, i.e., whether they allow asynchronity.
    :type tau: Dict[Model, int]
    :param alloc_u: First allocation.
    :type alloc_u: Allocation
    :param alloc_v: Second allocation.
    :type alloc_v: Allocation
    :return: Flag, whether the two allocations form a shadow pair.
    :rtype: bool
    """
    return tau.get(model, 0) > 0 and _allocs_disjoint(alloc_u, alloc_v)


def _compute_gen_phases(workload: Workload) -> List[Tuple[int, int, int, int, bool]]:
    """
    Compute generation sub-node phases from quantile breakpoints.

    Quantile breakpoints are (cum_frac, token_count) where token_count is the number of generation
    tokens produced. The cost model convention is that prefill produces the first token; the
    remaining T-1 tokens require T-1 decode steps. So token_count t → decode_step t-1.

    Returns list of (n_active, n_seqs, phase_steps, t_start, is_prefill) per phase.
    - n_active:    sequences still generating at the start of this phase
    - n_seqs:      sequences that complete during this phase (output to inference)
    - phase_steps: number of decode steps in this phase
    - t_start:     decode step offset (for average past length calc: avg = S + t_start + steps/2)
    - is_prefill:  True for the first phase (includes prefill cost)

    :param workload: Workload settings.
    :type workload: Workload
    :return: List of parameters per phase.
    :rtype: List[Tuple[int, int, int, int, bool]]
    """
    total_seqs = workload.B * workload.K
    quantiles = workload.gen_quantiles
    phases = []
    prev_decode_step = 0
    prev_seqs_done = 0

    for q_idx, (cum_frac, t_tokens) in enumerate(quantiles):
        seqs_done_now = round(cum_frac * total_seqs)
        n_seqs = seqs_done_now - prev_seqs_done
        n_active = total_seqs - prev_seqs_done
        # Token count → decode steps: first token from prefill
        decode_step = t_tokens - 1
        steps = decode_step - prev_decode_step
        if steps < 0:
            steps = 0
        phases.append((n_active, n_seqs, steps, prev_decode_step, q_idx == 0))
        prev_decode_step = decode_step
        prev_seqs_done = seqs_done_now

    return phases


def lower(
    rpeg: RPEG,
    unrolled: UnrolledRPEG,
    plan: ExecutionPlan,
    suite: ModelSuite,
    K_eval: int,
    workload: Optional[Workload] = None,
    intra_stage_fusion: bool = False,
    verbose: bool = False,
) -> ExecutableRPEG:
    """
    Lower an unrolled RPEG into an executable RPEG.

    :param rpeg: RLM Pipeline Execution Graph.
    :type rpeg: RPEG
    :param unrolled: Unrolled RPEG.
    :type unrolled: UnrolledRPEG
    :param plan: Execution plan.
    :type plan: ExecutionPlan
    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param K_eval: Number of iterations/steps to unroll.
    :type K_eval: int
    :param workload: Workload settings. Defaults to None.
    :type workload: Optional[Workload]
    :param intra_stage_fusion: Flag to indicate whether intra-stage fusion is allowed. Defaults to
                               False.
    :type intra_stage_fusion: bool
    :param verbose: Verbose flag for additional output. Defaults to False.
    :type verbose: bool
    :raises KeyError: When a destination node is not found.
    """
    nodes: Dict[str, ExecNode] = {}
    edges: List[ExecEdge] = []
    edge_seen: Set[Tuple[str, str]] = set()

    def add_edge(src: ExecNode, dst: ExecNode) -> None:
        """
        Add an edge to executable RPEG, if it is not already present.

        :param src: Origin vertex of the edge.
        :type src: ExecNode
        :param dst: Target vertex of the edge.
        :type dst: ExecNode
        """
        key = (src.id, dst.id)
        if key in edge_seen:
            return
        edge_seen.add(key)
        edges.append(ExecEdge(src=src, dst=dst))

    def make_reshard(u: ExecNode, v: ExecNode, model: Model, ctr: int) -> ExecNode:
        """
        Create an reshard vertex to be placed between u and v.

        :param u: First node, which uses the model.
        :type u: ExecNode
        :param v: Second node, which uses the model.
        :type v: ExecNode
        :param model: Model used in both vertices (u, v) that needs to be resharded.
        :type model: Model
        :param ctr: Counter for the reshard nodes.
        :type ctr: int
        :return: Reshard vertex.
        :rtype: ExecNode
        """
        payload = suite.n_params(model) * suite.specs[model].dtype_bytes
        return ExecNode(
            id = f"reshard{ctr}",
            kind = ExecNodeType.RESHARD,
            allocs = (u.allocs[0], v.allocs[0]),
            model = model,
            payload_in_bytes = payload,
            label = f"W_{model.value}",
        )

    # Compute generation phases from quantiles
    gen_phases: Optional[List] = None
    if workload and workload.gen_quantiles:
        gen_phases = _compute_gen_phases(workload)

    # Tracking for split nodes:
    #   gen_first[orig_id] → first generation sub-node id (for incoming publication edges)
    #   gen_last[orig_id]  → last generation sub-node id (for outgoing reshard/ordering)
    #   gen_subs[orig_id]  → list of all generation sub-node ids
    #   inf_first[orig_id] → first inference sub-node id (for incoming publication edges)
    #   inf_subs[orig_id]  → list of all inference sub-node ids
    gen_first: Dict[str, str] = {}
    gen_last:  Dict[str, str] = {}
    gen_subs:  Dict[str, List[str]] = {}
    inf_first: Dict[str, str] = {}
    inf_subs:  Dict[str, List[str]] = {}

    # Pre-scan: determine which geneneration vertices have at least one disjoint inference
    _gen_has_disjoint_inf: set = set()  # template vertex IDs where splitting pays off
    if gen_phases:
        for e in unrolled.data_edges:
            src_tpl = e.src.template_vertex
            dst_tpl = e.dst.template_vertex
            if src_tpl.kind.role == Role.GENERATION and dst_tpl.kind.role == Role.INFERENCE:
                if _allocs_disjoint(plan.sigma[src_tpl], plan.sigma[dst_tpl]):
                    _gen_has_disjoint_inf.add(src_tpl.id)

    # 1) OP nodes
    for x in unrolled.nodes:
        tpl = x.template_vertex
        nid = node_id(x)

        if gen_phases and tpl.kind.role == Role.GENERATION and tpl.id in _gen_has_disjoint_inf:
            # DISTRIBUTED: at least one downstream inference is on disjoint devices.
            # Split generation into Q phase sub-nodes for overlap benefit.
            sub_ids = []
            for q, (n_act, n_seq, steps, t_start, is_pf) in enumerate(gen_phases):
                sub_id = f"{tpl.id}@{x.k}:g{q}"
                nodes[sub_id] = ExecNode(
                    id = sub_id,
                    kind = ExecNodeType.OP,
                    allocs = (plan.sigma[tpl],),
                    op = x,
                    model = tpl.kind.model,
                    pi = plan.pi[tpl],
                    mb = plan.mb.get(tpl, None),
                    n_seqs = n_seq,
                    n_active = n_act,
                    phase_steps = steps,
                    phase_t_start = t_start,
                    is_prefill_phase = is_pf,
                )
                sub_ids.append(sub_id)
            # Chain generation sub-nodes sequentially (same devices)
            for q in range(1, len(sub_ids)):
                add_edge(nodes[sub_ids[q-1]], nodes[sub_ids[q]])
            gen_first[nid] = sub_ids[0]
            gen_last[nid] = sub_ids[-1]
            gen_subs[nid] = sub_ids
        else:
            nodes[nid] = ExecNode(
                id = nid,
                kind = ExecNodeType.OP,
                allocs = (plan.sigma[tpl],),
                op = x,
                model = tpl.kind.model,
                pi = plan.pi[tpl],
                mb = plan.mb.get(tpl, None),
            )

    # 2a) COMM insertion on data edges
    comm_ctr = 0

    def _add_data_edge_or_comm(u: ExecNode, v: ExecNode, payload: int, label: str) -> None:
        """
        Add an edge if the allocation of both vertices is the same. Otherwise add an communication
        node connected to both vertices with one edge respectively.

        :param u: Origin vertex.
        :type u: ExecNode
        :param v: Target vertex.
        :type v: ExecNode
        :param payload: Size of the data to communicate.
        :type payload: int
        :param label: Human-readable label for the communication node.
        :type label: str
        """
        nonlocal comm_ctr
        if u.allocs[0] == v.allocs[0]:
            add_edge(u, v)
        else:
            comm_ctr += 1
            c = ExecNode(
                id = f"comm{comm_ctr}",
                kind = ExecNodeType.COMM,
                allocs = (u.allocs[0], v.allocs[0]),
                payload_in_bytes = payload,
                label = label,
            )
            nodes[c.id] = c
            add_edge(u, c)
            add_edge(c, v)

    # Two-pass edge processing: generation→inference edges first (creates inference sub-nodes), then
    # all remaining edges (which may reference the newly created sub-nodes).
    _gen_inf_edges = []
    _other_edges = []
    for e in unrolled.data_edges:
        src_nid = node_id(e.src)
        if src_nid in gen_subs and e.dst.template_vertex.kind.role == Role.INFERENCE:
            _gen_inf_edges.append(e)
        else:
            _other_edges.append(e)

    for e in _gen_inf_edges + _other_edges:
        src_nid = node_id(e.src)
        dst_nid = node_id(e.dst)
        full_payload = e.container.bytes()
        label = e.container.name
        src_is_gen_split = src_nid in gen_subs
        src_is_inf_split = src_nid in inf_subs

        if src_is_gen_split and e.dst.template_vertex.kind.role == Role.INFERENCE:
            gen_alloc = plan.sigma[e.src.template_vertex]
            inf_alloc = plan.sigma[e.dst.template_vertex]
            devices_disjoint = _allocs_disjoint(gen_alloc, inf_alloc)

            if devices_disjoint:
                # DISTRIBUTED: generation and inference on different devices → split inference into
                # Q sub-nodes and wire gen_q → inf_q for overlap benefit.
                total_seqs = workload.B * workload.K
                if dst_nid not in inf_subs:
                    # Remove the original unsplit inf node (created in section 1)
                    nodes.pop(dst_nid, None)
                    isub_ids = []
                    for q, (_, n_seq, _, _, _) in enumerate(gen_phases):
                        isub_id = f"{e.dst.template_vertex.id}@{e.dst.k}:i{q}"
                        nodes[isub_id] = ExecNode(
                            id = isub_id,
                            kind = ExecNodeType.OP,
                            allocs = (inf_alloc,),
                            op = e.dst,
                            model = e.dst.template_vertex.kind.model,
                            pi = plan.pi[e.dst.template_vertex],
                            mb = plan.mb.get(e.dst.template_vertex, None),
                            n_seqs = n_seq,
                        )
                        isub_ids.append(isub_id)
                    # Chain inference sub-nodes (explicit ordering for publication edge correctness)
                    for q in range(1, len(isub_ids)):
                        add_edge(nodes[isub_ids[q-1]], nodes[isub_ids[q]])
                    inf_first[dst_nid] = isub_ids[0]
                    inf_subs[dst_nid] = isub_ids

                # Connect gen_q → inf_q with scaled payload
                for q in range(len(gen_phases)):
                    frac = gen_phases[q][1] / total_seqs
                    sub_payload = max(1, int(full_payload * frac))
                    _add_data_edge_or_comm(
                        nodes[gen_subs[src_nid][q]],
                        nodes[inf_subs[dst_nid][q]],
                        sub_payload, label,
                    )
            else:
                # Colocated: generation and inference share devices → no overlap possible. Wire last
                # generation sub-node → inference (original serial structure). Inference stays as a
                # single unsplit node (already created in section 1).
                v = nodes[dst_nid]
                _add_data_edge_or_comm(nodes[gen_last[src_nid]], v, full_payload, label)

        elif src_is_gen_split:
            # generation → non-inference (shouldn't happen in PPO, but handle gracefully): connect
            # last generation sub-node → dst
            v = nodes[dst_nid]
            _add_data_edge_or_comm(nodes[gen_last[src_nid]], v, full_payload, label)

        elif src_is_inf_split:
            # split inference → downstream (e.g., inference → training): connect last inference
            # sub-node only. Earlier sub-nodes are chained (i0→i1→i2), so the last one finishing
            # implies all are done. Connecting all would create redundant edges (and spurious COMM
            # nodes when allocations differ).
            dst_node = nodes.get(dst_nid)
            if dst_node is None:
                raise KeyError(f"Destination node {dst_nid} not found")
            last_inf_id = inf_subs[src_nid][-1]
            _add_data_edge_or_comm(nodes[last_inf_id], dst_node, full_payload, label)

        else:
            # Normal (unsplit) edge
            u = nodes[src_nid]
            v = nodes[dst_nid]
            _add_data_edge_or_comm(u, v, full_payload, label)

    # 2b) Colocated actor/critic fused-stage annotation
    # Compatible actor and critic PP operators form one semantic composite. The runtime lowers this
    # annotation to a coordinated bidirectional microtask stream; the analytical scheduler costs
    # that same fused stream.
    train_by_step: Dict[int, Dict[str, str]] = {}
    for x in unrolled.nodes:
        tpl = x.template_vertex
        if tpl.id in {"t_pi", "t_V"}:
            train_by_step.setdefault(x.k, {})[tpl.id] = node_id(x)
    for step, pair in train_by_step.items():
        if set(pair) != {"t_pi", "t_V"}:
            continue
        actor = nodes[pair["t_pi"]]
        critic = nodes[pair["t_V"]]
        compatible = (
            actor.allocs == critic.allocs
            and actor.pi == critic.pi
            and actor.mb == critic.mb
            and actor.pi is not None
            and int(actor.pi[1]) > 1
            and int(actor.mb or 1) >= int(actor.pi[1])
        )
        if compatible and intra_stage_fusion:
            group = f"actor_critic_train@{step}"
            nodes[actor.id] = dataclasses.replace(actor, fusion_group=group)
            nodes[critic.id] = dataclasses.replace(critic, fusion_group=group)

    # 3) Intra-step RESHARD (heuristic consecutive-in-topo)
    # Shadow pairs (tau > 0 and disjoint devices) are skipped: the two instances maintain
    # independent weight copies, no reshard needed. Data-reachable pairs on disjoint devices are
    # skipped: ordering is already guaranteed by data edges, and COMM nodes handle activation
    # transfer. Same-allocation pairs are kept (may need local reshard if strategies differ).
    topo = topo_order_template(rpeg)
    pairs = within_step_priors(topo, rpeg.trainable)
    data_reach = _data_reachable(rpeg)

    reshard_ctr = 0
    intra_step_resharted: set = set()

    for u_tpl, v_tpl, model in pairs:
        alloc_u = plan.sigma[u_tpl]
        alloc_v = plan.sigma[v_tpl]

        if _is_shadow(model, rpeg.tau, alloc_u, alloc_v):
            continue

        # If data edges already connect u → v and they're on different devices, the ordering is
        # guaranteed and no weight reshard is needed (each device holds its own weight copy via Φ or
        # publication edge).
        if (u_tpl, v_tpl) in data_reach and alloc_u != alloc_v:
            continue

        for k in range(K_eval):
            # Resolve split nodes: use last sub-node (generation/inference complete)
            u_nid = f"{u_tpl.id}@{k}"
            v_nid = f"{v_tpl.id}@{k}"
            # For u (predecessor): use last generation or last inference sub-node
            if u_nid in gen_last:
                u = nodes[gen_last[u_nid]]
            elif u_nid in inf_subs:
                u = nodes[inf_subs[u_nid][-1]]
            else:
                u = nodes[u_nid]
            # For v (successor): use first generation or first inference sub-node
            if v_nid in gen_first:
                v = nodes[gen_first[v_nid]]
            elif v_nid in inf_first:
                v = nodes[inf_first[v_nid]]
            else:
                v = nodes[v_nid]

            if u.allocs[0] == v.allocs[0] and u.pi == v.pi:
                continue

            if u.allocs[0] != v.allocs[0]:
                add_edge(u, v)
            else:
                reshard_ctr += 1
                r = make_reshard(u, v, model, reshard_ctr)
                nodes[r.id] = r
                add_edge(u, r)
                add_edge(r, v)
                intra_step_resharted.add(v_tpl)

    # 4a) Inter-step publication edges
    for e in unrolled.pub_edges:
        # Source is always training (never split)
        u = nodes[node_id(e.src)]

        # Destination may be a split generation or inference node → resolve to first sub-node
        dst_nid = node_id(e.dst)
        if dst_nid in gen_first:
            v = nodes[gen_first[dst_nid]]
        elif dst_nid in inf_first:
            v = nodes[inf_first[dst_nid]]
        else:
            v = nodes[dst_nid]
        model = u.model

        if v.op.template_vertex in intra_step_resharted:
            continue

        if u.allocs[0] == v.allocs[0] and u.pi == v.pi:
            add_edge(u, v)
            continue

        if u.allocs[0] == v.allocs[0]:
            reshard_ctr += 1
            r = make_reshard(u, v, model, reshard_ctr)
            nodes[r.id] = r
            add_edge(u, r)
            add_edge(r, v)
        else:
            comm_ctr += 1
            full_payload = suite.n_params(model) * suite.specs[model].dtype_bytes
            n_src = u.allocs[0].size()
            n_dst = v.allocs[0].size()
            n_overlap = _alloc_overlap_size(u.allocs[0], v.allocs[0])
            n_max = max(n_src, n_dst)
            payload = full_payload * (n_max - n_overlap) // n_max if n_max > 0 else full_payload
            c = ExecNode(
                id = f"comm{comm_ctr}",
                kind = ExecNodeType.COMM,
                allocs = (u.allocs[0], v.allocs[0]),
                payload_in_bytes = payload,
                label = f"W_{model.value}",
            )
            nodes[c.id] = c
            add_edge(u, c)
            add_edge(c, v)

    # 4a) Colocated trainable-layout lifetime barriers

    # A same-allocation training→serving reshard replaces the training module in its persistent
    # worker. If two trainable models share those GPUs, eagerly switching both workers to their
    # largest serving layouts retains both BF16 shards at once. The runtime memory model
    # deliberately charges one active trainable module shard at a time, and the scheduler cannot
    # overlap two same-GPU reshards anyway. Match that contract by returning an earlier data
    # producer to its training layout before publishing the next model's serving layout. For PPO
    # this gives the exact live range:
    #
    #   actor TRAIN→GEN, generation, actor GEN→TRAIN,
    #   critic TRAIN→INF, value inference, critic INF→TRAIN.
    #
    # Use semantic roles and data reachability rather than vertex names, so the invariant remains
    # valid for other multi-model RPEGs. Disjoint allocations retain their intended
    # publication/compute overlap.
    transition_predecessors: Dict[str, List[ExecNode]] = {}
    transition_successors: Dict[str, List[ExecNode]] = {}
    for edge in edges:
        transition_predecessors.setdefault(edge.dst.id, []).append(nodes[edge.src.id])
        transition_successors.setdefault(edge.src.id, []).append(nodes[edge.dst.id])

    returns_to_training: List[Tuple[ExecNode, ExecNode, ExecNode]] = []
    publications_to_serving: List[Tuple[ExecNode, ExecNode, ExecNode]] = []
    for transition in nodes.values():
        if transition.kind != ExecNodeType.RESHARD:
            continue
        predecessors = [
            node for node in transition_predecessors.get(transition.id, ())
            if node.kind == ExecNodeType.OP and node.op is not None
        ]
        successors = [
            node for node in transition_successors.get(transition.id, ())
            if node.kind == ExecNodeType.OP and node.op is not None
        ]
        if len(predecessors) != 1 or len(successors) != 1:
            continue
        predecessor, successor = predecessors[0], successors[0]
        if predecessor.model != successor.model or predecessor.model not in rpeg.trainable:
            continue
        predecessor_role = predecessor.op.template_vertex.kind.role
        successor_role = successor.op.template_vertex.kind.role
        if predecessor_role != Role.TRAINING and successor_role == Role.TRAINING:
            returns_to_training.append((transition, predecessor, successor))
        elif predecessor_role == Role.TRAINING and successor_role != Role.TRAINING:
            publications_to_serving.append((transition, predecessor, successor))

    for return_transition, serving_producer, _training_consumer in returns_to_training:
        for publication_transition, _training_producer, serving_consumer in publications_to_serving:
            if serving_producer.model == serving_consumer.model:
                continue
            if serving_producer.op.k != serving_consumer.op.k:
                continue
            if _allocs_disjoint(
                serving_producer.allocs[0],
                serving_consumer.allocs[0],
            ):
                continue
            if (
                serving_producer.op.template_vertex,
                serving_consumer.op.template_vertex,
            ) not in data_reach:
                continue
            add_edge(return_transition, publication_transition)

    # Some annotations (currently intra-stage fusion) replace immutable ExecNode instances after
    # data edges were created. Canonicalize every edge endpoint, so downstream schedulers observe
    # those annotations too.
    canonical_edges = tuple(
        ExecEdge(src=nodes[edge.src.id], dst=nodes[edge.dst.id])
        for edge in edges
    )
    composite_operators = []
    groups = sorted({node.fusion_group for node in nodes.values() if node.fusion_group is not None})
    for group in groups:
        members = sorted(
            (node for node in nodes.values() if node.fusion_group == group),
            key=lambda node: node.op.template_vertex.id if node.op is not None else node.id,
        )
        if len(members) != 2 or any(member.pi is None or member.mb is None for member in members):
            continue
        member_ids = frozenset(member.id for member in members)
        incoming = sorted({
            edge.src.id
            for edge in canonical_edges
            if edge.dst.id in member_ids and edge.src.id not in member_ids
        })
        outgoing = sorted({
            edge.dst.id
            for edge in canonical_edges
            if edge.src.id in member_ids and edge.dst.id not in member_ids
        })
        step = str(group).rsplit("@", 1)[-1]
        composite_operators.append(CompositeExecOperator(
            id=f"t_pi||V@{step}",
            member_ids=tuple(member.id for member in members),
            invoked_models=tuple(member.model for member in members if member.model is not None),
            allocation=members[0].allocs[0],
            parallelism=members[0].pi,
            microbatch=int(members[0].mb),
            incoming=tuple(incoming),
            outgoing=tuple(outgoing),
        ))
    lowered = ExecutableRPEG(
        nodes = tuple(nodes.values()),
        edges = canonical_edges,
        composite_operators = tuple(composite_operators),
    )

    if verbose:
        print("RPEG lowered successfully.")
        print(f"    Executable nodes: {len(lowered.nodes)}")
        print(f"    - OP nodes: {sum(n.kind == ExecNodeType.OP for n in lowered.nodes)}")
        print(f"    - COMM nodes: {sum(n.kind == ExecNodeType.COMM for n in lowered.nodes)}")
        print(f"    - RESHARD nodes: {sum(n.kind == ExecNodeType.RESHARD for n in lowered.nodes)}")
        if gen_phases:
            n_gen_subs = sum(len(s) for s in gen_subs.values())
            n_inf_subs = sum(len(s) for s in inf_subs.values())
            print(f"    - Gen sub-nodes: {n_gen_subs} ({len(gen_phases)} phases × {len(gen_subs)} gen vertices)")
            print(f"    - Inf sub-nodes: {n_inf_subs}")

    return lowered
