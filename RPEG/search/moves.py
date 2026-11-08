# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt
#
# Contributions: Pierre Pang

import random

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from core.types import (
    packing_geometry,
    rectangular_node_layouts,
    Allocation,
    DeviceSpace,
    TemplateVertex
)
from ir.plan import ExecutionPlan
from ir.rpeg import RPEG
from model.cost import NodeCostModel
from model.memory import NodeMemoryModel
from model.models import ModelSuite
from model.workload import Workload
from passes.lower import ExecNode, ExecNodeType, ExecutableRPEG
from passes.unroll import UnrolledNode
from schedule.trace import Trace
from search.evaluate import Configuration


@dataclass
class MoveStats:
    """
    Data class for diagnostic counters accumulated across all move_gen calls in one optimize() run.
    """
    n_calls:              int = 0  # total move_gen invocations
    n_placement_proposed: int = 0  # candidates that only change placement origin (pi+mb unchanged)
    n_strategy_proposed:  int = 0  # candidates that change (dp,pp,tp) or mb for some vertex
    n_fallback:           int = 0  # times _find_current_position used the middle-of-list fallback
    n_empty:              int = 0  # times move_gen returned no candidates at all
    # Multi-vertex kernel counters
    n_pair_proposed:      int = 0  # donor-receiver proposals generated
    n_pair_accepted:      int = 0
    n_pair_failed:        int = 0  # donor-receiver proposals that failed to sample
    n_pair_rejected:      int = 0  # pair proposals rejected by Metropolis-Hastings (MH)

    def summary(self, n_accepted_strategy: int, n_accepted_placement: int, n_rejected_sv: int) -> str:
        """
        Provide a summary as a string of the collected statistics.

        :param n_accepted_strategy: Number of times a strategy was accepted.
        :type n_accepted_strategy: int
        :param n_accepted_placement: Number of times a placement was accepted.
        :type n_accepted_placement: int
        :param n_rejected_sv: Number of times a single-vertex change was rejected.
        :type n_rejected_sv: int
        :return: Summary of the statistics.
        :rtype: str
        """
        # Single-vertex statistics
        sv_proposed = self.n_placement_proposed + self.n_strategy_proposed
        sv_accepted = n_accepted_strategy + n_accepted_placement
        sv_rejected = n_rejected_sv
        sv_failed = self.n_empty  # move_gen returned no candidates

        # Multi-vertex statistics
        mv_proposed = self.n_pair_proposed
        mv_accepted = self.n_pair_accepted
        mv_rejected = self.n_pair_rejected
        mv_failed = self.n_pair_failed

        all_proposed = sv_proposed + mv_proposed
        all_accepted = sv_accepted + mv_accepted
        all_rejected = sv_rejected + mv_rejected
        all_mh = all_accepted + all_rejected
        acc_rate = 100 * all_accepted / max(1, all_mh)

        # Consistent column layout: proposed / accepted / rejected / failed
        lines = [
            f"  Proposals              proposed  accepted  rejected  failed",
            f"    single-vertex      {sv_proposed:>8d}  {sv_accepted:>8d}  {sv_rejected:>8d}  {sv_failed:>6d}",
            f"      strategy         {self.n_strategy_proposed:>8d}  {n_accepted_strategy:>8d}",
            f"      placement        {self.n_placement_proposed:>8d}  {n_accepted_placement:>8d}",
        ]
        if mv_proposed or mv_failed:
            lines.append(f"    multi-vertex       {mv_proposed:>8d}  {mv_accepted:>8d}  {mv_rejected:>8d}  {mv_failed:>6d}")
            lines.append(f"      pair             {self.n_pair_proposed:>8d}  {self.n_pair_accepted:>8d}  {self.n_pair_rejected:>8d}  {self.n_pair_failed:>6d}")
        lines.append(f"  MH acceptance        {all_accepted:>8d} / {all_mh}  ({acc_rate:.1f}%)")
        return "\n".join(lines)


# CANDIDATE SPACE AUXILIARY FUNCTIONS
def valid_tp_degrees(G: int) -> List[int]:
    """
    Return all valid tensor parallelism degrees, i.e. all tp > 0 that divide G.

    :param G: Number of GPUs per compute node.
    :type G: int
    :return: List of valid tensor parallelism degrees.
    :rtype: List[int]
    """
    tp_degrees = [tp for tp in range(1, G + 1) if G % tp == 0]
    return tp_degrees

def valid_pp_degrees(R: int, N: int, G: int, L: int) -> List[int]:
    """
    Return all valid pipeline parallelism degrees.

    Even numbers (and 1) that divide L, up to R*N*G.
    Odd pp > 1 excluded: non-standard in practice (Megatron/DeepSpeed use even pipeline depths) and
    can cause asymmetric stage packing.

    :param R: Number of racks.
    :type R: int
    :param N: Number of compute nodes per rack.
    :type N: int
    :param G: Number of GPUs per compute node.
    :type G: int
    :param L: Number of model layers.
    :type L: int
    :return: List of valid pipeline parallelism degrees.
    :rtype: List[int]
    """
    pp_degrees = [d for d in range(1, R*N*G+1) if L % d == 0 and (d == 1 or d % 2 == 0)]
    return pp_degrees


def valid_dp_degrees(R: int, N: int, B: int, G: int = 1) -> List[int]:
    """
    Return all valid data parallelism degrees.

    Upper bound: total GPUs (R*N*G). Actual feasibility is enforced downstream by
    packing_geometry(dp, pp, tp, G) at each call site.

    :param R: Number of racks.
    :type R: int
    :param B: Batch size.
    :type B: int
    :param N: Number of compute nodes per rack.
    :type N: int
    :param G: Number of GPUs per compute node.
    :type G: int
    :return: List of valid data parallelism degrees.
    :rtype: List[int]
    """
    max_dp = R * N * G
    dp_degrees = [d for d in range(1, max_dp + 1) if B % d == 0]
    return dp_degrees


# ABLATION A3: set to True to use ReaL's fixed microbatching heuristic (mb=pp for all)
ABLATION_FIXED_MB = False
# ABLATION A6: set to True to use uniform vertex selection (no critical-path bias)
ABLATION_UNIFORM_VERTEX = False

MIN_B_LOCAL = 8 # minimum per-GPU microbatch count (validated operating regime)

def valid_mb_counts(pp: int, dp: int, B: int) -> List[int]:
    """
    Return valid microbatch counts.

    mb > 1 is only meaningful with pipeline parallelism; force mb=1 when pp=1

    :param pp: Pipeline parallism setting.
    :type pp: int
    :param dp: Data parallelism setting.
    :type dp: int
    :param B: Batch size.
    :type B: int
    :return: List of valid microbatch counts.
    :rtype: List[int]
    """
    local_batch_size = B // dp
    if local_batch_size < MIN_B_LOCAL and dp > 1:
        return []  # b_local too small for this dp — skip this config
    if pp == 1:
        return [1]
    if ABLATION_FIXED_MB:
        # ReaL heuristic: mb = pp (generation/inference) or mb = 2*pp (training)
        # Since we don't know the role here, use conservative mb = pp.
        mb = pp
        if local_batch_size % mb != 0:
            return [1]
        return [mb]
    # Enforce b_local = local_batch_size / mb >= MIN_B_LOCAL
    max_mb = local_batch_size // MIN_B_LOCAL if MIN_B_LOCAL > 0 else local_batch_size
    mb_counts = [mb for mb in range(pp, min(max_mb, local_batch_size) + 1)
                 if local_batch_size % mb == 0]
    if not mb_counts:
        # Cannot satisfy both pipeline fill (mb >= pp) and b_local >= MIN_B_LOCAL. This
        # configuration is infeasible for pipeline parallelism — return empty. The caller
        # (_enumerate_strategies, sample_expl) will skip this (dp, pp) combination.
        return []
    return mb_counts


# SAMPLE EXPLORATION
def sample_expl_in_region(
    v: TemplateVertex,
    suite: ModelSuite,
    device_space: DeviceSpace,
    workload: Workload,
    r_lo: int = 0,
    r_hi: Optional[int] = None,
) -> Optional[Tuple[Allocation, Tuple[int, int, int], int]]:
    """
    Sample a random valid (allocation, strategy, mb) for vertex v within the contiguous rack
    interval [r_lo, r_hi).

    :param v: Vertex in the template graph.
    :type v: TemplateVertex
    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param device_space: Available devices and their hierarchy.
    :type device_space: DeviceSpace
    :param workload: Workload settings.
    :type workload: Workload
    :param r_lo: Lower bound of the rack interval. Defaults to 0.
    :type r_lo: int
    :param r_high: Upper bound of the rack interval. Defaults to None, in which case the number of
                   racks is used.
    :type r_high: Optional[int]
    :return: Tuple of the allocation, strategy tuple (dp, pp, tp) and microbatch count or None if no
             valid allocation can be computed.
    :rtype: Optional[Tuple[Allocation, Tuple[int, int, int], int]]
    """
    R, N, G, B = device_space.R, device_space.N, device_space.G, workload.B
    L = suite.specs[v.kind.model].L
    if r_hi is None:
        r_hi = R
    R_eff = r_hi - r_lo  # number of racks in this region

    # Sample tp
    tp_candidates = valid_tp_degrees(G)
    if not tp_candidates:
        return None
    tp = random.choice(tp_candidates)

    # Sample pp (valid for full mesh; filtered below by region fit)
    pp_candidates = valid_pp_degrees(R, N, G, L)
    if not pp_candidates:
        return None
    pp = random.choice(pp_candidates)

    # Packing
    s = min(pp, G // tp)
    if pp % s != 0:
        return None

    # Sample dp — must fit within the region (R_eff racks × N nodes)
    dp_candidates = []
    for d in valid_dp_degrees(R, N, B, G):
        geom = packing_geometry(d, pp, tp, G)
        if geom is None:
            continue
        _, nodes_needed = geom
        if nodes_needed <= R_eff * N:
            dp_candidates.append(d)
    if not dp_candidates:
        return None
    dp = random.choice(dp_candidates)

    geom = packing_geometry(dp, pp, tp, G)
    g_ext, nodes_needed = geom

    # Exact rectangular embeddings within the selected rack region.
    layouts = [
        (r_ext, n_ext, g_ext)
        for r_ext, n_ext in rectangular_node_layouts(nodes_needed, R_eff, N)
    ]
    if not layouts:
        return None

    r_ext, n_ext, g_ext = random.choice(layouts)

    # Sample aligned origin within [r_lo, r_hi)
    r_origins = [r_lo + i * r_ext for i in range(R_eff // r_ext)]
    if not r_origins:
        return None
    r_o = random.choice(r_origins)
    n_o = random.choice([i * n_ext for i in range(N // n_ext)])
    g_o = random.choice([i * g_ext for i in range(G // g_ext)])
    alloc = Allocation(origin=(r_o, n_o, g_o), extent=(r_ext, n_ext, g_ext))

    # Sample microbatch counts
    mb_candidates = valid_mb_counts(pp, dp, B)
    if not mb_candidates:
        return None
    mb = random.choice(mb_candidates)

    return alloc, (dp, pp, tp), mb


# Convenience wrappers preserving the old call-sites
def sample_expl(
    v: TemplateVertex,
    suite: ModelSuite,
    device_space: DeviceSpace,
    workload: Workload
) -> Optional[Tuple[Allocation, Tuple[int, int, int], int]]:
    """
    Convenience wrapper preserving the old call-sites.

    Equivalent to calling sample_expl_in_region with the defaults (r_lo=0, r_hi=None).

    :param v: Vertex in the template graph.
    :type v: TemplateVertex
    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param device_space: Available devices and their hierarchy.
    :type device_space: DeviceSpace
    :param workload: Workload settings.
    :type workload: Workload
    :return: Tuple of the allocation, strategy tuple (dp, pp, tp) and microbatch count or None if no
             valid allocation can be computed.
    :rtype: Optional[Tuple[Allocation, Tuple[int, int, int], int]]
    """
    return sample_expl_in_region(v, suite, device_space, workload)


def sample_expl_on_rack(
    v: TemplateVertex,
    suite: ModelSuite,
    device_space: DeviceSpace,
    workload: Workload,
    rack: int,
    node_range: Optional[Tuple[int, int]] = None,
) -> Optional[Tuple[Allocation, Tuple[int, int, int], int]]:
    """
    Convenience wrapper preserving the old call-sites.

    Sample within one rack. Kept for backward compatbility with _sample_placement_split which uses
    the node_range parameter (not expressible via sample_expl_in_region alone).

    :param v: Vertex in the template graph.
    :type v: TemplateVertex
    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param device_space: Available devices and their hierarchy.
    :type device_space: DeviceSpace
    :param workload: Workload settings.
    :type workload: Workload
    :param rack: Rack number.
    :type rack: int
    :param node_range: Range of compute nodes inside the rack. Defaults to None, in which case the
                       whole rack is considered.
    :type node_range: Optional[Tuple[int, int]] = None
    :return: Tuple of the allocation, strategy tuple (dp, pp, tp) and microbatch count or None if no
             valid allocation can be computed.
    :rtype: Optional[Tuple[Allocation, Tuple[int, int, int], int]]
    """
    if node_range is None:
        return sample_expl_in_region(v, suite, device_space, workload, r_lo=rack, r_hi=rack + 1)

    # node_range path: restrict dp to fit within a node slice inside one rack.
    R, N, G, B = device_space.R, device_space.N, device_space.G, workload.B
    L = suite.specs[v.kind.model].L
    n_start, N_eff = node_range

    tp_candidates = valid_tp_degrees(G)
    if not tp_candidates:
        return None
    tp = random.choice(tp_candidates)
    pp_candidates = valid_pp_degrees(R, N, G, L)
    if not pp_candidates:
        return None
    pp = random.choice(pp_candidates)
    s = min(pp, G // tp)
    if pp % s != 0:
        return None
    dp_candidates = [d for d in valid_dp_degrees(R, N, B, G)
                     if packing_geometry(d, pp, tp, G) is not None
                     and packing_geometry(d, pp, tp, G)[1] <= N_eff]
    if not dp_candidates:
        return None
    dp = random.choice(dp_candidates)
    g_ext, nodes_needed = packing_geometry(dp, pp, tp, G)
    n_ext = nodes_needed
    n_o = n_start + random.choice([i * n_ext for i in range(N_eff // n_ext)])
    g_o = random.choice([i * g_ext for i in range(G // g_ext)])
    alloc = Allocation(origin=(rack, n_o, g_o), extent=(1, n_ext, g_ext))
    mb_candidates = valid_mb_counts(pp, dp, B)
    if not mb_candidates:
        return None
    mb = random.choice(mb_candidates)
    return alloc, (dp, pp, tp), mb


def _enumerate_strategies(
    v: TemplateVertex,
    suite: ModelSuite,
    device_space: DeviceSpace,
    workload: Workload
) -> List[Tuple[int, int, int, int, int, int]]:
    """
    Return all valid (dp, pp, tp, r_ext, n_ext, mb) tuples for vertex v.

    Enumerates the full support of the proposal distribution, so that every valid configuration has
    a non-zero probability.

    :param v: Vertex in the template graph.
    :type v: TemplateVertex
    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param device_space: Available devices and their hierarchy.
    :type device_space: DeviceSpace
    :param workload: Workload settings.
    :type workload: Workload
    :return: List of all valid strategies (dp, pp, tp, r_ext, n_ext, mb) for the vertex.
    :rtype: List[Tuple[int, int, int, int, int, int]]
    """
    R, N, G, B = device_space.R, device_space.N, device_space.G, workload.B
    L = suite.specs[v.kind.model].L
    strategies = []
    for tp in valid_tp_degrees(G):
        for pp in valid_pp_degrees(R, N, G, L):
            s = min(pp, G // tp)
            if pp % s != 0:
                continue
            for dp in valid_dp_degrees(R, N, B, G):
                mb_counts = valid_mb_counts(pp, dp, B)
                if not mb_counts:
                    continue
                geom = packing_geometry(dp, pp, tp, G)
                if geom is None:
                    continue
                g_ext, nodes_needed = geom
                layouts = rectangular_node_layouts(nodes_needed, R, N)
                for r_ext, n_ext in layouts:
                    for mb in mb_counts:
                        strategies.append((dp, pp, tp, r_ext, n_ext, mb))
    return strategies


def _make_mock_node(
    v: TemplateVertex,
    dp: int,
    pp: int,
    tp: int,
    r_ext: int,
    n_ext: int,
    mb: int,
    g_per_node: int
) -> ExecNode:
    """
    Create temporary vertex to check for memory feasibility and to compute a cost for the given
    configuration.

    g_per_node = s*tp where s = min(pp, floor(G/tp)); passed by caller to avoid needing G here. The
    allocation extent uses g_per_node (not just tp), so that the cost/memory models see the correct
    per-node GPU footprint.

    :param v: Vertex in the template graph.
    :type v: TemplateVertex
    :param dp: Data parallelism setting.
    :type dp: int
    :param pp: Pipeline parallism setting.
    :type pp: int
    :param tp: Tensor parallism setting.
    :type tp: int
    :param r_ext: Rack extent, i.e., how many racks are used.
    :type r_ext: int
    :param n_ext: Node extent, i.e., how many compute nodes per rack are used.
    :type n_ext: int
    :param mb: Microbatch count.
    :type mb: int
    :param g_per_node: GPUs per compute node.
    :type g_per_node: int
    :return: Temporary vertex.
    :rtype: ExecNode
    """
    mock_op = UnrolledNode(template_vertex=v, k=0)
    return ExecNode(
        id="_approx",
        kind=ExecNodeType.OP,
        allocs=(Allocation(origin=(0, 0, 0), extent=(r_ext, n_ext, g_per_node)),),
        op=mock_op,
        model=v.kind.model,
        pi=(dp, pp, tp),
        mb=mb,
    )


# ReaL-like sampling
def build_candidate_directory(
    rpeg: RPEG,
    suite: ModelSuite,
    device_space: DeviceSpace,
    workload: Workload,
    cost_model: NodeCostModel,
    mem_model: NodeMemoryModel,
    max_per_vertex: int = 1000
) -> Tuple[
    Dict[TemplateVertex, List[Tuple[int, int, int, int, int, int, float]]],
    Dict[TemplateVertex, Dict[Tuple[int, int, int, int, int, int], int]],
]:
    """
    Build the candidate directory used by move_gen.

    Returns a pair (table, index):
      table[v]  — list of (dp, pp, tp, r_ext, n_ext, mb, time), sorted ascending by time.
      index[v]  — dict mapping (dp, pp, tp, r_ext, n_ext, mb) -> position in table[v], used by
                  _find_current_position for O(1) lookup.

    :param rpeg: RLM Pipeline Execution Graph.
    :type rpeg: RPEG
    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param device_space: Available devices and their hierarchy.
    :type device_space: DeviceSpace
    :param workload: Workload settings.
    :type workload: Workload
    :param cost_model: Cost model to estimate the time for the graph nodes.
    :type cost_model: NodeCostModel
    :param mem_model: Memory model for the graph nodes.
    :type mem_model: NodeMemoryModel
    :param max_per_vertex: Maximum number of candidates per vertex. Defaults to 1000.
    :type max_per_vertex: int
    :return: Pair of per-vertex table and index.
    :rtype: Tuple[
                Dict[TemplateVertex, List[Tuple[int, int, int, int, int, int, float]]],
                Dict[TemplateVertex, Dict[Tuple[int, int, int, int, int, int], int]],
            ]
    """
    G = device_space.G
    table: Dict[TemplateVertex, List] = {}
    index: Dict[TemplateVertex, Dict] = {}
    for v in rpeg.vertices:
        candidates = []
        for strat in _enumerate_strategies(v, suite, device_space, workload):
            dp, pp, tp, r_ext, n_ext, mb = strat
            geom = packing_geometry(dp, pp, tp, G)
            if geom is None:
                continue
            g_ext, _ = geom
            mock_node  = _make_mock_node(v, dp, pp, tp, r_ext, n_ext, mb, g_ext)
            if mem_model.peak_bytes(mock_node) > mem_model._eff_capacity():
                continue
            t = cost_model.time(mock_node)
            candidates.append(strat + (t,))

        candidates.sort(key=lambda c: c[6])  # sort ascending by execution time
        candidates = candidates[:max_per_vertex]
        table[v] = candidates
        index[v] = {c[:6]: j for j, c in enumerate(candidates)}

    return table, index


def _find_current_position(
    v: TemplateVertex,
    plan: ExecutionPlan,
    index_v: Dict[Tuple[int, int, int, int, int, int], int],
    n_candidates: int,
    stats: Optional['MoveStats'] = None
) -> int:
    """
    O(1) lookup of the current plan's position in the sorted candidate list.

    :param v: Vertex in the template graph.
    :type v: TemplateVertex
    :param plan: Execution plan.
    :type plan: ExecutionPlan
    :param index_v: Index for the table lookup in the sorted candidate list.
    :type index_v: Dict[Tuple[int, int, int, int, int, int], int]
    :param n_candidates: Size of the sorted candidate list for the fallback mechanism.
    :type n_candidates: int
    :param stats: Diagnostic counters. Defaults to None.
    :type stats: Optional['MoveStats']
    :return: Position of the vertex in the current plan in the sorted candidate list or the middle
             of the list as a fallback.
    :rtype: int
    """
    dp, pp, tp = plan.pi[v]
    r_ext, n_ext, _ = plan.sigma[v].extent   # g_ext == s*tp; derived from (pp,tp,G) — not in key
    mb = plan.mb[v]
    key = (dp, pp, tp, r_ext, n_ext, mb)
    if key in index_v:
        return index_v[key]
    if stats is not None:
        stats.n_fallback += 1
    return n_candidates // 2  # fallback: treat as middle of the sorted list


# DIAGNOSTICS
def _critical_chain(executable: ExecutableRPEG, trace: Trace) -> Set[ExecNode]:
    """
    Approximates the critical chain by backtracking from the last node along the latest-predecessor
    edges. Considers data-dependencies only (device contention is handled by the scheduler).

    :param executable: Graph to execute.
    :type executable: ExecutableRPEG
    :param trace: Execution trace of the plan.
    :type trace: Trace
    :return: Set of nodes on the critical chain.
    :rtype: Set[ExecNode]
    """
    critic = []
    pred: Dict[ExecNode, List[ExecNode]] = {} # list of predecessors of each node (needed for backtracking)

    for v in executable.nodes: 
        pred[v] = []

    for e in executable.edges:
        u, v = e.src, e.dst 
        pred[v].append(u) 

    v_star = max(executable.nodes, key=lambda v: trace.finish[v])
    critic.append(v_star)

    while pred[v_star]: 
        v_star = max(pred[v_star], key=lambda v: trace.finish[v]) 
        critic.append(v_star)
    
    return set(critic) 


def _critical_mass(V_crit: Set[ExecNode], rpeg: RPEG, trace: Trace) -> Dict[TemplateVertex, float]:
    """
    For each node v on the critical chain, define its critical chain-mass as the sum of the T(u)
    where u originates from template vertex v, specifically sum T(u) for u in V_crit that belong to
    each template vertex.

    :param V_crit: Nodes on the critical chain.
    :type V_crit: Set[ExecNode]
    :param rpeg: RLM Pipeline Execution Graph.
    :type rpeg: RPEG
    :param trace: Execution trace of the plan.
    :type trace: Trace
    :return: Mass for each vertex on the critical chain.
    :rtype: Dict[TemplateVertex, float]
    """
    mass: Dict[TemplateVertex, float] = {v: 0.0 for v in rpeg.vertices}
    for v in V_crit: 
        if v.kind != ExecNodeType.OP: continue 
        mass[v.op.template_vertex] += trace.finish[v] - trace.start[v]

    return mass


def move_gen(
    config: Configuration,
    executable: ExecutableRPEG,
    trace: Trace,
    table: Dict[TemplateVertex, List],
    index: Dict[TemplateVertex, Dict],
    K_x: int,
    T_max: int,
    T: float,
    T_init: float,
    stats: Optional['MoveStats'] = None
) -> List[Configuration]:
    """
    Generates moves (strategy/placement changes) for potentially improving the current
    configuration.

    Candidates generated by move_gen are always valid by construction.

    :param config: Configuration (RPEG, execution plan, available hardware).
    :type config: Configuration
    :param executable: Graph to execute.
    :type executable: ExecutableRPEG
    :param trace: Execution trace of the plan.
    :type trace: Trace
    :param table: Sorted candidate list for each vertex in the RPEG.
    :type table: Dict[TemplateVertex, List]
    :param index: Index for the sorted candidate list for O(1) lookup.
    :type index: Dict[TemplateVertex, Dict]
    :param K_x: Target number of candidates to return.
    :type K_x: int
    :param T_max: Total sampling attempts limit (across all K_x slots).
    :type T_max: int
    :param T: Current Simulated Annealing (SA) temperature.
    :type T: float
    :param T_init: Initial SA temperature (cold restart), used for window normalisation.
    :type T_init: float
    :param stats: Diagnostic counters. Defaults to None.
    :type stats: Optional['MoveStats']
    :return: List of potential moves for changing the current configuration.
    :rtype: List[Configuration]
    """
    # Mixture vertex selection: 60% critical-chain biased, 40% uniform.
    # ABLATION A6: 100% uniform if ABLATION_UNIFORM_VERTEX is set.
    if not ABLATION_UNIFORM_VERTEX and random.random() >= 0.4:
        critic = _critical_chain(executable, trace)
        mass = _critical_mass(critic, config.rpeg, trace)
        weights = [mass[v] + 1e-6 for v in config.rpeg.vertices]
        v = random.choices(config.rpeg.vertices, weights=weights, k=1)[0]
    else:
        v = random.choice(config.rpeg.vertices)

    v_candidates = table[v]

    # Temperature-adaptive window: wide at high T (exploration), narrow at low T (exploitation).
    # frac = T/T_init in (0, 1]; at T=T_init the whole list is reachable, at T=T_min ≈ 10% of it.
    frac = min(1.0, T / T_init)
    w = max(5, int(len(v_candidates) * frac))

    # O(1) lookup of current position in the sorted candidate list
    j_star = _find_current_position(v, config.plan, index[v], len(v_candidates), stats)
    low, high = max(0, j_star - w), min(len(v_candidates) - 1, j_star + w)

    R, N, G = config.device_space.R, config.device_space.N, config.device_space.G

    def sample_origin(extent: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Sample an origin position within the available devices.

        :param extent: Available devices.
        :type extent: Tuple[int, int, int]
        :return: Origin position.
        :rtype: Tuple[int, int, int]
        """
        r_ext, n_ext, g_ext = extent
        r_o = random.choice([i * r_ext for i in range(R // r_ext)])
        n_o = random.choice([i * n_ext for i in range(N // n_ext)])
        g_o = random.choice([i * g_ext for i in range(G // g_ext)])
        return (r_o, n_o, g_o)

    candidates = []
    seen_moves = set()  # (pi_tuple, origin_tuple, mb) for vertex v — dedup within this call
    curr_alloc = config.plan.sigma[v]
    curr_pi = config.plan.pi[v]
    curr_mb = config.plan.mb[v]
    fused_peer = None
    if config.intra_stage_fusion and v.id in {"t_pi", "t_V"}:
        peer_id = "t_V" if v.id == "t_pi" else "t_pi"
        fused_peer = next((candidate for candidate in config.rpeg.vertices if candidate.id == peer_id), None)

    def updated_plan(
        allocation: Allocation,
        parallelism: Tuple[int, int, int],
        microbatch: int
    ) -> ExecutionPlan:
        """
        Create an updated executable plan with a composite operator.

        :param allocation: Allocation to use for the composite operator.
        :type allocation: Allocation
        :param parallelism: Parallelism strategy.
        :type parallelism: Tuple[int, int, int]
        :param microbatch: Microbatch count.
        :type microbatch: int
        :return: Executable plan.
        :rtype: ExecutionPlan
        """
        targets = (v,) if fused_peer is None else (v, fused_peer)
        sigma = {**config.plan.sigma, **{target: allocation for target in targets}}
        pi = {**config.plan.pi, **{target: parallelism for target in targets}}
        mb = {**config.plan.mb, **{target: microbatch for target in targets}}
        return ExecutionPlan(sigma=sigma, pi=pi, mb=mb)

    # Try placement-only (best effort, up to 5 attempts)
    for _ in range(min(5, T_max)):
        o = sample_origin(curr_alloc.extent)
        if o != curr_alloc.origin:
            key = (curr_pi, o, curr_mb)
            if key not in seen_moves:
                seen_moves.add(key)
                candidates.append(
                    Configuration(
                        config.rpeg,
                        updated_plan(
                            Allocation(o, curr_alloc.extent),
                            curr_pi,
                            curr_mb
                        ),
                        config.device_space,
                        intra_stage_fusion = config.intra_stage_fusion,
                        transfer_resource_policy = config.transfer_resource_policy
                    )
                )
            break  # got one placement candidate; done

    # Fill remaining K_x slots with strategy-change candidates
    for _ in range(T_max):
        if len(candidates) >= K_x:
            break

        j = random.randint(low, high)
        if j == j_star:
            continue
        dp, pp, tp, r_ext, n_ext, mb, _ = v_candidates[j]
        geom = packing_geometry(dp, pp, tp, G)
        if geom is None:
            continue
        if fused_peer is not None and not any(
            tuple(peer_candidate[:3]) == (dp, pp, tp)
            and int(peer_candidate[5]) == int(mb)
            for peer_candidate in table[fused_peer]
        ):
            continue
        g_ext, _ = geom
        r_o, n_o, g_o = sample_origin(extent=(r_ext, n_ext, g_ext))
        key = ((dp, pp, tp), (r_o, n_o, g_o), mb)
        if key in seen_moves:
            continue
        seen_moves.add(key)
        new_alloc = Allocation(origin=(r_o, n_o, g_o), extent=(r_ext, n_ext, g_ext))
        candidates.append(
            Configuration(
                config.rpeg,
                updated_plan(new_alloc, (dp, pp, tp), mb),
                config.device_space,
                intra_stage_fusion = config.intra_stage_fusion,
                transfer_resource_policy = config.transfer_resource_policy
            )
        )

    if stats is not None:
        stats.n_calls += 1
        for cand in candidates:
            # Placement-only: pi and mb are unchanged for all vertices
            if cand.plan.pi == config.plan.pi and cand.plan.mb == config.plan.mb:
                stats.n_placement_proposed += 1
            else:
                stats.n_strategy_proposed += 1
        if not candidates:
            stats.n_empty += 1

    return candidates


# Multi-vertex proposal kernels
def _alloc_devices_set(alloc: Allocation) -> Set[Tuple[int, int, int]]:
    """
    Enumerate all (r, n, g) in an allocation box.

    :param alloc: Input allocation.
    :type alloc: Allocation
    :return: Set of positions within the hardware allocation.
    "rtype: Set[Tuple[int, int, int]]
    """
    o, e = alloc.origin, alloc.extent
    return {(r, n, g)
            for r in range(o[0], o[0] + e[0])
            for n in range(o[1], o[1] + e[1])
            for g in range(o[2], o[2] + e[2])}


def multi_move_pair(
    config: Configuration, suite: ModelSuite, workload: Workload, max_retries: int = 100
) -> Optional[Configuration]:
    """
    Donor-receiver proposal: shrink one vertex's allocation and expand another into the freed
    devices.

    :param config: Configuration (RPEG, execution plan, available hardware).
    :type config: Configuration
    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param workload: Workload settings.
    :type workload: Workload
    :param max_retries: Number of retries for donor and receiver selection. Defaults to 100.
    :type max_retries: int
    :return: A single new configuration based on the allocation changes or None if sampling fails.
    :rtype: Optional[Configuration]
    """
    vertices = list(config.rpeg.vertices)
    R = config.device_space.R

    # 1. Select donor weighted by device count
    dev_counts = [config.plan.sigma[v].extent[0] * config.plan.sigma[v].extent[1] * config.plan.sigma[v].extent[2]
                  for v in vertices]
    # Only vertices spanning >1 rack or >1 node can meaningfully donate
    donor_weights = [float(c) for c in dev_counts]
    if sum(donor_weights) == 0:
        return None
    v_d = random.choices(vertices, weights=donor_weights, k=1)[0]

    # 2. Select receiver: must overlap with donor's current devices
    donor_devs = _alloc_devices_set(config.plan.sigma[v_d])
    overlap_candidates = [v for v in vertices
                          if v is not v_d and _alloc_devices_set(config.plan.sigma[v]) & donor_devs]
    if not overlap_candidates:
        return None
    v_r = random.choice(overlap_candidates)
    if config.intra_stage_fusion and ({v_d.id, v_r.id} & {"t_pi", "t_V"}):
        # Fused training is one plan variable. The single-vertex kernel above already moves both
        # members atomically; a donor/receiver move would assign contradictory layouts to the same
        # composite operator.
        return None

    # 3. Shrink donor
    d_alloc = config.plan.sigma[v_d]
    d_r_lo, d_r_ext = d_alloc.origin[0], d_alloc.extent[0]

    if d_r_ext > 1:
        # Shrink at rack level: pick a smaller contiguous sub-interval
        new_r_ext = random.randint(1, d_r_ext - 1)
        # Pick a random aligned origin within the donor's current rack span
        possible_origins = list(range(d_r_lo, d_r_lo + d_r_ext - new_r_ext + 1))
        new_r_lo = random.choice(possible_origins)
    else:
        # Already on one rack — use sample_expl_in_region on same rack (it will naturally pick a
        # potentially smaller node extent)
        new_r_lo, new_r_ext = d_r_lo, 1

    donor_sample = None
    for _ in range(max_retries):
        s = sample_expl_in_region(v_d, suite, config.device_space, workload,
                                  r_lo=new_r_lo, r_hi=new_r_lo + new_r_ext)
        if s is not None:
            donor_sample = s
            break
    if donor_sample is None:
        return None

    # 4. Expand receiver — sample in a region that overlaps the freed racks
    freed_racks = set(range(d_r_lo, d_r_lo + d_r_ext)) - set(range(new_r_lo, new_r_lo + new_r_ext))
    if freed_racks:
        # Include at least one freed rack plus optionally the receiver's current racks
        r_alloc = config.plan.sigma[v_r]
        recv_racks = set(range(r_alloc.origin[0], r_alloc.origin[0] + r_alloc.extent[0]))
        combined = recv_racks | freed_racks
        recv_r_lo = min(combined)
        recv_r_hi = max(combined) + 1
    else:
        # No racks freed (single-rack donor, node-level shrink) — just re-sample receiver freely
        recv_r_lo, recv_r_hi = 0, R

    recv_sample = None
    for _ in range(max_retries):
        s = sample_expl_in_region(v_r, suite, config.device_space, workload,
                                  r_lo=recv_r_lo, r_hi=recv_r_hi)
        if s is not None:
            recv_sample = s
            break
    if recv_sample is None:
        return None

    # 5. Build new plan with both changes
    d_alloc_new, d_pi, d_mb = donor_sample
    r_alloc_new, r_pi, r_mb = recv_sample
    new_plan = ExecutionPlan(
        sigma={**config.plan.sigma, v_d: d_alloc_new, v_r: r_alloc_new},
        pi={**config.plan.pi, v_d: d_pi, v_r: r_pi},
        mb={**config.plan.mb, v_d: d_mb, v_r: r_mb},
    )
    return Configuration(
        config.rpeg,
        new_plan,
        config.device_space,
        intra_stage_fusion = config.intra_stage_fusion,
        transfer_resource_policy = config.transfer_resource_policy
    )
