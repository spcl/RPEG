# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Authors: Leonard Schmidt
#          Pierre Pang

import math

from dataclasses import dataclass
from typing import Optional, Tuple

from core.types import (
    packing_geometry,
    rectangular_node_layouts,
    Allocation,
    DeviceSpace,
    RPEG_TRANSFER_RESOURCE_POLICY,
    TransferResourcePolicy
)
from ir.plan import ExecutionPlan
from ir.rpeg import RPEG
from passes.lower import lower
from passes.unroll import unroll
from model.cost import NodeCostModel
from model.hardware import Hardware
from model.memory import mem_feasible, mem_worst_violation, MemoryPolicy, NodeMemoryModel
from model.models import ModelSuite
from model.workload import Workload
from schedule.scheduler import schedule
from schedule.trace import Trace
from search.cache import config_hash, EvalCache


@dataclass
class Configuration:
    """
    Data class for a configuration consisting of an RPEG, an execution plan and the respective
    device space.
    """
    rpeg: RPEG
    plan: ExecutionPlan
    device_space: DeviceSpace
    intra_stage_fusion: bool = False
    transfer_resource_policy: Optional[TransferResourcePolicy] = None


def is_compatible(alloc: Allocation, pi: Tuple[int, int, int], device_space: DeviceSpace) -> bool:
    """
    Checks placement feasibility under nested 3D parallelism with stage packing and supports
    intra-node DP: when the pipeline fits within one node, multiple DP replicas can share a single
    node (g_ext = dp * pp * tp < G).

    :param allocation: Allocation to test.
    :type alloc: Allocation
    :param pi: 3D parallelism (dp, pp, tp) configuration.
    :type pi: Tuple[int, int, int]
    :param device_space: Available devices and their hierarchy.
    :type device_space: DeviceSpace
    :return: Decision whether the placement is feasible.
    :rtype: bool
    """
    dp, pp, tp = pi
    r_ext, n_ext, g_ext = alloc.extent
    r_o,   n_o,   g_o   = alloc.origin
    R, N, G = device_space.R, device_space.N, device_space.G

    # TP group must fit within one node
    if tp > G:
        return False

    # Packing geometry must be valid (no fractional nodes)
    geom = packing_geometry(dp, pp, tp, G)
    if geom is None:
        return False
    expected_g, nodes_needed = geom

    # GPU extent must match packing geometry
    if g_ext != expected_g:
        return False

    # Allocation must be an exact rectangle in the logical rack/node mesh
    if (r_ext, n_ext) not in rectangular_node_layouts(nodes_needed, R, N):
        return False

    # Allocation must lie within device bounds
    if r_o < 0 or n_o < 0 or g_o < 0:
        return False
    if r_o + r_ext > R or n_o + n_ext > N or g_o + g_ext > G:
        return False

    return True


def allocations_overlap(left: Allocation, right: Allocation) -> bool:
    """
    Return whether two rectangular logical-GPU allocations intersect.

    :param left: Allocation to the left in logical rack/node mesh.
    :type left: Allocation
    :param right: Allocation to the right in logical rack/node mesh.
    :type right: Allocation
    :return: True, if both allocations intersect. False if there is no intersection.
    :rtype: bool
    """
    return all(
        left.origin[axis] < right.origin[axis] + right.extent[axis]
        and right.origin[axis] < left.origin[axis] + left.extent[axis]
        for axis in range(3)
    )


def stale_publication_layout_valid(config: Configuration) -> bool:
    """
    Require a physical serving shadow for every positive-lag publication.

    A tau>0 consumer must retain an older immutable model version, while its publisher trains and
    publishes newer versions. Coalescing both endpoints onto intersecting GPUs turns that shadow
    into an in-place layout switch: the switch either destroys the stale version or doubles
    unmodeled HBM, and also serializes the two roles. Such a placement is therefore not an
    executable RPEG asynchronous plan.

    :param config: Configuration to check.
    :type config: Configuration
    :return: True, if the configuration is valid in regards to the publication layout. False
             otherwise.
    :rtype: bool
    """
    for edge in config.rpeg.pub_edges:
        model = edge.src.kind.model
        if int(config.rpeg.tau.get(model, 0)) <= 0:
            continue
        if allocations_overlap(
            config.plan.sigma[edge.src],
            config.plan.sigma[edge.dst],
        ):
            return False
    return True


def is_valid(
    config: Configuration,
    suite: ModelSuite,
    workload: Workload,
    mem_model: NodeMemoryModel
) -> bool:
    """
    Full validity gate, which checks for:
    - checks plan coverage,
    - divisibility invariants,
    - microbatching (mb) constraints,
    - packing/layout compatibility,
    - memory feasibility.

    :param config: Configuration to check.
    :type config: Configuration
    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param workload: Workload settings.
    :type workload: Workload
    :param mem_model: Memory model for the graph nodes.
    :type mem_model: NodeMemoryModel
    :return: Decision, whether the given configuration is valid within the constraints.
    :rtype: bool
    """
    B = workload.B
    device_space = config.device_space
    R, N, G = device_space.R, device_space.N, device_space.G

    for v in config.rpeg.vertices:
        # Plan coverage: every vertex must have a full assignment
        if v not in config.plan.sigma or v not in config.plan.pi or v not in config.plan.mb:
            return False

        alloc      = config.plan.sigma[v]
        dp, pp, tp = config.plan.pi[v]
        mb         = config.plan.mb[v]
        L          = suite.specs[v.kind.model].L

        # Divisibility: 
        if B % dp != 0 or G % tp != 0 or L % pp != 0:
            return False

        # Microbatching constraints (P6):
        if pp == 1:
            if mb != 1:
                return False
        else:
            if (B // dp) % mb != 0:
                return False

        # Packing and layout compatibility
        if not is_compatible(alloc, (dp, pp, tp), device_space):
            return False

    # Fused operators
    if not stale_publication_layout_valid(config):
        return False

    if config.intra_stage_fusion:
        by_id = {vertex.id: vertex for vertex in config.rpeg.vertices}
        actor = by_id.get("t_pi")
        critic = by_id.get("t_V")
        if actor is None or critic is None:
            return False
        if (
            config.plan.sigma[actor] != config.plan.sigma[critic]
            or config.plan.pi[actor] != config.plan.pi[critic]
            or config.plan.mb[actor] != config.plan.mb[critic]
            or int(config.plan.pi[actor][1]) <= 1
            or int(config.plan.mb[actor]) < int(config.plan.pi[actor][1])
        ):
            return False

    # Memory feasibility via the full lower pass
    try:
        unrolled = unroll(config.rpeg, K_eval=1)
        lowered = lower(
            config.rpeg,
            unrolled,
            config.plan,
            suite,
            K_eval = 1,
            workload = workload,
            intra_stage_fusion = config.intra_stage_fusion
        )
    except (KeyError, ValueError):
        return False

    return mem_feasible(lowered, mem_model)


def evaluate(
    config: Configuration,
    suite: ModelSuite,
    workload: Workload,
    hw: Hardware,
    K_eval: int,
    memory_policy: MemoryPolicy = MemoryPolicy()
) -> Tuple[float, Trace]:
    """
    Run the full deterministic pipeline: unroll → lower → cost → schedule → score.

    :param config: Initial configuration.
    :type config: Configuration
    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param workload: Workload settings.
    :type workload: Workload
    :param hw: Hardware settings.
    :type hw: Hardware
    :param K_eval: Number of iterations/steps to unroll.
    :type K_eval: int
    :param memory_policy: Policy to incorporate practical overheads. Defaults to MemoryPolicy().
    :type memory_policy: MemoryPolicy
    :return: Tuple of the score and the runtime trace.
    :rtype: Tuple[float, Trace]
    """
    B, K, S, T = workload.B, workload.K, workload.S, workload.T

    unrolled = unroll(config.rpeg, K_eval)
    lowered = lower(
        config.rpeg,
        unrolled,
        config.plan,
        suite,
        K_eval,
        workload = workload,
        intra_stage_fusion = config.intra_stage_fusion
    )
    cost_model = NodeCostModel(
        hw,
        suite,
        workload,
        optimizer_stage = memory_policy.optimizer_stage,
        activation_checkpointing = memory_policy.activation_checkpointing
    )
    trace = schedule(
        lowered,
        cost_model,
        config.device_space,
        transfer_resource_policy=(
            config.transfer_resource_policy or RPEG_TRANSFER_RESOURCE_POLICY
        )
    )
    score = trace.throughput(W_step=B*K*(S+T), K=K_eval)

    return (score, trace, lowered)


# OOM configs are no longer hard-rejected. Instead the raw throughput is always computed and then
# multiplied by an exponential memory penalty:
#   penalty_mult = exp(-beta * overflow_ratio), where overflow_ratio = peak/cap - 1
_OOM_BETA = 40.0
# ABLATION A8: set to True to use hard OOM rejection (score = 0) instead of soft penalty
ABLATION_HARD_OOM = False


def evaluation_cache_key(
    config: Configuration,
    memory_policy: MemoryPolicy = MemoryPolicy()
) -> str:
    """
    Create a unique key for caching based on the input configuration and the memory policy
    configuration.

    :param config: Input configuration.
    :type config: Configuration
    :param memory_policy: Policy to incorporate practical overheads. Defaults to MemoryPolicy().
    :type memory_policy: MemoryPolicy
    :return: Key for the evaluation cache.
    :rtype: str
    """
    return (
        f"{config_hash(config.rpeg, config.plan)}:mem:"
        f"fusion={int(config.intra_stage_fusion)}:"
        f"transfer={','.join((config.transfer_resource_policy or RPEG_TRANSFER_RESOURCE_POLICY).to_dict().values())}:"
        f"{memory_policy.optimizer_stage}:"
        f"{int(memory_policy.activation_checkpointing)}:"
        f"{memory_policy.runtime_overhead_fraction:.12g}:"
        f"{memory_policy.worker_overhead_bytes:.12g}:"
        f"activation_cache={memory_policy.semantic_activation_cache_bytes:.12g}:"
        f"groups={memory_policy.max_worker_groups_per_gpu}"
    )


def evaluate_cached(
    config: Configuration,
    cache: EvalCache,
    suite: ModelSuite,
    workload: Workload,
    hw: Hardware,
    K_eval: int,
    memory_policy: MemoryPolicy = MemoryPolicy()
) -> Tuple[float, Trace, bool, bool]:
    """
    Wrap evaluate with cache lookup and soft-infeasibility penalty.

    :param config: Initial configuration.
    :type config: Configuration
    :param cache: Evaluation cache of configurations.
    :type cache: EvalCache
    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param workload: Workload settings.
    :type workload: Workload
    :param hw: Hardware settings.
    :type hw: Hardware
    :param K_eval: Number of iterations/steps to unroll.
    :type K_eval: int
    :param memory_policy: Policy to incorporate practical overheads. Defaults to MemoryPolicy().
    :type memory_policy: MemoryPolicy
    :return: Tuple of the score, the runtime trace, the feasibility and whether the configuration
             was already in the cache.
    :rtype: Tuple[float, Trace, bool, bool]
    """
    h = evaluation_cache_key(config, memory_policy)
    if h in cache:
        score, trace, feasible = cache.get(h)
        return score, trace, feasible, False

    # Always compute raw throughput (scheduler runs regardless of memory). evaluate() returns the
    # lowered executable; reuse it for memory checking instead of redundantly calling unroll(1) +
    # lower(1). Per-device peaks are identical regardless of K_eval (all K copies of a template
    # vertex share the same pi/mb/alloc, and _device_peaks only inspects OP nodes).
    raw_score, trace, lowered = evaluate(config, suite, workload, hw, K_eval, memory_policy)

    # Positive-lag publication endpoints require distinct physical workers. This is a hard semantic
    # constraint, not a soft memory preference.
    if not stale_publication_layout_valid(config):
        cache.put(h, 0.0, trace, False)
        return 0.0, trace, False, True

    # Memory penalty: check worst node violation and scale score down.
    feasible = True
    mem_model = NodeMemoryModel.from_policy(
        hw,
        suite,
        workload,
        memory_policy,
        trainable = frozenset(config.rpeg.trainable),
        shadow_models = frozenset(m for m in config.rpeg.trainable if config.rpeg.tau.get(m, 0) > 0)
    )
    violation = mem_worst_violation(lowered, mem_model)

    if violation is None: # configuration is memory feasible
        penalty_mult = 1.0
    else:
        feasible = False
        if ABLATION_HARD_OOM:
            # A8: hard OOM rejection — score = 0 (effectively reject)
            penalty_mult = 0.0
        else:
            _, worst_req, worst_cap = violation
            overflow_ratio = max(0.0, worst_req / worst_cap - 1.0)
            penalty_mult = math.exp(-_OOM_BETA * overflow_ratio)
            penalty_mult = max(penalty_mult, 1e-12)

    score = raw_score * penalty_mult
    cache.put(h, score, trace, feasible)
    return score, trace, feasible, True  # cache miss


def _eval_worker(
    args: Tuple[Configuration, ModelSuite, Workload, Hardware, int, Optional[MemoryPolicy]
) -> Tuple[float, Trace, bool]:
    """
    Multiprocessing worker: evaluate a single config without cache.

    Performs the same computation as evaluate_cached on a cache miss: evaluate() → memory check →
    penalty. Does not touch EvalCache (the parent process owns it exclusively).

    :param args: Tuple (config, suite, workload, hw, K_eval, memory_policy) — all picklable.
    :type args: Tuple[Configuration, ModelSuite, Workload, Hardware, int, Optional[MemoryPolicy]]
    :return: Tuple of the score, the runtime trace and feasibility.
    :rtype: Tuple[float, Trace, bool]
    """
    if len(args) == 5:
        config, suite, workload, hw, K_eval = args
        memory_policy = MemoryPolicy()
    else:
        config, suite, workload, hw, K_eval, memory_policy = args
    raw_score, trace, lowered = evaluate(config, suite, workload, hw, K_eval, memory_policy)
    mem_model = NodeMemoryModel.from_policy(
        hw,
        suite,
        workload,
        memory_policy,
        trainable = frozenset(config.rpeg.trainable),
        shadow_models = frozenset(m for m in config.rpeg.trainable if config.rpeg.tau.get(m, 0) > 0)
    )
    violation = mem_worst_violation(lowered, mem_model)
    if violation is None:
        return raw_score, trace, True
    if ABLATION_HARD_OOM:
        return 0.0, trace, False
    _, worst_req, worst_cap = violation
    overflow_ratio = max(0.0, worst_req / worst_cap - 1.0)
    penalty_mult = max(math.exp(-_OOM_BETA * overflow_ratio), 1e-12)
    return raw_score * penalty_mult, trace, False
