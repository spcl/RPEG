#!/usr/bin/env python3

# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt

"""
RPEG — optimization pipeline entry point.

Reads a YAML configuration file and runs the full optimization pipeline:
  baseline evaluation -> local-search optimizer -> report.

Usage:
    python pipeline.py --config configs/pod.yaml

See configs/ for annotated example configuration files.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import re
import sys
import time
import yaml

from typing import Dict, List, Optional, Tuple

# Path setup
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "RPEG"))

from core.log import init_logging, print_run_header
from core.types import packing_geometry, Allocation, DeviceSpace, Model, Role, TemplateVertex
from ir.dpo import build_DPO_RPEG
from ir.grpo import build_GRPO_RPEG
from ir.plan import ExecutionPlan
from ir.ppo import build_PPO_RPEG
from ir.rpeg import RPEG
from ir.serialize import save_plan
from model.cost import NodeCostModel
from model.hardware import Hardware
from model.memory import _device_peaks, mem_worst_violation, NodeMemoryModel
from model.models import ModelSpec, ModelSuite
from model.workload import Workload
from passes.lower import lower, ExecNodeType
from passes.unroll import unroll
from schedule.scheduler import schedule
from search.cache import EvalCache
from search.evaluate import evaluate_cached, Configuration
from search.inter_model import DEFAULT_T_CAND
from search.moves import build_candidate_directory
from search.optimizer import optimize, polish, OptimizerConfig
from scripts.visualize import dot_to_img, exec_to_dot, rpeg_to_dot, unrolled_to_dot, safe_unlink, trace_to_gantt


# Configuration loading and validation
def _make_yaml_loader():
    """
    Return a SafeLoader that accepts bare scientific notation (e.g., 450e9).

    PyYAML's default resolver requires a sign character in the exponent (e.g., 4.5e+11) or a decimal
    point (e.g., 450.0). This loader relaxes that so users can write natural values like 450e9 or
    989e12 directly.
    """
    _FLOAT_RE = re.compile(
        r'''^(?:
            [-+]?(?:[0-9][0-9_]*)\.?[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?\.(?:inf|Inf|INF)
            |\.(?:nan|NaN|NAN)
        )$''', re.X)

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_implicit_resolver("tag:yaml.org,2002:float", _FLOAT_RE,
                                  list("-+0123456789."))
    return _Loader


def load_config(path: str) -> Dict:
    """
    Load configuration from a yaml file.

    :param path: Path to the input yaml file.
    :type path: str
    :return: Content of the yaml file as a dictionary.
    :rtype: Dict
    """
    with open(path) as f:
        return yaml.load(f, Loader=_make_yaml_loader())


def _require(d: Dict, *keys: str, section: str = "") -> None:
    """
    Checks whether the keys are present in the dictionary.

    :param d: Input dictionary.
    :type d: Dict
    :param *keys: Keys to check.
    :type *keys: str
    :param section: Identifier for the dictionary. Defaults to an empty string.
    :type section: str
    :raises ValueError: If a key is not present in the dictionary.
    """
    prefix = f"[{section}] " if section else ""
    for k in keys:
        if k not in d:
            raise ValueError(f"{prefix}Missing required field: '{k}'")


def validate_config(cfg: Dict) -> None:
    """
    Validate a configuration.

    :param cfg: Dictionary with the configuration.
    :type cfg: Dict
    :raises ValueError: If the validated constraints are not met.
    """
    _require(cfg, "hardware", "models", "workload", "optimizer")

    # Hardware
    hw = cfg["hardware"]
    _require(hw, "racks", "nodes_per_rack", "gpus_per_node",
             "flops_per_gpu", "memory_per_gpu", "memory_bandwidth",
             section="hardware")
    if hw["racks"] < 1 or hw["nodes_per_rack"] < 1 or hw["gpus_per_node"] < 1:
        raise ValueError("[hardware] racks, nodes_per_rack, gpus_per_node must all be >= 1")

    for f in ("alpha_coll_g", "beta_coll_g", "alpha_coll_n", "beta_coll_n",
              "alpha_coll_r", "beta_coll_r", "alpha_p2p_g", "beta_p2p_g",
              "alpha_p2p_n", "beta_p2p_n", "alpha_p2p_r", "beta_p2p_r"):
        _require(hw, f, section="hardware")

    # Policy — read early as model validation can depend on algorithm
    algo = cfg.get("policy", {}).get("algorithm", "ppo")

    # Models — required roles depend on algorithm:
    #   PPO:  actor, critic, reward   (+ reference = actor architecture)
    #   GRPO: actor, reward           (no critic)
    #   DPO:  actor                   (no critic, no reward)
    models = cfg["models"]
    if algo == "dpo":
        required_roles = ("actor",)
    elif algo == "grpo":
        required_roles = ("actor", "reward")
    else:
        required_roles = ("actor", "critic", "reward")
    for role in required_roles:
        _require(models, role, section="models")
        _require(models[role], "layers", "hidden_dim", "ffn_dim", "vocab_size",
                 section=f"models.{role}")

    # Workload
    w = cfg["workload"]
    _require(
        w,
        "batch_size",
        "samples_per_prompt",
        "context_length",
        "generation_length",
        section = "workload"
    )

    # Optimizer
    o = cfg["optimizer"]
    _require(o, "k_eval", section="optimizer")
    if o["k_eval"] < 1:
        raise ValueError("[optimizer] k_eval must be >= 1")

    # Policy (optional section — validated only if present)
    algo = cfg.get("policy", {}).get("algorithm", "ppo")
    if algo not in ("ppo", "grpo", "dpo"):
        raise ValueError(
            f"[policy] Unsupported algorithm '{algo}'. Currently supported: 'ppo', 'grpo', 'dpo'."
        )

    # Staleness candidates (optional)
    sc = o.get("staleness_candidates")
    if sc is not None:
        valid_names = {"actor", "critic", "ac_share"}
        for name in sc:
            if name not in valid_names:
                raise ValueError(
                    f"[optimizer.staleness_candidates] Unknown model '{name}'. "
                    f"Valid names: {sorted(valid_names)}"
                )


# Object builders
def build_device_space(cfg: Dict) -> DeviceSpace:
    """
    Initialize the DeviceSpace class from the configuration dictionary.

    :param cfg: Configuration dictionary.
    :type cfg: Dict
    :return: Initialized DeviceSpace class.
    :rtype: DeviceSpace
    """
    hw = cfg["hardware"]
    return DeviceSpace(R=hw["racks"], N=hw["nodes_per_rack"], G=hw["gpus_per_node"])


def build_hardware(cfg: Dict) -> Hardware:
    """
    Initialize the Hardware class from the configuration dictionary.

    :param cfg: Configuration dictionary.
    :type cfg: Dict
    :return: Initialized Hardware class.
    :rtype: Hardware
    """
    hw = cfg["hardware"]
    return Hardware(
        flops_per_gpu = hw["flops_per_gpu"],
        memory_per_gpu = hw["memory_per_gpu"],
        memory_bandwidth = hw["memory_bandwidth"],
        alpha_coll_g = hw["alpha_coll_g"],
        beta_coll_g = hw["beta_coll_g"],
        alpha_coll_n = hw["alpha_coll_n"],
        beta_coll_n = hw["beta_coll_n"],
        alpha_coll_r = hw["alpha_coll_r"],
        beta_coll_r = hw["beta_coll_r"],
        alpha_p2p_g = hw["alpha_p2p_g"],
        beta_p2p_g = hw["beta_p2p_g"],
        alpha_p2p_n = hw["alpha_p2p_n"],
        beta_p2p_n = hw["beta_p2p_n"],
        alpha_p2p_r = hw["alpha_p2p_r"],
        beta_p2p_r = hw["beta_p2p_r"],
        gemm_m_half = hw.get("gemm_m_half", 5.0),
        gemm_k_half = hw.get("gemm_k_half", 0.0),
        decode_bw_eff = hw.get("decode_bw_eff", 0.15),
        non_gemm_constant = hw.get("non_gemm_constant", 0.0),
        non_gemm_scaling = hw.get("non_gemm_scaling", 0.0),
        non_gemm_batch_cal = hw.get("non_gemm_batch_cal", 64),
    )


def build_suite(cfg: Dict) -> ModelSuite:
    """
    Initialze the models and their specifications from the configuration dictionary.

    :param cfg: Configuration dictionary.
    :type cfg: Dict
    :return: Initialized suite of models.
    :rtype: ModelSuite
    """
    m = cfg["models"]
    algo = cfg.get("policy", {}).get("algorithm", "ppo")

    def _spec(role: str) -> ModelSpec:
        """
        Set the model specification according to its role.

        :param role: Model role.
        :type role: str
        :return: Model specification.
        :rtype: ModelSpec
        """
        s = m[role]
        return ModelSpec(
            L = s["layers"],
            d = s["hidden_dim"],
            d_ff = s["ffn_dim"],
            V = s["vocab_size"],
            d_head = s.get("head_dim", 128)
        )

    actor_spec = _spec("actor")

    # Reference always shares the actor architecture.
    specs = {
        Model.ACTOR:     actor_spec,
        Model.REFERENCE: actor_spec,
    }

    if algo in ("ppo", "grpo"):
        # PPO and GRPO use a reward model; PPO also has a critic.
        specs[Model.REWARD] = _spec("reward")
    if algo == "ppo":
        specs[Model.CRITIC]   = _spec("critic")
        # Fused actor-critic uses actor architecture - set default here, will later be disabled if
        # necessary.
        specs[Model.AC_SHARE] = actor_spec
    # DPO: only actor + reference — no reward, no critic.

    return ModelSuite(specs)


def build_workload(cfg: Dict) -> Workload:
    """
    Initialize the Workload class from the configuration dictionary.

    :param cfg: Configuration dictionary.
    :type cfg: Dict
    :return: Initialized Workload class.
    :rtype: Workload
    """
    w = cfg["workload"]
    gq = w.get("gen_quantiles", None)
    gen_quantiles = tuple(tuple(q) for q in gq) if gq else None
    return Workload(
        B = w["batch_size"],
        K = w["samples_per_prompt"],
        S = w["context_length"],
        T = w["generation_length"],
        gen_quantiles = gen_quantiles,
    )


def build_optimizer_config(
    cfg: Dict,
    allow_ac_share: bool,
    algorithm: str = "ppo"
) -> OptimizerConfig:
    """
    Initialize the optimizer configuration from the general configuration dictionary.

    :param cfg: Configuration dictionary.
    :type cfg: Dict
    :param allow_ac_share: Flag indicating whether the actor and critic can share parts of their
                           weights.
    :type allow_ac_share: bool
    :param algorithm: Policy optimization algorithm to use. Defaults to 'ppo'.
    :type algorithm: str
    :return: Initialized OptimizerConfig class.
    :rtype: OptimizerConfig
    """
    o = cfg["optimizer"]

    # Build T_cand
    sc = o.get("staleness_candidates")

    if algorithm in ("grpo", "dpo"):
        # GRPO/DPO: only actor is trainable — no critic, no AC-share.
        t_cand = {Model.ACTOR: {0, 1}}
        if sc is not None and "actor" in sc:
            t_cand[Model.ACTOR] = set(sc["actor"])
    elif sc is None and allow_ac_share:
        # Use the built-in defaults unchanged.
        t_cand = None
    else:
        # Start from built-in defaults and apply user overrides.
        _name_to_model = {
            "actor": Model.ACTOR,
            "critic": Model.CRITIC,
            "ac_share": Model.AC_SHARE,
        }
        t_cand = copy.deepcopy(DEFAULT_T_CAND)
        if sc is not None:
            for name, vals in sc.items():
                t_cand[_name_to_model[name]] = set(vals)
        if not allow_ac_share:
            # Empty set -> _iter_staleness yields no AC-share combinations.
            t_cand[Model.AC_SHARE] = set()

    return OptimizerConfig(
        K_x = o.get("k_x", 6),
        T_max = o.get("t_max", 50),
        E_max = o.get("e_max", 5000),
        T_init = o.get("t_init", 0.30),
        T_warm = o.get("t_warm", 0.10),
        T_min = o.get("t_min", 0.03),
        E_struct = o.get("e_struct", 400),
        T_cand = t_cand,
        lambda_ac = o.get("lambda_ac", 0.15),
        score_eps = o.get("score_eps", 1e-4),
        stag_window = o.get("stag_window", 20),
        alpha_pair = o.get("alpha_pair", 0.28),
        max_throughput_loss = o.get("max_throughput_loss", 0.0),
        algorithm = algorithm,
        # Ablation flags
        fixed_mb = o.get("fixed_mb", False),
        uniform_vertex = o.get("uniform_vertex", False),
        no_polish = o.get("no_polish", False),
        hard_oom = o.get("hard_oom", False),
        geometric_cooling = o.get("geometric_cooling", False)
    )


# Baseline plan constructors
def build_valid_baseline_plan(
    rpeg: RPEG,
    device_space: DeviceSpace,
    suite: ModelSuite,
    workload: Workload,
    cost_model: NodeCostModel,
    mem_model: NodeMemoryModel
) -> ExecutionPlan:
    """
    Greedy-starting point: select for each vertex the best local plan.

    Baseline plan guaranteed to be valid (divisibility, packing, memory).

    Picks the fastest valid strategy for each vertex from the candidate directory (index-0 after
    ascending sort by per-node execution time) and places it at origin (0,0,0).

    :param rpeg: RLM Pipeline Execution Graph.
    :type rpeg: RPEG
    :param device_space: Available devices and their hierarchy.
    :type device_space: DeviceSpace
    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param workload: Workload settings.
    :type workload: Workload
    :param cost_model: Cost model to estimate the time for the graph nodes.
    :type cost_model: NodeCostModel
    :param mem_model: Memory model for the graph nodes.
    :type mem_model: NodeMemoryModel
    :return: Execution plan.
    :rtype: ExecutionPlan
    :raises RuntimeError: If no valid candidates for a vertex are found.
    """
    # Returns per-node strategies that are valid and mem_feasible
    table, _ = build_candidate_directory(rpeg, suite, device_space, workload, cost_model, mem_model)
    sigma, pi, mb = {}, {}, {}
    for v in rpeg.vertices:
        if not table[v]:
            raise RuntimeError(f"No valid candidates for vertex {v.id}; cannot construct baseline plan")
        dp, pp, tp, r_ext, n_ext, mb_v, _ = table[v][0] # greedy per-vertex selection
        geom = packing_geometry(dp, pp, tp, device_space.G)
        g_ext = geom[0] if geom else min(pp, device_space.G // tp) * tp
        sigma[v] = Allocation(origin=(0, 0, 0), extent=(r_ext, n_ext, g_ext))
        pi[v] = (dp, pp, tp)
        mb[v] = mb_v
    return ExecutionPlan(sigma, pi, mb)


# Concurrent-aware baseline.
def _find_concurrent_groups(rpeg: RPEG) -> List[List[TemplateVertex]]:
    """
    Partition RPEG vertices into sequential phases of mutually-independent vertices.

    Uses the data-edge DAG to compute the longest-path depth of each vertex. Vertices at the same
    depth are mutually independent (anti-chain) and can execute concurrently on disjoint device
    subsets.

    :param rpeg: RLM Pipeline Execution Graph.
    :type rpeg: RPEG
    :return: List of groups ordered by depth (phase 0, phase 1, ...).
    :rtype: List[List[TemplateVertex]]
    """
    # Build adjacency from data edges only (pub edges are inter-step).
    vset = set(rpeg.vertices)
    children: dict[TemplateVertex, list[TemplateVertex]] = {v: [] for v in rpeg.vertices}
    parents: dict[TemplateVertex, list[TemplateVertex]] = {v: [] for v in rpeg.vertices}
    for e in rpeg.data_edges:
        if e.src in vset and e.dst in vset:
            children[e.src].append(e.dst)
            parents[e.dst].append(e.src)

    # Longest-path depth: depth(v) = 0 if no parents, else max(depth(p) for p in parents) + 1.
    depth: dict[TemplateVertex, int] = {}
    def _depth(v: TemplateVertex) -> int:
        """
        Determine the depth of a vertex if it was not already computed.

        :param v: Vertex to determine the depth for.
        :type v: TemplateVertex
        :return: Depth of the vertex.
        :rtype: int
        """
        if v in depth:
            return depth[v]
        if not parents[v]:
            depth[v] = 0
        else:
            depth[v] = max(_depth(p) for p in parents[v]) + 1
        return depth[v]

    for v in rpeg.vertices:
        _depth(v)

    # Group by depth.
    max_d = max(depth.values()) if depth else 0
    groups: list[list[TemplateVertex]] = [[] for _ in range(max_d + 1)]
    for v in rpeg.vertices:
        groups[depth[v]].append(v)

    return groups


def _enumerate_device_subsets(device_space: DeviceSpace) -> List[Allocation]:
    """
    Enumerate useful contiguous device subsets for concurrent placement.

    Returns allocations representing natural partition units: full space, multi-rack slices,
    per-rack slices, multi-node slices, per-node slices, half-node slices.

    Multi-node and multi-rack slices use powers-of-2 groupings aligned
    to their size (e.g. nodes 0-1, 2-3, 0-3, 4-7, ...).

    :param device_space: Available devices and their hierarchy.
    :type device_space: DeviceSpace
    :return: List of possible allocations in the order described above.
    :rtype: List[Allocation]
    """
    R, N, G = device_space.R, device_space.N, device_space.G
    seen: set[tuple[tuple[int,int,int], tuple[int,int,int]]] = set()
    subsets: list[Allocation] = []

    def _add(origin: Tuple[int, int, int], extent: Tuple[int, int, int]) -> None:
        """
        Add an allocation to the list if it is not already part of the list.

        :param origin: Start device of the allocation.
        :type origin: Tuple[int, int, int]
        :param extent: Range of the allocation in all three dimensions.
        :type extent: Tuple[int, int, int]
        """
        key = (origin, extent)
        if key not in seen:
            seen.add(key)
            subsets.append(Allocation(origin=origin, extent=extent))

    # Full device space
    _add((0, 0, 0), (R, N, G))

    # Multi-rack slices (powers-of-2 rack groups, each with all N nodes)
    rk = 1
    while rk <= R:
        if rk < R: # skip full (already added)
            for r0 in range(0, R, rk):
                if r0 + rk <= R:
                    _add((r0, 0, 0), (rk, N, G))
        rk *= 2

    # Within each rack: multi-node slices (powers-of-2 node groups)
    for r in range(R):
        nk = 1
        while nk <= N:
            for n0 in range(0, N, nk):
                if n0 + nk <= N:
                    _add((r, n0, 0), (1, nk, G))
            nk *= 2

    # Half-node slices (G/2 GPUs) — only if G is even and >= 4
    if G >= 4 and G % 2 == 0:
        half = G // 2
        for r in range(R):
            for n in range(N):
                _add((r, n, 0),    (1, 1, half))
                _add((r, n, half), (1, 1, half))

    return subsets


def _allocs_disjoint(allocs: List[Allocation]) -> bool:
    """
    Return True if all allocations in the list occupy non-overlapping GPU sets.

    :param allocs: List of allocations to check.
    :type allocs: List[Allocation]
    :return: Flag indicating whether the allocations occupy non-overlapping GPU sets.
    :rtype: bool
    """
    seen: set[tuple[int, int, int]] = set()
    for a in allocs:
        ro, no, go = a.origin
        re, ne, ge = a.extent
        for r in range(ro, ro + re):
            for n in range(no, no + ne):
                for g in range(go, go + ge):
                    dev = (r, n, g)
                    if dev in seen:
                        return False
                    seen.add(dev)
    return True


def _best_strategy_in_alloc(
    v: TemplateVertex,
    table: Dict,
    alloc: Allocation,
    device_space: DeviceSpace,
    mem_model: Optional[NodeMemoryModel] = None
) -> Optional[Tuple]:
    """
    Find the fastest memory-feasible strategy for vertex v within the given allocation.

    :param v: Vertex to find the strategy for.
    :type v: TemplateVertex
    :param table: Dictionary with a list of strategies for the vertices.
    :type table: Dict
    :param alloc: Potential hardware that can be used for the strategy.
    :type alloc: Allocation
    :param device_space: Available devices and their hierarchy.
    :type device_space: DeviceSpace
    :param mem_model: Memory model for the graph nodes. Defaults to None.
    :type mem_model: Optional[NodeMemoryModel]
    :return: Tuple of (dp, pp, tp, r_ext, n_ext, mb, time, alloc_with_correct_origin) or None if no
             strategy is found.
    :rtype: Optional[Tuple]
    """
    from search.moves import _make_mock_node
    G = device_space.G
    a_ro, a_no, a_go = alloc.origin
    a_re, a_ne, a_ge = alloc.extent

    for strat_with_time in table[v]: # sorted ascending by time
        dp, pp, tp, r_ext, n_ext, mb_v, t = strat_with_time
        # Check that the strategy's extent fits within the allocation.
        if r_ext > a_re or n_ext > a_ne:
            continue
        geom = packing_geometry(dp, pp, tp, G)
        if geom is None:
            continue
        g_ext_strat, _ = geom
        if g_ext_strat > a_ge:
            continue
        # Compute a valid origin within the allocation bounds.
        r_o = a_ro
        n_o = a_no
        g_o = a_go
        if r_o + r_ext > a_ro + a_re:
            continue
        if n_o + n_ext > a_no + a_ne:
            continue
        if g_o + g_ext_strat > a_go + a_ge:
            continue
        fitted_alloc = Allocation(origin=(r_o, n_o, g_o), extent=(r_ext, n_ext, g_ext_strat))
        # Per-vertex memory feasibility check (catches single-vertex OOM).
        # Full plan-level OOM is caught after plan assembly in build_concurrent_baseline_plan.
        if mem_model is not None:
            mock = _make_mock_node(v, dp, pp, tp, r_ext, n_ext, mb_v, g_ext_strat)
            if not mem_model.is_locally_feasible(mock):
                continue
        return (dp, pp, tp, r_ext, n_ext, mb_v, t, fitted_alloc)

    return None # no strategy fits


def _enumerate_group_partitions(
    group: List[TemplateVertex],
    device_space: DeviceSpace,
) -> List[List[Allocation]]:
    """
    Enumerate partition schemes for a concurrent group.

    Returns a list of partition options. Each option is a list of Allocations, one per vertex in the
    group (same order as the group list).

    Includes: full serialization, 2-way splits, and k-way splits.

    Uses partition-scheme enumeration: enumerate allocation sizes (how many nodes each group gets)
    rather than all triples of positioned subsets. Two allocations with the same extent but
    different origin produce the same score (_best_strategy_in_alloc only depends on extent, not
    origin), so enumerating positions is redundant. This reduces k=3 candidates from O(S³) to
    O(log²(N)) — e.g. 2M → ~250 at 128 GPUs.

    :param group: List of vertices in the concurrent group.
    :type group: List[TemplateVertex]
    :param device_space: Available devices and their hierarchy.
    :type device_space: DeviceSpace
    :return: List of partition options as described above.
    :rtype: List[List[Allocation]]
    """
    from itertools import permutations as _perms

    R, N, G = device_space.R, device_space.N, device_space.G
    k = len(group)

    # Always include: full serialization (everyone gets the full device space).
    full_alloc = Allocation(origin=(0, 0, 0), extent=(R, N, G))
    partitions: list[list[Allocation]] = [[full_alloc] * k]

    if k == 1:
        return partitions # nothing to split

    total_nodes = R * N

    # Power-of-2 node counts that fit in the cluster
    node_sizes: list[int] = []
    s = 1
    while s <= total_nodes:
        node_sizes.append(s)
        s *= 2

    def _extent_for_nodes(n_nodes: int) -> Tuple[int, int, int]:
        """
        Compute (r_ext, n_ext, g_ext) for an allocation of n_nodes nodes.

        :param n_nodes: Number of nodes within the allocation.
        :type n_nodes: int
        :return: Tuple with the extent in the device space.
        :rtype: Tuple[int, int, int]
        """
        if n_nodes <= N:
            return (1, n_nodes, G)
        else:
            r_needed = (n_nodes + N - 1) // N # ceil(n_nodes / N)
            return (r_needed, N, G)

    def _pack_disjoint(sizes: List[int]) -> Optional[List[Allocation]]:
        """
        Greedily pack disjoint allocations of given node-sizes.

        Returns a list of Allocations (one per size) or None if they do not fit. Packing is
        first-fit aligned along node dimension, spilling to next rack.

        :param sizes: List of vertex sizes.
        :type sizes: List[int]
        :return: List of allocations or None if they do not fit.
        :rtype: Optional[List[Allocation]]
        """
        allocs: list[Allocation] = []
        cursor_r, cursor_n = 0, 0

        for n_nodes in sizes:
            r_ext, n_ext, _ = _extent_for_nodes(n_nodes)

            if r_ext > 1:
                # Multi-rack allocation: start at a rack-aligned boundary.
                if cursor_n > 0:
                    cursor_r += 1
                    cursor_n = 0
                if cursor_r % r_ext != 0:
                    cursor_r += r_ext - (cursor_r % r_ext)
                if cursor_r + r_ext > R:
                    return None
                allocs.append(Allocation(origin=(cursor_r, 0, 0), extent=(r_ext, N, G)))
                cursor_r += r_ext
                cursor_n = 0
            else:
                # Single-rack: align to n_ext boundary within current rack.
                if cursor_n % n_ext != 0:
                    cursor_n += n_ext - (cursor_n % n_ext)
                if cursor_n + n_ext > N:
                    cursor_r += 1
                    cursor_n = 0
                if cursor_r >= R:
                    return None
                allocs.append(Allocation(origin=(cursor_r, cursor_n, 0), extent=(1, n_ext, G)))
                cursor_n += n_ext

        return allocs

    # Build partitions from size schemes
    seen: set[tuple] = set() # deduplicate by (extent_per_vertex) tuple

    def _try_disjoint(sizes: List[int]) -> None:
        """
        Try all vertex permutations of a fully-disjoint size scheme.

        Updates the seen set for checking whether the partition has already been processed and
        updates the partition list.

        :param sizes: List of vertex sizes.
        :type sizes: List[int]
        """
        packed = _pack_disjoint(sizes)
        if packed is None:
            return
        for perm in _perms(range(k)):
            key = tuple(packed[perm[i]].extent for i in range(k))
            if key in seen:
                continue
            seen.add(key)
            partitions.append([packed[perm[i]] for i in range(k)])

    def _try_2way(s_solo: int, s_shared: int, k: int) -> None:
        """
        Try 2-way splits: one vertex solo, the rest serialized together.

        Updates the seen set for checking whether the partition has already been processed and
        updates the partition list.

        :param s_solo: Size of a solo allocation for a single graph node.
        :type s_solo: int
        :param s_shared: Size of the serialized allocation.
        :type s_shared: int
        :param k: Number of groups.
        :type k: int
        """
        packed = _pack_disjoint([s_solo, s_shared])
        if packed is None:
            return
        solo_alloc, shared_alloc = packed[0], packed[1]
        for solo_idx in range(k):
            part = [shared_alloc] * k # all share by default (same object = serialized)
            part[solo_idx] = solo_alloc
            key = tuple(a.extent for a in part) + (solo_idx,)
            if key in seen:
                continue
            seen.add(key)
            partitions.append(part)
            # Also try reversed assignment
            part2 = [solo_alloc] * k # all share the solo alloc (serialized)
            # Actually: reverse means solo gets the shared region, others get the solo region
            packed_rev = _pack_disjoint([s_shared, s_solo])
            if packed_rev is None:
                continue
            part2 = [packed_rev[1]] * k
            part2[solo_idx] = packed_rev[0]
            key2 = tuple(a.extent for a in part2) + (solo_idx,)
            if key2 in seen:
                continue
            seen.add(key2)
            partitions.append(part2)

    if k == 2:
        for s1 in node_sizes:
            for s2 in node_sizes:
                if s1 + s2 <= total_nodes:
                    _try_disjoint([s1, s2])
        # Half-node splits (G/2 GPUs each, same node)
        if G >= 4 and G % 2 == 0:
            half = G // 2
            for r in range(R):
                for n in range(N):
                    a = Allocation(origin=(r, n, 0), extent=(1, 1, half))
                    b = Allocation(origin=(r, n, half), extent=(1, 1, half))
                    partitions.append([a, b])
                    partitions.append([b, a])

    elif k == 3:
        # 3-way disjoint: all three on separate partitions
        for s1 in node_sizes:
            for s2 in node_sizes:
                if s1 + s2 > total_nodes:
                    continue
                for s3 in node_sizes:
                    if s1 + s2 + s3 <= total_nodes:
                        _try_disjoint([s1, s2, s3])

        # 2-way: one vertex solo, two serialized on the same partition
        for s_solo in node_sizes:
            for s_shared in node_sizes:
                if s_solo + s_shared <= total_nodes:
                    _try_2way(s_solo, s_shared, k)

    else:
        # k > 3: full serialization (already added) + per-node splits
        if total_nodes >= k:
            node_allocs = []
            for r in range(R):
                for n in range(N):
                    node_allocs.append(Allocation(origin=(r, n, 0), extent=(1, 1, G)))
            if len(node_allocs) >= k:
                partitions.append(node_allocs[:k])

    return partitions


def _score_group_partition(
    group: List[TemplateVertex],
    allocs: List[Allocation],
    table: Dict,
    device_space: DeviceSpace,
    mem_model: Optional[NodeMemoryModel] = None
) -> Tuple[float, Optional[Dict]]:
    """
    Score a partition for a concurrent group.

    For vertices sharing the same allocation (serialized): sum their times. For vertices on disjoint
    allocations: maximum of their times.

    Returns (group_time, assignment_dict) where assignment_dict maps vertex -> (dp, pp, tp, mb,
    Allocation) or None if infeasible.

    :param group: List of vertices in the concurrent group.
    :type group: List[TemplateVertex]
    :param allocs: List of allocations for the concurrent group.
    :type allocs: List[Allocation]
    :param table: Dictionary with a list of strategies for the vertices.
    :type table: Dict
    :param device_space: Available devices and their hierarchy.
    :type device_space: DeviceSpace
    :param mem_model: Memory model for the graph nodes. Defaults to None.
    :type mem_model: Optional[NodeMemoryModel]
    :return: Tuple of execution time of the group and assignment dictionary as described above or
             'inf', None, if infeasible.
    :rtype: Tuple[float, Optional[Dict]]
    """
    # Group vertices by their allocation (same allocation = serialized).
    from collections import defaultdict
    alloc_groups: dict[Allocation, list[tuple[TemplateVertex, int]]] = defaultdict(list)
    for idx, (v, a) in enumerate(zip(group, allocs)):
        alloc_groups[a].append((v, idx))

    assignment: dict[TemplateVertex, tuple] = {}
    alloc_times: list[float] = []

    for alloc, members in alloc_groups.items():
        serial_time = 0.0
        for v, idx in members:
            result = _best_strategy_in_alloc(v, table, alloc, device_space, mem_model)
            if result is None:
                return float('inf'), None  # infeasible
            dp, pp, tp, r_ext, n_ext, mb_v, t, fitted_alloc = result
            assignment[v] = (dp, pp, tp, mb_v, fitted_alloc)
            serial_time += t
        alloc_times.append(serial_time)

    # Vertices on disjoint allocations run concurrently: group_time = max.
    group_time = max(alloc_times)

    return group_time, assignment


def build_concurrent_baseline_plan(
    rpeg: RPEG,
    device_space: DeviceSpace,
    suite: ModelSuite,
    workload: Workload,
    cost_model: NodeCostModel,
    mem_model: NodeMemoryModel
) -> Tuple[ExecutionPlan, bool]:
    """
    Concurrency-aware baseline: partitions device space among independent vertices.

    Identifies concurrent groups (anti-chains) from the RPEG DAG, enumerates device partitions per
    group, and picks the combination that minimizes the estimated step time (sum of per-group
    bottleneck times).

    Falls back to build_valid_baseline_plan if no concurrent plan improves on the serialized
    baseline.

    :param rpeg: RLM Pipeline Execution Graph.
    :type rpeg: RPEG
    :param device_space: Available devices and their hierarchy.
    :type device_space: DeviceSpace
    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param workload: Workload settings.
    :type workload: Workload
    :param cost_model: Cost model to estimate the time for the graph nodes.
    :type cost_model: NodeCostModel
    :param mem_model: Memory model for the graph nodes.
    :type mem_model: NodeMemoryModel
    :return: Tuple of the execution plan and a flag indicating whether the plan was found by the
             concurrency-aware baseline.
    :rtype: Tuple[ExecutionPlan, bool]
    """
    table, _ = build_candidate_directory(rpeg, suite, device_space, workload, cost_model, mem_model)
    groups = _find_concurrent_groups(rpeg)

    print(f"\n  Concurrent groups ({len(groups)} phases):")
    for i, g in enumerate(groups):
        print(f"    phase {i}: [{', '.join(v.id for v in g)}]")

    # For each group, enumerate all feasible partitions ranked by concurrency (most concurrent
    # first, then by time within same concurrency level). This lets us cascade: if 3-way concurrent
    # OOMs the assembled plan, try 2-way, then serialized — without falling back to a weak greedy
    # baseline.
    ranked_per_group: list[list[tuple[float, dict, str]]] = [] # per group: [(time, assign, label), ...]

    for gi, group in enumerate(groups):
        partitions = _enumerate_group_partitions(group, device_space)
        candidates: list[tuple[int, float, dict, str]] = [] # (neg_concurrency, time, assign, label)

        for pi_idx, allocs in enumerate(partitions):
            t, assign = _score_group_partition(group, allocs, table, device_space, mem_model)
            if assign is None:
                continue
            unique_allocs = set(id(a) for a in allocs)
            n_concurrent = len(unique_allocs)

            if n_concurrent == 1:
                label = "serialized"
            elif n_concurrent == len(group):
                label = f"{len(group)}-way concurrent"
            else:
                label = f"{n_concurrent}-way concurrent"

            # Sort key: prefer more concurrency (negative for descending), then lower time.
            candidates.append((-n_concurrent, t, assign, label))

        candidates.sort(key=lambda x: (x[0], x[1])) # most concurrent first, fastest within tier
        ranked_per_group.append([(t, assign, label) for _, t, assign, label in candidates])

        if not candidates:
            print(f"  ⚠ No feasible partition for concurrent group {gi}: "
                  f"[{', '.join(v.id for v in group)}]. Falling back to greedy baseline.")
            return build_valid_baseline_plan(rpeg, device_space, suite, workload, cost_model, mem_model), False

    # Assemble the plan using the best (most concurrent) partition per group. If the assembled plan
    # is OOM, cascade: try the next-best partition for each group until we find a memory-feasible
    # combination. Strategy: start with index [0, 0, ..., 0] (best per group). On OOM, increment the
    # index of the group with the largest allocation footprint (most likely to cause the OOM). This
    # avoids enumerating all combinations.
    group_indices = [0] * len(groups)
    max_attempts = sum(len(r) for r in ranked_per_group) # upper bound

    for attempt in range(max_attempts):
        # Check if current indices are valid
        valid = all(group_indices[gi] < len(ranked_per_group[gi])
                    for gi in range(len(groups)))
        if not valid:
            break

        # Assemble plan from current indices
        sigma, pi_map, mb_map = {}, {}, {}
        total_time = 0.0
        labels = []
        for gi in range(len(groups)):
            t, assign, label = ranked_per_group[gi][group_indices[gi]]
            total_time += t
            labels.append(label)
            for v, (dp, pp, tp, mb_v, alloc) in assign.items():
                sigma[v] = alloc
                pi_map[v] = (dp, pp, tp)
                mb_map[v] = mb_v

        plan = ExecutionPlan(sigma, pi_map, mb_map)

        # Memory feasibility check on the assembled plan
        unrolled_check = unroll(rpeg, 1, verbose=False)
        lowered_check = lower(rpeg, unrolled_check, plan, suite, 1, workload=workload, verbose=False)
        violation = mem_worst_violation(lowered_check, mem_model)

        if violation is None:
            # Feasible! Print and return.
            serial_time = sum(table[v][0][6] for v in rpeg.vertices if table[v])
            speedup = serial_time / total_time if total_time > 0 else 1.0
            for gi, group in enumerate(groups):
                t, assign, label = ranked_per_group[gi][group_indices[gi]]
                print(f"    phase {gi} best: {label}  time={t:.4e}s  "
                      f"({', '.join(f'{v.id}={assign[v][4].extent}' for v in group)})")
            print(f"\n  Concurrent baseline estimate: {total_time:.4e}s  "
                  f"(serialized: {serial_time:.4e}s, speedup: {speedup:.2f}x)")
            return plan, True # feasible

        # OOM — cascade to less-concurrent partition for the most concurrent group.
        # Find the group with the highest concurrency index that can be demoted.
        _, peak, cap = violation
        demoted = False
        for gi in range(len(groups)):
            if group_indices[gi] + 1 < len(ranked_per_group[gi]):
                group_indices[gi] += 1
                demoted = True
                overflow_pct = (peak / cap - 1) * 100
                print(f"    phase {gi}: {labels[gi]} OOM (+{overflow_pct:.0f}%), "
                      f"demoting to {ranked_per_group[gi][group_indices[gi]][2]}")
                break
        if not demoted:
            break  # all groups exhausted

    # All concurrent combinations exhausted — fall back to greedy.
    print(f"  ⚠ All concurrent combinations OOM. Falling back to greedy baseline.")
    return build_valid_baseline_plan(rpeg, device_space, suite, workload, cost_model, mem_model), False # OOM fallback


def build_fullycolocated_baseline_plan(
    rpeg: RPEG,
    device_space: DeviceSpace,
    suite: ModelSuite,
    workload: Workload,
    cost_model: NodeCostModel,
    mem_model: NodeMemoryModel
) -> ExecutionPlan:
    """
    Fully colocated baseline: every vertex uses the entire device space.

    All operators are placed on all GPUs (origin=0, extent=full), running fully sequentially with no
    concurrency. The best-fitting strategy (dp, pp, tp, mb) for the full allocation is chosen per
    vertex.

    :param rpeg: RLM Pipeline Execution Graph.
    :type rpeg: RPEG
    :param device_space: Available devices and their hierarchy.
    :type device_space: DeviceSpace
    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param workload: Workload settings.
    :type workload: Workload
    :param cost_model: Cost model to estimate the time for the graph nodes.
    :type cost_model: NodeCostModel
    :param mem_model: Memory model for the graph nodes.
    :type mem_model: NodeMemoryModel
    :return: Execution plan.
    :rtype: ExecutionPlan
    :raises RuntimeError: If no valid full-device strategy for a vertex is found.
    """
    table, _ = build_candidate_directory(rpeg, suite, device_space, workload, cost_model, mem_model)
    R, N, G = device_space.R, device_space.N, device_space.G

    sigma, pi, mb_map = {}, {}, {}
    for v in rpeg.vertices:
        # Find the fastest strategy whose extent covers all GPUs.
        best = None
        for dp, pp, tp, r_ext, n_ext, mb_v, t in table[v]:
            geom = packing_geometry(dp, pp, tp, G)
            if geom is None:
                continue
            g_ext_v, _ = geom
            # Must use the full device space: all racks, all nodes, all GPUs.
            if r_ext == R and n_ext == N and g_ext_v == G:
                best = (dp, pp, tp, r_ext, n_ext, g_ext_v, mb_v)
                break # table is sorted by time, first match is fastest
        if best is None:
            raise RuntimeError(
                f"No valid full-device strategy for vertex {v.id} in "
                f"({R}×{N}×{G}); cannot construct fully-colocated baseline"
            )
        dp, pp, tp, r_ext, n_ext, g_ext, mb_v = best
        sigma[v] = Allocation(origin=(0, 0, 0), extent=(r_ext, n_ext, g_ext))
        pi[v] = (dp, pp, tp)
        mb_map[v] = mb_v

    return ExecutionPlan(sigma, pi, mb_map)


def build_naive_concurrent_plan(
    rpeg: RPEG,
    device_space: DeviceSpace,
    suite: ModelSuite,
    workload: Workload,
    cost_model: NodeCostModel,
    mem_model: NodeMemoryModel
) -> ExecutionPlan:
    """
    Human-tuned concurrent schedule: partitions GPUs for concurrent execution.

    Assigns disjoint GPU subsets so that independent vertices run in parallel:
      - Generation (g_pi):        all GPUs
      - Inference (i_R, i_V):     half GPUs each (concurrent pair)
      - Inference (i_ref):        all GPUs (after i_R || i_V)
      - Training (t_pi, t_V):     half GPUs each (concurrent pair)

    Picks for each vertex the fastest strategy from the candidate table that fits within the
    assigned allocation.

    :param rpeg: RLM Pipeline Execution Graph.
    :type rpeg: RPEG
    :param device_space: Available devices and their hierarchy.
    :type device_space: DeviceSpace
    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param workload: Workload settings.
    :type workload: Workload
    :param cost_model: Cost model to estimate the time for the graph nodes.
    :type cost_model: NodeCostModel
    :param mem_model: Memory model for the graph nodes.
    :type mem_model: NodeMemoryModel
    :return: Execution plan.
    :rtype: ExecutionPlan
    :raises RuntimeError: If no feasible strategy for a vertex is found.
    """
    table, _ = build_candidate_directory(rpeg, suite, device_space, workload, cost_model, mem_model)
    R, N, G = device_space.R, device_space.N, device_space.G
    half_G = G // 2

    # Device assignments: map vertex id prefix → allocation
    # Concurrent pairs share the time slot but use disjoint GPU halves.
    alloc_full = Allocation(origin=(0, 0, 0), extent=(R, N, G))
    alloc_lo = Allocation(origin=(0, 0, 0), extent=(R, N, half_G))
    alloc_hi = Allocation(origin=(0, 0, half_G), extent=(R, N, half_G))

    assign = {}
    for v in rpeg.vertices:
        role  = v.kind.role
        model = v.kind.model
        if role == Role.GENERATION:
            assign[v] = alloc_full
        elif role == Role.INFERENCE:
            if model == Model.REWARD:
                assign[v] = alloc_lo # i_R on first half
            elif model == Model.CRITIC:
                assign[v] = alloc_hi # i_V on second half (concurrent with i_R)
            else: # REFERENCE
                assign[v] = alloc_full # i_ref on all GPUs (after concurrent pair)
        elif role == Role.TRAINING:
            if model == Model.ACTOR:
                assign[v] = alloc_lo # t_pi on first half
            else: # CRITIC
                assign[v] = alloc_hi # t_V on second half (concurrent with t_pi)

    # Pick best strategy per vertex that fits its assigned allocation.
    sigma, pi_map, mb_map = {}, {}, {}
    for v in rpeg.vertices:
        alloc = assign[v]
        _, _, g_alloc = alloc.extent
        best = None
        for dp, pp, tp, r_ext, n_ext, mb_v, t in table[v]:
            geom = packing_geometry(dp, pp, tp, G)
            if geom is None:
                continue
            g_ext_v, _ = geom
            # Strategy must fit within the assigned allocation.
            if r_ext <= alloc.extent[0] and n_ext <= alloc.extent[1] and g_ext_v <= g_alloc:
                best = (dp, pp, tp, mb_v)
                break # table sorted by time → first match is fastest
        if best is None:
            raise RuntimeError(
                f"No feasible strategy for {v.id} on allocation "
                f"{alloc.extent}; cannot build naive concurrent plan")
        dp, pp, tp, mb_v = best
        sigma[v]  = alloc
        pi_map[v] = (dp, pp, tp)
        mb_map[v] = mb_v

    return ExecutionPlan(sigma, pi_map, mb_map)


# Formatting helpers
def _fmt_alloc(a: Allocation) -> str:
    """
    Returns a formatted string representing an allocation.

    :param a: Allocation to format.
    :type a: Allocation
    :return: Formatted string.
    :rtype: str
    """
    def rng(start: int, size: int, prefix: str) -> str:
        """
        Format the prefix (device dimension), the start of the allocation in that dimension and its
        size if not one.

        :param start: Start of the allocation in that dimension.
        :type start: int
        :param size: Size of the allocation in that dimension.
        :type size: int
        :param prefix: Identification of the dimension.
        :type prefix: str
        :return: Formatted string.
        :rtype: str
        """
        return f"{prefix}{start}" if size == 1 else f"{prefix}[{start}-{start+size-1}]"

    or_, on, og = a.origin
    er, en, eg = a.extent
    return f"{rng(or_, er, 'r')} {rng(on, en, 'n')} {rng(og, eg, 'g')}"


def _fmt_pi(pi: Tuple[int, int, int]) -> str:
    """
    Returns a formatted string representing a 3D strategy.

    :param pi: 3D strategy.
    :type pi: Tuple[int, int, int]
    :return: Formatted string.
    :rtype: str
    """
    dp, pp, tp = pi
    return f"({dp},{pp},{tp})"


def main():
    """
    Main function executing the full pipeline with the various baselines for the given
    configuration.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True, metavar="PATH",
        help="path to a YAML configuration file (see configs/ for examples)",
    )
    parser.add_argument(
        "--hw", metavar="PATH", default=None,
        help="path to calibrated hardware YAML (from calibrate.py). "
             "Overrides hardware section in --config.",
    )
    parser.add_argument(
        "--plots", action="store_true", default=None,
        help="generate DOT graphs and Gantt charts (overrides output.plots in YAML)",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="enable verbose debug output (cost model, memory, schedule details, candidate counts)",
    )
    parser.add_argument(
        "--seed-plan", metavar="PATH", default=None,
        help="seed the optimizer from a previously saved plan YAML (two-stage optimization)",
    )
    parser.add_argument(
        "--time", metavar="SECONDS", type=float, default=None,
        help="SA optimization loop time budget in seconds (auto-derives E_max/E_struct "
             "via calibration). Does not include baseline construction, candidate directory, "
             "or polish phase. Overrides e_max/e_struct from config.",
    )
    args = parser.parse_args()

    # Load and validate
    cfg = load_config(args.config)

    # Merge calibrated hardware YAML if provided
    if args.hw:
        with open(args.hw) as f:
            hw_lines = [l for l in f if not l.strip().startswith('#') and l.strip()]
        hw_cal = {}
        for line in hw_lines:
            if ':' in line:
                key, val = line.split(':', 1)
                key = key.strip()
                val = val.split('#')[0].strip()
                try:
                    hw_cal[key] = float(val)
                except ValueError:
                    pass
        cfg.setdefault("hardware", {}).update(hw_cal)

    validate_config(cfg)

    out = cfg.get("output", {})
    pol = cfg.get("policy", {})
    o = cfg["optimizer"]

    # Random seed
    seed = out.get("seed")
    if seed is not None:
        random.seed(seed)

    # Plan files
    base_plan_file = out.get("base_plan")
    opt_plan_file = out.get("opt_plan")

    debug = args.debug

    # Plots
    plots = args.plots if args.plots is not None else out.get("plots", False)
    plots_dir = out.get("plots_dir", "figures/runs/")
    if plots:
        os.makedirs(plots_dir, exist_ok=True)
        
    def plot_path(name: str) -> str:
        """
        Returns the full relative path to the output file.

        :param name: Name of the output file.
        :type name: str
        :return: Full relative path to the output file.
        :rtype: str
        """
        return os.path.join(plots_dir, name)

    # Policy / algorithm
    algo = pol.get("algorithm", "ppo")
    allow_ac_share = pol.get("allow_ac_share", False) and algo == "ppo"

    # Build objects
    device_space = build_device_space(cfg)
    suite = build_suite(cfg)
    workload = build_workload(cfg)
    hw = build_hardware(cfg)
    K_eval = o["k_eval"]
    optim = build_optimizer_config(cfg, allow_ac_share, algorithm=algo)

    # Time-based budget (--time)
    # Sets a wall-clock deadline on the SA optimization loop. E_max/E_struct from the configuration
    # still apply as safety caps; the deadline is an additional termination criterion (whichever
    # fires first wins).
    if args.time is not None:
        from search.inter_model import Iter_inter_model_candidates
        n_structs = sum(1 for _ in Iter_inter_model_candidates(
            suite, workload, optim.T_cand, algorithm=optim.algorithm))
        n_structs = max(n_structs, 1)

        # Deadline is set just before optimize() is called (see below).
        # Store the user's time budget and struct count for later.
        _time_budget = args.time
        _n_structs = n_structs
        print(f"\n  Time budget: {args.time:.0f}s (SA loop wall-clock limit)")
        print(f"    structural settings: {n_structs}")
        print(f"    time per struct: {args.time / n_structs:.1f}s")

    tau = {}
    if optim.T_cand:
        for m, s in optim.T_cand.items():
            if s: # non-empty set
                tau[m] = next(iter(s))
    trainable = {Model.ACTOR} if algo in ("grpo", "dpo") else {Model.ACTOR, Model.CRITIC}

    cost_model = NodeCostModel(hw, suite, workload)
    mem_model = NodeMemoryModel(
        hw,
        suite,
        workload,
        trainable = frozenset(trainable),
        shadow_models = frozenset(m for m in trainable if tau.get(m, 0) > 0),
    )

    # Logging
    log_file = out.get("log_file")
    init_logging(log_file)

    # Pass the full staleness candidate sets so the header shows the search budget per model.
    t_cand_display = optim.T_cand if optim.T_cand is not None else DEFAULT_T_CAND
    print_run_header(suite, workload, hw, device_space, optim, t_cand_display, trainable, K_eval)

    # Build baseline RPEG and plans
    if algo == "dpo":
        base_rpeg = build_DPO_RPEG(suite, workload, tau)
    elif algo == "grpo":
        base_rpeg = build_GRPO_RPEG(suite, workload, tau)
    else:
        base_rpeg = build_PPO_RPEG(suite, workload, tau)

    # Synchronous baseline (for reporting only)
    # The baseline Gantt and speedup are always measured against synchronous execution (tau=0 for
    # all models), regardless of the user's tau setting. This is a synthetic reference point — it is
    # not part of the search space.
    tau_sync = {m: 0 for m in tau}
    if algo == "dpo":
        sync_rpeg = build_DPO_RPEG(suite, workload, tau_sync)
    elif algo == "grpo":
        sync_rpeg = build_GRPO_RPEG(suite, workload, tau_sync)
    else:
        sync_rpeg = build_PPO_RPEG(suite, workload, tau_sync)

    conc_plan, baseline_feasible = build_concurrent_baseline_plan(
        sync_rpeg, device_space, suite, workload, cost_model, mem_model
    )

    # Greedy baseline: per-vertex greedy (fastest feasible strategy each, always feasible).
    greedy_plan = build_valid_baseline_plan(
        sync_rpeg, device_space, suite, workload, cost_model, mem_model
    )

    # Baseline lowering uses the sync RPEG (for Gantt and evaluation)
    sync_unrolled = unroll(sync_rpeg, K_eval, verbose=False)
    lowered = lower(
        sync_rpeg, sync_unrolled, conc_plan, suite, K_eval, workload=workload, verbose=False
    )

    # Optimizer seed
    # Default: seed from the concurrent baseline.
    # --seed-plan: seed from a previously saved plan (for two-stage optimization).
    if args.seed_plan:
        from ir.serialize import load_plan
        base_plan = load_plan(args.seed_plan, base_rpeg)
        print(f"  Seeded optimizer from: {args.seed_plan}")
    else:
        base_plan = conc_plan # seed strategy/placement from baseline

    if debug:
        table_dbg, _ = build_candidate_directory(base_rpeg, suite, device_space, workload, cost_model, mem_model)
        print("\n  Candidate directory (baseline):")
        for v in base_rpeg.vertices:
            print(f"    {v.id}: {len(table_dbg[v])} candidates")

    if plots:
        base_rpeg_dot = plot_path("base_rpeg.dot")
        base_unrolled_dot = plot_path("base_unrolled.dot")
        base_exec_dot = plot_path("base_executable.dot")

        rpeg_to_dot(sync_rpeg, base_rpeg_dot)
        unrolled_to_dot(sync_unrolled, base_unrolled_dot, rpeg=sync_rpeg)
        exec_to_dot(lowered, base_exec_dot)

        dot_to_img([
            (base_rpeg_dot, plot_path("rpeg.pdf")),
            (base_unrolled_dot, plot_path("unrolled.pdf")),
            (base_exec_dot, plot_path("executable.pdf")),
        ])
        print("Generated baseline plots (PDF)")
        safe_unlink([base_rpeg_dot, base_unrolled_dot, base_exec_dot])

    op_nodes = [n for n in lowered.nodes if n.kind == ExecNodeType.OP]

    # Baseline cost model (debug only)
    if debug:
        print("\n  Cost model (baseline):")
        for n in op_nodes:
            t = cost_model.time(n)
            print(f"    {n.id:30s}  time={t:.3e} s")

    # Baseline memory model (debug only)
    if debug:
        cap = mem_model._eff_capacity()
        peaks = _device_peaks(lowered, mem_model)
        print(f"\n  Memory model (baseline):")
        for d in sorted(peaks):
            ok = peaks[d] <= cap
            print(f"    device {d}  peak={peaks[d]/1e9:.2f} GB / {cap/1e9:.2f} GB  {'OK' if ok else 'OOM'}")

    # Baseline schedule
    trace = schedule(executable=lowered, cost_model=cost_model, device_space=device_space)
    base_makespan = trace.makespan()

    if debug:
        print(f"\n  Schedule (baseline, sorted by start):")
        print(f"    {'node':30s}  {'kind':8s}  {'start':>12s}  {'finish':>12s}  {'dur':>12s}")
        print(f"    {'─'*30}  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*12}")
        for v in sorted(trace.start.keys(), key=lambda v: trace.start[v]):
            s, f = trace.start[v], trace.finish[v]
            print(f"    {v.id:30s}  {v.kind.value:8s}  {s:12.3e}  {f:12.3e}  {f-s:12.3e}")

    if plots:
        trace_to_gantt(lowered, trace, device_space, plot_path("base_gantt.pdf"),
                       title="Initial Schedule (Baseline)", t_max=base_makespan)
        print("Generated base_gantt.pdf")

    # Greedy baseline Gantt
    greedy_lowered = lower(sync_rpeg, sync_unrolled, greedy_plan, suite, K_eval, workload=workload, verbose=False)
    greedy_trace_obj = schedule(executable=greedy_lowered, cost_model=cost_model, device_space=device_space)
    if plots:
        trace_to_gantt(
            greedy_lowered,
            greedy_trace_obj,
            device_space,
            plot_path("greedy_gantt.pdf"),
            title = "Greedy Baseline",
            t_max = greedy_trace_obj.makespan()
        )
        print("Generated greedy_gantt.pdf")

    # Evaluate baselines
    W_step = workload.B * workload.K * (workload.S + workload.T)
    # Baseline configurations use sync RPEG (tau=0) for reporting.
    conc_config = Configuration(sync_rpeg, conc_plan, device_space)
    greedy_config = Configuration(sync_rpeg, greedy_plan, device_space)
    base_config = Configuration(sync_rpeg, base_plan, device_space)
    cache = EvalCache()
    greedy_score, greedy_trace, _, _ = evaluate_cached(greedy_config, cache, suite, workload, hw, K_eval)
    conc_score, conc_trace, _, _ = evaluate_cached(conc_config, cache, suite, workload, hw, K_eval)
    base_score, base_trace, _, _ = evaluate_cached(base_config, cache, suite, workload, hw, K_eval)

    # Baseline plan
    if base_plan_file:
        save_plan(base_config, base_plan_file)

    # Optimizer seed: uses the user's async RPEG (not the sync baseline)
    opt_seed_config = Configuration(base_rpeg, base_plan, device_space)

    # Search log legend + header
    print(f"\n{'─'*90}")
    print(f"  Simulated Annealing Search")
    print(f"{'─'*90}")
    conc_makespan  = conc_trace.makespan() if conc_trace else float('nan')
    print(f"  Concurrent heuristic: {conc_score:.4e} tokens/s  |  makespan: {conc_makespan:.4e} s")
    if base_score == -float('inf'):
        print(f"  Baseline is memory-infeasible — searching from scratch")
    print(f"")
    print(f"  Iteration log format:")
    print(f"    s=structure  r=restart  i=iter | vertex  move_type | curr ↑/↓ proposed  α=accept_prob  T=temp | ACC/REJ  E=evals E_s=struct_evals")
    print(f"    ↑ = proposed is better   ↓ = proposed is worse   ACC/REJ = Metropolis-Hastings accept/reject")
    print(f"    ★ = new global best")
    print(f"{'─'*90}")

    # Set wall-clock deadline if --time was specified
    if args.time is not None:
        optim.deadline = time.perf_counter() + _time_budget
        optim.E_max = 10**9 # unlimited — wall-clock deadline is the binding constraint
        optim.E_struct = 10**9

    t_optimize_start = time.perf_counter()
    opt_config = optimize(opt_seed_config, suite, workload, hw, K_eval, optim, cache)
    t_optimize_wall = time.perf_counter() - t_optimize_start

    # Polish phase (skipped if A7 ablation)
    if optim.no_polish:
        print(f"\n  Polish: skipped (A7 ablation)")
    else:
        _, _, _pol_feasible, _ = evaluate_cached(opt_config, cache, suite, workload, hw, K_eval)
        if _pol_feasible:
            print(f"\n  Polish phase:")
            _pol_ac_share = Model.AC_SHARE in opt_config.rpeg.trainable
            _pol_table, _pol_index = build_candidate_directory(opt_config.rpeg, suite, device_space, workload, cost_model, mem_model)
            opt_config = polish(opt_config, suite, workload, hw, K_eval, optim, cache, _pol_table, _pol_index, ac_share=_pol_ac_share)
        else:
            print(f"\n  Polish: skipped (infeasible config)")

    # Build optimized executable
    opt_unrolled = unroll(opt_config.rpeg, K_eval, verbose=False)
    opt_lowered = lower(opt_config.rpeg, opt_unrolled, opt_config.plan, suite, K_eval, workload=workload, verbose=False)

    if plots:
        opt_rpeg_dot = plot_path("opt_rpeg.dot")
        opt_unrolled_dot = plot_path("opt_unrolled.dot")
        opt_exec_dot = plot_path("opt_executable.dot")

        rpeg_to_dot(opt_config.rpeg, opt_rpeg_dot)
        unrolled_to_dot(opt_unrolled, opt_unrolled_dot, rpeg=opt_config.rpeg)
        exec_to_dot(opt_lowered, opt_exec_dot)

        dot_to_img([
            (opt_rpeg_dot, plot_path("opt_rpeg.pdf")),
            (opt_unrolled_dot, plot_path("opt_unrolled.pdf")),
            (opt_exec_dot, plot_path("opt_executable.pdf")),
        ])
        print("Generated optimized plots (PDF)")
        safe_unlink([opt_rpeg_dot, opt_unrolled_dot, opt_exec_dot])

    opt_score, opt_trace, _, _ = evaluate_cached(opt_config, cache, suite, workload, hw, K_eval)
    opt_ac_share = Model.AC_SHARE in opt_config.rpeg.trainable
    opt_tau = opt_config.rpeg.tau

    # Optimal memory model (debug only)
    if debug:
        cap = mem_model._eff_capacity()
        peaks = _device_peaks(opt_lowered, mem_model)
        print(f"\n  Memory model (optimal):")
        for d in sorted(peaks):
            ok = peaks[d] <= cap
            print(f"    device {d}  peak={peaks[d]/1e9:.2f} GB / {cap/1e9:.2f} GB  {'OK' if ok else 'OOM'}")

    # Optimal schedule (debug only)
    if debug and opt_trace is not None:
        print(f"\n  Schedule (optimal, sorted by start):")
        hdr = f"    {'node':30s}  {'kind':8s}  {'model':5s}  {'start':>12s}  {'finish':>12s}  {'dur':>12s}"
        print(hdr)
        print(f"    {'─'*30}  {'─'*8}  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*12}")
        for v in sorted(opt_trace.start.keys(), key=lambda v: opt_trace.start[v]):
            s, f = opt_trace.start[v], opt_trace.finish[v]
            m = v.model.value if v.model else (v.op.template_vertex.kind.model.value if v.op else "-")
            print(f"    {v.id:30s}  {v.kind.value:8s}  {m:5s}  {s:12.3e}  {f:12.3e}  {f-s:12.3e}")

    if plots and opt_trace is not None:
        trace_to_gantt(
            opt_lowered,
            opt_trace,
            device_space,
            plot_path("opt_gantt.pdf"),
            title = "Optimized Schedule (Output)",
            t_max = base_makespan
        )
        print("Generated opt_gantt.pdf")

    # FOOTER — all final stats in one clean block
    opt_makespan = opt_trace.makespan() if opt_trace else None
    improvement = ((opt_score - base_score) / abs(base_score) * 100
                   if base_score not in (-float('inf'), 0.0) else None)

    tau_str = ", ".join(f"{m.value}:{opt_tau[m]}" for m in opt_tau) or "none"

    print(f"\n{'═'*90}")
    print(f"  Results")
    print(f"{'═'*90}")

    # Search statistics
    print(f"\n  Search statistics")
    print(f"    wall time   {t_optimize_wall:.1f}s")
    print(f"    cache       {cache.stats()}")

    # Structure
    print(f"\n  Structure")
    print(f"    ac_share    {opt_ac_share}")
    print(f"    tau         {{{tau_str}}}")

    # Per-vertex plan comparison
    print(f"\n  Per-vertex plan (base → optimal):")
    base_vids = {v.id for v in base_rpeg.vertices}
    opt_vids = {v.id for v in opt_config.rpeg.vertices}
    if base_vids != opt_vids:
        print(f"    Structural change: {sorted(base_vids)} → {sorted(opt_vids)}")
    else:
        print(f"    {'':2s}  {'vertex':6s}  {'parallelism (dp,pp,tp)':22s}  {'mb':>5s}  {'placement (rack, nodes, gpus)'}")
        print(f"    {'':2s}  {'─'*6}  {'─'*22}  {'─'*5}  {'─'*42}")
        for v in sorted(base_rpeg.vertices, key=lambda v: v.id):
            b_pi = base_config.plan.pi[v]
            o_pi = opt_config.plan.pi[v]
            b_mb = base_config.plan.mb[v]
            o_mb = opt_config.plan.mb[v]
            b_sigma = base_config.plan.sigma[v]
            o_sigma = opt_config.plan.sigma[v]
            changed = b_pi != o_pi or b_mb != o_mb or b_sigma != o_sigma
            tag     = " *" if changed else "  "
            pi_str  = f"{_fmt_pi(b_pi)} -> {_fmt_pi(o_pi)}" if b_pi != o_pi else _fmt_pi(b_pi)
            mb_str  = f"{b_mb}->{o_mb}" if b_mb != o_mb else str(b_mb)
            pl_str  = (f"{_fmt_alloc(b_sigma)}  ->  {_fmt_alloc(o_sigma)}"
                       if b_sigma != o_sigma else _fmt_alloc(b_sigma))
            print(f"    {tag}  {v.id:6s}  {pi_str:22s}  {mb_str:>5s}  {pl_str}")

    # Metrics
    def _pct(a: float, b: float) -> str:
        """
        Returns a string of the improvements of a over b in percent.

        :param a: Optimized value.
        :type a: float
        :param b: Baseline value.
        :type b: float
        :return: String with the improvements of a over b in percent.
        :rtype: str
        """
        return f"{(a - b) / abs(b) * 100:+.1f}%" if b not in (-float('inf'), 0.0) else "N/A"

    # Total FLOP per step (sum of work_cost across all template vertices)
    # Use the optimized RPEG's vertices (may differ from base if structural change).
    # Compute for both base and opt configs to handle structural differences.
    from passes.lower import ExecNodeType as _ENT
    def _total_flops_per_step(
        rpeg: RPEG,
        plan: ExecutionPlan,
        suite_obj: ModelSuite,
    ) -> float:
        """
        Sum _work_cost for each template vertex (= one step's total FLOP).

        :param rpeg: RLM Pipeline Execution Graph.
        :type rpeg: RPEG
        :param plan: Execution plan.
        :type plan: ExecutionPlan
        :param suite_obj: Model hyperparameters.
        :type suite_obj: ModelSuite
        :return: Total work of a step in FLOP.
        :rtype: float
        """
        from model.cost import NodeCostModel as _NCM
        from passes.unroll import UnrolledNode
        total = 0.0
        for v in rpeg.vertices:
            # Build a mock node to query _work_cost
            dp, pp, tp = plan.pi[v]
            alloc = plan.sigma[v]
            mock_op = UnrolledNode(template_vertex=v, k=0)
            mock_node = ExecNode(
                id = "_flops",
                kind = _ENT.OP,
                allocs = (alloc,),
                op = mock_op,
                model = v.kind.model,
                pi = (dp, pp, tp),
                mb = plan.mb[v]
            )
            total += cost_model._work_cost(mock_node)
        return total

    from passes.lower import ExecNode
    flops_per_step = _total_flops_per_step(opt_config.rpeg, opt_config.plan, suite)
    conc_flops = _total_flops_per_step(base_rpeg, conc_plan, suite)

    conc_pflops = conc_flops / (conc_makespan * 1e15) if conc_makespan > 0 else 0
    opt_pflops = flops_per_step / (opt_makespan * 1e15) if opt_makespan and opt_makespan > 0 else 0

    # Greedy baseline metrics
    greedy_makespan = greedy_trace.makespan() if greedy_trace else float('nan')
    greedy_flops = _total_flops_per_step(sync_rpeg, greedy_plan, suite)
    greedy_pflops = greedy_flops / (greedy_makespan * 1e15) if greedy_makespan > 0 else 0
    greedy_util = greedy_trace.utilization() if greedy_trace else 0
    greedy_nactive = greedy_trace.n_active() if greedy_trace else 0

    # Concurrent heuristic metrics
    conc_util = conc_trace.utilization() if conc_trace else 0
    conc_nactive = conc_trace.n_active() if conc_trace else 0
    opt_util = opt_trace.utilization() if opt_trace else 0
    opt_nactive = opt_trace.n_active() if opt_trace else 0

    print(f"\n  {'Metric':<28s}  {'Greedy':>16s}  {'Conc. Heuristic':>16s}  {'Optimized':>16s}  {'Δ (vs greedy)':>14s}")
    print(f"  {'─'*28}  {'─'*16}  {'─'*16}  {'─'*16}  {'─'*14}")
    print(f"  {'Throughput (tokens/s)':<28s}  {greedy_score:>16.4e}  {conc_score:>16.4e}  {opt_score:>16.4e}  {_pct(opt_score, greedy_score) if greedy_score > 0 else 'N/A':>14s}")
    print(f"  {'Makespan (s)':<28s}  {greedy_makespan:>16.4e}  {conc_makespan:>16.4e}  {opt_makespan if opt_makespan else float('nan'):>16.4e}  {''}")
    print(f"  {'Throughput (PFLOP/s)':<28s}  {greedy_pflops:>16.4f}  {conc_pflops:>16.4f}  {opt_pflops:>16.4f}  {_pct(opt_pflops, greedy_pflops) if greedy_pflops > 0 else 'N/A':>14s}")
    print(f"  {'GPU utilization':<28s}  {greedy_util*100:>15.1f}%  {conc_util*100:>15.1f}%  {opt_util*100:>15.1f}%  {''}")
    print(f"  {'Active GPUs':<28s}  {greedy_nactive:>16d}  {conc_nactive:>16d}  {opt_nactive:>16d}  {''}")

    # Machine-readable baseline status for plot scripts
    print(f"  baseline_feasible: {baseline_feasible}")

    # Warn if baseline is OOM (throughput speed-up is inflated)
    if not baseline_feasible:
        print(f"\n  ⚠ WARNING: Baseline is memory-infeasible (OOM fallback). Throughput speed-up is")
        print(f"    inflated and should NOT be used for comparison. Use PFLOP/s gain instead.")

    print(f"\n{'═'*90}")

    # ── optimized plan ───────────────────────────────────────────────
    if opt_plan_file:
        save_plan(opt_config, opt_plan_file)


if __name__ == "__main__":
    main()
