# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Authors: Leonard Schmidt
#          Pierre Pang

import math
import random

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Set, Tuple

from core.types import packing_geometry, Allocation
from ir.plan import ExecutionPlan
from ir.rpeg import RPEG
from model.cost import NodeCostModel
from model.hardware import Hardware
from model.memory import MemoryPolicy, NodeMemoryModel
from model.models import ModelSuite, Model
from model.workload import Workload
from passes.unroll import unroll
from passes.lower import lower
from search.cache import EvalCache
from search.evaluate import evaluation_cache_key, evaluate_cached, is_valid, _eval_worker, Configuration
from search.inter_model import Iter_inter_model_candidates, DEFAULT_T_CAND
from search.moves import build_candidate_directory, move_gen, multi_move_pair, sample_expl, MoveStats


@dataclass
class OptimizerConfig:
    """
    Data class representing the configuration of the optimizer.
    """
    K_x:      int = 6     # candidates per focused vertex
    T_max:    int = 50    # attempt limit for sample_expl
    E_max:    int = 5000  # global evaluation budget (cache misses)
    E_struct: int = 400   # per-structural-setting evaluation budget (cache misses)
    T_cand:   Optional[Dict[Model, Set[int]]] = None  # staleness candidates per model; None = DEFAULT_T_CAND
    # Restrict the inter-model structures considered by the optimizer. The default preserves the
    # original search space; fixed-ablation studies can use one or more of: separate, fused, ac_share.
    structural_candidates: Optional[FrozenSet[str]] = None
    lambda_ac:    float = 0.15   # AC-share penalty
    score_eps:    float = 1e-4   # ε for global-best / struct-best guard (strict improvement threshold)
    # Metropolis-Hastings Simulated Annealing (MH-SA) parameters
    # Temperature is dimensionless (fraction of current score magnitude). At T=0.20 a move that is
    # 20% worse is accepted with prob exp(-1) ≈ 0.37. Temperature resets to T_init at every restart
    # (independent sampling chains). Cooling: T_cool is auto-derived, so that the chain reaches
    # T_min at ~70% of I_max iterations, leaving ~30% for near-greedy exploitation at T_min.
    T_init:  float = 0.30   # initial temperature for cold/random restarts
    T_warm:  float = 0.05   # initial temperature for warm restarts (exploit near elite)
    T_min:   float = 0.01   # temperature floor (near-greedy exploitation)
    K_elite: int   = 3      # elite restart set: keep top-K distinct feasible plans per family.
    # Multi-vertex proposal mixture weights (maximum, attained at T=T_init). Actual weight at
    # temperature T is alpha * (T / T_init).
    alpha_pair:      float = 0.28  # donor-receiver kernel: coordinated 2-vertex resource transfer
    # Stagnation-triggered restart: restart after stag_window consecutive rejected proposals.
    # This is the primary chain-termination criterion; I_max is not a cap. At T_min=0.01 a truly
    # stuck chain accumulates 20 consecutive rejections quickly; a chain still finding improvements
    # rarely goes 20 iterations without any accept.
    stag_window:     int   = 20
    # Utilization-aware efficiency objective (optional). When max_throughput_loss (delta) > 0, the
    # optimizer maximises throughput × utilization, subject to throughput ≥ (1 - delta) × best.
    # delta=0.0 (default) → pure throughput. delta=0.10 → accept ≤10% slowdown.
    max_throughput_loss: float = 0.0
    # Auto-derived (not user-settable)
    # I_max: calibrates the T_cool schedule. Estimated from E_struct and K_x: each SA iteration
    #   evaluates ~(K_x+1)/2 candidates on average, so estimated_iters = E_struct / ((K_x+1)/2).
    #   I_max = 50% of that.
    # R_max: safety cap on restarts per structural setting. Never reached in practice since
    #   E_max/E_struct drive termination.
    I_max:    int   = 0      # 0 = auto-derive in __post_init__
    R_max:    int   = 10000
    T_cool:   float = 0.0    # auto-derived in __post_init__
    algorithm: str  = "ppo"  # "ppo", "dpo" or "grpo"
    # Wall-clock deadline (optional). When set, the optimizer terminates when time.perf_counter() >=
    # deadline, regardless of E_max. Set by pipeline.py --time flag. None = use E_max only.
    deadline: Optional[float] = None
    # Per-structural-setting time share. Auto-derived from deadline / n_structs.
    deadline_per_struct: Optional[float] = None
    # Ablation flags (configuration-driven, default=off)
    fixed_mb: bool = False          # A3: use ReaL's fixed mb heuristic
    uniform_vertex: bool = False    # A6: uniform vertex selection (no critical-path bias)
    no_polish: bool = False         # A7: skip polish phase
    hard_oom: bool = False          # A8: hard OOM rejection (no soft penalty)
    geometric_cooling: bool = False # A9: optional geometric cooling instead of constant temperature
    # Runtime settings
    memory_optimizer_stage: int = 2
    memory_activation_checkpointing: bool = True
    memory_runtime_overhead_fraction: float = 0.0
    memory_worker_overhead_bytes: float = 0.0
    memory_semantic_activation_cache_bytes: float = 0.0
    memory_max_worker_groups_per_gpu: int = 0

    def __post_init__(self) -> None:
        """
        Post-initialization of the dataclass to partially validate the parameters, initalize the
        memory policy, and to derive I_max and T_cool automatically.

        :raises ValueError: If one of the parameters has an invalid setting or are not set.
        """
        allowed_structures = frozenset({"separate", "fused", "ac_share"})
        if self.structural_candidates is not None:
            unknown = self.structural_candidates - allowed_structures
            if unknown:
                raise ValueError(
                    "unknown structural_candidates: " + ", ".join(sorted(unknown))
                )
            if not self.structural_candidates:
                raise ValueError("structural_candidates must not be empty")
        MemoryPolicy(
            optimizer_stage = self.memory_optimizer_stage,
            activation_checkpointing = self.memory_activation_checkpointing,
            runtime_overhead_fraction = self.memory_runtime_overhead_fraction,
            worker_overhead_bytes = self.memory_worker_overhead_bytes,
            semantic_activation_cache_bytes = self.memory_semantic_activation_cache_bytes,
            max_worker_groups_per_gpu = self.memory_max_worker_groups_per_gpu
        )

        # Derive automatically I_max from E_struct and K_x if not explicitly set.
        if self.I_max <= 0:
            estimated_iters = self.E_struct / ((self.K_x + 1) / 2)
            self.I_max = int(estimated_iters * 0.5)
        # Compute automatically T_cool, so the chain reaches T_min at 70% of I_max iterations.
        # T_cool = (T_min/T_init)^(1 / (0.7 * I_max))
        self.T_cool = (self.T_min / self.T_init) ** (1.0 / (0.7 * self.I_max))


# Quality constraints
def _ac_penalty(score: float, lambda_ac: float) -> float:
    """
    Apply actor-critc share penalty based on whether it is enabled.

    :param score: Score to potentially penalize.
    :type score: float
    :param lambda_ac: Penalty factor.
    :type lambda_ac: float
    :return: Penalized score.
    :rtype: float
    """
    if lambda_ac == 0.0:
        return score
    return score * (1.0 - lambda_ac)


def penalized_score(raw_score: float, rpeg: RPEG, opt_config: 'OptimizerConfig', ac_share: bool) -> float:
    """
    Return the penalized score the optimizer uses internally for ranking configs.

    raw_score is the throughput returned by evaluate_cached (no penalties applied). Currently only
    the AC-share penalty is active; staleness penalty was removed because the user constrains tau
    directly via staleness_candidates.

    :param raw_score: Original throughput score.
    :type raw_score: float
    :param rpeg: RLM Pipeline Execution Graph.
    :type rpeg: RPEG
    :param opt_config: Optimizer configuration.
    :type opt_config: 'OptimizerConfig'
    :param ac_share: Flag to indicate, whether the AC-share penalty should be applied.
    :type ac_share: bool
    :return: Penalized score.
    :rtype: float
    """
    s = raw_score
    if ac_share:
        s = _ac_penalty(s, opt_config.lambda_ac)
    return s


def _plan_fp(plan: ExecutionPlan) -> Tuple:
    """
    Canonical fingerprint of an execution plan for elite-set deduplication.

    Two plans are considered identical if every vertex has the same (dp, pp, tp) strategy, the same
    placement origin, and the same microbatch count.  All three components affect the raw
    throughput, so differing in any one of them yields a distinct plan.

    :param plan: Execution plan.
    :type plan: ExecutionPlan
    :return: Sorted tuple of vertex ID, 3D strategy, placement origin and microbatch count for each
             vertex.
    :rtype: Tuple
    """
    verts = sorted(plan.pi.keys(), key=lambda v: v.id)
    return tuple(
        (v.id, plan.pi[v], plan.sigma[v].origin if v in plan.sigma else None, plan.mb.get(v))
        for v in verts
    )


def _elite_insert(
    family: Tuple, raw: float, plan: ExecutionPlan, elite: Dict, seen: Dict, K: int
) -> None:
    """
    Insert (raw, plan) into the per-family elite list if it qualifies (score among the best top-K
    seen). The elite list is kept sorted descending by raw throughput and capped at K entries.

    :param family: Identifier for the elite family.
    :type family: Tuple
    :param raw: Raw throughtput score.
    :type raw: float
    :param plan: Execution plan.
    :type plan: ExecutionPlan
    :param elite: Per-family elite list.
    :type elite: Dict
    :param seen: Plans in the elite lists.
    :type seen: Dict
    :param K: Number of entries for top-K.
    :type K: int
    """
    fp = _plan_fp(plan)
    fp_set = seen.setdefault(family, set())
    if fp in fp_set:
        return # already tracked; don't insert again even if score differs
    
    lst = elite.get(family, [])
    if len(lst) >= K and raw <= lst[-1][0]:
        return # not good enough to enter the elite set
    elite[family] =  sorted(lst + [(raw, plan)], key=lambda x: -x[0])[:K] # insert to elite list and sort
    fp_set.add(fp)


def _sample_all_vertices(
    config: Configuration, suite: ModelSuite, workload: Workload
) -> ExecutionPlan:
    """
    Assigns each TemplateVertex a new placement, 3D-parallel strategy, and microbatch count, which
    generates a new execution plan.

    :param config: Configuration (RPEG, execution plan, available hardware).
    :type config: Configuration
    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param workload: Workload settings.
    :type workload: Workload
    :return: New execution plan.
    :rtype: ExecutionPlan
    :raises RuntimeError: If sampling of a new configuration for a vertex fails after 100 retries or
                          a compostie operator encounters a problematic setting.
    """
    new_sigma, new_pi, new_mb = {}, {}, {}
    vertices = config.rpeg.vertices
    # Process composite operators first, so that we can exclude them subsequently
    fused_pair = {}
    if config.intra_stage_fusion:
        fused_pair = {vertex.id: vertex for vertex in vertices if vertex.id in {"t_pi", "t_V"}}
        if set(fused_pair) != {"t_pi", "t_V"}:
            raise RuntimeError("intra-stage fusion requires t_pi and t_V")
        actor = fused_pair["t_pi"]
        critic = fused_pair["t_V"]
        for _ in range(100):
            fused_sample = sample_expl(actor, suite, config.device_space, workload)
            if fused_sample is None:
                continue
            alloc, (_dp, pp, _tp), _mb = fused_sample
            if suite.specs[critic.kind.model].L % pp == 0:
                break
        else:
            raise RuntimeError("failed to sample one actor/critic-compatible fused layout")
        for vertex in (actor, critic):
            new_sigma[vertex] = alloc
            new_pi[vertex] = fused_sample[1]
            new_mb[vertex] = fused_sample[2]
    for v in vertices:
        if v in fused_pair.values():
            continue
        for _ in range(100):  # max retries per vertex
            x_sample = sample_expl(v, suite, config.device_space, workload)  
            if x_sample is not None:
                break
        else:
            raise RuntimeError(f"Random restart failed due to sample_expl failing after 100 retries for vertex {v.id}")

        alloc, (dp, pp, tp), mb = x_sample  # memory feasible plus all constraints hold
        new_sigma[v] = alloc
        new_pi[v] = (dp, pp, tp)
        new_mb[v] = mb

    return ExecutionPlan(new_sigma, new_pi, new_mb)


def _restart(
    config: Configuration,
    suite: ModelSuite,
    workload: Workload,
    mem_model: NodeMemoryModel,
    max_attempts: int = 50
) -> Optional[Configuration]:
    """
    Samples a fully new plan completely randomly for each vertex. Only used if placement_restart and
    warm_restart fail.

    :param config: Original configuration (RPEG, execution plan, available hardware).
    :type config: Configuration
    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param workload: Workload settings.
    :type workload: Workload
    :param mem_model: Memory model for the graph nodes.
    :type mem_model: NodeMemoryModel
    :param max_attempts: Maximum number of attempts. Defaults to 50.
    :type max_attempts: int
    :return: New execution plan or None, if no valid configuration is found.
    :rtype: Optional[Configuration]
    """
    for _ in range(max_attempts):
        new_plan = _sample_all_vertices(config, suite, workload)
        new_config = Configuration(
            config.rpeg,
            new_plan,
            config.device_space,
            intra_stage_fusion = config.intra_stage_fusion,
            transfer_resource_policy = config.transfer_resource_policy
        )
        if is_valid(new_config, suite, workload, mem_model):
            return new_config
    return None # caller must handle None (skip this restart)


def _warm_restart(
        rpeg: RPEG,
        warm_plan: Optional[ExecutionPlan],
        config: Configuration,
        suite: ModelSuite,
        workload: Workload,
        mem_model: NodeMemoryModel
) -> Tuple[Configuration, str]:
    """
    Seed a warm restart from elite[0] — the best feasible plan found so far for this family.

    :param rpeg: RLM Pipeline Execution Graph.
    :type rpeg: RPEG
    :param warm_plan: Best feasible plan for this family.
    :type warm_plan: Optional[ExecutionPlan]
    :param config: Configuraton, used only for the available hardware and as fallback seed.
    :type config: Configuration
    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param workload: Workload settings.
    :type workload: Workload
    :param mem_model: Memory model for the graph nodes.
    :type mem_model: NodeMemoryModel
    :return: Tuple of a configuration with best feasible plan and a reason.
    :rtype: Tuple[Configuration, str]
    """
    if warm_plan is not None:
        warm = Configuration(
            rpeg,
            warm_plan,
            config.device_space,
            intra_stage_fusion = config.intra_stage_fusion,
            transfer_resource_policy = config.transfer_resource_policy
        )
        if is_valid(warm, suite, workload, mem_model):
            return warm, "warm (best feasible)"
        
    restarted = _restart(config, suite, workload, mem_model) # fallback to random restart 
    reason = "random (no feasible plan)" if warm_plan is None else "random (warm invalid)"
    return restarted, reason


def optimize(
    base_config: Configuration,
    suite: ModelSuite,
    workload: Workload,
    hw: Hardware,
    K_eval: int,
    opt_config: OptimizerConfig,
    cache: EvalCache,
) -> Configuration:
    """
    Optimize a configuration.

    Outer loop over (ac_share, tau), inner loop = Metropolis-Hastings Simulated Annealing (MH-SA)
    coordinate search. Each restart is an independent MH-SA chain. Temperature T_i decays
    geometrically within a restart and resets to T_init at each restart (multi-chain SA). This lets
    the chain escape locally-colocated optima and discover distributed placement strategies by
    accepting temporarily worse single-vertex moves that unlock true inter-vertex concurrency.

    :param base_config: Original configuration to optimize.
    :type base_config: Configuration
    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param workload: Workload settings.
    :type workload: Workload
    :param hw: Hardware settings.
    :type hw: Hardware
    :param K_eval: Number of iterations/steps to unroll.
    :type K_eval: int
    :param opt_config: Optimizer configuration.
    :type opt_config: OptimizerConfig
    :param cache: Evaluation cache of traces. Shared cache, passed in so caller can inspect it.
    :type cache: EvalCache
    :return: Optimized configuration.
    :rtype: Configuration
    :raises ValueError: If no structural settings can be found.
    """
    eps_global = opt_config.score_eps # strict epsilon for global-best / struct-best guard
    eps_noise = 1e-6 # below this is floating-point noise → skip MH

    # Ablation flags: propagate to module-level toggles before forking workers so that subprocess
    # workers inherit the same values via fork.
    import search.evaluate as _eval_mod
    import search.moves as _moves_mod
    _moves_mod.ABLATION_FIXED_MB = opt_config.fixed_mb
    _moves_mod.ABLATION_UNIFORM_VERTEX = opt_config.uniform_vertex
    _eval_mod.ABLATION_HARD_OOM = opt_config.hard_oom

    # Parallel candidate evaluation: pre-fork K_x workers. Each SA iteration evaluates up to K_x
    # candidates independently; dispatching cache misses to the pool gives ~3.5× speedup.
    import multiprocessing as _mp
    import os as _os
    _n_workers = min(opt_config.K_x, _os.cpu_count() or 1)
    _pool = _mp.Pool(processes=_n_workers) if _n_workers > 1 else None

    def _stop_eval_pool() -> None:
        """
        Synchronously reap candidate workers before returning from search.
        """
        nonlocal _pool
        if _pool is None:
            return
        # All dispatches use blocking map(), so no evaluation is outstanding at an optimizer return
        # point. In certain settings, Pool.terminate() can spin while its handler threads and forked
        # workers are idle; an orderly sentinel shutdown is both sufficient and reliable.
        _pool.close()
        _pool.join()
        _pool = None

    memory_policy = MemoryPolicy(
        optimizer_stage = opt_config.memory_optimizer_stage,
        activation_checkpointing = opt_config.memory_activation_checkpointing,
        runtime_overhead_fraction = opt_config.memory_runtime_overhead_fraction,
        worker_overhead_bytes = opt_config.memory_worker_overhead_bytes,
        semantic_activation_cache_bytes = opt_config.memory_semantic_activation_cache_bytes,
        max_worker_groups_per_gpu = opt_config.memory_max_worker_groups_per_gpu
    )
    # Build cost and memory models from the same runtime policy.
    cost_model = NodeCostModel(
        hw,
        suite,
        workload,
        optimizer_stage = memory_policy.optimizer_stage,
        activation_checkpointing = memory_policy.activation_checkpointing
    )
    mem_model = NodeMemoryModel.from_policy(
        hw,
        suite,
        workload,
        memory_policy,
        trainable = frozenset(base_config.rpeg.trainable),
        shadow_models = frozenset(
            m for m in base_config.rpeg.trainable if base_config.rpeg.tau.get(m, 0) > 0
        ),
    )

    # Efficiency objective helper
    _delta = opt_config.max_throughput_loss
    _best_raw = [0.0]  # mutable for closure

    def _eff(raw: float, trace: Trace) -> float:
        """
        Apply utilization weighting if efficiency mode is on.

        Score = raw × utilization above the throughput floor.
        The floor prevents degradation beyond delta; the utilization factor steers the SA toward
        plans with fewer idle gaps.

        :param raw: Throughput score.
        :type raw: float
        :param trace: Execution trace.
        :type trace: Trace
        :return: Weighted score.
        :rtype: float
        """
        if _delta <= 0.0 or trace is None:
            return raw
        if raw > _best_raw[0]:
            _best_raw[0] = raw
        floor = (1.0 - _delta) * _best_raw[0]
        if raw < floor:
            return raw * 1e-12  # hard penalty: below floor
        return raw * max(trace.utilization(), 1e-6)

    # Initialise: evaluate the base configuration
    eval_base, trace_base, feasible, _ = evaluate_cached(
        base_config, cache, suite, workload, hw, K_eval, memory_policy
    )
    best_score = _eff(eval_base, trace_base)

    best_config = base_config

    if feasible:
        best_feasible_config, best_feasible_score = base_config, best_score
    else:
        best_feasible_config, best_feasible_score = None, -float('inf')

    # Elite restart set: top-K distinct feasible plans per family, ranked by throughput. Warm
    # restarts always seed from elite[0] (the best known throughput plan). For feasible
    # configurations the memory penalty is 1.0, so the stored score equals raw throughput.
    # _elite = {
    #       False: [(score0, plan0), (score1, plan1), ...], (RPEG without weight sharing)
    #       True:  [(score0, plan0), ...],                  (AC-share PPO RPEG)
    # }
    # NOTE: tau is not used as a key because tau variants within the same ac_share family share the
    #   same plan space. While tau does change the raw-score, a good plan under one tau is still a
    #   structurally sound warm-start seed for another. Shared elites let every tau variant benefit
    #   from whatever the earlier ones discovered.
    _elite:      dict = {}
    # _elite_seen deduplicates by fingerprint, so that the same plan reached via different
    # structural settings only occupies one slot.
    _elite_seen: dict = {}
    if feasible:
        _elite_insert(
            (False, base_config.intra_stage_fusion),
            eval_base,
            base_config.plan,
            _elite,
            _elite_seen,
            opt_config.K_elite
        )

    E = 1
    print(f"  [init] base score = {best_score:.4e}  (E={E})")

    t_cand = opt_config.T_cand if opt_config.T_cand is not None else DEFAULT_T_CAND

    _table_vertices = None  
    table = None
    index = None

    _move_stats = MoveStats()
    _n_accepted_strategy  = 0
    _n_accepted_placement = 0
    _n_rejected = 0

    # Pre-collect structural settings to know count (for time-based splitting).
    _all_structs = list(Iter_inter_model_candidates(suite, workload, t_cand, algorithm=opt_config.algorithm))
    if opt_config.structural_candidates is not None:
        def _structure_name(inter_cfg) -> str:
            if inter_cfg.ac_share:
                return "ac_share"
            if inter_cfg.intra_stage_fusion:
                return "fused"
            return "separate"

        _all_structs = [
            item for item in _all_structs
            if _structure_name(item[1]) in opt_config.structural_candidates
        ]
    if not _all_structs:
        raise ValueError(
            "structural_candidates exclude every inter-model candidate for "
            f"algorithm={opt_config.algorithm}"
        )
    _n_structs = len(_all_structs)

    _s_idx = 0  # structural setting counter
    for rpeg, inter_cfg in _all_structs:
        label = (
            f"ac_share={inter_cfg.ac_share}  "
            f"intra_stage_fusion={inter_cfg.intra_stage_fusion}  "
            + "tau={"
            + ", ".join(f"{m.value}:{inter_cfg.tau[m]}" for m in inter_cfg.tau)
            + "}"
        )

        # Update per-struct deadline: split remaining time evenly among remaining structs.
        if opt_config.deadline is not None:
            import time as _time_mod2
            _remaining_time = opt_config.deadline - _time_mod2.perf_counter()
            _remaining_structs = _n_structs - _s_idx
            if _remaining_structs > 0 and _remaining_time > 0:
                opt_config.deadline_per_struct = _time_mod2.perf_counter() + _remaining_time / _remaining_structs
            else:
                opt_config.deadline_per_struct = _time_mod2.perf_counter()  # no time left

        # Rebuild the table only when the vertex set actually changes (i.e. PPO -> AC_share).
        if rpeg.vertices is not _table_vertices:
            table, index = build_candidate_directory(rpeg, suite, base_config.device_space, workload, cost_model, mem_model)
            _table_vertices = rpeg.vertices

        # Log the penalty multiplier so the reader can convert displayed scores to raw throughput
        penalty_mult = 1.0
        if inter_cfg.ac_share:
            penalty_mult *= (1.0 - opt_config.lambda_ac)
        if penalty_mult < 1.0:
            print(f"  ── s={_s_idx}  {label}  penalty={penalty_mult:.4f}")
        else:
            print(f"  ── s={_s_idx}  {label}")

        # Try feasible start; fall back to infeasible baseline if needed.
        base_candidate = Configuration(
            rpeg,
            base_config.plan,
            base_config.device_space,
            intra_stage_fusion = inter_cfg.intra_stage_fusion,
            transfer_resource_policy = base_config.transfer_resource_policy
        )
        if is_valid(base_candidate, suite, workload, mem_model):
            config = base_candidate
        else:
            bootstrap = Configuration(
                rpeg,
                ExecutionPlan({}, {}, {}),
                base_config.device_space,
                intra_stage_fusion = inter_cfg.intra_stage_fusion,
                transfer_resource_policy = base_config.transfer_resource_policy
            )
            config = _restart(bootstrap, suite, workload, mem_model)
            if config is None:
                plan_vertices = set(base_candidate.plan.sigma) & set(base_candidate.plan.pi) & set(base_candidate.plan.mb)
                if not set(rpeg.vertices).issubset(plan_vertices):
                    # A PPO plan cannot be evaluated against the structurally different AC-share
                    # RPEG. If random restart cannot build a complete plan, this structural family
                    # is infeasible; feeding the old plan to lower() would be a KeyError, not a
                    # meaningful soft-memory penalty.
                    print("  no complete structural plan found — skipping")
                    _s_idx += 1
                    continue
                # No feasible starting point found — start from infeasible baseline.
                # The soft OOM penalty will guide the SA toward feasibility.
                config = base_candidate
                print(f"  starting from infeasible baseline (soft OOM penalty will guide search)")

        E_s = 0 # evaluations used for this structural setting
        struct_best_score = -float('inf') # best penalized score seen in this structural setting
        struct_exhausted = False

        # Restart loop: primary termination is E_max / E_struct (budget). R_max is a safety cap only
        # to prevent infinite loops on degenerate configurations.
        # Restart strategy: r=0 cold (concurrent baseline),
        #   r>0   → warm restart from elite[0] (exploitation, T_warm).
        # Rationale: cold/random restarts waste 93% of budget finding nothing (measured over 108
        # scaling configs). Warm restarts hold the final best in 73% of runs — every restart should
        # refine the best known plan.
        r = 0
        no_progress_restarts = 0
        while r < opt_config.R_max:
            if struct_exhausted:
                break
            restart_E_s = E_s

            if r > 0:
                # All restart are warm: seed from elite[0] (best raw-score plan).
                family = (inter_cfg.ac_share, inter_cfg.intra_stage_fusion)
                current_structure_elites = _elite.get(family, [])
                warm_plan = current_structure_elites[0][1] if current_structure_elites else None
                restarted, _wkind = _warm_restart(rpeg, warm_plan, config, suite, workload, mem_model)
                kind = (f"warm (elite #1/{len(current_structure_elites)},  raw={current_structure_elites[0][0]:.4e})"
                        if warm_plan is not None and "warm" in _wkind else _wkind)

                print(f"  restart {r}: {kind}")
                if restarted is None:
                    print(f"  restart {r}: no feasible restart — skipping")
                    r += 1
                    no_progress_restarts += 1
                    if no_progress_restarts >= 2:
                        print("  structural search found no feasible restart twice — stopping")
                        struct_exhausted = True
                    continue

                config = restarted
                no_progress_restarts = 0

            eval_curr, trace_curr, feasible, miss = evaluate_cached(
                config, cache, suite, workload, hw, K_eval, memory_policy
            )
            score_curr = _eff(eval_curr, trace_curr)
            if inter_cfg.ac_share:
                score_curr = _ac_penalty(score_curr, opt_config.lambda_ac)
            if miss: E += 1; E_s += 1

            if feasible:
                # For feasible configs, eval_curr == raw throughput (memory penalty = 1.0).
                family = (inter_cfg.ac_share, inter_cfg.intra_stage_fusion)
                _elite_insert(family, eval_curr, config.plan, _elite, _elite_seen, opt_config.K_elite)

            print(f"  restart {r}: score={score_curr:.4e}  E={E}  E_s={E_s}")

            if score_curr > struct_best_score:
                struct_best_score = score_curr

            if score_curr > best_score * (1.0 + eps_global):
                best_config, best_score = config, score_curr
                print(f"  ★ new global best = {best_score:.4e}")

            if feasible and score_curr > best_feasible_score:
                best_feasible_config, best_feasible_score = config, score_curr

            # r=0 starts cold (T_init); all r>0 start warm (T_warm).
            T = opt_config.T_warm if r > 0 else opt_config.T_init
            executable_curr = None # lazily computed before first move_gen; reset on chain move
            _consec_rej = 0 # consecutive rejections (for stagnation detection)

            # Inner SA chain: runs until stagnation (primary) or budget exhaustion.
            # I_max is not a cap here — it only calibrates the T_cool schedule.
            import time as _time_mod
            for i in range(10**8): # effectively unbounded; stagnation drives termination
                _now = _time_mod.perf_counter()
                if opt_config.deadline is not None and _now >= opt_config.deadline:
                    _stop_eval_pool()
                    print(f"  time limit reached (E={E})")
                    print(f"\n  Move statistics")
                    print(_move_stats.summary(_n_accepted_strategy, _n_accepted_placement, _n_rejected))
                    return best_feasible_config if best_feasible_config is not None else best_config
                if E >= opt_config.E_max:
                    _stop_eval_pool()
                    print(f"  budget exhausted: E={E} >= E_max={opt_config.E_max}")
                    print(f"\n  Move statistics")
                    print(_move_stats.summary(_n_accepted_strategy, _n_accepted_placement, _n_rejected))
                    return best_feasible_config if best_feasible_config is not None else best_config
                if opt_config.deadline_per_struct is not None and _now >= opt_config.deadline_per_struct:
                    print(f"  struct time limit reached (E_s={E_s})")
                    struct_exhausted = True
                    break
                if E_s >= opt_config.E_struct:
                    print(f"  struct budget exhausted: E_s={E_s} >= E_struct={opt_config.E_struct}")
                    struct_exhausted = True
                    break

                # Kernel selection: temperature-dependent mixture
                frac = min(1.0, T / opt_config.T_init)
                p_pair = opt_config.alpha_pair * frac
                roll = random.random()

                if roll < p_pair:
                    # Donor-receiver proposal: coordinated 2-vertex resource transfer
                    _kernel_name = "pair"
                    neighbor = multi_move_pair(config, suite, workload)
                    if neighbor is None:
                        _move_stats.n_pair_failed += 1
                        T = max(opt_config.T_min, T * opt_config.T_cool)
                        continue
                    _move_stats.n_pair_proposed += 1
                    candidates = [neighbor]
                else:
                    # Single-vertex proposal
                    _kernel_name = "single"
                    if executable_curr is None:
                        _exec_unrolled = unroll(config.rpeg, K_eval)
                        executable_curr = lower(
                            config.rpeg,
                            _exec_unrolled,
                            config.plan,
                            suite,
                            K_eval,
                            workload = workload,
                            intra_stage_fusion = config.intra_stage_fusion
                        )
                    candidates = move_gen(
                        config,
                        executable_curr,
                        trace_curr,
                        table,
                        index,
                        opt_config.K_x,
                        opt_config.T_max,
                        T,
                        opt_config.T_init,
                        _move_stats
                    )

                if not candidates:
                    print(f"  r={r} i={i:02d} | chain stuck — no candidates")
                    break

                # Best-of-K proposal: evaluate all candidates, then apply one MH step to the
                # highest-scoring one.
                #
                # Option C parallel dispatch: parent checks cache, dispatches only misses to the
                # worker pool, then merges results sequentially. The EvalCache is never shared —
                # parent owns it exclusively.
                best_n_score, best_n, best_n_trace = -float('inf'), None, None

                # Step 1: parent checks cache for each candidate
                _cached_results = []   # (neighbor, score, trace, feasible) for hits
                _to_eval = []          # (neighbor, hash_key) for misses
                for neighbor in candidates:
                    h = evaluation_cache_key(neighbor, memory_policy)
                    if h in cache:
                        s, t, f = cache.get(h)
                        _cached_results.append((neighbor, s, t, f))
                    else:
                        _to_eval.append((neighbor, h))

                # Step 2: dispatch misses to worker pool (parallel) or evaluate sequentially
                _fresh_results = []   # (neighbor, hash_key, score, trace, feasible)
                if _to_eval:
                    if _pool is not None and len(_to_eval) > 1:
                        # Parallel: dispatch all misses at once
                        _worker_args = [(nb, suite, workload, hw, K_eval, memory_policy) for nb, _ in _to_eval]
                        _worker_out = _pool.map(_eval_worker, _worker_args)
                        for (nb, h), (s, t, f) in zip(_to_eval, _worker_out):
                            _fresh_results.append((nb, h, s, t, f))
                    else:
                        # Sequential fallback (single candidate or no pool)
                        for nb, h in _to_eval:
                            s, t, f = _eval_worker((nb, suite, workload, hw, K_eval, memory_policy))
                            _fresh_results.append((nb, h, s, t, f))

                # Step 3: merge results — cache store, budget accounting, bookkeeping. Process
                # cached hits first (no budget cost), then fresh misses.
                for neighbor, eval_n, trace_n, feasible in _cached_results:
                    score_n = _eff(eval_n, trace_n)
                    if inter_cfg.ac_share:
                        score_n = _ac_penalty(score_n, opt_config.lambda_ac)
                    # No E increment for cache hits.
                    if feasible:
                        family = (inter_cfg.ac_share, inter_cfg.intra_stage_fusion)
                        _elite_insert(
                            family, eval_n, neighbor.plan, _elite, _elite_seen, opt_config.K_elite
                        )
                    if score_n > best_score * (1.0 + eps_global):
                        best_config, best_score = neighbor, score_n
                        print(f"  ★ new global best = {best_score:.4e}  (s={_s_idx} r={r} i={i:02d} {_kernel_name})")
                    if feasible and score_n > best_feasible_score:
                        best_feasible_config, best_feasible_score = neighbor, score_n
                    if score_n > struct_best_score:
                        struct_best_score = score_n
                    if score_n > best_n_score:
                        best_n_score, best_n, best_n_trace = score_n, neighbor, trace_n

                for neighbor, h, eval_n, trace_n, feasible in _fresh_results:
                    cache.put(h, eval_n, trace_n, feasible)
                    E += 1; E_s += 1
                    score_n = _eff(eval_n, trace_n)
                    if inter_cfg.ac_share:
                        score_n = _ac_penalty(score_n, opt_config.lambda_ac)
                    if feasible:
                        family = (inter_cfg.ac_share, inter_cfg.intra_stage_fusion)
                        _elite_insert(
                            family, eval_n, neighbor.plan, _elite, _elite_seen, opt_config.K_elite
                        )
                    if score_n > best_score * (1.0 + eps_global):
                        best_config, best_score = neighbor, score_n
                        print(f"  ★ new global best = {best_score:.4e}  (s={_s_idx} r={r} i={i:02d} {_kernel_name})")
                    if feasible and score_n > best_feasible_score:
                        best_feasible_config, best_feasible_score = neighbor, score_n
                    if score_n > struct_best_score:
                        struct_best_score = score_n
                    if score_n > best_n_score:
                        best_n_score, best_n, best_n_trace = score_n, neighbor, trace_n

                # Budget check after batch (may overshoot by up to K_x-1 at boundary)
                _now = _time_mod.perf_counter()
                if opt_config.deadline is not None and _now >= opt_config.deadline:
                    _stop_eval_pool()
                    print(f"  time limit reached (E={E})")
                    print(f"\n  Move statistics")
                    print(_move_stats.summary(_n_accepted_strategy, _n_accepted_placement, _n_rejected))
                    return best_feasible_config if best_feasible_config is not None else best_config
                if E >= opt_config.E_max:
                    _stop_eval_pool()
                    print(f"  budget exhausted: E={E} >= E_max={opt_config.E_max}")
                    print(f"\n  Move statistics")
                    print(_move_stats.summary(_n_accepted_strategy, _n_accepted_placement, _n_rejected))
                    return best_feasible_config if best_feasible_config is not None else best_config
                if opt_config.deadline_per_struct is not None and _now >= opt_config.deadline_per_struct:
                    struct_exhausted = True
                if E_s >= opt_config.E_struct:
                    struct_exhausted = True

                # One MH step: propose the best candidate found across all K_x evaluations.
                if best_n is not None and not struct_exhausted:
                    # Detect which vertices changed and categorize the move.
                    _changed = []
                    _move_type = "placement"
                    for _vv in config.rpeg.vertices:
                        _pi_changed = config.plan.pi.get(_vv) != best_n.plan.pi.get(_vv)
                        _mb_changed = config.plan.mb.get(_vv) != best_n.plan.mb.get(_vv)
                        _sig_changed = config.plan.sigma.get(_vv) != best_n.plan.sigma.get(_vv)
                        if _pi_changed or _mb_changed:
                            _changed.append(_vv.id)
                            _move_type = "strategy"
                        elif _sig_changed:
                            _changed.append(_vv.id)
                    _chv_id = "+".join(_changed) if _changed else "?"
                    if _kernel_name != "single":
                        _move_type = _kernel_name  # override: "pair"

                    delta = (best_n_score - score_curr) / (abs(score_curr) + 1e-12)
                    if abs(delta) >= eps_noise:
                        if delta >= 0.0:
                            accept_prob = 1.0
                        else:
                            accept_prob = math.exp(delta / T)

                        _accepted = random.random() < accept_prob
                        if _accepted:
                            if _kernel_name == "pair":
                                _move_stats.n_pair_accepted += 1
                            else:
                                _n_accepted_strategy += (_move_type == "strategy")
                                _n_accepted_placement += (_move_type == "placement")
                        else:
                            if _kernel_name == "pair":
                                _move_stats.n_pair_rejected += 1
                            else:
                                _n_rejected += 1

                        # Single compact line per iteration
                        arrow = "up" if delta >= 0 else "down"
                        verdict = "ACC" if _accepted else "REJ"
                        print(f"  s={_s_idx} r={r} i={i:02d} | {_chv_id:12s} {_move_type:9s} | "
                              f"{score_curr:.4e} {arrow} {best_n_score:.4e}  "
                              f"α={accept_prob:.2f}  T={T:.3f} | "
                              f"{verdict}  E={E} E_s={E_s}")

                        if _accepted:
                            config, score_curr, trace_curr = best_n, best_n_score, best_n_trace
                            executable_curr = None # config changed; recompute next iter
                            _consec_rej = 0
                        else:
                            _consec_rej += 1

                if opt_config.geometric_cooling:
                    # Optional A9 path: T_i = max(T_min, T_{i-1} x gamma).
                    T = max(opt_config.T_min, T * opt_config.T_cool)

                if struct_exhausted:
                    break

                # Stagnation detection: restart after N consecutive rejections.
                if opt_config.stag_window > 0 and _consec_rej >= opt_config.stag_window:
                    print(f"  s={_s_idx} r={r} i={i:02d} | stagnation ({_consec_rej} consecutive rejections) - restarting")
                    break

            r += 1
            if E_s == restart_E_s:
                no_progress_restarts += 1
                if no_progress_restarts >= 2:
                    print("  structural search made no new evaluations across two restarts — stopping")
                    struct_exhausted = True
            else:
                no_progress_restarts = 0

        _s_idx += 1

    _stop_eval_pool()
    print(f"\n  Move statistics")
    print(_move_stats.summary(_n_accepted_strategy, _n_accepted_placement, _n_rejected))
    return best_feasible_config if best_feasible_config is not None else best_config


def polish(
    best_config: Configuration,
    suite: ModelSuite,
    workload: Workload,
    hw: Hardware,
    K_eval: int,
    opt_config: OptimizerConfig,
    cache: EvalCache,
    table: Dict[TemplateVertex, List],
    index: Dict[TemplateVertex, Dict],
    ac_share: bool = False,
    max_rounds: int = 50
) -> Configuration:
    """
    Deterministic polish phase after optimize: deterministic greedy coordinate descent on the best
    plan returned by optimize().

    For each template vertex v (in stable id-sorted order), evaluates:
      1) all valid aligned origins for the current (dp,pp,tp,mb,extent) — placement-only moves;
      2) all (strategy, origin) pairs in the sorted-table window [j_star-w, j_star+w] around the
         current position — local strategy changes.
    Applies the best strict improvement found for that vertex, then moves to the next. Repeats
    passes until a full pass yields no improvement or max_rounds is reached.

    :param best_config: Best configuration found by optimize().
    :type best_config: Configuration
    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param workload: Workload settings.
    :type workload: Workload
    :param hw: Hardware settings.
    :type hw: Hardware
    :param K_eval: Number of iterations/steps to unroll.
    :type K_eval: int
    :param opt_config: Optimizer configuration.
    :type opt_config: OptimizerConfig
    :param cache: Evaluation cache of traces. Shared cache, passed in so caller can inspect it.
    :type cache: EvalCache
    :param table: Sorted candidate list for each vertex in the RPEG.
    :type table: Dict[TemplateVertex, List]
    :param index: Index for the sorted candidate list for O(1) lookup.
    :type index: Dict[TemplateVertex, Dict]
    :param ac_share: Flag to indicate, whether actor and critic share their weights. Defaults to
                     False.
    :type ac_share: bool
    :param max_rounds: Maximum number of polishing passes. Defaults to 50.
    :type max_rounds: int
    :return: Optimized configuration.
    :rtype: Configuration
    """
    config = best_config
    R, N, G = config.device_space.R, config.device_space.N, config.device_space.G
    # RPEG is fixed for all configurations in this structural setting — capture once.
    rpeg = best_config.rpeg

    # Parallel evaluation pool (same Option C pattern as optimize).
    import multiprocessing as _mp
    import os as _os
    _n_workers = min(opt_config.K_x, _os.cpu_count() or 1)
    _pool = _mp.Pool(processes=_n_workers) if _n_workers > 1 else None

    def _stop_eval_pool() -> None:
        """
        Synchronously reap candidate workers before returning from polish.
        """
        nonlocal _pool
        if _pool is None:
            return
        _pool.close()
        _pool.join()
        _pool = None

    _delta_pol = opt_config.max_throughput_loss
    _best_raw_pol = [0.0]

    def _score(raw: float, trace: Optional[Trace] = None) -> float:
        """
        Apply utilization weighting if efficiency mode is on.

        Score = raw × utilization above the throughput floor.
        The floor prevents degradation beyond delta; the utilization factor steers the SA toward
        plans with fewer idle gaps.

        :param raw: Throughput score.
        :type raw: float
        :param trace: Execution trace. Defaults to None.
        :type trace: Optional[Trace]
        :return: Weighted score.
        :rtype: float
        """
        s = raw
        if _delta_pol > 0.0 and trace is not None:
            if raw > _best_raw_pol[0]:
                _best_raw_pol[0] = raw
            floor = (1.0 - _delta_pol) * _best_raw_pol[0]
            if raw < floor:
                s = raw * 1e-12
            else:
                s = raw * max(trace.utilization(), 1e-6)
        if ac_share:
            s = _ac_penalty(s, opt_config.lambda_ac)
        return s

    # Sort by v.id for a stable, deterministic traversal order regardless of RPEG construction.
    vertices_ordered = sorted(rpeg.vertices, key=lambda vert: vert.id)

    # Window fraction: same as move_gen at T_min (fully-cooled exploitation window).
    w_frac = opt_config.T_min / opt_config.T_init

    memory_policy = MemoryPolicy(
        optimizer_stage = opt_config.memory_optimizer_stage,
        activation_checkpointing = opt_config.memory_activation_checkpointing,
        runtime_overhead_fraction = opt_config.memory_runtime_overhead_fraction,
        worker_overhead_bytes = opt_config.memory_worker_overhead_bytes,
        semantic_activation_cache_bytes = opt_config.memory_semantic_activation_cache_bytes,
        max_worker_groups_per_gpu = opt_config.memory_max_worker_groups_per_gpu
    )
    raw, _tr, _, _ = evaluate_cached(config, cache, suite, workload, hw, K_eval, memory_policy)
    score = _score(raw, _tr)
    print(f"  [polish] start score = {score:.4e}")

    for round_idx in range(max_rounds):
        improved_this_pass = False

        for v in vertices_ordered:
            best_n_score, best_n = score, None # must strictly exceed current score

            v_table = table[v]
            n = len(v_table)
            w = max(5, int(n * w_frac))

            # Current position in the sorted table — O(1) via index.
            dp_c, pp_c, tp_c = config.plan.pi[v]
            r_ext_c, n_ext_c, g_ext_c = config.plan.sigma[v].extent
            mb_c = config.plan.mb[v]
            j_star = index[v].get((dp_c, pp_c, tp_c, r_ext_c, n_ext_c, mb_c), n // 2)

            # 1. Placement-only: skip
            # Scores are origin-independent: the cost model depends on (dp, pp, tp, mb, extent), but
            # not on the origin within the device space. The PP communication model reads
            # alloc.extent to compute intra/inter edge splits; hw.tier() yields the same tier for
            # any aligned origin with the same extent. Verified empirically across all vertices,
            # strategies, and origin combinations on 64-GPU and 128-GPU configurations. Therefore
            # placement-only moves (same strategy, different origin) cannot improve the score — skip
            # entirely.

            # 2. Strategy window: [j_star-w, j_star+w], one origin per strategy
            # Since scores are origin-independent (see above), we evaluate each (dp, pp, tp, mb)
            # exactly once with origin (0, 0, 0). If it improves the score, we record it. The origin
            # is valid, because extent is guaranteed to fit within the device space (enforced by
            # build_candidate_directory).
            #
            # Option C parallel dispatch: build all neighbors, separate cache hits from misses,
            # dispatch misses to pool, merge results.
            low  = max(0, j_star - w)
            high = min(n - 1, j_star + w)

            # Build all candidate configs for this vertex
            _pol_candidates = []
            for j in range(low, high + 1):
                if j == j_star:
                    continue
                dp, pp, tp, r_ext, n_ext, mb = v_table[j][:6]
                geom = packing_geometry(dp, pp, tp, G)
                if geom is None:
                    continue
                g_ext, _ = geom
                targets = (v,)
                if config.intra_stage_fusion and v.id in {"t_pi", "t_V"}:
                    peer_id = "t_V" if v.id == "t_pi" else "t_pi"
                    peer = next(candidate for candidate in rpeg.vertices if candidate.id == peer_id)
                    if suite.specs[peer.kind.model].L % pp:
                        continue
                    targets = (v, peer)
                allocation = Allocation(origin=(0, 0, 0), extent=(r_ext, n_ext, g_ext))
                new_plan = ExecutionPlan(
                    sigma={**config.plan.sigma, **{target: allocation for target in targets}},
                    pi={**config.plan.pi, **{target: (dp, pp, tp) for target in targets}},
                    mb={**config.plan.mb, **{target: mb for target in targets}}
                )
                _pol_candidates.append(Configuration(
                    rpeg,
                    new_plan,
                    config.device_space,
                    intra_stage_fusion = config.intra_stage_fusion,
                    transfer_resource_policy = config.transfer_resource_policy
                ))

            # Check cache, dispatch misses to pool
            _pol_cached = []
            _pol_to_eval = []
            for nb in _pol_candidates:
                h = evaluation_cache_key(nb, memory_policy)
                if h in cache:
                    s, t, f = cache.get(h)
                    _pol_cached.append((nb, s, t, f))
                else:
                    _pol_to_eval.append((nb, h))

            _pol_fresh = []
            if _pol_to_eval:
                if _pool is not None and len(_pol_to_eval) > 1:
                    _worker_args = [(nb, suite, workload, hw, K_eval, memory_policy) for nb, _ in _pol_to_eval]
                    _worker_out = _pool.map(_eval_worker, _worker_args)
                    for (nb, h), (s, t, f) in zip(_pol_to_eval, _worker_out):
                        _pol_fresh.append((nb, h, s, t, f))
                else:
                    for nb, h in _pol_to_eval:
                        s, t, f = _eval_worker((nb, suite, workload, hw, K_eval, memory_policy))
                        _pol_fresh.append((nb, h, s, t, f))

            # Merge: process hits then fresh results
            for nb, eval_n, trace_n, feasible in _pol_cached:
                if not feasible:
                    continue
                score_n = _score(eval_n, trace_n)
                if score_n > best_n_score:
                    best_n_score, best_n = score_n, nb

            for nb, h, eval_n, trace_n, feasible in _pol_fresh:
                cache.put(h, eval_n, trace_n, feasible)
                if not feasible:
                    continue
                score_n = _score(eval_n, trace_n)
                if score_n > best_n_score:
                    best_n_score, best_n = score_n, nb

            if best_n is not None:
                print(f"  [polish  round {round_idx}  v={v.id}] {score:.4e} -> {best_n_score:.4e}")
                config, score = best_n, best_n_score
                improved_this_pass = True

        if not improved_this_pass:
            print(f"  [polish] converged after {round_idx + 1} round(s)")
            break

    _stop_eval_pool()
    print(f"  [polish] final score = {score:.4e}")

    return config
