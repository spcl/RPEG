#!/usr/bin/env python3
"""Optimizer benchmarking via exhaustive enumeration on small instances.

This script:
  1) Loads an instance from a YAML config file (same format as pipeline.py).
  2) Enumerates the full per-vertex candidate Cartesian product to find the
     global optimum under the analytical model (schedule + cost + memory).
  3) Runs the optimizer over multiple RNG seeds and reports
        (i) optimality-gap distribution,
        (ii) success rate of hitting the optimum,
        (iii) runtime and number of unique evaluations (cache misses).

Usage:
    python evaluation/optimizer_bench.py --config configs/micro2.yaml --n_seeds 50
    python evaluation/optimizer_bench.py --config configs/micro2.yaml --n_seeds 200 --out_csv results.csv

Config format:
    Same YAML structure as pipeline.py configs (hardware / models / workload /
    policy / optimizer sections), plus an optional 'evaluation' section that
    controls bench-specific settings such as max_per_vertex and apply_polish.
    See configs/micro2.yaml for a fully annotated example.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import yaml


# ── Path setup ───────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_SRC = os.path.join(_ROOT, "src")
sys.path.insert(0, _SRC)
sys.path.insert(0, _ROOT)  # for pipeline.py imports


from core.types import Allocation, DeviceSpace, Model
from ir.ppo import build_PPO_RPEG
from ir.plan import ExecutionPlan
from model.cost import NodeCostModel
from model.memory import NodeMemoryModel, _device_peaks
from model.models import ModelSuite
from model.workload import Workload
from model.hardware import Hardware
from passes.lower import ExecNodeType, lower
from passes.unroll import unroll
from schedule.scheduler import schedule
from search.cache import EvalCache
from search.evaluate import Configuration
from search.moves import build_candidate_directory
from search.optimizer import OptimizerConfig, optimize, penalized_score, polish
from ir.serialize import save_plan

# Reuse pipeline.py builders — same YAML format, no duplication.
from pipeline import (
    _make_yaml_loader,
    build_device_space,
    build_hardware,
    build_suite,
    build_workload,
)
import search.evaluate as eval_mod
import search.moves as moves_mod
import search.optimizer as opt_mod


# ─────────────────────────────────────────────────────────────────────────────
# Preset instances (kept for backward-compat with budget_scaling.py etc.)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MicroInstance:
    suite: ModelSuite
    workload: Workload
    hw: Hardware
    ds: DeviceSpace
    k_eval: int


def _preset_instance(name: str) -> MicroInstance:
    """Return a small hardcoded instance preset (micro2, micro4, twonode2)."""
    from model.models import ModelSpec
    name = name.lower()

    spec_actor = ModelSpec(L=4, d=256, d_ff=1024, V=1000)
    spec_small = ModelSpec(L=4, d=128, d_ff=512, V=1000)

    suite = ModelSuite({
        Model.ACTOR:     spec_actor,
        Model.REFERENCE: spec_actor,
        Model.CRITIC:    spec_small,
        Model.REWARD:    spec_small,
        Model.AC_SHARE:  spec_actor,
    })
    workload = Workload(B=4, K=2, S=16, T=8)
    hw = Hardware(
        flops_per_gpu=100e12, memory_per_gpu=80e9,
        memory_bandwidth=3.35e12,
        alpha_coll_g=0.5e-6, beta_coll_g=1 / 450e9,
        alpha_coll_n=5.0e-6, beta_coll_n=1 / 50e9,
        alpha_coll_r=10.0e-6, beta_coll_r=1 / 25e9,
        alpha_p2p_g=0.5e-6, beta_p2p_g=1 / 450e9,
        alpha_p2p_n=5.0e-6, beta_p2p_n=1 / 50e9,
        alpha_p2p_r=10.0e-6, beta_p2p_r=1 / 25e9,
    )

    if name in {"micro2", "micro2gpu", "2gpu"}:
        ds, k_eval = DeviceSpace(R=1, N=1, G=2), 4
    elif name in {"micro4", "micro4gpu", "4gpu"}:
        ds, k_eval = DeviceSpace(R=1, N=1, G=4), 4
    elif name in {"twonode2", "two-node-2", "twonode2gpu"}:
        ds, k_eval = DeviceSpace(R=1, N=2, G=2), 4
    else:
        raise ValueError(f"Unknown preset '{name}'")

    return MicroInstance(suite=suite, workload=workload, hw=hw, ds=ds, k_eval=k_eval)


def _make_opt_cfg(
    *,
    enable_structural_search: bool,
    e_max: int,
    e_struct: int,
    r_max: int,
    i_max: int,
    lambda_ac: float = 0.0,
) -> OptimizerConfig:
    """Build an OptimizerConfig from explicit budget kwargs (used by budget_scaling.py etc.)."""
    if enable_structural_search:
        t_cand = None
    else:
        t_cand = {Model.ACTOR: {0}, Model.CRITIC: {0}, Model.AC_SHARE: set()}

    return OptimizerConfig(
        E_max=e_max, E_struct=e_struct, R_max=r_max, I_max=i_max,
        lambda_ac=lambda_ac,
        T_cand=t_cand,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=_make_yaml_loader())


def _build_opt_cfg(cfg: dict, *, enable_structural_search: bool) -> OptimizerConfig:
    """Build OptimizerConfig from the 'optimizer' YAML section.

    When enable_structural_search is False (the default for micro benchmarks),
    staleness and AC-share search are disabled so that the exhaustive enumeration
    and the optimizer operate on the same restricted search space.
    """
    o = cfg["optimizer"]

    if enable_structural_search:
        t_cand = None  # use DEFAULT_T_CAND (tau in {0,1} for each model)
    else:
        # Restrict to a single structural setting: no staleness, no AC-share.
        t_cand = {
            Model.ACTOR: {0},
            Model.CRITIC: {0},
            Model.AC_SHARE: set(),
        }

    return OptimizerConfig(
        K_x      = o.get("k_x",      6),
        T_max    = o.get("t_max",    50),
        I_max    = o.get("i_max",    40),
        R_max    = o.get("r_max",     4),
        E_max    = o.get("e_max",   400),
        E_struct = o.get("e_struct", 400),
        T_init   = o.get("t_init",  0.30),
        T_warm   = o.get("t_warm",  0.10),
        T_min    = o.get("t_min",   0.03),
        T_cool   = o.get("t_cool",  -1.0),
        lambda_ac  = o.get("lambda_ac", 0.0),
        score_eps  = o.get("score_eps",  1e-4),
        T_cand   = t_cand,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Baseline plan constructors
# ─────────────────────────────────────────────────────────────────────────────


def _baseline_plan(rpeg, ds: DeviceSpace) -> ExecutionPlan:
    """Fully-colocated baseline (all vertices share the whole device space)."""
    R, N, G = ds.R, ds.N, ds.G
    alloc = Allocation(origin=(0, 0, 0), extent=(R, N, G))
    sigma, pi, mb = {}, {}, {}
    for v in rpeg.vertices:
        sigma[v] = alloc
        pi[v] = (R, N, G)
        mb[v] = 1
    return ExecutionPlan(sigma=sigma, pi=pi, mb=mb)


def _baseline_plan_from_candidates(
    rpeg,
    *,
    suite: ModelSuite,
    workload: Workload,
    hw: Hardware,
    ds: DeviceSpace,
    max_per_vertex: int,
) -> ExecutionPlan:
    """Baseline plan guaranteed to lie inside the per-vertex candidate sets.

    Picks the fastest (index-0) candidate for each vertex and places it at origin (0,0,0).
    """
    cost_model = NodeCostModel(hw=hw, suite=suite, workload=workload)
    mem_model = NodeMemoryModel(hw=hw, suite=suite, workload=workload,
                                trainable=frozenset(rpeg.trainable),
                                shadow_models=frozenset(m for m in rpeg.trainable if rpeg.tau.get(m, 0) > 0))
    table, _index = build_candidate_directory(
        rpeg, suite, ds, workload, cost_model, mem_model,
        max_per_vertex=max_per_vertex,
    )

    sigma, pi, mb = {}, {}, {}
    for v in rpeg.vertices:
        if not table[v]:
            raise RuntimeError(f"No candidates for vertex {v.id}; cannot construct baseline")
        dp, pp, tp, r_ext, n_ext, mb_v, _t = table[v][0]
        s = min(pp, ds.G // tp)
        g_ext = s * tp
        sigma[v] = Allocation(origin=(0, 0, 0), extent=(r_ext, n_ext, g_ext))
        pi[v] = (dp, pp, tp)
        mb[v] = mb_v
    return ExecutionPlan(sigma=sigma, pi=pi, mb=mb)


# ─────────────────────────────────────────────────────────────────────────────
# Exhaustive enumeration
# ─────────────────────────────────────────────────────────────────────────────


def _aligned_origins(extent: Tuple[int, int, int], ds: DeviceSpace) -> Iterable[Tuple[int, int, int]]:
    """Enumerate aligned origins exactly like sample_expl/move_gen (multiples of extent)."""
    r_ext, n_ext, g_ext = extent
    for r_o in (i * r_ext for i in range(ds.R // r_ext)):
        for n_o in (i * n_ext for i in range(ds.N // n_ext)):
            for g_o in (i * g_ext for i in range(ds.G // g_ext)):
                yield (r_o, n_o, g_o)


@dataclass(frozen=True)
class CandidateChoice:
    alloc: Allocation
    pi: Tuple[int, int, int]
    mb: int


def _expand_vertex_candidates(
    table_v: Sequence[Tuple[int, int, int, int, int, int, float]],
    ds: DeviceSpace,
) -> List[CandidateChoice]:
    """Expand directory entries into concrete per-vertex choices including all aligned origins."""
    seen = set()
    out: List[CandidateChoice] = []
    for dp, pp, tp, r_ext, n_ext, mb, _t in table_v:
        s = min(pp, ds.G // tp)
        g_ext = s * tp
        extent = (r_ext, n_ext, g_ext)
        for origin in _aligned_origins(extent, ds):
            key = (origin, extent, dp, pp, tp, mb)
            if key in seen:
                continue
            seen.add(key)
            out.append(CandidateChoice(
                alloc=Allocation(origin=origin, extent=extent),
                pi=(dp, pp, tp),
                mb=mb,
            ))
    out.sort(key=lambda c: (c.pi, c.mb, c.alloc.extent, c.alloc.origin))
    return out


@dataclass
class ExhaustiveResult:
    best_feasible: Configuration
    best_feasible_score_raw: float
    best_feasible_score_pen: float
    n_total: int
    n_feasible: int


def _evaluate_plan_once(
    rpeg,
    plan: ExecutionPlan,
    ds: DeviceSpace,
    suite: ModelSuite,
    workload: Workload,
    hw: Hardware,
    *,
    k_eval: int,
    unrolled,
    cost_model: NodeCostModel,
    mem_model: NodeMemoryModel,
) -> Tuple[float, bool]:
    """Evaluate one plan (no cache), returning (score_like_evaluate_cached, feasible)."""
    lowered = lower(rpeg=rpeg, unrolled=unrolled, plan=plan, suite=suite, K_eval=k_eval)
    trace = schedule(executable=lowered, cost_model=cost_model, device_space=ds)
    W_step = workload.B * workload.K * (workload.S + workload.T)
    raw = trace.throughput(W_step=W_step, K=k_eval)

    cap = mem_model._eff_capacity()
    peaks = _device_peaks(lowered, mem_model)
    worst_peak = max(peaks.values()) if peaks else 0.0
    feasible = worst_peak <= cap

    if feasible:
        return raw, True

    overflow_ratio = max(0.0, worst_peak / cap - 1.0)
    penalty_mult = math.exp(-eval_mod._OOM_BETA * overflow_ratio)
    penalty_mult = max(penalty_mult, 1e-12)
    return raw * penalty_mult, False


def exhaustive_global_optimum(
    rpeg,
    suite: ModelSuite,
    workload: Workload,
    hw: Hardware,
    ds: DeviceSpace,
    *,
    opt_cfg: OptimizerConfig,
    k_eval: int,
    max_per_vertex: int,
    progress_every: int = 2000,
) -> ExhaustiveResult:
    """Compute the global optimum by enumerating the full per-vertex candidate product."""
    cost_model = NodeCostModel(hw=hw, suite=suite, workload=workload)
    mem_model = NodeMemoryModel(hw=hw, suite=suite, workload=workload,
                                trainable=frozenset(rpeg.trainable),
                                shadow_models=frozenset(m for m in rpeg.trainable if rpeg.tau.get(m, 0) > 0))
    unrolled = unroll(rpeg=rpeg, K_eval=k_eval)

    table, _index = build_candidate_directory(
        rpeg, suite, ds, workload, cost_model, mem_model,
        max_per_vertex=max_per_vertex,
    )

    vertices = list(rpeg.vertices)
    expanded: List[List[CandidateChoice]] = [_expand_vertex_candidates(table[v], ds) for v in vertices]

    n_total = 1
    for choices in expanded:
        n_total *= max(1, len(choices))

    print("\nExhaustive enumeration setup:")
    for v, choices in zip(vertices, expanded):
        print(f"  {v.id:8s}: {len(choices):4d} concrete choices")
    print(f"  Total Cartesian product size: {n_total} configurations")

    best_cfg: Optional[Configuration] = None
    best_raw = -float("inf")
    best_pen = -float("inf")
    n_feasible = 0
    t0 = time.perf_counter()

    for i, picks in enumerate(itertools.product(*expanded), start=1):
        sigma = {v: p.alloc for v, p in zip(vertices, picks)}
        pi    = {v: p.pi    for v, p in zip(vertices, picks)}
        mb    = {v: p.mb    for v, p in zip(vertices, picks)}
        plan  = ExecutionPlan(sigma=sigma, pi=pi, mb=mb)

        score_like_cached, feasible = _evaluate_plan_once(
            rpeg, plan, ds, suite, workload, hw,
            k_eval=k_eval, unrolled=unrolled,
            cost_model=cost_model, mem_model=mem_model,
        )
        if not feasible:
            continue
        n_feasible += 1

        score_pen = penalized_score(
            raw_score=score_like_cached,
            rpeg=rpeg,
            opt_config=opt_cfg,
            ac_share=(Model.AC_SHARE in rpeg.trainable),
        )

        if score_pen > best_pen:
            best_pen = score_pen
            best_raw = score_like_cached
            best_cfg = Configuration(rpeg=rpeg, plan=plan, device_space=ds)

        if progress_every and (i % progress_every == 0):
            dt = time.perf_counter() - t0
            rate = i / max(dt, 1e-9)
            print(
                f"  enumerated {i}/{n_total} (feasible {n_feasible}) "
                f"[{rate:.1f} cfg/s] best_pen_score={best_pen:.4e}"
            )

    if best_cfg is None:
        raise RuntimeError("No feasible configuration found during exhaustive enumeration")

    dt = time.perf_counter() - t0
    n_infeasible = n_total - n_feasible
    print(f"\nExhaustive enumeration done in {dt:.2f}s")
    print(f"  total_cartesian    = {n_total}")
    print(f"  n_feasible         = {n_feasible}  ({n_feasible / n_total * 100:.1f}%)")
    print(f"  n_infeasible       = {n_infeasible}  ({n_infeasible / n_total * 100:.1f}%)")
    print(f"  Best feasible raw score:       {best_raw:.6e}")
    print(f"  Best feasible penalized score: {best_pen:.6e}")

    return ExhaustiveResult(
        best_feasible=best_cfg,
        best_feasible_score_raw=best_raw,
        best_feasible_score_pen=best_pen,
        n_total=n_total,
        n_feasible=n_feasible,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer evaluation over multiple seeds
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SeedRun:
    seed: int
    score_raw: float
    score_pen: float
    gap: float
    hit_opt: bool
    evals: int
    wall_s: float
    coverage_cartesian: float
    coverage_feasible: float


def _patch_optimizer_candidate_dir(max_per_vertex: int) -> None:
    """Force optimizer.optimize() to use a limited candidate directory."""
    original = moves_mod.build_candidate_directory

    def _wrapped(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["max_per_vertex"] = max_per_vertex
        return original(*args, **kwargs)

    opt_mod.build_candidate_directory = _wrapped  # type: ignore[attr-defined]


def run_optimizer_seeds(
    *,
    base_config: Configuration,
    suite: ModelSuite,
    workload: Workload,
    hw: Hardware,
    k_eval: int,
    opt_cfg: OptimizerConfig,
    n_seeds: int,
    seed0: int,
    max_per_vertex: int,
    opt_pen_best: float,
    hit_rtol: float,
    n_total_cartesian: int,
    n_feasible_enum: int,
    save_plans_dir: Optional[str] = None,
    log_trajectory_dir: Optional[str] = None,
    apply_polish: bool = False,
) -> List[SeedRun]:
    """Run optimize() for multiple RNG seeds and collect per-run metrics."""
    _patch_optimizer_candidate_dir(max_per_vertex)

    runs: List[SeedRun] = []
    for i in range(n_seeds):
        seed = seed0 + i
        random.seed(seed)

        cache = EvalCache()
        traj = []
        orig_eval_cached = opt_mod.evaluate_cached  # type: ignore[attr-defined]

        if log_trajectory_dir:
            os.makedirs(log_trajectory_dir, exist_ok=True)

            def _wrapped_eval_cached(config, cache, suite, workload, hw, K_eval):
                score_raw, trace, feasible, miss = eval_mod.evaluate_cached(
                    config, cache, suite, workload, hw, K_eval
                )
                if miss:
                    score_pen = penalized_score(
                        raw_score=score_raw,
                        rpeg=config.rpeg,
                        opt_config=opt_cfg,
                        ac_share=(Model.AC_SHARE in config.rpeg.trainable),
                    )
                    traj.append({
                        "eval": len(traj) + 1,
                        "score_raw": float(score_raw),
                        "score_pen": float(score_pen),
                        "feasible": bool(feasible),
                    })
                return score_raw, trace, feasible, miss

            opt_mod.evaluate_cached = _wrapped_eval_cached  # type: ignore[attr-defined]

        t0 = time.perf_counter()
        try:
            best = optimize(base_config, suite, workload, hw, k_eval, opt_cfg, cache)
        finally:
            opt_mod.evaluate_cached = orig_eval_cached  # type: ignore[attr-defined]

        if apply_polish:
            _, _, _pol_feasible, _ = eval_mod.evaluate_cached(best, cache, suite, workload, hw, k_eval)
            if _pol_feasible:
                _cost = NodeCostModel(hw=hw, suite=suite, workload=workload)
                _mem  = NodeMemoryModel(hw=hw, suite=suite, workload=workload,
                                        trainable=frozenset(best.rpeg.trainable),
                                        shadow_models=frozenset(m for m in best.rpeg.trainable if best.rpeg.tau.get(m, 0) > 0))
                _pol_table, _pol_index = build_candidate_directory(
                    best.rpeg, suite, best.device_space, workload, _cost, _mem
                )
                best = polish(
                    best, suite, workload, hw, k_eval, opt_cfg, cache,
                    _pol_table, _pol_index,
                    ac_share=(Model.AC_SHARE in best.rpeg.trainable),
                )

        wall = time.perf_counter() - t0

        score_raw, _trace, _feasible, _miss = eval_mod.evaluate_cached(
            best, cache, suite, workload, hw, k_eval
        )
        score_pen = penalized_score(
            raw_score=score_raw,
            rpeg=best.rpeg,
            opt_config=opt_cfg,
            ac_share=(Model.AC_SHARE in best.rpeg.trainable),
        )

        gap = 0.0 if opt_pen_best <= 0 else max(0.0, (opt_pen_best - score_pen) / opt_pen_best)
        hit = math.isfinite(score_pen) and (
            abs(score_pen - opt_pen_best) <= hit_rtol * max(1.0, abs(opt_pen_best))
        )

        coverage_cartesian = cache.misses / max(1, n_total_cartesian)
        coverage_feasible  = cache.misses / n_feasible_enum if n_feasible_enum > 0 else 0.0

        run = SeedRun(
            seed=seed,
            score_raw=float(score_raw),
            score_pen=float(score_pen),
            gap=float(gap),
            hit_opt=bool(hit),
            evals=int(cache.misses),
            wall_s=float(wall),
            coverage_cartesian=float(coverage_cartesian),
            coverage_feasible=float(coverage_feasible),
        )

        if save_plans_dir:
            os.makedirs(save_plans_dir, exist_ok=True)
            save_plan(
                best,
                os.path.join(save_plans_dir, f"seed_{seed}.yaml"),
                score_raw=float(score_raw),
            )

        if log_trajectory_dir:
            best_so_far = -float("inf")
            for t in traj:
                best_so_far = max(best_so_far, t["score_pen"])
                t["best_so_far_pen"] = best_so_far
            with open(os.path.join(log_trajectory_dir, f"trajectory_seed_{seed}.json"), "w") as f:
                json.dump(traj, f, indent=2)

        runs.append(run)

    return runs


def _summarize_runs(runs: Sequence[SeedRun]) -> dict:
    gaps     = [r.gap for r in runs]
    evals    = [r.evals for r in runs]
    times    = [r.wall_s for r in runs]
    hits     = [r.hit_opt for r in runs]
    cov_cart = [r.coverage_cartesian for r in runs]
    cov_feas = [r.coverage_feasible for r in runs]

    def _pct(xs, q):
        xs = sorted(xs)
        if not xs:
            return float("nan")
        k = int(round((len(xs) - 1) * q))
        return xs[k]

    def _dist(xs):
        if not xs:
            nan = float("nan")
            return {"mean": nan, "median": nan, "p25": nan, "p75": nan, "min": nan, "max": nan}
        return {
            "mean":   statistics.fmean(xs),
            "median": statistics.median(xs),
            "p25":    _pct(xs, 0.25),
            "p75":    _pct(xs, 0.75),
            "min":    min(xs),
            "max":    max(xs),
        }

    return {
        "n": len(runs),
        "success_rate": sum(1 for h in hits if h) / max(1, len(hits)),
        "gap":               _dist(gaps),
        "evals":             _dist(evals),
        "wall_s":            _dist(times),
        "coverage_cartesian": _dist(cov_cart),
        "coverage_feasible":  _dist(cov_feas),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--config", required=True,
        help="YAML config file (hardware / models / workload / optimizer / evaluation sections). "
             "See configs/micro2.yaml for an annotated example.",
    )

    # ── Run-specific flags (not in YAML) ──────────────────────────────────────
    ap.add_argument("--n_seeds",  type=int,   default=25,   help="Number of optimizer seeds to run")
    ap.add_argument("--seed0",    type=int,   default=0,    help="First RNG seed (seeds = seed0, seed0+1, ...)")
    ap.add_argument("--hit_rtol", type=float, default=1e-9, help="Relative tolerance for counting a run as 'hit optimum'")

    # ── Optional overrides for evaluation section ─────────────────────────────
    ap.add_argument("--base_from_candidates", action="store_true",
                    help="Choose baseline plan from the candidate directory instead of fully-colocated")

    # ── Output flags ──────────────────────────────────────────────────────────
    ap.add_argument("--out_json",           default=None)
    ap.add_argument("--out_csv",            default=None)
    ap.add_argument("--save_opt_plan",      default=None, help="Save the exhaustive optimum plan (yaml)")
    ap.add_argument("--save_seed_plans_dir",default=None, help="Save per-seed optimizer plans into this directory")
    ap.add_argument("--log_trajectory_dir", default=None, help="Write per-seed convergence trajectories here")
    args = ap.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = _load_config(args.config)

    ds       = build_device_space(cfg)
    hw       = build_hardware(cfg)
    suite    = build_suite(cfg)
    workload = build_workload(cfg)

    o    = cfg["optimizer"]
    ev   = cfg.get("evaluation", {})
    k_eval                   = o.get("k_eval", 4)
    max_per_vertex           = ev.get("max_per_vertex", 1000)
    apply_polish             = ev.get("apply_polish", False)
    enable_structural_search = ev.get("enable_structural_search", False)

    opt_cfg = _build_opt_cfg(cfg, enable_structural_search=enable_structural_search)

    # ── Build baseline ────────────────────────────────────────────────────────
    tau0 = {Model.ACTOR: 0, Model.CRITIC: 0}
    rpeg = build_PPO_RPEG(suite, workload, tau=tau0)

    if args.base_from_candidates:
        base_plan = _baseline_plan_from_candidates(
            rpeg, suite=suite, workload=workload, hw=hw, ds=ds,
            max_per_vertex=max_per_vertex,
        )
    else:
        base_plan = _baseline_plan(rpeg, ds)

    base_config = Configuration(rpeg=rpeg, plan=base_plan, device_space=ds)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Optimizer benchmark ===")
    print(f"  config:       {args.config}")
    print(f"  device_space: R={ds.R} N={ds.N} G={ds.G} (total={ds.num_devices()})")
    print(f"  workload:     B={workload.B} K={workload.K} S={workload.S} T={workload.T}")
    print(f"  k_eval:       {k_eval}")
    print(f"  max_per_vertex:           {max_per_vertex}")
    print(f"  enable_structural_search: {enable_structural_search}")
    print(f"  apply_polish:             {apply_polish}")
    print(f"  n_seeds={args.n_seeds}  seed0={args.seed0}  hit_rtol={args.hit_rtol:.1e}")

    # ── Exhaustive enumeration ────────────────────────────────────────────────
    print("\n=== Exhaustive enumeration (global optimum) ===")
    ex = exhaustive_global_optimum(
        rpeg, suite, workload, hw, ds,
        opt_cfg=opt_cfg,
        k_eval=k_eval,
        max_per_vertex=max_per_vertex,
    )

    if args.save_opt_plan:
        os.makedirs(os.path.dirname(args.save_opt_plan) or ".", exist_ok=True)
        save_plan(
            ex.best_feasible,
            args.save_opt_plan,
            score_raw=ex.best_feasible_score_raw,
        )

    # ── Optimizer seeds ───────────────────────────────────────────────────────
    print("\n=== Optimizer over seeds ===")
    runs = run_optimizer_seeds(
        base_config=base_config,
        suite=suite,
        workload=workload,
        hw=hw,
        k_eval=k_eval,
        opt_cfg=opt_cfg,
        n_seeds=args.n_seeds,
        seed0=args.seed0,
        max_per_vertex=max_per_vertex,
        opt_pen_best=ex.best_feasible_score_pen,
        hit_rtol=args.hit_rtol,
        n_total_cartesian=ex.n_total,
        n_feasible_enum=ex.n_feasible,
        save_plans_dir=args.save_seed_plans_dir,
        log_trajectory_dir=args.log_trajectory_dir,
        apply_polish=apply_polish,
    )

    # ── Results ───────────────────────────────────────────────────────────────
    summary = _summarize_runs(runs)
    print("\nSummary:")
    print(f"  n_seeds:      {summary['n']}")
    print(f"  success_rate: {summary['success_rate'] * 100:.1f}%")
    print(
        "  optimality_gap (relative): "
        f"median={summary['gap']['median']:.3e} "
        f"p25={summary['gap']['p25']:.3e} p75={summary['gap']['p75']:.3e} "
        f"min={summary['gap']['min']:.3e} max={summary['gap']['max']:.3e}"
    )
    print(
        "  evals (cache misses): "
        f"median={summary['evals']['median']:.0f} "
        f"min={summary['evals']['min']:.0f} max={summary['evals']['max']:.0f}"
    )
    print(
        "  wall time (s): "
        f"median={summary['wall_s']['median']:.3f} "
        f"min={summary['wall_s']['min']:.3f} max={summary['wall_s']['max']:.3f}"
    )

    n_infeasible = ex.n_total - ex.n_feasible
    cf = summary["coverage_feasible"]
    cc = summary["coverage_cartesian"]
    print("\nSearch-space coverage:")
    print(f"  total_cartesian = {ex.n_total}")
    print(f"  n_feasible      = {ex.n_feasible}  ({ex.n_feasible / ex.n_total * 100:.1f}%)")
    print(f"  n_infeasible    = {n_infeasible}  ({n_infeasible / ex.n_total * 100:.1f}%)")
    print(
        "  coverage_feasible  (unique_evals / n_feasible):     "
        f"median={cf['median'] * 100:.2f}%  "
        f"p25={cf['p25'] * 100:.2f}%  p75={cf['p75'] * 100:.2f}%  "
        f"min={cf['min'] * 100:.2f}%  max={cf['max'] * 100:.2f}%"
    )
    print(
        "  coverage_cartesian (unique_evals / total_cartesian): "
        f"median={cc['median'] * 100:.2f}%  "
        f"p25={cc['p25'] * 100:.2f}%  p75={cc['p75'] * 100:.2f}%  "
        f"min={cc['min'] * 100:.2f}%  max={cc['max'] * 100:.2f}%"
    )

    # ── CSV output ────────────────────────────────────────────────────────────
    if args.out_csv:
        import csv
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seed", "score_raw", "score_pen", "gap", "hit_opt", "evals", "wall_s",
                         "coverage_cartesian", "coverage_feasible"])
            for r in runs:
                w.writerow([r.seed, r.score_raw, r.score_pen, r.gap, int(r.hit_opt), r.evals, r.wall_s,
                             r.coverage_cartesian, r.coverage_feasible])
        print(f"\nWrote results CSV -> {args.out_csv}")

    # ── JSON output ───────────────────────────────────────────────────────────
    if args.out_json:
        payload = {
            "instance": {
                "config": args.config,
                "device_space": {"R": ds.R, "N": ds.N, "G": ds.G},
                "workload": {"B": workload.B, "K": workload.K, "S": workload.S, "T": workload.T},
                "k_eval": k_eval,
                "max_per_vertex": max_per_vertex,
                "structural_search": enable_structural_search,
                "optimizer_cfg": {
                    "E_max":      opt_cfg.E_max,
                    "E_struct":   opt_cfg.E_struct,
                    "R_max":      opt_cfg.R_max,
                    "I_max":      opt_cfg.I_max,
                    "lambda_ac":  opt_cfg.lambda_ac,
                    "T_cand": None if opt_cfg.T_cand is None
                              else {m.value: sorted(list(s)) for m, s in opt_cfg.T_cand.items()},
                },
            },
            "exhaustive": {
                "n_total":      ex.n_total,
                "n_feasible":   ex.n_feasible,
                "n_infeasible": ex.n_total - ex.n_feasible,
                "feasible_ratio": ex.n_feasible / max(1, ex.n_total),
                "best_raw": ex.best_feasible_score_raw,
                "best_pen": ex.best_feasible_score_pen,
            },
            "runs":    [r.__dict__ for r in runs],
            "summary": summary,
        }
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote results JSON -> {args.out_json}")


if __name__ == "__main__":
    main()
