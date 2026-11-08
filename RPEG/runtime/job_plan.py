# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shlex

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.types import Allocation, DeviceSpace, Model, packing_geometry
from ir.ac_share import build_AC_share_RPEG
from ir.dpo import build_DPO_RPEG
from ir.grpo import build_GRPO_RPEG
from ir.plan import ExecutionPlan
from ir.ppo import build_PPO_RPEG
from ir.rpeg import RPEG
from ir.serialize import execution_schedule_to_dict, load_plan, optimized_plan_to_dict, save_optimized_plan
from model.cost import NodeCostModel
from model.hardware import Hardware
from model.memory import MemoryPolicy, NodeMemoryModel, optimizer_stage
from model.models import ModelSpec, ModelSuite
from model.workload import Workload
from passes.lower import lower
from passes.unroll import unroll
from runtime.compiler import compile_runtime_plan
from runtime.errors import RuntimeConfigError
from runtime.hf import infer_model_spec, resolve_model_path
from runtime.io import dump_mapping, load_mapping, sha256_file
from runtime.plot_trace import analytical_trace_to_plot, render_trace_svg
from runtime.schema import OptimizedPlan
from runtime.transfer_policy import transfer_resource_policy_from_job_config
from search.cache import EvalCache
from search.evaluate import Configuration, is_valid
from search.moves import build_candidate_directory
from search.optimizer import OptimizerConfig, optimize
from schedule.scheduler import schedule


_NAME_TO_MODEL = {
    "actor": Model.ACTOR,
    "critic": Model.CRITIC,
    "reference": Model.REFERENCE,
    "reward": Model.REWARD,
    "ac_share": Model.AC_SHARE,
}


@dataclass(frozen=True)
class PlannedRuntimeJob:
    baseline_plan_path: Path
    plan_path: Path
    run_path: Path
    sbatch_path: Path | None
    plan_hash: str | None
    optimizer_input_plot_path: Path
    optimizer_output_plot_path: Path
    analytical_summary_path: Path


def plan_runtime_job(config_path: str | Path, out_dir: str | Path | None = None, *, emit_sbatch: bool = False) -> PlannedRuntimeJob:
    cfg_path = Path(config_path).resolve()
    cfg = load_mapping(cfg_path)
    root = Path(out_dir or cfg.get("out_dir") or cfg_path.with_suffix("")).resolve()
    root.mkdir(parents=True, exist_ok=True)

    algorithm = str((cfg.get("policy") or {}).get("algorithm") or (cfg.get("algorithm") or {}).get("name") or "ppo").lower()
    if algorithm not in {"ppo", "grpo", "dpo"}:
        raise RuntimeConfigError(f"unsupported runtime job algorithm {algorithm!r}")
    workload = _workload(cfg, cfg_path.parent)
    device_space = _device_space(cfg)
    suite = _model_suite(cfg, algorithm)
    rpeg = _base_rpeg(cfg, algorithm, suite, workload)
    hardware = _hardware(cfg)
    memory_policy = _memory_policy(cfg)
    cost_model = NodeCostModel(
        hardware,
        suite,
        workload,
        optimizer_stage=memory_policy.optimizer_stage,
        activation_checkpointing=memory_policy.activation_checkpointing,
    )
    mem_model = NodeMemoryModel.from_policy(
        hardware, suite, workload, memory_policy,
        trainable=frozenset(rpeg.trainable),
        shadow_models=frozenset(m for m in rgeg.trainable if rpeg.tau.get(m, 0) > 0),
    )
    transfer_resource_policy = transfer_resource_policy_from_job_config(cfg)
    base_plan = _baseline_plan(rpeg, device_space, suite, workload, cost_model, mem_model)
    base_config = Configuration(
        rpeg=rpeg,
        plan=base_plan,
        device_space=device_space,
        transfer_resource_policy=transfer_resource_policy,
    )
    baseline_feasible = is_valid(base_config, suite, workload, mem_model)
    optimizer_seed_config, optimizer_seed_plan_path = _optimizer_seed_config(
        cfg,
        cfg_path,
        rpeg,
        device_space,
        suite,
        workload,
        mem_model,
        transfer_resource_policy,
        default=base_config,
    )
    optimizer_cfg = _optimizer_config(cfg, algorithm)
    optimizer_seed = int(
        (cfg.get("outputs") or {}).get(
            "seed",
            (cfg.get("algorithm") or {}).get("seed", 0),
        )
    )
    optimizer_options = cfg.get("optimizer") or {}
    feasibility_attempts = int(optimizer_options.get("feasibility_attempts", 3))
    if feasibility_attempts < 1:
        raise RuntimeConfigError("optimizer.feasibility_attempts must be >= 1")
    optimized = None
    optimizer_selected_seed = None
    optimizer_attempt_count = 0
    for attempt in range(feasibility_attempts):
        attempt_seed = optimizer_seed + attempt
        random.seed(attempt_seed)
        candidate = optimize(
            optimizer_seed_config,
            suite,
            workload,
            hardware,
            K_eval=int(optimizer_options.get("k_eval", 1)),
            opt_config=optimizer_cfg,
            cache=EvalCache(),
        )
        optimizer_attempt_count = attempt + 1
        candidate_feasible = is_valid(candidate, suite, workload, mem_model)
        print(
            f"optimizer attempt {optimizer_attempt_count} seed={attempt_seed} "
            f"outer_feasible={candidate_feasible}"
        )
        if candidate_feasible:
            optimized = candidate
            optimizer_selected_seed = attempt_seed
            break
        save_optimized_plan(
            candidate,
            str(root / f"optimizer_infeasible_attempt_{optimizer_attempt_count}.yaml"),
            algorithm=algorithm,
            workload=workload,
            source_config_hash=_source_config_hash(cfg),
        )
    optimized_feasible = optimized is not None
    if not optimized_feasible:
        attempted_seeds = tuple(optimizer_seed + offset for offset in range(feasibility_attempts))
        raise RuntimeConfigError(
            "optimizer did not find a globally memory-feasible execution plan "
            f"after seeds={attempted_seeds}"
        )
    assert optimized is not None

    execution_horizon = int((cfg.get("training") or {}).get("max_steps", 1))
    if execution_horizon < 1:
        raise RuntimeConfigError("training.max_steps must be >= 1")
    source_hash = _source_config_hash(cfg)
    baseline_executable = lower(
        base_config.rpeg,
        unroll(base_config.rpeg, K_eval=execution_horizon),
        base_config.plan,
        suite,
        K_eval=execution_horizon,
        workload=workload,
        intra_stage_fusion=base_config.intra_stage_fusion,
    )
    baseline_trace = schedule(
        baseline_executable,
        cost_model,
        device_space,
        transfer_resource_policy=base_config.transfer_resource_policy,
    )
    baseline_schedule = execution_schedule_to_dict(baseline_trace, horizon=execution_horizon)
    baseline_plan_path = root / "baseline_plan.yaml"
    save_optimized_plan(
        base_config,
        str(baseline_plan_path),
        algorithm=algorithm,
        workload=workload,
        source_config_hash=source_hash,
        execution_schedule=baseline_schedule,
    )
    executable = lower(
        optimized.rpeg,
        unroll(optimized.rpeg, K_eval=execution_horizon),
        optimized.plan,
        suite,
        K_eval=execution_horizon,
        workload=workload,
        intra_stage_fusion=optimized.intra_stage_fusion,
    )
    analytical_trace = schedule(
        executable,
        cost_model,
        device_space,
        transfer_resource_policy=optimized.transfer_resource_policy,
    )
    execution_schedule = execution_schedule_to_dict(analytical_trace, horizon=execution_horizon)
    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    optimizer_input_plot_path = plots_dir / "optimizer_input.gantt.svg"
    optimizer_output_plot_path = plots_dir / "optimizer_output.gantt.svg"
    shared_t_max = max(baseline_trace.makespan(), analytical_trace.makespan())
    render_trace_svg(
        analytical_trace_to_plot(
            baseline_executable,
            baseline_trace,
            run_id=f"{cfg.get('run_id', root.name)}:optimizer_input",
        ),
        optimizer_input_plot_path,
        title="Optimizer input schedule",
        t_max=shared_t_max,
    )
    render_trace_svg(
        analytical_trace_to_plot(
            executable,
            analytical_trace,
            run_id=f"{cfg.get('run_id', root.name)}:optimizer_output",
        ),
        optimizer_output_plot_path,
        title="Optimizer output schedule",
        t_max=shared_t_max,
    )
    analytical_summary_path = root / "metrics" / "analytical_comparison.json"
    analytical_summary_path.parent.mkdir(parents=True, exist_ok=True)
    analytical_summary_path.write_text(
        json.dumps(
            {
                "schema_version": "rpeg.analytical_comparison.v1",
                "run_id": str(cfg.get("run_id") or root.name),
                "optimizer_seed": optimizer_seed,
                "optimizer_seed_plan_path": optimizer_seed_plan_path,
                "optimizer_selected_seed": optimizer_selected_seed,
                "optimizer_attempt_count": optimizer_attempt_count,
                "horizon": execution_horizon,
                "transfer_resource_policy": transfer_resource_policy.to_dict(),
                "optimizer_input": {
                    "plan_path": str(baseline_plan_path),
                    "plot_path": str(optimizer_input_plot_path),
                    "predicted_makespan_s": baseline_trace.makespan(),
                    "predicted_utilization": baseline_trace.utilization(),
                    "node_count": len(baseline_executable.nodes),
                    "feasible": baseline_feasible,
                },
                "optimizer_output": {
                    "plan_path": str(root / "optimized_plan.yaml"),
                    "plot_path": str(optimizer_output_plot_path),
                    "predicted_makespan_s": analytical_trace.makespan(),
                    "predicted_utilization": analytical_trace.utilization(),
                    "node_count": len(executable.nodes),
                    "feasible": optimized_feasible,
                },
                "comparison_status": "comparable" if baseline_feasible else "baseline_infeasible",
                "predicted_makespan_improvement_fraction": (
                    None
                    if not baseline_feasible
                    else 0.0
                    if baseline_trace.makespan() <= 0.0
                    else 1.0 - analytical_trace.makespan() / baseline_trace.makespan()
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    plan_path = root / "optimized_plan.yaml"
    save_optimized_plan(
        optimized,
        str(plan_path),
        algorithm=algorithm,
        workload=workload,
        source_config_hash=source_hash,
        execution_schedule=execution_schedule,
    )
    plan_dict = optimized_plan_to_dict(
        optimized,
        algorithm=algorithm,
        workload=workload,
        source_config_hash=source_hash,
        execution_schedule=execution_schedule,
    )
    plan_file_hash = sha256_file(plan_path)
    run = _run_config(
        cfg,
        algorithm,
        workload,
        device_space,
        plan_path,
        plan_file_hash,
        root,
        intra_stage_fusion=optimized.intra_stage_fusion,
    )
    run_path = root / "run.yaml"
    dump_mapping(run, run_path)
    compile_runtime_plan(run_path, dry_run=True)

    sbatch_path = None
    if emit_sbatch or bool((cfg.get("outputs") or {}).get("emit_sbatch", False)):
        sbatch_path = root / "submit.sbatch"
        sbatch_path.write_text(_sbatch_script(cfg, run_path, device_space, root))
    return PlannedRuntimeJob(
        baseline_plan_path=baseline_plan_path,
        plan_path=plan_path,
        run_path=run_path,
        sbatch_path=sbatch_path,
        plan_hash=plan_dict.get("plan_hash"),
        optimizer_input_plot_path=optimizer_input_plot_path,
        optimizer_output_plot_path=optimizer_output_plot_path,
        analytical_summary_path=analytical_summary_path,
    )


def _source_config_hash(cfg: dict[str, Any]) -> str:
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _optimizer_seed_config(
    cfg: dict[str, Any],
    cfg_path: Path,
    rpeg: RPEG,
    device_space: DeviceSpace,
    suite: ModelSuite,
    workload: Workload,
    mem_model: NodeMemoryModel,
    transfer_resource_policy,
    *,
    default: Configuration,
) -> tuple[Configuration, str | None]:
    """
    Load an optional feasible incumbent without weakening plan exactness.
    """
    raw_path = (cfg.get("optimizer") or {}).get("seed_plan_path")
    if raw_path is None:
        return default, None
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = cfg_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise RuntimeConfigError(f"optimizer.seed_plan_path not found: {path}")

    artifact = load_mapping(path)
    parsed = OptimizedPlan.from_dict(artifact)
    expected_algorithm = str((cfg.get("policy") or {}).get("algorithm", "ppo")).lower()
    if parsed.algorithm != expected_algorithm:
        raise RuntimeConfigError(
            "optimizer.seed_plan_path algorithm does not match the optimizer input"
        )
    expected_workload = (
        workload.B,
        workload.K,
        workload.S,
        workload.T,
        workload.gen_quantiles,
    )
    observed_workload = (
        parsed.workload.batch_size,
        parsed.workload.samples_per_prompt,
        parsed.workload.context_length,
        parsed.workload.generation_length,
        parsed.workload.gen_quantiles,
    )
    if observed_workload != expected_workload:
        raise RuntimeConfigError(
            "optimizer.seed_plan_path workload does not match the optimizer input"
        )
    expected_space = {
        "racks": device_space.R,
        "nodes_per_rack": device_space.N,
        "gpus_per_node": device_space.G,
    }
    if artifact.get("device_space") != expected_space:
        raise RuntimeConfigError(
            "optimizer.seed_plan_path device_space does not match the optimizer input"
        )
    expected_vertices = {vertex.id for vertex in rpeg.vertices}
    plan_vertices = set((artifact.get("plan") or {}).keys())
    if plan_vertices != expected_vertices:
        raise RuntimeConfigError(
            "optimizer.seed_plan_path vertices do not match the optimizer RPEG: "
            f"expected={sorted(expected_vertices)}, observed={sorted(plan_vertices)}"
        )
    structure = artifact.get("structure") or {}
    expected_tau = {
        next(name for name, model in _NAME_TO_MODEL.items() if model == key): int(value)
        for key, value in rpeg.tau.items()
    }
    observed_tau = {str(key): int(value) for key, value in (structure.get("tau") or {}).items()}
    if observed_tau != expected_tau:
        raise RuntimeConfigError(
            "optimizer.seed_plan_path staleness structure does not match the optimizer input"
        )
    expected_ac_share = Model.AC_SHARE in rpeg.trainable
    if bool(structure.get("ac_share", False)) != expected_ac_share:
        raise RuntimeConfigError(
            "optimizer.seed_plan_path actor-critic sharing does not match the optimizer input"
        )
    observed_transfer_policy = structure.get("transfer_resource_policy")
    if observed_transfer_policy != transfer_resource_policy.to_dict():
        raise RuntimeConfigError(
            "optimizer.seed_plan_path transfer resource policy does not match the optimizer input"
        )

    seed = Configuration(
        rpeg=rpeg,
        plan=load_plan(str(path), rpeg),
        device_space=device_space,
        intra_stage_fusion=bool(structure.get("intra_stage_fusion", False)),
        transfer_resource_policy=transfer_resource_policy,
    )
    if not is_valid(seed, suite, workload, mem_model):
        raise RuntimeConfigError(
            "optimizer.seed_plan_path is not feasible under the current memory policy"
        )
    return seed, str(path)


def _workload(cfg: dict[str, Any], base_dir: Path | None = None) -> Workload:
    w = cfg.get("workload") or {}
    quantiles = w.get("gen_quantiles")
    profile_ref = cfg.get("generation_profile") or {}
    if quantiles is None and profile_ref.get("path"):
        profile_path = Path(str(profile_ref["path"]))
        if not profile_path.is_absolute() and base_dir is not None:
            profile_path = base_dir / profile_path
        profile = json.loads(profile_path.read_text())
        if profile.get("schema_version") != "rpeg.generation_profile.v1":
            raise RuntimeConfigError(f"unsupported generation profile schema in {profile_path}")
        if int(profile.get("max_new_tokens", -1)) != int(w["generation_length"]):
            raise RuntimeConfigError(
                f"generation profile max_new_tokens does not match workload.generation_length in {profile_path}"
            )
        quantiles = profile.get("quantiles")
    return Workload(
        B=int(w["batch_size"]),
        K=int(w.get("samples_per_prompt", 1)),
        S=int(w["context_length"]),
        T=int(w["generation_length"]),
        gen_quantiles=tuple(tuple(q) for q in quantiles) if quantiles else None,
    )


def _device_space(cfg: dict[str, Any]) -> DeviceSpace:
    ds = cfg.get("device_space") or cfg.get("hardware") or {}
    return DeviceSpace(
        R=int(ds.get("racks", 1)),
        N=int(ds.get("nodes_per_rack", 1)),
        G=int(ds.get("gpus_per_node", 1)),
    )


def _hardware(cfg: dict[str, Any]) -> Hardware:
    hw = cfg.get("hardware") or {}
    if "alpha_coll_g" in hw:
        return Hardware(
            flops_per_gpu=float(hw["flops_per_gpu"]),
            memory_per_gpu=float(hw["memory_per_gpu"]),
            memory_bandwidth=float(hw["memory_bandwidth"]),
            alpha_coll_g=float(hw["alpha_coll_g"]),
            beta_coll_g=float(hw["beta_coll_g"]),
            alpha_coll_n=float(hw["alpha_coll_n"]),
            beta_coll_n=float(hw["beta_coll_n"]),
            alpha_coll_r=float(hw["alpha_coll_r"]),
            beta_coll_r=float(hw["beta_coll_r"]),
            alpha_p2p_g=float(hw["alpha_p2p_g"]),
            beta_p2p_g=float(hw["beta_p2p_g"]),
            alpha_p2p_n=float(hw["alpha_p2p_n"]),
            beta_p2p_n=float(hw["beta_p2p_n"]),
            alpha_p2p_r=float(hw["alpha_p2p_r"]),
            beta_p2p_r=float(hw["beta_p2p_r"]),
            gemm_m_half=float(hw.get("gemm_m_half", 5.0)),
            gemm_k_half=float(hw.get("gemm_k_half", 0.0)),
            decode_bw_eff=float(hw.get("decode_bw_eff", 0.15)),
            non_gemm_constant=float(hw.get("non_gemm_constant", 0.0)),
            non_gemm_scaling=float(hw.get("non_gemm_scaling", 0.0)),
            non_gemm_batch_cal=float(hw.get("non_gemm_batch_cal", 64)),
            op_generation_scale=float(hw.get("op_generation_scale", 1.0)),
            op_inference_scale=float(hw.get("op_inference_scale", 1.0)),
            op_training_scale=float(hw.get("op_training_scale", 1.0)),
            activation_comm_overhead_s=float(hw.get("activation_comm_overhead_s", 0.0)),
            weight_comm_scale=float(hw.get("weight_comm_scale", 1.0)),
            reshard_scale=float(hw.get("reshard_scale", 1.0)),
        )
    net = hw.get("network") or {}
    intra = net.get("intra_node") or {"latency": 1e-6, "bandwidth": 270e9}
    rack = net.get("intra_rack") or {"latency": 50e-6, "bandwidth": 22e9}
    inter = net.get("inter_rack") or {"latency": 100e-6, "bandwidth": 11e9}
    return Hardware(
        flops_per_gpu=float(hw.get("flops_per_gpu", 280e12)),
        memory_per_gpu=float(hw.get("memory_per_gpu", hw.get("memory_per_gpu_bytes", 80e9))),
        memory_bandwidth=float(hw.get("memory_bandwidth", 1.8e12)),
        alpha_coll_g=float(intra["latency"]),
        beta_coll_g=1.0 / float(intra["bandwidth"]),
        alpha_coll_n=float(rack["latency"]),
        beta_coll_n=1.0 / float(rack["bandwidth"]),
        alpha_coll_r=float(inter["latency"]),
        beta_coll_r=1.0 / float(inter["bandwidth"]),
        alpha_p2p_g=float(intra["latency"]),
        beta_p2p_g=1.0 / float(intra["bandwidth"]),
        alpha_p2p_n=float(rack["latency"]),
        beta_p2p_n=1.0 / float(rack["bandwidth"]),
        alpha_p2p_r=float(inter["latency"]),
        beta_p2p_r=1.0 / float(inter["bandwidth"]),
        gemm_m_half=float(hw.get("gemm_m_half", 5.0)),
        decode_bw_eff=float(hw.get("decode_bw_eff", 0.15)),
        op_generation_scale=float(hw.get("op_generation_scale", 1.0)),
        op_inference_scale=float(hw.get("op_inference_scale", 1.0)),
        op_training_scale=float(hw.get("op_training_scale", 1.0)),
        activation_comm_overhead_s=float(hw.get("activation_comm_overhead_s", 0.0)),
        weight_comm_scale=float(hw.get("weight_comm_scale", 1.0)),
        reshard_scale=float(hw.get("reshard_scale", 1.0)),
    )


def _model_suite(cfg: dict[str, Any], algorithm: str) -> ModelSuite:
    models = cfg.get("models") or {}
    actor = _model_spec(models, "actor")
    specs = {
        Model.ACTOR: actor,
        Model.REFERENCE: _model_spec(models, "reference", fallback=actor),
    }
    if algorithm in {"ppo", "grpo"}:
        specs[Model.REWARD] = _model_spec(models, "reward", fallback=actor)
    if algorithm == "ppo":
        specs[Model.CRITIC] = _model_spec(models, "critic", fallback=actor)
        specs[Model.AC_SHARE] = actor
    return ModelSuite(specs)


def _model_spec(models: dict[str, Any], role: str, fallback: ModelSpec | None = None) -> ModelSpec:
    m = models.get(role)
    if m is None or str(m.get("source", "")).lower() == "actor":
        if fallback is not None:
            return fallback
        raise RuntimeConfigError(f"runtime job config requires models.{role}")
    if {"layers", "hidden_dim", "ffn_dim", "vocab_size"} <= set(m):
        return ModelSpec(
            L=int(m["layers"]),
            d=int(m["hidden_dim"]),
            d_ff=int(m["ffn_dim"]),
            V=int(m["vocab_size"]),
            dtype_bytes=int(m.get("dtype_bytes", 2)),
            d_head=int(m.get("head_dim", 128)),
            parameter_count=(
                None if m.get("parameter_count") is None else int(m["parameter_count"])
            ),
        )
    path = resolve_model_path(
        str(m.get("source", "local")).lower(),
        m.get("repo"),
        m.get("checkpoint_path"),
        revision=m.get("revision"),
    )
    return infer_model_spec(path)


def _base_rpeg(cfg: dict[str, Any], algorithm: str, suite: ModelSuite, workload: Workload):
    tau = {
        _NAME_TO_MODEL[name]: int(value)
        for name, value in ((cfg.get("policy") or {}).get("tau") or {}).items()
        if name in _NAME_TO_MODEL
    }
    if algorithm == "dpo":
        return build_DPO_RPEG(suite, workload, tau=tau or None)
    if algorithm == "grpo":
        return build_GRPO_RPEG(suite, workload, tau=tau or None)
    if bool((cfg.get("policy") or {}).get("ac_share", False)):
        return build_AC_share_RPEG(suite, workload, tau=tau or None)
    return build_PPO_RPEG(suite, workload, tau=tau or None)


def _baseline_plan(rpeg: RPEG, device_space: DeviceSpace, suite: ModelSuite, workload: Workload, cost_model, mem_model) -> ExecutionPlan:
    table, _ = build_candidate_directory(rpeg, suite, device_space, workload, cost_model, mem_model)
    sigma, pi, mb = {}, {}, {}
    for vertex in rpeg.vertices:
        if not table[vertex]:
            raise RuntimeConfigError(f"no feasible strategy candidates for vertex {vertex.id}")
        dp, pp, tp, r_ext, n_ext, mb_v, _time = table[vertex][0]
        geom = packing_geometry(int(dp), int(pp), int(tp), device_space.G)
        if geom is None:
            raise RuntimeConfigError(f"infeasible packing geometry for vertex {vertex.id}: dp={dp} pp={pp} tp={tp}")
        g_ext, _nodes_needed = geom
        sigma[vertex] = Allocation(origin=(0, 0, 0), extent=(int(r_ext), int(n_ext), int(g_ext)))
        pi[vertex] = (int(dp), int(pp), int(tp))
        mb[vertex] = int(mb_v)
    return ExecutionPlan(sigma, pi, mb)


def _optimizer_config(cfg: dict[str, Any], algorithm: str) -> OptimizerConfig:
    o = cfg.get("optimizer") or {}
    memory_policy = _memory_policy(cfg)
    staleness = o.get("staleness_candidates")
    structural_candidates = o.get("structural_candidates")
    if structural_candidates is not None:
        if not isinstance(structural_candidates, (list, tuple, set, frozenset)):
            raise RuntimeConfigError("optimizer.structural_candidates must be a list")
        structural_candidates = frozenset(str(value) for value in structural_candidates)
    t_cand = None
    if staleness is not None:
        t_cand = {
            _NAME_TO_MODEL[name]: {int(v) for v in values}
            for name, values in staleness.items()
            if name in _NAME_TO_MODEL
        }
        required = {Model.ACTOR}
        if algorithm == "ppo":
            required.update({Model.CRITIC, Model.AC_SHARE})
        for model in required:
            t_cand.setdefault(model, {0})
    try:
        return OptimizerConfig(
            K_x=int(o.get("k_x", 1)),
            T_max=int(o.get("t_max", 10)),
            E_max=int(o.get("e_max", 8)),
            E_struct=int(o.get("e_struct", 4)),
            T_init=float(o.get("t_init", 0.30)),
            T_warm=float(o.get("t_warm", 0.05)),
            T_min=float(o.get("t_min", 0.01)),
            K_elite=int(o.get("k_elite", 3)),
            T_cand=t_cand,
            structural_candidates=structural_candidates,
            lambda_ac=float(o.get("lambda_ac", 0.15)),
            score_eps=float(o.get("score_eps", 1e-4)),
            stag_window=int(o.get("stag_window", 4)),
            alpha_pair=float(o.get("alpha_pair", 0.28)),
            max_throughput_loss=float(o.get("max_throughput_loss", 0.0)),
            algorithm=algorithm,
            fixed_mb=bool(o.get("fixed_mb", False)),
            uniform_vertex=bool(o.get("uniform_vertex", False)),
            no_polish=bool(o.get("no_polish", False)),
            hard_oom=bool(o.get("hard_oom", False)),
            geometric_cooling=bool(o.get("geometric_cooling", False)),
            memory_optimizer_stage=memory_policy.optimizer_stage,
            memory_activation_checkpointing=memory_policy.activation_checkpointing,
            memory_runtime_overhead_fraction=memory_policy.runtime_overhead_fraction,
            memory_worker_overhead_bytes=memory_policy.worker_overhead_bytes,
            memory_semantic_activation_cache_bytes=memory_policy.semantic_activation_cache_bytes,
            memory_max_worker_groups_per_gpu=memory_policy.max_worker_groups_per_gpu,
        )
    except ValueError as exc:
        raise RuntimeConfigError(str(exc)) from exc


def _memory_policy(cfg: dict[str, Any]) -> MemoryPolicy:
    optimizer_runtime = cfg.get("optimizer_runtime") or {}
    optimizer = cfg.get("optimizer") or {}
    runtime_hardware = cfg.get("runtime_hardware") or {}
    mode = str(optimizer_runtime.get("distributed_optimizer", "zero1"))
    try:
        stage = optimizer_stage(mode)
    except ValueError as exc:
        raise RuntimeConfigError(str(exc)) from exc
    return MemoryPolicy(
        optimizer_stage=stage,
        activation_checkpointing=bool(optimizer_runtime.get("activation_checkpointing", True)),
        runtime_overhead_fraction=float(
            optimizer.get("memory_reserve_fraction", runtime_hardware.get("memory_reserve_fraction", 0.10))
        ),
        worker_overhead_bytes=float(runtime_hardware.get("worker_overhead_bytes", 10 * 1024**3)),
        semantic_activation_cache_bytes=float(
            runtime_hardware.get("semantic_activation_cache_bytes", 0)
        ),
        max_worker_groups_per_gpu=int(runtime_hardware.get("max_worker_groups_per_gpu", 0)),
    )


def _run_config(
    cfg: dict[str, Any],
    algorithm: str,
    workload: Workload,
    device_space: DeviceSpace,
    plan_path: Path,
    expected_hash: str | None,
    root: Path,
    *,
    intra_stage_fusion: bool = False,
) -> dict[str, Any]:
    run_id = str(cfg.get("run_id") or root.name)
    runtime = dict(cfg.get("runtime") or {})
    checkpoints = dict(cfg.get("checkpoints") or {})
    cluster = dict(cfg.get("cluster") or {})
    launcher = dict(cluster.get("launcher") or {})
    if launcher.get("type") == "slurm":
        launcher.setdefault("shared_step", True)
        cluster["launcher"] = launcher
    data = dict(cfg.get("data") or {})
    monitoring = {
        "summary_path": str(root / "metrics" / f"{run_id}.summary.json"),
        "metrics_path": str(root / "metrics" / f"{run_id}.prom"),
        "alert_path": str(root / "metrics" / f"{run_id}.alerts.json"),
        "dashboard_path": str(root / "metrics" / f"{run_id}.dashboard.html"),
        **dict(cfg.get("monitoring") or {}),
    }
    models = _runtime_models(cfg, algorithm)
    default_backend = runtime.pop("default_backend", None)
    if default_backend is None:
        if (cluster.get("launcher") or {}).get("type") == "slurm" or algorithm == "grpo":
            default_backend = "megatron"
        elif algorithm == "dpo":
            default_backend = "torch_dpo"
        else:
            default_backend = "torch_ppo"
    planner_hardware = _hardware(cfg)
    runtime_hardware = {
        "gpu_type": (cfg.get("hardware") or {}).get("gpu_type", "unknown"),
        "memory_per_gpu_bytes": int(planner_hardware.memory_per_gpu),
        "worker_overhead_bytes": int(_memory_policy(cfg).worker_overhead_bytes),
        "semantic_activation_cache_bytes": int(
            _memory_policy(cfg).semantic_activation_cache_bytes
        ),
        "max_worker_groups_per_gpu": _memory_policy(cfg).max_worker_groups_per_gpu,
        "memory_reserve_fraction": _memory_policy(cfg).runtime_overhead_fraction,
    }
    runtime_hardware.update(dict(cfg.get("runtime_hardware") or {}))
    return {
        "schema_version": "rpeg.run.v1",
        "run_id": run_id,
        "optimized_plan": {
            "path": str(plan_path),
            "expected_hash": expected_hash,
            "require_exact_match": True,
        },
        "algorithm": {"name": algorithm, **dict(cfg.get("algorithm") or {})},
        "training": {"max_steps": 1, "save_every_steps": 1, "eval_every_steps": 1, "log_every_steps": 1, **dict(cfg.get("training") or {})},
        "workload": {
            "batch_size": workload.B,
            "samples_per_prompt": workload.K,
            "context_length": workload.S,
            "generation_length": workload.T,
            "gen_quantiles": [list(q) for q in workload.gen_quantiles] if workload.gen_quantiles else None,
        },
        "models": models,
        "tokenizer": _tokenizer_config(cfg, models["actor"]),
        "device_space": {"racks": device_space.R, "nodes_per_rack": device_space.N, "gpus_per_node": device_space.G},
        "hardware": runtime_hardware,
        "cluster": {"scheduler": "local", "launcher": {"type": "local"}, **cluster},
        "runtime": {
            "default_backend": default_backend,
            "intra_stage_fusion": bool(intra_stage_fusion),
            "dry_run": False,
            "log_dir": str(root / "logs"),
            "store_dir": str(root / "store"),
            "trace_path": str(root / "traces" / f"{run_id}.jsonl"),
            **runtime,
            "require_optimized_schedule": True,
        },
        "data": data,
        "optimizer_runtime": {
            "learning_rate_actor": 1e-6,
            "learning_rate_critic": 5e-6,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "activation_checkpointing": True,
            "distributed_optimizer": "zero1",
            **dict(cfg.get("optimizer_runtime") or {}),
        },
        "checkpoints": {
            "output_dir": str(root / "checkpoints"),
            "interval_steps": 100,
            "format": "megatron_dist",
            **checkpoints,
        },
        "monitoring": monitoring,
    }


def _runtime_models(cfg: dict[str, Any], algorithm: str) -> dict[str, dict[str, Any]]:
    models = cfg.get("models") or {}
    actor = _runtime_model_config(models, "actor")
    out = {"actor": actor}
    reference = models.get("reference")
    out["reference"] = _runtime_model_config(models, "reference") if reference else {"source": "actor", "role": "frozen", "dtype": actor.get("dtype", "bf16")}
    if algorithm in {"ppo", "grpo"}:
        out["reward"] = _runtime_model_config(models, "reward")
    if algorithm == "ppo":
        out["critic"] = _runtime_model_config(models, "critic")
    return out


def _runtime_model_config(models: dict[str, Any], role: str) -> dict[str, Any]:
    src = dict(models.get(role) or {})
    allowed = {"source", "role", "repo", "revision", "checkpoint_path", "dtype", "trust_remote_code"}
    if "source" not in src:
        if src.get("checkpoint_path"):
            src["source"] = "local"
        elif src.get("repo"):
            src["source"] = "huggingface"
        elif role == "actor":
            raise RuntimeConfigError(
                "runtime job config requires models.actor.checkpoint_path or models.actor.repo "
                "so the generated run config can load a real model"
            )
        else:
            src["source"] = "actor"
    if role == "actor" and str(src.get("source", "")).lower() == "actor":
        raise RuntimeConfigError("models.actor.source cannot be 'actor'")
    src.setdefault("role", "frozen" if role in {"reference", "reward"} else "trainable")
    src.setdefault("dtype", "bf16")
    return {key: value for key, value in src.items() if key in allowed and value is not None}


def _tokenizer_config(cfg: dict[str, Any], actor_model: dict[str, Any]) -> dict[str, Any]:
    tokenizer = dict(cfg.get("tokenizer") or {})
    if not tokenizer:
        tokenizer = {
            "source": actor_model.get("source", "local"),
            "repo": actor_model.get("repo"),
            "revision": actor_model.get("revision"),
            "checkpoint_path": actor_model.get("checkpoint_path"),
        }
    tokenizer.setdefault("padding_side", "left")
    tokenizer.setdefault("truncation_side", "left")
    return {key: value for key, value in tokenizer.items() if value is not None}


def _sbatch_script(cfg: dict[str, Any], run_path: Path, device_space: DeviceSpace, root: Path) -> str:
    cluster = cfg.get("cluster") or {}
    launcher = cluster.get("launcher") or {}
    nodes = device_space.R * device_space.N
    tasks = nodes * device_space.G
    environment_script = str(launcher.get("environment_script") or "scripts/clariden/runtime_env.sh")
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={cfg.get('run_id', 'rpeg-runtime')}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks={tasks}",
        f"#SBATCH --ntasks-per-node={device_space.G}",
        f"#SBATCH --gpus-per-node={device_space.G}",
        "#SBATCH --exclusive",
    ]
    if launcher.get("partition"):
        lines.append(f"#SBATCH --partition={launcher['partition']}")
    if launcher.get("account"):
        lines.append(f"#SBATCH --account={launcher['account']}")
    if launcher.get("uenv"):
        lines.append(f"#SBATCH --uenv={launcher['uenv']}")
    environment_exports = []
    for key, value in (launcher.get("env") or {}).items():
        key = str(key)
        if not key.replace("_", "a").isalnum() or key[0].isdigit():
            raise RuntimeConfigError(f"invalid launcher environment variable name {key!r}")
        environment_exports.append(f"export {key}={shlex.quote(str(value))}")
    if launcher.get("view"):
        lines.append(f"#SBATCH --view={launcher['view']}")
    if launcher.get("time_limit"):
        lines.append(f"#SBATCH --time={launcher['time_limit']}")
    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"cd \"${{REPO_ROOT:-{Path.cwd()}}}\"",
            *environment_exports,
            f"source {shlex.quote(environment_script)}",
            "export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}",
            f"mkdir -p '{root / 'slurm'}'",
            f"python -m runtime.run --config '{run_path}'",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan an optimized RPEG runtime job from a high-level config")
    parser.add_argument("--config", required=True, help="High-level optimizer/runtime config")
    parser.add_argument("--out-dir", default=None, help="Directory for optimized_plan.yaml, run.yaml, and optional submit.sbatch")
    parser.add_argument("--emit-sbatch", action="store_true", help="Write submit.sbatch for Slurm execution")
    args = parser.parse_args(argv)
    try:
        job = plan_runtime_job(args.config, args.out_dir, emit_sbatch=args.emit_sbatch)
    except RuntimeConfigError as exc:
        print(f"runtime job planning error: {exc}")
        return 2
    print(
        json.dumps(
            {
                "baseline_plan_path": str(job.baseline_plan_path),
                "plan_path": str(job.plan_path),
                "run_path": str(job.run_path),
                "sbatch_path": None if job.sbatch_path is None else str(job.sbatch_path),
                "plan_hash": job.plan_hash,
                "optimizer_input_plot_path": str(job.optimizer_input_plot_path),
                "optimizer_output_plot_path": str(job.optimizer_output_plot_path),
                "analytical_summary_path": str(job.analytical_summary_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
