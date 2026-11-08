# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.types import (
    Allocation,
    DeviceSpace,
    Model,
    Role,
    TemplateVertex,
    TransferResourcePolicy,
    packing_geometry,
)
from ir.ac_share import build_AC_share_RPEG
from ir.dpo import build_DPO_RPEG
from ir.grpo import build_GRPO_RPEG
from ir.plan import ExecutionPlan
from ir.ppo import build_PPO_RPEG
from ir.rpeg import RPEG
from model.hardware import Hardware
from model.memory import MemoryPolicy, NodeMemoryModel, mem_worst_violation, optimizer_stage
from model.models import ModelSpec, ModelSuite
from model.workload import Workload
from passes.lower import ExecNodeType, ExecutableRPEG, lower
from passes.transfer_resources import reserved_allocations
from passes.unroll import UnrolledRPEG, unroll
from runtime.cluster import ClusterInventory, GPURef, load_inventory
from runtime.errors import RuntimeConfigError
from runtime.hf import infer_model_spec, resolve_model_path
from runtime.io import load_mapping, sha256_file
from runtime.schema import OptimizedPlan, RuntimeRunConfig
from runtime.transfer_policy import transfer_resource_policy_from_runtime
from search.evaluate import (
    Configuration,
    allocations_overlap,
    is_compatible,
    stale_publication_layout_valid,
)


_MODEL_TO_NAME = {
    Model.ACTOR: "actor",
    Model.CRITIC: "critic",
    Model.REFERENCE: "reference",
    Model.REWARD: "reward",
    Model.AC_SHARE: "ac_share",
}
_NAME_TO_MODEL = {v: k for k, v in _MODEL_TO_NAME.items()}
_ROLE_TO_VALUE = {
    Role.GENERATION: "GEN",
    Role.INFERENCE: "INF",
    Role.TRAINING: "TRAIN",
}


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    vertex_id: str
    model: str
    role: str
    backend: str
    allocation: Allocation
    devices: tuple[GPURef, ...]
    parallelism: tuple[int, int, int]
    microbatch: int
    rendezvous_port: int
    vertex_ids: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    vertex_parallelisms: tuple[tuple[str, tuple[int, int, int]], ...] = ()
    vertex_microbatches: tuple[tuple[str, int], ...] = ()
    vertex_roles: tuple[tuple[str, str], ...] = ()

    @property
    def owned_vertex_ids(self) -> tuple[str, ...]:
        return self.vertex_ids or (self.vertex_id,)

    @property
    def owns_training(self) -> bool:
        return "TRAIN" in (self.roles or (self.role,))

    def parallelism_for(self, vertex_id: str) -> tuple[int, int, int]:
        return dict(self.vertex_parallelisms).get(vertex_id, self.parallelism)

    def microbatch_for(self, vertex_id: str) -> int:
        return dict(self.vertex_microbatches).get(vertex_id, self.microbatch)

    def role_for(self, vertex_id: str) -> str:
        return dict(self.vertex_roles).get(vertex_id, self.role)

    @property
    def has_layout_transitions(self) -> bool:
        return len({parallelism for _vertex, parallelism in self.vertex_parallelisms}) > 1


@dataclass(frozen=True)
class TransferSpec:
    node_id: str
    kind: str
    label: str | None
    payload_in_bytes: int
    src_allocation: Allocation
    dst_allocation: Allocation


@dataclass(frozen=True)
class CompiledRuntimePlan:
    run_config: RuntimeRunConfig
    optimized_plan: OptimizedPlan
    device_space: DeviceSpace
    model_suite: ModelSuite
    workload: Workload
    config: Configuration
    unrolled: UnrolledRPEG
    lowered: ExecutableRPEG
    inventory: ClusterInventory
    workers: tuple[WorkerSpec, ...]
    transfers: tuple[TransferSpec, ...]
    execution_order: tuple[str, ...] | None = None

    def runtime_graph(self):
        from runtime.graph import build_runtime_graph

        return build_runtime_graph(
            self.lowered,
            self.workers,
            preferred_order=self.execution_order,
            transfer_resource_policy=self.config.transfer_resource_policy,
            checkpoint_barrier_steps=tuple(
                step - 1
                for step in range(1, self.run_config.training.max_steps + 1)
                if self.run_config.checkpoints.interval_steps > 0
                and step % self.run_config.checkpoints.interval_steps == 0
            ),
        )

    def summary(self) -> dict[str, Any]:
        runtime_graph = self.runtime_graph()
        return {
            "run_id": self.run_config.run_id,
            "algorithm": self.run_config.algorithm.name,
            "workers": [
                {
                    "worker_id": w.worker_id,
                    "vertex_id": w.vertex_id,
                    "vertex_ids": list(w.owned_vertex_ids),
                    "model": w.model,
                    "role": w.role,
                    "backend": w.backend,
                    "parallelism": {"dp": w.parallelism[0], "pp": w.parallelism[1], "tp": w.parallelism[2]},
                    "vertex_parallelisms": {
                        vertex_id: {"dp": pi[0], "pp": pi[1], "tp": pi[2]}
                        for vertex_id, pi in w.vertex_parallelisms
                    },
                    "microbatch": w.microbatch,
                    "devices": [f"{d.hostname}:cuda:{d.local_cuda_index}" for d in w.devices],
                    "rendezvous_port": w.rendezvous_port,
                }
                for w in self.workers
            ],
            "transfers": [
                {
                    "node_id": t.node_id,
                    "kind": t.kind,
                    "label": t.label,
                    "payload_in_bytes": t.payload_in_bytes,
                }
                for t in self.transfers
            ],
            "lowered_nodes": len(self.lowered.nodes),
            "lowered_edges": len(self.lowered.edges),
            "runtime_graph": {
                "node_count": len(runtime_graph.nodes),
                "edge_count": len(runtime_graph.edges),
                "topological_order": list(runtime_graph.topological_order),
                "counts": runtime_graph.summary()["counts"],
            },
        }


def _workload_from_config(run: RuntimeRunConfig) -> Workload:
    w = run.workload
    return Workload(
        B=w.batch_size,
        K=w.samples_per_prompt,
        S=w.context_length,
        T=w.generation_length,
        gen_quantiles=w.gen_quantiles,
    )


def _device_space_from_config(run: RuntimeRunConfig) -> DeviceSpace:
    ds = run.device_space
    return DeviceSpace(R=ds.racks, N=ds.nodes_per_rack, G=ds.gpus_per_node)


def _tau_from_plan(plan: OptimizedPlan) -> dict[Model, int]:
    tau_d = plan.structure.get("tau") or {}
    out = {}
    for name, value in tau_d.items():
        model = _NAME_TO_MODEL.get(str(name))
        if model is not None:
            out[model] = int(value)
    return out


def _build_rpeg(run: RuntimeRunConfig, opt: OptimizedPlan, suite: ModelSuite):
    tau = _tau_from_plan(opt)
    workload = _workload_from_config(run)
    if run.algorithm.name == "ppo":
        if bool(opt.structure.get("ac_share", False)):
            return build_AC_share_RPEG(suite, workload, tau=tau or None)
        return build_PPO_RPEG(suite, workload, tau=tau or None)
    if run.algorithm.name == "grpo":
        return build_GRPO_RPEG(suite, workload, tau=tau or None)
    if run.algorithm.name == "dpo":
        return build_DPO_RPEG(suite, workload, tau=tau or None)
    raise RuntimeConfigError(f"Unsupported algorithm {run.algorithm.name!r}")


def _role_model_name(v: TemplateVertex) -> str:
    return _MODEL_TO_NAME[v.kind.model]


def _build_execution_plan(rpeg: RPEG, opt: OptimizedPlan) -> ExecutionPlan:
    vertices = {v.id: v for v in rpeg.vertices}
    sigma, pi, mb = {}, {}, {}
    missing = set(vertices) - set(opt.vertices)
    extra = set(opt.vertices) - set(vertices)
    if missing:
        raise RuntimeConfigError(f"optimized plan missing vertices: {sorted(missing)}")
    if extra:
        raise RuntimeConfigError(f"optimized plan has unknown vertices: {sorted(extra)}")
    for vid, v in vertices.items():
        pv = opt.vertices[vid]
        expected_model = _role_model_name(v)
        expected_role = _ROLE_TO_VALUE[v.kind.role]
        if pv.model != expected_model:
            raise RuntimeConfigError(f"plan.{vid}.model={pv.model!r}, expected {expected_model!r}")
        if pv.role != expected_role:
            raise RuntimeConfigError(f"plan.{vid}.role={pv.role!r}, expected {expected_role!r}")
        sigma[v] = Allocation(origin=pv.origin, extent=pv.extent)
        pi[v] = pv.parallelism
        mb[v] = pv.microbatch
    return ExecutionPlan(sigma=sigma, pi=pi, mb=mb)


def _model_spec_for_role(run: RuntimeRunConfig, role_name: str, fallback: ModelSpec | None = None) -> ModelSpec:
    cfg = run.models.get(role_name)
    if cfg is None:
        if fallback is not None:
            return fallback
        raise RuntimeConfigError(f"models.{role_name} is required")
    if cfg.source == "actor":
        if fallback is None:
            return _model_spec_for_role(run, "actor")
        return fallback
    model_dir = resolve_model_path(cfg.source, cfg.repo, cfg.checkpoint_path, revision=cfg.revision)
    return infer_model_spec(model_dir)


def _build_model_suite(run: RuntimeRunConfig) -> ModelSuite:
    actor = _model_spec_for_role(run, "actor")
    reference = _model_spec_for_role(run, "reference", actor)
    critic = _model_spec_for_role(run, "critic", actor) if "critic" in run.models else actor
    reward = _model_spec_for_role(run, "reward", actor) if "reward" in run.models else actor
    return ModelSuite(
        {
            Model.ACTOR: actor,
            Model.REFERENCE: reference,
            Model.CRITIC: critic,
            Model.REWARD: reward,
            Model.AC_SHARE: actor,
        }
    )


def _validate_run_against_plan(run: RuntimeRunConfig, opt: OptimizedPlan, plan_path: Path) -> None:
    if run.optimized_plan.expected_hash:
        actual = sha256_file(plan_path)
        if actual != run.optimized_plan.expected_hash:
            raise RuntimeConfigError(f"optimized plan hash mismatch: expected {run.optimized_plan.expected_hash}, got {actual}")
    if opt.schema_version != "rpeg.optimized_plan.v1":
        raise RuntimeConfigError("runtime requires optimized plan schema_version rpeg.optimized_plan.v1")
    if run.algorithm.name != opt.algorithm:
        raise RuntimeConfigError(f"algorithm mismatch: run={run.algorithm.name}, plan={opt.algorithm}")
    rw, pw = run.workload, opt.workload
    if rw != pw:
        raise RuntimeConfigError(f"workload mismatch between run config and optimized plan: {rw} != {pw}")
    rds, pds = run.device_space, opt.device_space
    if rds != pds:
        raise RuntimeConfigError(f"device_space mismatch between run config and optimized plan: {rds} != {pds}")
    planned_fusion = bool(opt.structure.get("intra_stage_fusion", False))
    if run.optimized_plan.require_exact_match and run.runtime.intra_stage_fusion != planned_fusion:
        raise RuntimeConfigError(
            "intra-stage fusion mismatch between run config and optimized plan: "
            f"run={run.runtime.intra_stage_fusion}, plan={planned_fusion}"
        )
    planned_transfer_policy = opt.structure.get("transfer_resource_policy")
    if run.optimized_plan.require_exact_match and planned_transfer_policy is not None:
        try:
            planned_policy = TransferResourcePolicy.from_dict(planned_transfer_policy)
        except ValueError as exc:
            raise RuntimeConfigError(str(exc)) from exc
        runtime_policy = transfer_resource_policy_from_runtime(run.runtime)
        if planned_policy != runtime_policy:
            raise RuntimeConfigError(
                "transfer resource policy mismatch between run config and optimized plan: "
                f"run={runtime_policy.to_dict()}, plan={planned_policy.to_dict()}"
            )
    if run.algorithm.name == "dpo" and not run.data.preferences:
        raise RuntimeConfigError("DPO requires data.preferences")
    if run.algorithm.name in {"ppo", "grpo"}:
        if not run.data.train_prompts:
            raise RuntimeConfigError(f"{run.algorithm.name.upper()} requires data.train_prompts")
        if "reward" not in run.models:
            raise RuntimeConfigError(f"{run.algorithm.name.upper()} requires models.reward or an explicit reward adapter")


def _validate_exactness(
    config: Configuration,
    device_space: DeviceSpace,
    suite: ModelSuite,
    workload: Workload,
    run: RuntimeRunConfig,
) -> None:
    rollout_batch = workload.B * workload.K
    configured_minibatch = run.algorithm.minibatch_size
    global_minibatch = rollout_batch if configured_minibatch is None else configured_minibatch
    if rollout_batch % global_minibatch != 0:
        raise RuntimeConfigError(
            f"rollout batch B*K={rollout_batch} must be divisible by algorithm.minibatch_size={global_minibatch}"
        )
    for v in config.rpeg.vertices:
        alloc = config.plan.sigma[v]
        dp, pp, tp = config.plan.pi[v]
        mb = config.plan.mb[v]
        if dp <= 0 or pp <= 0 or tp <= 0:
            raise RuntimeConfigError(f"{v.id}: dp, pp, and tp must all be > 0")
        if mb <= 0:
            raise RuntimeConfigError(f"{v.id}: microbatch must be > 0")
        if rollout_batch % dp != 0:
            raise RuntimeConfigError(
                f"{v.id}: rollout batch B*K={rollout_batch} must be divisible by data parallel size {dp}"
            )
        if v.kind.role == Role.TRAINING and global_minibatch % dp != 0:
            raise RuntimeConfigError(
                f"{v.id}: algorithm.minibatch_size={global_minibatch} must be divisible by data parallel size {dp}"
            )
        if device_space.G % tp != 0:
            raise RuntimeConfigError(
                f"{v.id}: GPUs per node {device_space.G} must be divisible by tensor parallel size {tp}"
            )
        layers = suite.specs[v.kind.model].L
        if layers % pp != 0:
            raise RuntimeConfigError(
                f"{v.id}: model layers {layers} must be divisible by pipeline parallel size {pp}"
            )
        if alloc.size() != dp * pp * tp:
            raise RuntimeConfigError(
                f"{v.id}: allocation size {alloc.size()} does not equal dp*pp*tp={dp * pp * tp}"
            )
        geom = packing_geometry(dp, pp, tp, device_space.G)
        if geom is None:
            raise RuntimeConfigError(f"{v.id}: invalid packing geometry for dp={dp}, pp={pp}, tp={tp}")
        if not is_compatible(alloc, (dp, pp, tp), device_space):
            raise RuntimeConfigError(f"{v.id}: placement is incompatible with device_space")
        if pp == 1 and mb != 1:
            raise RuntimeConfigError(f"{v.id}: pp=1 requires microbatch=1")
        local_step_batch = rollout_batch // dp
        local_minibatch = global_minibatch // dp if v.kind.role == Role.TRAINING else local_step_batch
        if pp > 1 and local_minibatch % mb != 0:
            raise RuntimeConfigError(
                f"{v.id}: local execution batch {local_minibatch} must be divisible by pipeline microbatch {mb}"
            )
    # In-place Megatron layout switching cannot retain an independently live stale shadow. CPU/local
    # backends may keep historical state dictionaries without a second GPU allocation and remain
    # useful for semantic tests.
    if (
        run.runtime.default_backend == "megatron"
        and not stale_publication_layout_valid(config)
    ):
        conflicts = [
            f"{edge.src.id}->{edge.dst.id}"
            for edge in config.rpeg.pub_edges
            if int(config.rpeg.tau.get(edge.src.kind.model, 0)) > 0
            and allocations_overlap(
                config.plan.sigma[edge.src],
                config.plan.sigma[edge.dst],
            )
        ]
        raise RuntimeConfigError(
            "positive-lag RPEG publication requires non-overlapping publisher "
            "and serving allocations; conflicts=" + ",".join(conflicts)
        )


def _validated_execution_order(
    run: RuntimeRunConfig,
    opt: OptimizedPlan,
    lowered: ExecutableRPEG,
) -> tuple[str, ...] | None:
    schedule = opt.execution_schedule
    if schedule is None:
        if run.runtime.require_optimized_schedule:
            raise RuntimeConfigError(
                "runtime.require_optimized_schedule=true, but optimized plan has no execution_schedule"
            )
        return None
    if schedule.horizon != run.training.max_steps:
        raise RuntimeConfigError(
            f"execution_schedule.horizon={schedule.horizon} does not match training.max_steps={run.training.max_steps}"
        )
    expected = {node.id for node in lowered.nodes}
    actual = set(schedule.order)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RuntimeConfigError(
            f"execution_schedule node set mismatch: missing={missing[:20]} extra={extra[:20]}"
        )
    timing = {node.node_id: node for node in schedule.nodes}
    tolerance = 1e-9
    for edge in lowered.edges:
        src = timing[edge.src.id]
        dst = timing[edge.dst.id]
        if src.predicted_finish_s > dst.predicted_start_s + tolerance:
            raise RuntimeConfigError(
                "execution_schedule violates dependency timing "
                f"{edge.src.id}->{edge.dst.id}: {src.predicted_finish_s} > {dst.predicted_start_s}"
            )
    by_id = {node.id: node for node in lowered.nodes}
    predecessors = {node.id: set() for node in lowered.nodes}
    successors = {node.id: set() for node in lowered.nodes}
    for edge in lowered.edges:
        predecessors[edge.dst.id].add(edge.src.id)
        successors[edge.src.id].add(edge.dst.id)
    policy = transfer_resource_policy_from_runtime(run.runtime)
    device_intervals: dict[tuple[int, int, int], list[tuple[float, float, str]]] = {}
    for node in lowered.nodes:
        allocations = reserved_allocations(
            node,
            policy=policy,
            predecessors=predecessors,
            successors=successors,
            by_id=by_id,
        )
        interval = timing[node.id]
        node_devices: set[tuple[int, int, int]] = set()
        for allocation in allocations:
            r0, n0, g0 = allocation.origin
            re, ne, ge = allocation.extent
            for r in range(r0, r0 + re):
                for n in range(n0, n0 + ne):
                    for g in range(g0, g0 + ge):
                        node_devices.add((r, n, g))
        for device in node_devices:
            device_intervals.setdefault(device, []).append(
                (interval.predicted_start_s, interval.predicted_finish_s, node.id)
            )
    for device, intervals in device_intervals.items():
        previous_start, previous_finish, previous_id = sorted(intervals)[0]
        for start, finish, node_id in sorted(intervals)[1:]:
            fused_overlap = (
                by_id[previous_id].fusion_group is not None
                and by_id[previous_id].fusion_group == by_id[node_id].fusion_group
                and abs(start - previous_start) <= tolerance
                and abs(finish - previous_finish) <= tolerance
            )
            if start < previous_finish - tolerance and not fused_overlap:
                raise RuntimeConfigError(
                    "execution_schedule violates transfer resource timing on logical GPU "
                    f"{device}: {previous_id} [{previous_start}, {previous_finish}] overlaps "
                    f"{node_id} [{start}, {finish}] under {policy.to_dict()}"
                )
            if finish > previous_finish:
                previous_start, previous_finish, previous_id = start, finish, node_id
    return schedule.order


def _build_worker_specs(
    lowered: ExecutableRPEG,
    inventory: ClusterInventory,
    backend: str,
    port_base: int,
    model_lag: dict[Model, int],
    reverse_critic_pipeline: bool = False,
) -> tuple[WorkerSpec, ...]:
    vertices = {}
    for node in lowered.nodes:
        if node.kind != ExecNodeType.OP or node.op is None:
            continue
        tpl = node.op.template_vertex
        if tpl.id in vertices:
            continue
        vertices[tpl.id] = node
    grouped: dict[tuple[Any, ...], list[tuple[str, Any]]] = {}
    for vertex_id, node in vertices.items():
        allocation = node.allocs[0]
        # One model allocation owns one physical model instance. Positive lag only creates a shadow
        # when the producer and consumer allocations are disjoint; colocated vertices switch layouts
        # in place.
        key = (node.model, allocation.origin, allocation.extent)
        grouped.setdefault(key, []).append((vertex_id, node))
    workers = []
    ordered_groups = sorted(grouped.values(), key=lambda items: tuple(sorted(vertex_id for vertex_id, _ in items)))
    for idx, items in enumerate(ordered_groups):
        _initial_vertex_id, initial_node = items[0]
        items = sorted(items, key=lambda item: item[0])
        vertex_ids = tuple(vertex_id for vertex_id, _ in items)
        training = [item for item in items if item[1].op.template_vertex.kind.role == Role.TRAINING]
        vertex_id, node = training[0] if training else items[0]
        assert node.model is not None and node.pi is not None
        allocation = node.allocs[0]
        roles = tuple(sorted({item[1].op.template_vertex.kind.role.value for item in items}))
        worker_id = f"wg_{vertex_id}"
        devices = inventory.resolve_allocation(allocation)
        if (
            reverse_critic_pipeline
            and node.model == Model.CRITIC
            and training
        ):
            train_dp, train_pp, train_tp = training[0][1].pi
            reversed_devices = [None] * len(devices)
            for dp_rank in range(int(train_dp)):
                for pp_rank in range(int(train_pp)):
                    for tp_rank in range(int(train_tp)):
                        logical_rank = (dp_rank * int(train_pp) + pp_rank) * int(train_tp) + tp_rank
                        physical_rank = (
                            (dp_rank * int(train_pp) + (int(train_pp) - pp_rank - 1))
                            * int(train_tp)
                            + tp_rank
                        )
                        reversed_devices[logical_rank] = devices[physical_rank]
            if any(device is None for device in reversed_devices):
                raise RuntimeConfigError(f"{worker_id}: failed to reverse critic pipeline placement")
            devices = tuple(reversed_devices)
        workers.append(
            WorkerSpec(
                worker_id=worker_id,
                vertex_id=vertex_id,
                model=_MODEL_TO_NAME[node.model],
                role="TRAIN" if training else node.op.template_vertex.kind.role.value,
                backend=backend,
                allocation=allocation,
                devices=devices,
                parallelism=initial_node.pi,
                microbatch=initial_node.mb or 1,
                rendezvous_port=port_base + idx,
                vertex_ids=vertex_ids,
                roles=roles,
                vertex_parallelisms=tuple(
                    (item_vertex_id, tuple(int(x) for x in item_node.pi))
                    for item_vertex_id, item_node in items
                ),
                vertex_microbatches=tuple(
                    (item_vertex_id, int(item_node.mb or 1))
                    for item_vertex_id, item_node in items
                ),
                vertex_roles=tuple(
                    (item_vertex_id, item_node.op.template_vertex.kind.role.value)
                    for item_vertex_id, item_node in items
                ),
            )
        )
    return tuple(workers)


def _build_transfer_specs(lowered: ExecutableRPEG) -> tuple[TransferSpec, ...]:
    transfers = []
    for node in lowered.nodes:
        if node.kind not in {ExecNodeType.COMM, ExecNodeType.RESHARD}:
            continue
        transfers.append(
            TransferSpec(
                node_id=node.id,
                kind=node.kind.value,
                label=node.label,
                payload_in_bytes=node.payload_in_bytes,
                src_allocation=node.allocs[0],
                dst_allocation=node.allocs[1],
            )
        )
    return tuple(transfers)


def _runtime_memory_hardware(run: RuntimeRunConfig, config_path: str | Path) -> Hardware | None:
    explicit = run.hardware.memory_per_gpu_bytes
    calibrated = None
    if run.hardware.calibrated_hw_path:
        path = Path(run.hardware.calibrated_hw_path)
        if not path.is_absolute():
            path = Path(config_path).resolve().parent / path
        if not path.exists():
            raise RuntimeConfigError(f"calibrated hardware profile not found: {path}")
        profile = load_mapping(path)
        calibrated_value = profile.get("memory_per_gpu") or profile.get("memory_per_gpu_bytes")
        if calibrated_value is None:
            raise RuntimeConfigError(f"calibrated hardware profile has no memory_per_gpu: {path}")
        calibrated = int(float(calibrated_value))
    if explicit is not None and calibrated is not None and run.hardware.require_calibration_match:
        relative_error = abs(int(explicit) - calibrated) / max(calibrated, 1)
        if relative_error > 0.01:
            raise RuntimeConfigError(
                "hardware.memory_per_gpu_bytes does not match calibrated hardware profile "
                f"({int(explicit)} vs {calibrated})"
            )
    capacity = int(explicit) if explicit is not None else calibrated
    if capacity is None:
        return None
    local_capacity = _slurm_local_cuda_memory_capacity()
    if local_capacity is not None and capacity > int(local_capacity * 1.01):
        raise RuntimeConfigError(
            "hardware.memory_per_gpu_bytes exceeds the CUDA capacity visible in this Slurm allocation "
            f"({capacity} configured vs {local_capacity} detected)"
        )
    if capacity <= 0:
        raise RuntimeConfigError("hardware memory capacity must be positive")
    return Hardware(
        flops_per_gpu=0.0,
        memory_per_gpu=float(capacity),
        memory_bandwidth=0.0,
        alpha_coll_g=0.0,
        beta_coll_g=0.0,
        alpha_coll_n=0.0,
        beta_coll_n=0.0,
        alpha_coll_r=0.0,
        beta_coll_r=0.0,
        alpha_p2p_g=0.0,
        beta_p2p_g=0.0,
        alpha_p2p_n=0.0,
        beta_p2p_n=0.0,
        alpha_p2p_r=0.0,
        beta_p2p_r=0.0,
    )


def _slurm_local_cuda_memory_capacity() -> int | None:
    import os

    if not os.environ.get("SLURM_JOB_ID"):
        return None
    try:
        import torch
    except ModuleNotFoundError:
        return None
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        return None
    return min(
        int(torch.cuda.get_device_properties(index).total_memory)
        for index in range(int(torch.cuda.device_count()))
    )


def _validate_runtime_memory(
    run: RuntimeRunConfig,
    config_path: str | Path,
    lowered: ExecutableRPEG,
    suite: ModelSuite,
    workload: Workload,
    rpeg: RPEG,
) -> None:
    hardware = _runtime_memory_hardware(run, config_path)
    if hardware is None:
        return
    try:
        stage = optimizer_stage(run.optimizer_runtime.distributed_optimizer)
    except ValueError as exc:
        raise RuntimeConfigError(str(exc)) from exc
    policy = MemoryPolicy(
        optimizer_stage=stage,
        activation_checkpointing=run.optimizer_runtime.activation_checkpointing,
        runtime_overhead_fraction=run.hardware.memory_reserve_fraction,
        worker_overhead_bytes=run.hardware.worker_overhead_bytes,
        semantic_activation_cache_bytes=run.hardware.semantic_activation_cache_bytes,
        max_worker_groups_per_gpu=run.hardware.max_worker_groups_per_gpu,
    )
    memory = NodeMemoryModel.from_policy(
        hardware,
        suite,
        workload,
        policy,
        trainable=frozenset(rpeg.trainable),
        shadow_models=frozenset(model for model in rpeg.trainable if rpeg.tau.get(model, 0) > 0),
    )
    violation = mem_worst_violation(lowered, memory)
    if violation is None:
        return
    device, peak, capacity = violation
    if peak == float("inf") and policy.max_worker_groups_per_gpu > 0:
        raise RuntimeConfigError(
            "optimized plan exceeds runtime worker-group colocation cap on logical GPU "
            f"{device}: max_worker_groups_per_gpu={policy.max_worker_groups_per_gpu}"
        )
    raise RuntimeConfigError(
        "optimized plan exceeds runtime HBM capacity on logical GPU "
        f"{device}: predicted peak={peak / 2**30:.2f} GiB, "
        f"effective capacity={capacity / 2**30:.2f} GiB "
        f"(optimizer_stage={stage}, activation_checkpointing={policy.activation_checkpointing})"
    )


def compile_runtime_plan(config_path: str | Path, dry_run: bool | None = None) -> CompiledRuntimePlan:
    run = RuntimeRunConfig.load(config_path)
    if dry_run is not None:
        object.__setattr__(run.runtime, "dry_run", dry_run)
    direct_transports = {
        run.runtime.activation_transport,
        run.runtime.weight_transport,
    } & {"nccl_direct", "ucx_direct"}
    if (
        run.cluster.scheduler == "slurm"
        and run.cluster.launcher.type == "slurm"
        and direct_transports
        and not run.cluster.launcher.shared_step
    ):
        raise RuntimeConfigError(
            "Slurm UCX/NCCL direct transports require cluster.launcher.shared_step=true "
            "so all RPEG roles share one Slingshot VNI/security context"
        )
    plan_path = Path(run.optimized_plan.path)
    if not plan_path.is_absolute():
        plan_path = Path(config_path).resolve().parent / plan_path
    opt = OptimizedPlan.load(plan_path)
    _validate_run_against_plan(run, opt, plan_path)

    device_space = _device_space_from_config(run)
    workload = _workload_from_config(run)
    suite = _build_model_suite(run)
    rpeg = _build_rpeg(run, opt, suite)
    execution_plan = _build_execution_plan(rpeg, opt)
    config = Configuration(
        rpeg=rpeg,
        plan=execution_plan,
        device_space=device_space,
        intra_stage_fusion=run.runtime.intra_stage_fusion,
        transfer_resource_policy=transfer_resource_policy_from_runtime(run.runtime),
    )
    _validate_exactness(config, device_space, suite, workload, run)

    execution_horizon = run.training.max_steps
    if execution_horizon < 1:
        raise RuntimeConfigError("training.max_steps must be >= 1")
    unrolled = unroll(rpeg, K_eval=execution_horizon)
    lowered = lower(
        rpeg,
        unrolled,
        execution_plan,
        suite,
        K_eval=execution_horizon,
        workload=workload,
        intra_stage_fusion=run.runtime.intra_stage_fusion,
    )
    fused_training_groups = {
        node.fusion_group
        for node in lowered.nodes
        if node.fusion_group is not None and node.kind == ExecNodeType.OP
    }
    if run.runtime.intra_stage_fusion and not fused_training_groups:
        raise RuntimeConfigError(
            "runtime.intra_stage_fusion=true requires colocated actor/critic training with "
            "identical DP/PP/TP and microbatch count, PP>1, and enough microbatches"
        )
    critic_device_order = run.runtime.critic_pipeline_device_order
    if run.runtime.intra_stage_fusion and critic_device_order == "canonical":
        raise RuntimeConfigError(
            "runtime.intra_stage_fusion=true requires reverse critic pipeline device order"
        )
    reverse_critic_pipeline = (
        run.runtime.intra_stage_fusion
        if critic_device_order == "auto"
        else critic_device_order == "reverse"
    )
    _validate_runtime_memory(run, config_path, lowered, suite, workload, rpeg)
    execution_order = _validated_execution_order(run, opt, lowered)
    inventory = load_inventory(run.cluster.inventory_path, device_space, scheduler=run.cluster.scheduler)
    workers = _build_worker_specs(
        lowered,
        inventory,
        backend=run.runtime.default_backend,
        port_base=run.runtime.rendezvous_port_base,
        model_lag=config.rpeg.tau,
        reverse_critic_pipeline=reverse_critic_pipeline,
    )
    transfers = _build_transfer_specs(lowered)
    return CompiledRuntimePlan(
        run_config=run,
        optimized_plan=opt,
        device_space=device_space,
        model_suite=suite,
        workload=workload,
        config=config,
        unrolled=unrolled,
        lowered=lowered,
        inventory=inventory,
        workers=workers,
        transfers=transfers,
        execution_order=execution_order,
    )
