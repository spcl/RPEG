# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import dataclasses
import hashlib
import heapq
import json

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping

from runtime.errors import RuntimeConfigError


@dataclass(frozen=True, order=True)
class FusedPipelineTask:
    """
    One forward/backward microtask on one physical pipeline stage.
    """
    model: str
    microbatch: int
    phase: str
    stage: int
    device_stage: int
    duration_s: float
    activation_bytes: int

    def __post_init__(self) -> None:
        if not self.model:
            raise RuntimeConfigError("fused pipeline task model must be non-empty")
        if self.microbatch < 0 or self.stage < 0 or self.device_stage < 0:
            raise RuntimeConfigError("fused pipeline task indices must be non-negative")
        if self.phase not in {"F", "B"}:
            raise RuntimeConfigError(f"invalid fused pipeline task phase {self.phase!r}")
        if self.duration_s <= 0.0:
            raise RuntimeConfigError("fused pipeline task duration must be positive")
        if self.activation_bytes < 0:
            raise RuntimeConfigError("fused pipeline activation bytes must be non-negative")

    @property
    def task_id(self) -> str:
        return f"{self.model}:{self.phase}{self.microbatch}:s{self.stage}"

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "model": self.model,
            "microbatch": self.microbatch,
            "phase": self.phase,
            "stage": self.stage,
            "device_stage": self.device_stage,
            "duration_s": self.duration_s,
            "activation_bytes": self.activation_bytes,
        }


@dataclass(frozen=True)
class FusedPipelineModel:
    """
    Profile and placement of one model participating in stage fusion.

    reverse=True maps logical stage zero to the last physical stage. A two-model RPEG training
    fusion must contain one forward and one reverse pipeline, following RLHFuse/Chimera's
    bidirectional construction.
    """
    name: str
    pipeline_stages: int
    microbatches: int
    forward_s: tuple[float, ...]
    backward_s: tuple[float, ...]
    activation_bytes: tuple[int, ...]
    reverse: bool = False

    @classmethod
    def uniform(
        cls,
        name: str,
        *,
        pipeline_stages: int,
        microbatches: int,
        forward_s: float,
        backward_s: float,
        activation_bytes: int,
        reverse: bool = False,
    ) -> "FusedPipelineModel":
        return cls(
            name=name,
            pipeline_stages=int(pipeline_stages),
            microbatches=int(microbatches),
            forward_s=(float(forward_s),) * int(pipeline_stages),
            backward_s=(float(backward_s),) * int(pipeline_stages),
            activation_bytes=(int(activation_bytes),) * int(pipeline_stages),
            reverse=bool(reverse),
        )

    def __post_init__(self) -> None:
        if not self.name:
            raise RuntimeConfigError("fused pipeline model name must be non-empty")
        if self.pipeline_stages <= 1:
            raise RuntimeConfigError("intra-stage fusion requires pipeline_stages > 1")
        if self.microbatches < self.pipeline_stages:
            raise RuntimeConfigError(
                f"{self.name}: microbatches={self.microbatches} must be >= "
                f"pipeline_stages={self.pipeline_stages}"
            )
        for field_name, values in (
            ("forward_s", self.forward_s),
            ("backward_s", self.backward_s),
            ("activation_bytes", self.activation_bytes),
        ):
            if len(values) != self.pipeline_stages:
                raise RuntimeConfigError(
                    f"{self.name}: {field_name} has {len(values)} entries, expected "
                    f"{self.pipeline_stages}"
                )
        if any(value <= 0.0 for value in self.forward_s + self.backward_s):
            raise RuntimeConfigError(f"{self.name}: forward/backward durations must be positive")
        if any(value < 0 for value in self.activation_bytes):
            raise RuntimeConfigError(f"{self.name}: activation bytes must be non-negative")

    def device_stage(self, logical_stage: int) -> int:
        if not 0 <= logical_stage < self.pipeline_stages:
            raise RuntimeConfigError(
                f"{self.name}: logical stage {logical_stage} is outside pipeline"
            )
        if self.reverse:
            return self.pipeline_stages - logical_stage - 1
        return logical_stage


@dataclass(frozen=True)
class FusedPipelineSchedule:
    """
    Validated fused pipeline instruction stream and analytical timings.

    Validation includes progress under NCCL's large-message host-blocking P2P launch semantics, not
    only acyclicity of the offline compute-task DAG.
    """
    models: tuple[FusedPipelineModel, ...]
    tasks: tuple[FusedPipelineTask, ...]
    dependencies: tuple[tuple[str, str], ...]
    device_order: tuple[tuple[str, ...], ...]
    start_s: tuple[tuple[str, float], ...]
    finish_s: tuple[tuple[str, float], ...]
    peak_activation_bytes: tuple[int, ...]
    makespan_s: float
    optimizer: str = "greedy_backward_work"
    optimizer_candidates: int = 1
    baseline_makespan_s: float | None = None

    @property
    def schedule_hash(self) -> str:
        encoded = json.dumps(self.to_dict(include_timings=True), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def to_dict(self, *, include_timings: bool = True) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": "rpeg.fused_pipeline_schedule.v1",
            "models": [
                {
                    "name": model.name,
                    "pipeline_stages": model.pipeline_stages,
                    "microbatches": model.microbatches,
                    "forward_s": list(model.forward_s),
                    "backward_s": list(model.backward_s),
                    "activation_bytes": list(model.activation_bytes),
                    "reverse": model.reverse,
                }
                for model in self.models
            ],
            "tasks": [task.to_dict() for task in self.tasks],
            "dependencies": [list(edge) for edge in self.dependencies],
            "device_order": [list(order) for order in self.device_order],
            "peak_activation_bytes": list(self.peak_activation_bytes),
            "makespan_s": self.makespan_s,
            "optimizer": self.optimizer,
            "optimizer_candidates": self.optimizer_candidates,
            "baseline_makespan_s": self.baseline_makespan_s,
        }
        if include_timings:
            out["start_s"] = dict(self.start_s)
            out["finish_s"] = dict(self.finish_s)
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "FusedPipelineSchedule":
        if raw.get("schema_version") != "rpeg.fused_pipeline_schedule.v1":
            raise RuntimeConfigError(
                f"unsupported fused pipeline schedule schema {raw.get('schema_version')!r}"
            )
        raw_models = raw.get("models")
        raw_tasks = raw.get("tasks")
        raw_dependencies = raw.get("dependencies")
        raw_device_order = raw.get("device_order")
        raw_start = raw.get("start_s")
        raw_finish = raw.get("finish_s")
        raw_peak = raw.get("peak_activation_bytes")
        if not isinstance(raw_models, list) or not isinstance(raw_tasks, list):
            raise RuntimeConfigError("fused pipeline schedule models/tasks must be lists")
        if not isinstance(raw_dependencies, list) or not isinstance(raw_device_order, list):
            raise RuntimeConfigError("fused pipeline dependencies/device_order must be lists")
        if not isinstance(raw_start, dict) or not isinstance(raw_finish, dict):
            raise RuntimeConfigError("fused pipeline schedule must include start_s and finish_s")
        if not isinstance(raw_peak, list):
            raise RuntimeConfigError("fused pipeline peak_activation_bytes must be a list")
        models = tuple(
            FusedPipelineModel(
                name=str(item["name"]),
                pipeline_stages=int(item["pipeline_stages"]),
                microbatches=int(item["microbatches"]),
                forward_s=tuple(float(value) for value in item["forward_s"]),
                backward_s=tuple(float(value) for value in item["backward_s"]),
                activation_bytes=tuple(int(value) for value in item["activation_bytes"]),
                reverse=bool(item.get("reverse", False)),
            )
            for item in raw_models
            if isinstance(item, dict)
        )
        if len(models) != len(raw_models):
            raise RuntimeConfigError("fused pipeline schedule contains a malformed model")
        tasks = tuple(
            FusedPipelineTask(
                model=str(item["model"]),
                microbatch=int(item["microbatch"]),
                phase=str(item["phase"]),
                stage=int(item["stage"]),
                device_stage=int(item["device_stage"]),
                duration_s=float(item["duration_s"]),
                activation_bytes=int(item["activation_bytes"]),
            )
            for item in raw_tasks
            if isinstance(item, dict)
        )
        if len(tasks) != len(raw_tasks):
            raise RuntimeConfigError("fused pipeline schedule contains a malformed task")
        schedule = cls(
            models=models,
            tasks=tasks,
            dependencies=tuple(
                (str(edge[0]), str(edge[1]))
                for edge in raw_dependencies
                if isinstance(edge, list) and len(edge) == 2
            ),
            device_order=tuple(
                tuple(str(task_id) for task_id in order)
                for order in raw_device_order
                if isinstance(order, list)
            ),
            start_s=tuple(sorted((str(key), float(value)) for key, value in raw_start.items())),
            finish_s=tuple(sorted((str(key), float(value)) for key, value in raw_finish.items())),
            peak_activation_bytes=tuple(int(value) for value in raw_peak),
            makespan_s=float(raw.get("makespan_s", 0.0)),
            optimizer=str(raw.get("optimizer", "greedy_backward_work")),
            optimizer_candidates=int(raw.get("optimizer_candidates", 1)),
            baseline_makespan_s=(
                None
                if raw.get("baseline_makespan_s") is None
                else float(raw["baseline_makespan_s"])
            ),
        )
        if len(schedule.dependencies) != len(raw_dependencies):
            raise RuntimeConfigError("fused pipeline schedule contains a malformed dependency")
        if len(schedule.device_order) != len(raw_device_order):
            raise RuntimeConfigError("fused pipeline schedule contains malformed device order")
        validate_fused_pipeline_schedule(schedule)
        return schedule

    def local_instructions(self, model: str, device_stage: int) -> tuple[dict[str, object], ...]:
        by_id = {task.task_id: task for task in self.tasks}
        if not 0 <= device_stage < len(self.device_order):
            raise RuntimeConfigError(f"physical pipeline stage {device_stage} is out of range")
        return tuple(
            by_id[task_id].to_dict()
            for task_id in self.device_order[device_stage]
            if by_id[task_id].model == model
        )


def build_fused_pipeline_schedule(
    models: Iterable[FusedPipelineModel],
    *,
    activation_capacity_bytes: int | tuple[int, ...] | None = None,
    optimization: str = "portfolio",
) -> FusedPipelineSchedule:
    """
    Return a cached offline schedule for one immutable fused profile.
    """
    resolved_capacity = (
        activation_capacity_bytes
        if activation_capacity_bytes is None or isinstance(activation_capacity_bytes, int)
        else tuple(int(value) for value in activation_capacity_bytes)
    )
    return _build_fused_pipeline_schedule_cached(
        tuple(models),
        activation_capacity_bytes=resolved_capacity,
        optimization=optimization,
    )


@lru_cache(maxsize=4096)
def _build_fused_pipeline_schedule_cached(
    models: tuple[FusedPipelineModel, ...],
    *,
    activation_capacity_bytes: int | tuple[int, ...] | None = None,
    optimization: str = "portfolio",
) -> FusedPipelineSchedule:
    """
    Build a deterministic, deadlock-free bidirectional microtask schedule.

    This is a list scheduler over the same constraints used by RLHFuse's released FlexFusion solver:
    microbatch data dependencies, one instruction stream per physical stage, and bounded live
    activation memory. It is a deterministic runtime baseline. The default offline portfolio
    evaluates several dependency-aware priority rules and retains the shortest validated instruction
    stream without changing the schedule schema or worker contract. Each candidate also has
    phase-monotonic stage waves and is simulated as a blocking NCCL P2P state machine before it can
    reach a worker.
    """
    resolved = tuple(models)
    if len(resolved) != 2:
        raise RuntimeConfigError("RPEG actor/critic intra-stage fusion requires exactly two models")
    if len({model.name for model in resolved}) != len(resolved):
        raise RuntimeConfigError("fused pipeline model names must be unique")
    stage_counts = {model.pipeline_stages for model in resolved}
    if len(stage_counts) != 1:
        raise RuntimeConfigError(
            "runtime fused execution currently requires equal actor/critic pipeline stage counts"
        )
    if {model.reverse for model in resolved} != {False, True}:
        raise RuntimeConfigError("fused actor/critic pipelines must run in opposite directions")
    num_devices = resolved[0].pipeline_stages

    priority_modes = (
        "greedy_backward_work",
        "critical_path",
        "forward_wave",
        "microbatch_wave",
        "model_serial",
        "model_serial_reverse",
        "phase_af_cf_ab_cb",
        "phase_af_cf_cb_ab",
        "phase_cf_af_ab_cb",
        "phase_cf_af_cb_ab",
    )
    if optimization == "portfolio":
        candidates: list[FusedPipelineSchedule] = []
        errors: list[RuntimeConfigError] = []
        for priority_mode in priority_modes:
            try:
                candidates.append(
                    build_fused_pipeline_schedule(
                        resolved,
                        activation_capacity_bytes=activation_capacity_bytes,
                        optimization=priority_mode,
                    )
                )
            except RuntimeConfigError as exc:
                errors.append(exc)
        if not candidates:
            if errors:
                raise errors[0]
            raise RuntimeConfigError("fused pipeline optimizer produced no candidate schedules")
        # A priority rule can produce an acyclic task DAG that is unsafe under measured blocking
        # NCCL host-launch semantics. Such a candidate is intentionally absent; do not let an
        # unavailable analytical baseline prevent the safe portfolio from selecting another
        # validated order.
        baseline = next(
            (
                candidate
                for candidate in candidates
                if candidate.optimizer == "greedy_backward_work"
            ),
            None,
        )
        selected = min(
            candidates,
            key=lambda candidate: (
                candidate.makespan_s,
                max(candidate.peak_activation_bytes, default=0),
                sum(candidate.peak_activation_bytes),
                candidate.device_order,
            ),
        )
        optimized = dataclasses.replace(
            selected,
            optimizer="deterministic_blocking_safe_portfolio_v2",
            optimizer_candidates=len(candidates),
            baseline_makespan_s=(
                None if baseline is None else baseline.makespan_s
            ),
        )
        validate_fused_pipeline_schedule(
            optimized,
            activation_capacity_bytes=activation_capacity_bytes,
        )
        return optimized
    if optimization not in priority_modes:
        raise RuntimeConfigError(
            f"unsupported fused pipeline optimization {optimization!r}; "
            f"expected portfolio or one of {priority_modes}"
        )

    tasks: dict[str, FusedPipelineTask] = {}
    dependencies: set[tuple[str, str]] = set()
    successors: dict[str, set[str]] = {}
    predecessors: dict[str, set[str]] = {}
    model_work: dict[str, float] = {}
    for model in resolved:
        model_work[model.name] = model.microbatches * sum(model.forward_s + model.backward_s)
        for microbatch in range(model.microbatches):
            chain: list[str] = []
            for stage in range(model.pipeline_stages):
                task = FusedPipelineTask(
                    model=model.name,
                    microbatch=microbatch,
                    phase="F",
                    stage=stage,
                    device_stage=model.device_stage(stage),
                    duration_s=float(model.forward_s[stage]),
                    activation_bytes=int(model.activation_bytes[stage]),
                )
                tasks[task.task_id] = task
                chain.append(task.task_id)
            for stage in reversed(range(model.pipeline_stages)):
                task = FusedPipelineTask(
                    model=model.name,
                    microbatch=microbatch,
                    phase="B",
                    stage=stage,
                    device_stage=model.device_stage(stage),
                    duration_s=float(model.backward_s[stage]),
                    activation_bytes=int(model.activation_bytes[stage]),
                )
                tasks[task.task_id] = task
                chain.append(task.task_id)
            dependencies.update(zip(chain, chain[1:]))

        # NCCL P2P launch is not host-nonblocking for a large unmatched send. Keep every logical
        # stage in a forward wave followed by a backward wave. This prevents one PP link from
        # waiting on a forward receive, while its peer waits on a backward receive, while still
        # allowing the oppositely placed actor and critic waves to overlap on every device.
        for stage in range(model.pipeline_stages):
            stage_order = [
                f"{model.name}:F{microbatch}:s{stage}"
                for microbatch in range(model.microbatches)
            ] + [
                f"{model.name}:B{microbatch}:s{stage}"
                for microbatch in range(model.microbatches)
            ]
            dependencies.update(zip(stage_order, stage_order[1:]))

    for task_id in tasks:
        successors[task_id] = set()
        predecessors[task_id] = set()
    for source, destination in dependencies:
        successors[source].add(destination)
        predecessors[destination].add(source)

    # Compute longest remaining paths iteratively. The PP18/MB32 405B fused profiles have dependency
    # paths deeper than Python's recursion limit; recursive memoization incorrectly classified those
    # valid plans as planning failures. Kahn order plus reverse dynamic programming has the same
    # O(V+E) semantics without a call-stack bound.
    topological_indegree = {
        task_id: len(values) for task_id, values in predecessors.items()
    }
    topological_ready = [
        task_id for task_id, count in topological_indegree.items() if count == 0
    ]
    heapq.heapify(topological_ready)
    topological_order: list[str] = []
    while topological_ready:
        task_id = heapq.heappop(topological_ready)
        topological_order.append(task_id)
        for successor in successors[task_id]:
            topological_indegree[successor] -= 1
            if topological_indegree[successor] == 0:
                heapq.heappush(topological_ready, successor)
    if len(topological_order) != len(tasks):
        cyclic = sorted(
            task_id for task_id, count in topological_indegree.items() if count > 0
        )
        raise RuntimeConfigError(
            f"fused pipeline dependency cycle; cyclic tasks={cyclic[:20]}"
        )
    remaining_path_s: dict[str, float] = {}
    for task_id in reversed(topological_order):
        remaining_path_s[task_id] = tasks[task_id].duration_s + max(
            (remaining_path_s[successor] for successor in successors[task_id]),
            default=0.0,
        )

    remaining_predecessors = {task_id: len(values) for task_id, values in predecessors.items()}
    predecessor_finish = {task_id: 0.0 for task_id in tasks}
    device_available = [0.0] * num_devices
    device_order: list[list[str]] = [[] for _ in range(num_devices)]
    start: dict[str, float] = {}
    finish: dict[str, float] = {}
    ready = {task_id for task_id, count in remaining_predecessors.items() if count == 0}
    barrier_orders = {
        "model_serial": ((0, "F"), (0, "B"), (1, "F"), (1, "B")),
        "model_serial_reverse": ((1, "F"), (1, "B"), (0, "F"), (0, "B")),
        "phase_af_cf_ab_cb": ((0, "F"), (1, "F"), (0, "B"), (1, "B")),
        "phase_af_cf_cb_ab": ((0, "F"), (1, "F"), (1, "B"), (0, "B")),
        "phase_cf_af_ab_cb": ((1, "F"), (0, "F"), (0, "B"), (1, "B")),
        "phase_cf_af_cb_ab": ((1, "F"), (0, "F"), (1, "B"), (0, "B")),
    }
    barrier_phase_group = (
        {
            (resolved[model_index].name, phase): index
            for index, (model_index, phase) in enumerate(barrier_orders[optimization])
        }
        if optimization in barrier_orders
        else None
    )

    while ready:
        def candidate_key(task_id: str) -> tuple[object, ...]:
            task = tasks[task_id]
            earliest = max(predecessor_finish[task_id], device_available[task.device_stage])
            common = (task.model, task.stage, task.task_id)
            if barrier_phase_group is not None:
                # Blocking-P2P fallbacks. A global model/phase barrier makes every admitted wave
                # independently deadlock-free. The phase variants can still pipeline adjacent waves
                # in the same direction; the two model-serial variants are the guaranteed last
                # resort. Faster bidirectional candidates remain preferred whenever their exact
                # blocking simulation succeeds.
                return (
                    barrier_phase_group[(task.model, task.phase)],
                    earliest,
                    task.microbatch,
                    *common,
                )
            if optimization == "critical_path":
                return (
                    earliest,
                    -remaining_path_s[task_id],
                    0 if task.phase == "B" else 1,
                    -model_work[task.model],
                    task.microbatch,
                    *common,
                )
            if optimization == "forward_wave":
                return (
                    earliest,
                    0 if task.phase == "F" else 1,
                    task.microbatch,
                    -remaining_path_s[task_id],
                    *common,
                )
            if optimization == "microbatch_wave":
                return (
                    earliest,
                    task.microbatch,
                    0 if task.phase == "B" else 1,
                    -remaining_path_s[task_id],
                    *common,
                )
            # Backward first reduces live activations. Prefer the larger model when both are
            # otherwise ready, matching FlexFusion's greedy seed.
            return (
                earliest,
                0 if task.phase == "B" else 1,
                -model_work[task.model],
                task.microbatch,
                *common,
            )

        task_id = min(ready, key=candidate_key)
        ready.remove(task_id)
        task = tasks[task_id]
        task_start = max(predecessor_finish[task_id], device_available[task.device_stage])
        task_finish = task_start + task.duration_s
        start[task_id] = task_start
        finish[task_id] = task_finish
        device_available[task.device_stage] = task_finish
        device_order[task.device_stage].append(task_id)
        for successor in successors[task_id]:
            remaining_predecessors[successor] -= 1
            predecessor_finish[successor] = max(predecessor_finish[successor], task_finish)
            if remaining_predecessors[successor] == 0:
                ready.add(successor)

    if len(finish) != len(tasks):
        missing = sorted(set(tasks) - set(finish))
        raise RuntimeConfigError(f"fused pipeline dependency cycle; unscheduled tasks={missing[:20]}")

    peak_activation = _peak_activation_bytes(tasks, start, finish, num_devices)
    capacities = _activation_capacities(activation_capacity_bytes, num_devices)
    if capacities is not None:
        violations = [
            (device, peak, capacities[device])
            for device, peak in enumerate(peak_activation)
            if peak > capacities[device]
        ]
        if violations:
            details = ", ".join(
                f"stage {device}: peak={peak} capacity={capacity}"
                for device, peak, capacity in violations
            )
            raise RuntimeConfigError(f"fused pipeline activation memory exceeds capacity: {details}")

    schedule = FusedPipelineSchedule(
        models=resolved,
        tasks=tuple(sorted(tasks.values(), key=lambda task: task.task_id)),
        dependencies=tuple(sorted(dependencies)),
        device_order=tuple(tuple(order) for order in device_order),
        start_s=tuple(sorted(start.items())),
        finish_s=tuple(sorted(finish.items())),
        peak_activation_bytes=tuple(peak_activation),
        makespan_s=max(finish.values(), default=0.0),
        optimizer=optimization,
        optimizer_candidates=1,
        baseline_makespan_s=None,
    )
    validate_fused_pipeline_schedule(schedule, activation_capacity_bytes=activation_capacity_bytes)
    return schedule


def _validate_blocking_p2p_progress(schedule: FusedPipelineSchedule) -> None:
    """
    Prove progress when NCCL send launch blocks until its peer receives.

    Each model/stage is a sequential host process. A task first waits for the preceding
    physical-device instruction, then rendezvous-receives, computes and records its completion
    marker, and finally rendezvous-sends. The colocated model may advance after that marker even
    while this process is blocked in send. Simulating those exact states catches communication
    cycles that ordinary task-DAG acyclicity cannot represent.
    """
    tasks = {task.task_id: task for task in schedule.tasks}
    models = {model.name: model for model in schedule.models}
    processes = {
        (model.name, stage): tuple(
            task_id
            for task_id in schedule.device_order[model.device_stage(stage)]
            if tasks[task_id].model == model.name
        )
        for model in schedule.models
        for stage in range(model.pipeline_stages)
    }
    device_predecessor = {
        task_id: (order[index - 1] if index else None)
        for order in schedule.device_order
        for index, task_id in enumerate(order)
    }
    indices = {process: 0 for process in processes}
    states = {process: "new" for process in processes}
    compute_completed: set[str] = set()
    waiting_sends: dict[tuple[str, str, int, int], tuple[str, int]] = {}
    waiting_recvs: dict[tuple[str, str, int, int], tuple[str, int]] = {}

    def has_recv(task: FusedPipelineTask) -> bool:
        stages = models[task.model].pipeline_stages
        return (task.phase == "F" and task.stage > 0) or (
            task.phase == "B" and task.stage < stages - 1
        )

    def has_send(task: FusedPipelineTask) -> bool:
        stages = models[task.model].pipeline_stages
        return (task.phase == "F" and task.stage < stages - 1) or (
            task.phase == "B" and task.stage > 0
        )

    def send_channel(task: FusedPipelineTask) -> tuple[str, str, int, int]:
        link = task.stage if task.phase == "F" else task.stage - 1
        return task.model, task.phase, task.microbatch, link

    def recv_channel(task: FusedPipelineTask) -> tuple[str, str, int, int]:
        link = task.stage - 1 if task.phase == "F" else task.stage
        return task.model, task.phase, task.microbatch, link

    def finish_compute(process: tuple[str, int], task: FusedPipelineTask) -> None:
        compute_completed.add(task.task_id)
        if has_send(task):
            channel = send_channel(task)
            if channel in waiting_sends:
                raise RuntimeConfigError(
                    f"fused pipeline duplicates blocking send channel {channel}"
                )
            states[process] = "send"
            waiting_sends[channel] = process
        else:
            indices[process] += 1
            states[process] = "new"

    max_transitions = len(tasks) * 8 + 1
    for _transition in range(max_transitions):
        progress = False
        for process in sorted(processes):
            sequence = processes[process]
            if indices[process] >= len(sequence) or states[process] != "new":
                continue
            task = tasks[sequence[indices[process]]]
            predecessor = device_predecessor[task.task_id]
            if predecessor is not None and predecessor not in compute_completed:
                continue
            if has_recv(task):
                channel = recv_channel(task)
                if channel in waiting_recvs:
                    raise RuntimeConfigError(
                        f"fused pipeline duplicates blocking recv channel {channel}"
                    )
                states[process] = "recv"
                waiting_recvs[channel] = process
            else:
                finish_compute(process, task)
            progress = True

        for channel in sorted(set(waiting_sends) & set(waiting_recvs)):
            sender = waiting_sends.pop(channel)
            receiver = waiting_recvs.pop(channel)
            receiver_task = tasks[processes[receiver][indices[receiver]]]
            indices[sender] += 1
            states[sender] = "new"
            finish_compute(receiver, receiver_task)
            progress = True

        if all(
            indices[process] >= len(sequence)
            for process, sequence in processes.items()
        ):
            return
        if not progress:
            blocked = [
                {
                    "process": f"{process[0]}:s{process[1]}",
                    "state": states[process],
                    "task": processes[process][indices[process]],
                }
                for process in sorted(processes)
                if indices[process] < len(processes[process])
            ]
            raise RuntimeConfigError(
                "fused pipeline blocking P2P execution deadlocks: "
                f"{blocked[:16]}"
            )
    raise RuntimeConfigError("fused pipeline blocking P2P validation did not converge")


def validate_fused_pipeline_schedule(
    schedule: FusedPipelineSchedule,
    *,
    activation_capacity_bytes: int | tuple[int, ...] | None = None,
) -> None:
    """
    Reject incomplete, overlapping, cyclic, or memory-invalid schedules.
    """
    if len(schedule.models) != 2:
        raise RuntimeConfigError("fused pipeline schedule must contain exactly two models")
    if len({model.name for model in schedule.models}) != 2:
        raise RuntimeConfigError("fused pipeline schedule model names must be unique")
    stage_counts = {model.pipeline_stages for model in schedule.models}
    if len(stage_counts) != 1 or len(schedule.device_order) != schedule.models[0].pipeline_stages:
        raise RuntimeConfigError(
            "fused pipeline schedule must use one physical device per common pipeline stage"
        )
    if {model.reverse for model in schedule.models} != {False, True}:
        raise RuntimeConfigError("fused pipeline schedule models must run in opposite directions")

    tasks = {task.task_id: task for task in schedule.tasks}
    if len(tasks) != len(schedule.tasks):
        raise RuntimeConfigError("fused pipeline schedule contains duplicate task IDs")
    expected_tasks: dict[str, FusedPipelineTask] = {}
    expected_dependencies: set[tuple[str, str]] = set()
    for model in schedule.models:
        for microbatch in range(model.microbatches):
            chain: list[str] = []
            for phase, stages in (
                ("F", range(model.pipeline_stages)),
                ("B", reversed(range(model.pipeline_stages))),
            ):
                for stage in stages:
                    expected = FusedPipelineTask(
                        model=model.name,
                        microbatch=microbatch,
                        phase=phase,
                        stage=stage,
                        device_stage=model.device_stage(stage),
                        duration_s=(
                            float(model.forward_s[stage])
                            if phase == "F"
                            else float(model.backward_s[stage])
                        ),
                        activation_bytes=int(model.activation_bytes[stage]),
                    )
                    expected_tasks[expected.task_id] = expected
                    chain.append(expected.task_id)
            expected_dependencies.update(zip(chain, chain[1:]))
        for stage in range(model.pipeline_stages):
            stage_order = [
                f"{model.name}:F{microbatch}:s{stage}"
                for microbatch in range(model.microbatches)
            ] + [
                f"{model.name}:B{microbatch}:s{stage}"
                for microbatch in range(model.microbatches)
            ]
            expected_dependencies.update(zip(stage_order, stage_order[1:]))
    if tasks != expected_tasks:
        missing = sorted(set(expected_tasks) - set(tasks))
        extra = sorted(set(tasks) - set(expected_tasks))
        mismatched = sorted(
            task_id
            for task_id in set(tasks) & set(expected_tasks)
            if tasks[task_id] != expected_tasks[task_id]
        )
        raise RuntimeConfigError(
            "fused pipeline task/model mismatch: "
            f"missing={missing[:8]} extra={extra[:8]} mismatched={mismatched[:8]}"
        )
    if set(schedule.dependencies) != expected_dependencies:
        raise RuntimeConfigError(
            "fused pipeline data dependencies do not match F/B and blocking-safe stage chains"
        )
    start = dict(schedule.start_s)
    finish = dict(schedule.finish_s)
    if set(start) != set(tasks) or set(finish) != set(tasks):
        raise RuntimeConfigError("fused pipeline timings do not cover every task exactly once")
    ordered_ids = [task_id for order in schedule.device_order for task_id in order]
    if len(ordered_ids) != len(set(ordered_ids)) or set(ordered_ids) != set(tasks):
        raise RuntimeConfigError("fused pipeline device orders do not cover every task exactly once")
    for device_stage, order in enumerate(schedule.device_order):
        if any(tasks[task_id].device_stage != device_stage for task_id in order):
            raise RuntimeConfigError(
                f"fused pipeline device order {device_stage} contains a task for another stage"
            )

    graph: dict[str, set[str]] = {task_id: set() for task_id in tasks}
    indegree = {task_id: 0 for task_id in tasks}
    edges = set(schedule.dependencies)
    for order in schedule.device_order:
        edges.update(zip(order, order[1:]))
    for source, destination in edges:
        if source not in tasks or destination not in tasks:
            raise RuntimeConfigError(f"fused pipeline edge references unknown task {source}->{destination}")
        if destination not in graph[source]:
            graph[source].add(destination)
            indegree[destination] += 1
        if finish[source] > start[destination] + 1.0e-9:
            raise RuntimeConfigError(
                f"fused pipeline timing violates dependency {source}->{destination}: "
                f"{finish[source]} > {start[destination]}"
            )
    for task_id, task in tasks.items():
        if start[task_id] < 0.0 or abs(
            finish[task_id] - start[task_id] - task.duration_s
        ) > 1.0e-9:
            raise RuntimeConfigError(f"fused pipeline task {task_id} has invalid timing/duration")

    ready = [task_id for task_id, count in indegree.items() if count == 0]
    visited = 0
    while ready:
        task_id = ready.pop()
        visited += 1
        for successor in graph[task_id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if visited != len(tasks):
        raise RuntimeConfigError("fused pipeline device/dependency graph contains a deadlock cycle")
    _validate_blocking_p2p_progress(schedule)

    peak = tuple(_peak_activation_bytes(tasks, start, finish, len(schedule.device_order)))
    if peak != schedule.peak_activation_bytes:
        raise RuntimeConfigError(
            f"fused pipeline peak activation metadata mismatch: {schedule.peak_activation_bytes} != {peak}"
        )
    capacities = _activation_capacities(activation_capacity_bytes, len(schedule.device_order))
    if capacities is not None and any(value > capacities[idx] for idx, value in enumerate(peak)):
        raise RuntimeConfigError("fused pipeline schedule exceeds activation capacity")
    if abs(max(finish.values(), default=0.0) - schedule.makespan_s) > 1.0e-9:
        raise RuntimeConfigError("fused pipeline makespan metadata mismatch")
    if not schedule.optimizer:
        raise RuntimeConfigError("fused pipeline optimizer metadata must be non-empty")
    if schedule.optimizer_candidates < 1:
        raise RuntimeConfigError("fused pipeline optimizer candidate count must be positive")
    if (
        schedule.baseline_makespan_s is not None
        and schedule.baseline_makespan_s + 1.0e-9 < schedule.makespan_s
    ):
        raise RuntimeConfigError(
            "fused pipeline optimized makespan exceeds its recorded baseline"
        )


def _activation_capacities(
    value: int | tuple[int, ...] | None,
    num_devices: int,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, int):
        if value < 0:
            raise RuntimeConfigError("fused pipeline activation capacity must be non-negative")
        return (value,) * num_devices
    capacities = tuple(int(item) for item in value)
    if len(capacities) != num_devices or any(item < 0 for item in capacities):
        raise RuntimeConfigError(
            f"fused pipeline activation capacity must have {num_devices} non-negative entries"
        )
    return capacities


def _peak_activation_bytes(
    tasks: Mapping[str, FusedPipelineTask],
    start: Mapping[str, float],
    finish: Mapping[str, float],
    num_devices: int,
) -> list[int]:
    del start
    events: list[list[tuple[float, int, str]]] = [[] for _ in range(num_devices)]
    for task_id, task in tasks.items():
        delta = task.activation_bytes if task.phase == "F" else -task.activation_bytes
        events[task.device_stage].append((finish[task_id], delta, task_id))
    peaks = [0] * num_devices
    for device, device_events in enumerate(events):
        live = 0
        # At the same timestamp conservatively account allocations before releases; an executor may
        # not have reclaimed the backward state yet.
        for _timestamp, delta, _task_id in sorted(
            device_events,
            key=lambda item: (item[0], 0 if item[1] > 0 else 1, item[2]),
        ):
            live += delta
            if live < 0:
                raise RuntimeConfigError(
                    f"fused pipeline stage {device} releases activation before its forward"
                )
            peaks[device] = max(peaks[device], live)
        if live != 0:
            raise RuntimeConfigError(
                f"fused pipeline stage {device} leaves {live} activation bytes live"
            )
    return peaks
