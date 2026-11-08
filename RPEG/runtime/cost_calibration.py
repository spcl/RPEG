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
import math
import statistics

from pathlib import Path
from typing import Any

from runtime.errors import RuntimeConfigError
from runtime.io import load_mapping


_ROLE_BUCKETS = {
    "GEN": "generation",
    "INF": "inference",
    "TRAIN": "training",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _vertex_id(node_id: str) -> str:
    return node_id.split("@", 1)[0]


def _hardware_mapping(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    loaded = load_mapping(path)
    nested = loaded.get("hardware")
    return dict(nested) if isinstance(nested, dict) else dict(loaded)


def build_runtime_cost_calibration(
    plan_path: str | Path,
    trace_path: str | Path,
    *,
    hardware_path: str | Path | None = None,
    exclude_warmup_steps: int = 0,
) -> dict[str, Any]:
    """
    Fit robust end-to-end corrections from one actual graph execution.

    The isolated-kernel cost equations remain responsible for extrapolation across DP/PP/TP
    strategies. This calibration measures only the residual backend/runtime factor between those
    equations and an actual RPEG trace. Failed nodes are excluded, making a completed prefix useful
    without pretending it is a complete validation run. Optional warmup exclusion fits steady-state
    RPEG costs without folding first-use kernel/cache/model effects into every logical step.
    """
    if int(exclude_warmup_steps) < 0:
        raise RuntimeConfigError("exclude_warmup_steps must be >= 0")

    plan_path = Path(plan_path).resolve()
    trace_path = Path(trace_path).resolve()
    hw_path = None if hardware_path is None else Path(hardware_path).resolve()
    plan = load_mapping(plan_path)
    schedule = plan.get("execution_schedule")
    if not isinstance(schedule, dict) or not isinstance(schedule.get("nodes"), list):
        raise RuntimeConfigError("cost calibration requires an optimized plan execution_schedule")
    vertex_plan = plan.get("plan")
    if not isinstance(vertex_plan, dict):
        raise RuntimeConfigError("cost calibration requires optimized plan vertex metadata")

    predicted: dict[str, float] = {}
    for item in schedule["nodes"]:
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("node_id"))
        duration = float(item.get("predicted_finish_s", 0.0)) - float(
            item.get("predicted_start_s", 0.0)
        )
        if not math.isfinite(duration) or duration <= 0.0:
            raise RuntimeConfigError(f"invalid predicted duration for {node_id}: {duration}")
        predicted[node_id] = duration

    successful: dict[str, dict[str, Any]] = {}
    planned_starts: dict[str, dict[str, Any]] = {}
    lifecycle_durations: dict[str, list[float]] = {}
    failure_count = 0
    with trace_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            node_id = str(event.get("node_id"))
            if event.get("event") == "start" and node_id in predicted:
                planned_starts[node_id] = event
                continue
            if event.get("event") != "finish":
                continue
            duration = float((event.get("counters") or {}).get("duration_s", 0.0))
            if (
                node_id.startswith("runtime.")
                and event.get("status") == "ok"
                and math.isfinite(duration)
                and duration > 0.0
            ):
                lifecycle_durations.setdefault(node_id, []).append(duration)
            if node_id not in predicted:
                continue
            if event.get("status") != "ok":
                failure_count += 1
                continue
            if math.isfinite(duration) and duration > 0.0:
                successful[node_id] = event

    calibration_successful = {
        node_id: event
        for node_id, event in successful.items()
        if event.get("step") is None or int(event["step"]) > int(exclude_warmup_steps)
    }
    excluded_warmup_node_count = len(successful) - len(calibration_successful)

    ratios: dict[str, list[float]] = {
        "generation": [],
        "inference": [],
        "training": [],
        "weight_comm": [],
        "reshard": [],
    }
    activation_residuals: list[float] = []
    unclassified: list[str] = []
    observations: list[dict[str, Any]] = []
    for node_id, event in calibration_successful.items():
        actual = float(event["counters"]["duration_s"])
        modeled = predicted[node_id]
        counters = event.get("counters") or {}
        bucket = None
        vertex = vertex_plan.get(_vertex_id(node_id))
        if isinstance(vertex, dict):
            bucket = _ROLE_BUCKETS.get(str(vertex.get("role")))
        elif node_id.startswith("reshard"):
            bucket = "reshard"
        elif isinstance(counters.get("weight_transfer"), dict):
            bucket = "weight_comm"
        elif isinstance(counters.get("activation_transfer"), dict):
            bucket = "activation_comm"
        if bucket is None:
            unclassified.append(node_id)
            continue
        ratio = actual / modeled
        if bucket == "activation_comm":
            activation_residuals.append(max(0.0, actual - modeled))
        else:
            ratios[bucket].append(ratio)
        observations.append({
            "node_id": node_id,
            "bucket": bucket,
            "predicted_duration_s": modeled,
            "actual_duration_s": actual,
            "actual_to_predicted_ratio": ratio,
        })

    stats = {
        bucket: _summary(values)
        for bucket, values in ratios.items()
        if values
    }
    if activation_residuals:
        stats["activation_comm_overhead_s"] = _summary(activation_residuals)

    # Keep cold and recurring observations independently inspectable even when the aggregate fit
    # excludes warmup steps. RPEG steps can overlap under staleness, so wall intervals are reported
    # per logical step without claiming that they sum to the graph makespan.
    per_step_values: dict[int, dict[str, list[float]]] = {}
    per_step_activation_residuals: dict[int, list[float]] = {}
    per_step_intervals: dict[int, list[float]] = {}
    for node_id, event in successful.items():
        raw_step = event.get("step")
        if raw_step is None:
            continue
        step = int(raw_step)
        vertex = vertex_plan.get(_vertex_id(node_id))
        bucket = _ROLE_BUCKETS.get(str(vertex.get("role"))) if isinstance(vertex, dict) else None
        counters = event.get("counters") or {}
        if bucket is None:
            if node_id.startswith("reshard"):
                bucket = "reshard"
            elif isinstance(counters.get("weight_transfer"), dict):
                bucket = "weight_comm"
            elif isinstance(counters.get("activation_transfer"), dict):
                bucket = "activation_comm"
        actual = float(counters["duration_s"])
        modeled = predicted[node_id]
        if bucket == "activation_comm":
            per_step_activation_residuals.setdefault(step, []).append(max(0.0, actual - modeled))
        elif bucket is not None:
            per_step_values.setdefault(step, {}).setdefault(bucket, []).append(actual / modeled)
        start = planned_starts.get(node_id)
        if start is not None and start.get("timestamp") is not None and event.get("timestamp") is not None:
            per_step_intervals.setdefault(step, []).extend(
                (float(start["timestamp"]), float(event["timestamp"]))
            )
    statistics_by_step: dict[str, dict[str, Any]] = {}
    for step in sorted(set(per_step_values) | set(per_step_activation_residuals) | set(per_step_intervals)):
        entry: dict[str, Any] = {
            bucket: _summary(values)
            for bucket, values in sorted(per_step_values.get(step, {}).items())
        }
        activation_values = per_step_activation_residuals.get(step, [])
        if activation_values:
            entry["activation_comm_overhead_s"] = _summary(activation_values)
        interval = per_step_intervals.get(step, [])
        if interval:
            entry["observed_wall_interval_s"] = max(interval) - min(interval)
        statistics_by_step[str(step)] = entry

    base = _hardware_mapping(hw_path)
    overrides: dict[str, float] = {}
    for bucket, field in (
        ("generation", "op_generation_scale"),
        ("inference", "op_inference_scale"),
        ("training", "op_training_scale"),
        ("weight_comm", "weight_comm_scale"),
        ("reshard", "reshard_scale"),
    ):
        if bucket in stats:
            overrides[field] = float(base.get(field, 1.0)) * float(stats[bucket]["median"])
    if activation_residuals:
        overrides["activation_comm_overhead_s"] = (
            float(base.get("activation_comm_overhead_s", 0.0))
            + statistics.median(activation_residuals)
        )

    return {
        "schema_version": "rpeg.runtime_cost_calibration.v1",
        "plan_path": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "trace_path": str(trace_path),
        "trace_sha256": _sha256(trace_path),
        "hardware_path": None if hw_path is None else str(hw_path),
        "hardware_sha256": None if hw_path is None else _sha256(hw_path),
        "planned_node_count": len(predicted),
        "successful_planned_node_count": len(successful),
        "failed_planned_node_count": failure_count,
        "partial_trace": len(successful) != len(predicted),
        "exclude_warmup_steps": int(exclude_warmup_steps),
        "excluded_warmup_node_count": excluded_warmup_node_count,
        "calibration_observation_count": len(calibration_successful),
        "unclassified_successful_nodes": sorted(unclassified),
        "statistics": stats,
        "statistics_by_step": statistics_by_step,
        "runtime_lifecycle_duration_s": {
            node_id: _summary(values)
            for node_id, values in sorted(lifecycle_durations.items())
        },
        "suggested_hardware_overrides": overrides,
        "observations": sorted(observations, key=lambda item: item["node_id"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit analytical cost corrections from an actual RPEG runtime trace."
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--hardware")
    parser.add_argument("--exclude-warmup-steps", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build_runtime_cost_calibration(
        args.plan,
        args.trace,
        hardware_path=args.hardware,
        exclude_warmup_steps=args.exclude_warmup_steps,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output.resolve()),
        "partial_trace": result["partial_trace"],
        "suggested_hardware_overrides": result["suggested_hardware_overrides"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
