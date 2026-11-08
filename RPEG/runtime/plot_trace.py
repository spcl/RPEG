# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

from collections import defaultdict, deque
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Iterable

from core.types import DeviceSpace
from runtime.cluster import load_inventory
from runtime.errors import RuntimeConfigError
from runtime.metrics import load_trace_events
from runtime.schema import OptimizedPlan, RuntimeRunConfig


_GPU_RESOURCE = re.compile(r"^gpu:(\d+):(\d+):(\d+)$")
_COLORS = {
    "actor": "#69AEE8",
    "critic": "#F07B72",
    "reference": "#F2B45E",
    "reward": "#65C77A",
    "ac_share": "#C58AE0",
    "comm": "#AEB8C2",
    "reshard": "#D49AC9",
    "unknown": "#87929D",
}
_LEGEND_LABELS = {
    "actor": "Actor",
    "critic": "Critic",
    "reference": "Reference",
    "reward": "Reward",
    "ac_share": "AC-share",
    "comm": "Communication",
    "reshard": "Reshard",
    "unknown": "Other",
}


@dataclass(frozen=True)
class TraceInterval:
    run_id: str
    node_id: str
    worker_id: str | None
    step: int | None
    attempt: int
    start: float
    finish: float
    status: str
    kind: str
    model: str | None
    role: str | None
    vertex_id: str | None
    parallelism: tuple[int, int, int] | None
    microbatch: int | None
    label: str | None
    resources: tuple[str, ...]
    batch_members: tuple[str, ...]
    counters: dict[str, Any]

    @property
    def duration_s(self) -> float:
        return max(0.0, self.finish - self.start)


@dataclass(frozen=True)
class TracePlotContext:
    run_id: str | None
    algorithm: str | None
    vertices: dict[str, dict[str, Any]]
    gpu_labels: dict[str, str]
    gpu_resources: tuple[str, ...]


@dataclass(frozen=True)
class TraceParseResult:
    intervals: tuple[TraceInterval, ...]
    replay_markers: tuple[tuple[float, int], ...]
    orphan_finishes: int
    incomplete_starts: int


def analytical_trace_to_plot(
    executable: Any,
    trace: Any,
    *,
    run_id: str,
) -> TraceParseResult:
    """
    Convert an analytical scheduler trace to the runtime plotting schema.
    """
    events: list[dict[str, Any]] = []
    for node in executable.nodes:
        if node not in trace.start or node not in trace.finish:
            continue
        allocations = node.allocs[-1:] if node.kind.value in {"comm", "reshard"} else node.allocs
        resources = []
        for allocation in allocations:
            r0, n0, g0 = allocation.origin
            re, ne, ge = allocation.extent
            resources.extend(
                f"gpu:{rack}:{host}:{gpu}"
                for rack in range(r0, r0 + re)
                for host in range(n0, n0 + ne)
                for gpu in range(g0, g0 + ge)
            )
        op = node.op
        vertex = None if op is None else op.template_vertex
        model = node.model
        if model is None and vertex is not None:
            model = vertex.kind.model
        role = None if vertex is None else vertex.kind.role.value
        vertex_id = None if vertex is None else vertex.id
        counters: dict[str, Any] = {
            "kind": node.kind.value,
            "resources": sorted(set(resources)),
            "batch_scheduled_nodes": [node.id],
        }
        for key, value in (
            ("model", None if model is None else model.value),
            ("role", role),
            ("vertex_id", vertex_id),
            ("parallelism", node.pi),
            ("microbatch", node.mb),
            ("label", node.label),
        ):
            if value is not None:
                counters[key] = value
        step = None if op is None else int(op.k) + 1
        common = {
            "run_id": run_id,
            "node_id": node.id,
            "status": "ok",
            "step": step,
        }
        events.append(
            {
                **common,
                "event": "start",
                "timestamp": float(trace.start[node]),
                "counters": counters,
            }
        )
        events.append(
            {
                **common,
                "event": "finish",
                "timestamp": float(trace.finish[node]),
                "counters": {
                    "kind": node.kind.value,
                    "duration_s": float(trace.finish[node] - trace.start[node]),
                },
            }
        )
    return parse_trace_events(events)


def _counters(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("counters")
    return dict(value) if isinstance(value, dict) else {}


def _event_key(event: dict[str, Any]) -> tuple[str, str | None, int | None]:
    step = event.get("step")
    return (
        str(event.get("node_id", "")),
        None if event.get("worker_id") is None else str(event["worker_id"]),
        None if step is None else int(step),
    )


def _parallelism(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return tuple(int(item) for item in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _vertex_id(node_id: str, counters: dict[str, Any]) -> str | None:
    explicit = counters.get("vertex_id")
    if explicit is not None:
        return str(explicit)
    if "@" in node_id:
        return node_id.split("@", 1)[0]
    return None


def _infer_model(node_id: str, vertex_id: str | None, counters: dict[str, Any]) -> str | None:
    explicit = counters.get("model")
    if explicit is not None:
        normalized = str(explicit).lower()
        aliases = {"pi": "actor", "v": "critic", "ref": "reference", "r": "reward", "ac": "ac_share"}
        return aliases.get(normalized, normalized)
    vertex = vertex_id or node_id.split("@", 1)[0]
    if vertex.startswith(("g_ac", "i_ac", "t_ac")):
        return "ac_share"
    if vertex.startswith(("g_pi", "i_pi", "t_pi")):
        return "actor"
    if vertex.startswith(("i_V", "t_V")):
        return "critic"
    if vertex.startswith("i_ref"):
        return "reference"
    if vertex.startswith("i_R"):
        return "reward"
    lowered = f"{node_id} {counters.get('label', '')}".lower()
    for token, model in (
        ("ac_share", "ac_share"),
        ("actor", "actor"),
        ("critic", "critic"),
        ("reference", "reference"),
        ("reward", "reward"),
    ):
        if token in lowered:
            return model
    return None


def _classify_kind(node_id: str, counters: dict[str, Any]) -> str:
    kind = str(counters.get("kind", "")).lower()
    if kind in {"op", "comm", "reshard"}:
        return kind
    lowered = node_id.lower()
    if lowered.startswith("comm"):
        return "comm"
    if "reshard" in lowered or lowered.startswith("rs"):
        return "reshard"
    return "op"


def parse_trace_events(
    events: Iterable[dict[str, Any]],
    *,
    context: TracePlotContext | None = None,
) -> TraceParseResult:
    ordered = list(events)
    observed_end = max((float(event.get("timestamp", 0.0)) for event in ordered), default=0.0)
    open_events: dict[tuple[str, str | None, int | None], deque[tuple[int, dict[str, Any]]]] = defaultdict(deque)
    intervals: list[TraceInterval] = []
    replays: list[tuple[float, int]] = []
    attempt = 0
    orphan_finishes = 0

    def make_interval(
        start_event: dict[str, Any],
        finish_event: dict[str, Any] | None,
        interval_attempt: int,
    ) -> TraceInterval:
        start_counters = _counters(start_event)
        finish_counters = _counters(finish_event or {})
        merged = {**start_counters, **finish_counters}
        node_id = str(start_event.get("node_id", ""))
        vertex_id = _vertex_id(node_id, merged)
        plan_vertex = None if context is None or vertex_id is None else context.vertices.get(vertex_id)
        model = _infer_model(node_id, vertex_id, merged)
        role = merged.get("role")
        if plan_vertex is not None:
            model = model or str(plan_vertex["model"])
            role = role or plan_vertex["role"]
        pi = _parallelism(merged.get("parallelism"))
        if pi is None and plan_vertex is not None:
            pi = _parallelism(plan_vertex.get("parallelism"))
        microbatch = merged.get("microbatch")
        if microbatch is None and plan_vertex is not None:
            microbatch = plan_vertex.get("microbatch")
        resources = start_counters.get("resources") or merged.get("resources") or ()
        batch = start_counters.get("batch_scheduled_nodes") or (node_id,)
        finish = observed_end if finish_event is None else float(finish_event.get("timestamp", observed_end))
        return TraceInterval(
            run_id=str(start_event.get("run_id", "")),
            node_id=node_id,
            worker_id=None if start_event.get("worker_id") is None else str(start_event["worker_id"]),
            step=None if start_event.get("step") is None else int(start_event["step"]),
            attempt=interval_attempt,
            start=float(start_event.get("timestamp", 0.0)),
            finish=max(float(start_event.get("timestamp", 0.0)), finish),
            status="incomplete" if finish_event is None else str(finish_event.get("status", "unknown")),
            kind=_classify_kind(node_id, merged),
            model=model,
            role=None if role is None else str(role),
            vertex_id=vertex_id,
            parallelism=pi,
            microbatch=None if microbatch is None else int(microbatch),
            label=None if merged.get("label") is None else str(merged["label"]),
            resources=tuple(str(resource) for resource in resources),
            batch_members=tuple(str(member) for member in batch),
            counters=merged,
        )

    for event in ordered:
        event_type = event.get("event")
        if event_type == "replay":
            replay_attempt = int(_counters(event).get("attempt", attempt + 1))
            attempt = replay_attempt
            replays.append((float(event.get("timestamp", 0.0)), replay_attempt))
            continue
        key = _event_key(event)
        if event_type == "start":
            event_attempt = int(_counters(event).get("attempt", attempt))
            open_events[key].append((event_attempt, event))
        elif event_type == "finish":
            queue = open_events.get(key)
            if not queue:
                orphan_finishes += 1
                continue
            event_attempt, start_event = queue.popleft()
            intervals.append(make_interval(start_event, event, event_attempt))

    incomplete = 0
    for queue in open_events.values():
        while queue:
            event_attempt, start_event = queue.popleft()
            intervals.append(make_interval(start_event, None, event_attempt))
            incomplete += 1
    intervals.sort(key=lambda item: (item.start, item.finish, item.node_id))
    return TraceParseResult(tuple(intervals), tuple(replays), orphan_finishes, incomplete)


def load_plot_context(config_path: str | Path) -> TracePlotContext:
    path = Path(config_path).resolve()
    run = RuntimeRunConfig.load(path)
    plan_path = Path(run.optimized_plan.path)
    if not plan_path.is_absolute():
        plan_path = path.parent / plan_path
    plan = OptimizedPlan.load(plan_path)
    vertices = {
        vertex_id: {
            "model": vertex.model,
            "role": vertex.role,
            "parallelism": vertex.parallelism,
            "microbatch": vertex.microbatch,
        }
        for vertex_id, vertex in plan.vertices.items()
    }
    ds = DeviceSpace(
        R=run.device_space.racks,
        N=run.device_space.nodes_per_rack,
        G=run.device_space.gpus_per_node,
    )
    gpu_resources = tuple(
        f"gpu:{rack}:{node}:{gpu}"
        for rack in range(ds.R)
        for node in range(ds.N)
        for gpu in range(ds.G)
    )
    gpu_labels: dict[str, str] = {}
    inventory_path = run.cluster.inventory_path
    if inventory_path is not None:
        inventory = Path(inventory_path)
        if not inventory.is_absolute():
            inventory = path.parent / inventory
        resolved_inventory = load_inventory(str(inventory), ds, scheduler=run.cluster.scheduler)
    elif run.cluster.scheduler == "slurm" and os.environ.get("SLURM_JOB_NODELIST"):
        resolved_inventory = load_inventory(None, ds, scheduler="slurm")
    else:
        resolved_inventory = None
    if resolved_inventory is not None:
        for key, gpu_ref in resolved_inventory.gpus.items():
            resource = f"gpu:{key[0]}:{key[1]}:{key[2]}"
            gpu_labels[resource] = f"{gpu_ref.hostname}:cuda:{gpu_ref.local_cuda_index}"
    return TracePlotContext(
        run_id=run.run_id,
        algorithm=run.algorithm.name,
        vertices=vertices,
        gpu_labels=gpu_labels,
        gpu_resources=gpu_resources,
    )


def _gpu_key(resource: str) -> tuple[int, int, int] | None:
    match = _GPU_RESOURCE.match(resource)
    if match is None:
        return None
    return tuple(int(value) for value in match.groups())  # type: ignore[return-value]


def _resource_sort_key(resource: str) -> tuple[int, int, int, str]:
    key = _gpu_key(resource)
    return (sys.maxsize, sys.maxsize, sys.maxsize, resource) if key is None else (*key, "")


def _node_resource(resource: str) -> str | None:
    key = _gpu_key(resource)
    return None if key is None else f"node:{key[0]}:{key[1]}"


def _nice_tick(span: float, target_ticks: int = 10) -> float:
    raw = max(span / target_ticks, 1e-12)
    exponent = math.floor(math.log10(raw))
    fraction = raw / (10**exponent)
    nice = 1.0 if fraction <= 1.0 else 2.0 if fraction <= 2.0 else 5.0 if fraction <= 5.0 else 10.0
    return nice * (10**exponent)


def _bar_category(interval: TraceInterval) -> str:
    if interval.kind in {"comm", "reshard"}:
        return interval.kind
    return interval.model if interval.model in _COLORS else "unknown"


def _tooltip(interval: TraceInterval, elapsed_start: float, elapsed_finish: float, resource: str) -> str:
    details = [
        interval.node_id,
        f"GPU resource: {resource}",
        f"elapsed: {elapsed_start:.6f}s - {elapsed_finish:.6f}s",
        f"duration: {interval.duration_s:.6f}s",
        f"attempt: {interval.attempt}",
        f"status: {interval.status}",
        f"kind: {interval.kind}",
    ]
    for label, value in (
        ("model", interval.model),
        ("role", interval.role),
        ("worker", interval.worker_id),
        ("step", interval.step),
        ("parallelism", interval.parallelism),
        ("microbatch", interval.microbatch),
        ("label", interval.label),
    ):
        if value is not None:
            details.append(f"{label}: {value}")
    return "\n".join(details)


def render_trace_svg(
    parsed: TraceParseResult,
    output_path: str | Path,
    *,
    context: TracePlotContext | None = None,
    title: str | None = None,
    group_by: str = "gpu",
    attempt: str = "all",
    show_comm: bool = True,
    show_reshard: bool = True,
    labels: str = "auto",
    width: int = 1800,
    row_height: int = 28,
    t_min: float | None = None,
    t_max: float | None = None,
) -> dict[str, Any]:
    if group_by not in {"gpu", "node"}:
        raise ValueError("group_by must be 'gpu' or 'node'")
    if labels not in {"auto", "all", "none"}:
        raise ValueError("labels must be 'auto', 'all', or 'none'")
    trace_interval_count = len(parsed.intervals)
    intervals = list(parsed.intervals)
    if attempt != "all":
        selected_attempt = max((item.attempt for item in intervals), default=0) if attempt == "latest" else int(attempt)
        intervals = [item for item in intervals if item.attempt == selected_attempt]
    if not show_comm:
        intervals = [item for item in intervals if item.kind != "comm"]
    if not show_reshard:
        intervals = [item for item in intervals if item.kind != "reshard"]
    # Controller lifecycle intervals (worker startup, model initialization, initial weight
    # publication, and transport prewarm) deliberately share the raw trace with graph execution, but
    # have no GPU allocation. They remain available for end-to-end metrics and calibration; a
    # runtime execution Gantt should contain only GPU-backed graph work.
    intervals = [
        item
        for item in intervals
        if any(_gpu_key(resource) is not None for resource in item.resources)
    ]
    if not intervals:
        raise ValueError("trace selection contains no GPU-backed start/finish intervals")

    # Match the analytical Gantt convention: elapsed time starts when graph execution first invokes
    # a model, not when the controller begins loading it. A communication-only trace falls back to
    # its first GPU interval.
    model_invocations = [item for item in intervals if item.kind == "op"]
    origin_intervals = model_invocations or intervals
    global_origin = min(item.start for item in origin_intervals)
    origin_policy = "first_model_invocation" if model_invocations else "first_gpu_interval"
    observed_resources = {
        resource
        for item in intervals
        for resource in item.resources
        if _gpu_key(resource) is not None
    }
    resources = set(observed_resources)
    if context is not None:
        resources.update(context.gpu_resources)
    if not resources:
        raise ValueError("trace contains no gpu:rack:node:gpu resources")

    if group_by == "gpu":
        rows = sorted(resources, key=_resource_sort_key)
        row_for_resource = {resource: resource for resource in rows}
        row_labels = {
            resource: (
                f"{context.gpu_labels[resource]} ({resource})"
                if context is not None and resource in context.gpu_labels
                else resource
            )
            for resource in rows
        }
    else:
        rows = sorted({_node_resource(resource) for resource in resources if _node_resource(resource) is not None})
        row_for_resource = {resource: _node_resource(resource) for resource in resources}
        row_labels = {}
        for row in rows:
            assert row is not None
            member_labels = sorted(
                {
                    context.gpu_labels[resource].split(":cuda:", 1)[0]
                    for resource in resources
                    if row_for_resource[resource] == row
                    and context is not None
                    and resource in context.gpu_labels
                }
            )
            row_labels[row] = f"{member_labels[0]} ({row})" if len(member_labels) == 1 else row

    elapsed_starts = [item.start - global_origin for item in intervals]
    elapsed_finishes = [item.finish - global_origin for item in intervals]
    # Pre-invocation communication, if present, is preserved in the raw trace, but clipped from the
    # default execution view. Callers can still request a negative relative bound explicitly with
    # --t-min.
    plot_start = max(0.0, min(elapsed_starts)) if t_min is None else float(t_min)
    plot_finish = max(elapsed_finishes) if t_max is None else float(t_max)
    if plot_finish <= plot_start:
        plot_finish = plot_start + max(1e-9, max((item.duration_s for item in intervals), default=1.0))
    span = plot_finish - plot_start

    longest_label = max(len(row_labels[row]) for row in rows)
    left = min(430, max(145, 8 * longest_label + 24))
    right = 35
    top = 92
    bottom = 64
    plot_width = max(400, width - left - right)
    height = top + len(rows) * row_height + bottom
    row_index = {row: index for index, row in enumerate(rows)}
    clip_id = "trace-bars-clip"

    def x_coord(elapsed: float) -> float:
        return left + (elapsed - plot_start) / span * plot_width

    run_id = next((item.run_id for item in intervals if item.run_id), context.run_id if context else None)
    chart_title = title or f"RPEG runtime trace{f': {run_id}' if run_id else ''}"
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="chart-title chart-desc">',
        "<style>",
        "text{font-family:DejaVu Sans,Arial,sans-serif;fill:#17212B;letter-spacing:0}",
        ".axis{font-size:12px}.row{font-size:11px}.bar-label{font-size:10px;pointer-events:none}",
        ".grid{stroke:#CBD2D9;stroke-width:1}.separator{stroke:#87929D;stroke-width:1.5}",
        "</style>",
        f'<title id="chart-title">{escape(chart_title)}</title>',
        f'<desc id="chart-desc">Runtime operations by {escape(group_by)} over elapsed wall-clock time.</desc>',
        f'<defs><clipPath id="{clip_id}"><rect x="{left}" y="{top}" width="{plot_width}" '
        f'height="{len(rows) * row_height}"/></clipPath></defs>',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        f'<text x="{left}" y="30" font-size="20" font-weight="600">{escape(chart_title)}</text>',
        f'<text x="{left}" y="51" font-size="12" fill="#52606D">'
        f'{len(intervals)} invocations, {len(rows)} {escape(group_by)} rows, '
        f'{span:.6g}s visible span</text>',
    ]

    categories = []
    for item in intervals:
        category = _bar_category(item)
        if category not in categories:
            categories.append(category)
    legend_x = left
    for category in categories:
        legend_x += 112
        parts.append(
            f'<rect x="{legend_x - 106}" y="62" width="12" height="12" rx="1" '
            f'fill="{_COLORS[category]}" stroke="#3E4C59" stroke-width="0.6"/>'
        )
        parts.append(
            f'<text x="{legend_x - 89}" y="72" font-size="11">{escape(_LEGEND_LABELS[category])}</text>'
        )

    tick = _nice_tick(span)
    first_tick = math.ceil(plot_start / tick - 1e-12) * tick
    tick_value = first_tick
    while tick_value <= plot_finish + tick * 1e-9:
        x = x_coord(tick_value)
        parts.append(
            f'<line class="grid" x1="{x:.3f}" y1="{top}" x2="{x:.3f}" '
            f'y2="{top + len(rows) * row_height}"/>'
        )
        parts.append(
            f'<text class="axis" x="{x:.3f}" y="{top + len(rows) * row_height + 22}" '
            f'text-anchor="middle">{tick_value:g}</text>'
        )
        tick_value += tick
    parts.append(
        f'<text class="axis" x="{left + plot_width / 2:.3f}" y="{height - 13}" '
        'text-anchor="middle">Elapsed time (s)</text>'
    )

    previous_node: tuple[int, int] | None = None
    for index, row in enumerate(rows):
        y = top + index * row_height
        parts.append(
            f'<rect x="{left}" y="{y}" width="{plot_width}" height="{row_height}" '
            f'fill="{"#F2F4F7" if index % 2 == 0 else "#FAFBFC"}"/>'
        )
        parts.append(
            f'<text class="row" x="{left - 8}" y="{y + row_height * 0.67:.3f}" '
            f'text-anchor="end">{escape(row_labels[row])}</text>'
        )
        if group_by == "gpu":
            key = _gpu_key(row)
            node_key = None if key is None else key[:2]
            if previous_node is not None and node_key != previous_node:
                parts.append(
                    f'<line class="separator" x1="0" y1="{y}" x2="{width}" y2="{y}"/>'
                )
            previous_node = node_key

    replay_markers = parsed.replay_markers if attempt == "all" else ()
    for timestamp, replay_attempt in replay_markers:
        elapsed = timestamp - global_origin
        if not plot_start <= elapsed <= plot_finish:
            continue
        x = x_coord(elapsed)
        parts.append(
            f'<line x1="{x:.3f}" y1="{top}" x2="{x:.3f}" y2="{top + len(rows) * row_height}" '
            'stroke="#B83280" stroke-width="1.5" stroke-dasharray="5 3"/>'
        )
        parts.append(
            f'<text x="{x + 4:.3f}" y="{top - 7}" font-size="10" fill="#B83280">'
            f'replay {replay_attempt}</text>'
        )

    rendered_bars = 0
    for item in intervals:
        item_start = item.start - global_origin
        item_finish = item.finish - global_origin
        clipped_start = max(plot_start, item_start)
        clipped_finish = min(plot_finish, item_finish)
        if clipped_finish < clipped_start:
            continue
        item_resources = sorted(
            {
                row_for_resource[resource]
                for resource in item.resources
                if resource in row_for_resource and row_for_resource[resource] in row_index
            },
            key=lambda row: row_index[row],
        )
        if not item_resources:
            continue
        batch_count = max(1, len(item.batch_members))
        try:
            batch_index = item.batch_members.index(item.node_id)
        except ValueError:
            batch_index = 0
        inner_height = max(2.0, (row_height - 6) / batch_count)
        bar_width = max(0.8, x_coord(clipped_finish) - x_coord(clipped_start))
        color = _COLORS[_bar_category(item)]
        stroke = "#B42318" if item.status == "error" else "#3E4C59"
        dash = ' stroke-dasharray="4 2"' if item.status == "incomplete" else ""
        for row in item_resources:
            y = top + row_index[row] * row_height + 3 + batch_index * inner_height
            tooltip = _tooltip(item, item_start, item_finish, row)
            parts.append(
                f'<g clip-path="url(#{clip_id})"><rect x="{x_coord(clipped_start):.3f}" '
                f'y="{y:.3f}" width="{bar_width:.3f}" height="{max(1.5, inner_height - 1):.3f}" '
                f'rx="1" fill="{color}" stroke="{stroke}" stroke-width="'
                f'{"2" if item.status == "error" else "0.7"}"{dash}>'
                f"<title>{escape(tooltip)}</title></rect>"
            )
            label_width = len(item.node_id) * 6
            if labels == "all" or (labels == "auto" and bar_width >= label_width + 8 and inner_height >= 11):
                parts.append(
                    f'<text class="bar-label" x="{x_coord(clipped_start) + 4:.3f}" '
                    f'y="{y + max(1.5, inner_height - 1) * 0.70:.3f}">{escape(item.node_id)}</text>'
                )
            parts.append("</g>")
            rendered_bars += 1

    parts.append(
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{len(rows) * row_height}" '
        'fill="none" stroke="#52606D" stroke-width="1"/>'
    )
    parts.append("</svg>")
    output = Path(output_path)
    if output.suffix.lower() != ".svg":
        raise ValueError("runtime trace output must use the .svg extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n")
    return {
        "output": str(output),
        "run_id": run_id,
        "intervals": len(intervals),
        "rendered_bars": rendered_bars,
        "rows": len(rows),
        "group_by": group_by,
        "attempt": attempt,
        "trace_intervals": trace_interval_count,
        "excluded_non_gpu_intervals": trace_interval_count - len(intervals),
        "elapsed_origin_policy": origin_policy,
        "elapsed_origin_unix_s": global_origin,
        "visible_start_s": plot_start,
        "visible_finish_s": plot_finish,
        "orphan_finishes": parsed.orphan_finishes,
        "incomplete_starts": parsed.incomplete_starts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render an RPEG runtime JSONL trace as a per-GPU or per-node SVG Gantt chart."
    )
    parser.add_argument("--trace", required=True, help="Runtime JSONL trace path")
    parser.add_argument("--output", help="Output SVG path (default: TRACE.gantt.svg)")
    parser.add_argument("--config", help="Optional rpeg.run.v1 config for idle GPUs, model metadata, and host labels")
    parser.add_argument("--run-id", help="Select one run_id when a JSONL file contains multiple runs")
    parser.add_argument("--group-by", choices=("gpu", "node"), default="gpu")
    parser.add_argument("--attempt", default="all", help="all, latest, or a zero-based replay attempt")
    parser.add_argument("--title")
    parser.add_argument("--labels", choices=("auto", "all", "none"), default="auto")
    parser.add_argument("--hide-comm", action="store_true")
    parser.add_argument("--hide-reshard", action="store_true")
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--row-height", type=int, default=28)
    parser.add_argument("--t-min", type=float, help="Visible elapsed-time lower bound in seconds")
    parser.add_argument("--t-max", type=float, help="Visible elapsed-time upper bound in seconds")
    args = parser.parse_args(argv)

    try:
        events = load_trace_events(args.trace)
        if not events:
            raise ValueError(f"trace is missing or empty: {args.trace}")
        run_ids = sorted({str(event.get("run_id")) for event in events if event.get("run_id") is not None})
        if args.run_id is not None:
            events = [event for event in events if str(event.get("run_id")) == args.run_id]
            if not events:
                raise ValueError(f"run_id {args.run_id!r} does not occur in the trace")
        elif len(run_ids) > 1:
            raise ValueError(f"trace contains multiple run_ids {run_ids}; select one with --run-id")
        if args.width < 700:
            raise ValueError("--width must be at least 700")
        if args.row_height < 14:
            raise ValueError("--row-height must be at least 14")
        context = load_plot_context(args.config) if args.config else None
        parsed = parse_trace_events(events, context=context)
        output = args.output or str(Path(args.trace).with_suffix(".gantt.svg"))
        summary = render_trace_svg(
            parsed,
            output,
            context=context,
            title=args.title,
            group_by=args.group_by,
            attempt=args.attempt,
            show_comm=not args.hide_comm,
            show_reshard=not args.hide_reshard,
            labels=args.labels,
            width=args.width,
            row_height=args.row_height,
            t_min=args.t_min,
            t_max=args.t_max,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (OSError, RuntimeConfigError, ValueError, json.JSONDecodeError) as exc:
        print(f"trace plot error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
