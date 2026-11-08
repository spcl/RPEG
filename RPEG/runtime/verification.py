# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from passes.lower import ExecNodeType
from runtime.compiler import CompiledRuntimePlan
from runtime.errors import RuntimeExecutionError
from runtime.graph import RuntimeGraph, RuntimeNodeSpec
from runtime.weight_versions import consumer_weight_version, uses_shadow_copies

_RPEG_WEIGHT_LABEL_TO_MODEL = {
    "W_Pi": "actor",
    "W_V": "critic",
    "W_Ref": "reference",
    "W_R": "reward",
    "W_AC": "ac_share",
}

# Semantic backends emit controller lifecycle intervals into the same trace as the optimized graph.
# They are deliberately outside RuntimeGraph: the search objective models steady-state RPEG work,
# while startup remains observable for end-to-end validation and calibration.
_RUNTIME_LIFECYCLE_NODE_IDS = frozenset({
    "runtime.workers.start",
    "runtime.workers.initialize",
    "runtime.weights.publish_initial",
    "runtime.transports.prewarm",
})


def is_runtime_lifecycle_node(node_id: object) -> bool:
    return node_id in _RUNTIME_LIFECYCLE_NODE_IDS


def _fail(message: str) -> None:
    raise RuntimeExecutionError(f"runtime trace verification failed: {message}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def graph_node_step(node: RuntimeNodeSpec) -> int | None:
    if node.logical_step is not None:
        return int(node.logical_step)
    if node.exec_node.op is not None:
        return int(node.exec_node.op.k) + 1
    for node_id in (*node.dependencies, *node.successors):
        if "@" not in node_id:
            continue
        try:
            return int(node_id.rsplit("@", 1)[1].split(":", 1)[0]) + 1
        except ValueError:
            continue
    return None


def _event_counters(event: dict[str, Any]) -> dict[str, Any]:
    counters = event.get("counters")
    return counters if isinstance(counters, dict) else {}


def _recovery_step(event: dict[str, Any]) -> int:
    counters = _event_counters(event)
    try:
        return int(counters.get("step", 0))
    except (TypeError, ValueError):
        return 0


def _checkpoint_replays_node(node: RuntimeNodeSpec, recovered_step: int) -> bool:
    step = graph_node_step(node)
    return (
        step is None
        or int(step) > int(recovered_step)
        or (
            int(step) == int(recovered_step)
            and node.kind in {ExecNodeType.COMM, ExecNodeType.RESHARD}
            and str(node.label or "").startswith("W_")
        )
    )


def _model_for_version(node: RuntimeNodeSpec, counters: dict[str, Any]) -> str | None:
    loaded = counters.get("loaded_weight")
    if isinstance(loaded, dict) and loaded.get("model") is not None:
        return str(loaded["model"])
    if counters.get("model") is not None:
        return str(counters["model"])
    return node.model


def _require_shadow_flag(
    compiled: CompiledRuntimePlan,
    *,
    node_id: str,
    model: str,
    raw_shadow: Any,
    counter_name: str,
) -> bool:
    _require(raw_shadow is not None, f"{node_id} {counter_name} for {model} is missing shadow flag")
    actual = bool(raw_shadow)
    expected = uses_shadow_copies(compiled, model)
    _require(
        actual == expected,
        f"{node_id} {counter_name} for {model} reports shadow={actual}, expected {expected}",
    )
    return actual


def _split_part_index(node_id: str) -> int | None:
    marker = ":g"
    if marker not in node_id:
        marker = ":i"
    if marker not in node_id:
        return None
    raw = node_id.rsplit(marker, 1)[1]
    try:
        return int(raw)
    except ValueError:
        return None


def _quantile_batch_offset(compiled: CompiledRuntimePlan, part: int) -> int:
    if part <= 0:
        return 0
    quantiles = compiled.run_config.workload.gen_quantiles or ()
    total = compiled.run_config.workload.batch_size * compiled.run_config.workload.samples_per_prompt
    return sum(
        round(float(quantiles[idx][0]) * total) - (round(float(quantiles[idx - 1][0]) * total) if idx else 0)
        for idx in range(min(part, len(quantiles)))
    )


def _verify_split_counters(
    compiled: CompiledRuntimePlan,
    node: RuntimeNodeSpec,
    event: dict[str, Any],
) -> dict[str, int]:
    if node.kind != ExecNodeType.OP or node.exec_node.n_seqs is None:
        return {}
    counters = _event_counters(event)
    checked: dict[str, int] = {}

    _require(
        counters.get("partial_n_seqs") is not None,
        f"{node.node_id} is a split node but did not report partial_n_seqs",
    )
    actual_n_seqs = int(counters["partial_n_seqs"])
    _require(
        actual_n_seqs >= 0,
        f"{node.node_id} partial_n_seqs must be nonnegative, got {actual_n_seqs}",
    )
    checked["partial_n_seqs"] = actual_n_seqs

    for attr, counter_name in (
        ("n_active", "partial_n_active"),
        ("phase_steps", "partial_phase_steps"),
        ("phase_t_start", "partial_phase_t_start"),
    ):
        expected = getattr(node.exec_node, attr)
        if expected is None:
            continue
        _require(
            counters.get(counter_name) is not None,
            f"{node.node_id} is missing {counter_name}",
        )
        actual = int(counters[counter_name])
        _require(actual == int(expected), f"{node.node_id} {counter_name}={actual}, expected {expected}")
        checked[counter_name] = actual

    _require(
        counters.get("partial_is_prefill_phase") is not None,
        f"{node.node_id} is missing partial_is_prefill_phase",
    )
    actual_prefill = bool(counters["partial_is_prefill_phase"])
    _require(
        actual_prefill == bool(node.exec_node.is_prefill_phase),
        f"{node.node_id} partial_is_prefill_phase={actual_prefill}, expected {node.exec_node.is_prefill_phase}",
    )

    part = _split_part_index(node.node_id)
    if part is not None:
        _require(counters.get("artifact_part") is not None, f"{node.node_id} is missing artifact_part")
        actual_part = int(counters["artifact_part"])
        _require(actual_part == part, f"{node.node_id} artifact_part={actual_part}, expected {part}")
        checked["artifact_part"] = actual_part

        expected_parts = len(compiled.run_config.workload.gen_quantiles or ())
        _require(counters.get("artifact_parts") is not None, f"{node.node_id} is missing artifact_parts")
        actual_parts = int(counters["artifact_parts"])
        _require(actual_parts == expected_parts, f"{node.node_id} artifact_parts={actual_parts}, expected {expected_parts}")
        checked["artifact_parts"] = actual_parts

        expected_offset = _quantile_batch_offset(compiled, part)
        offset_counter = (
            "planned_batch_offset"
            if counters.get("planned_batch_offset") is not None
            else "partial_batch_offset"
        )
        if counters.get(offset_counter) is not None:
            actual_offset = int(counters[offset_counter])
            _require(
                actual_offset == expected_offset,
                f"{node.node_id} planned_batch_offset={actual_offset}, expected {expected_offset}",
            )
            checked["partial_batch_offset"] = actual_offset

        if node.vertex_id in {"g_pi", "g_ac"} and compiled.run_config.workload.gen_quantiles:
            expected_length = int(compiled.run_config.workload.gen_quantiles[part][1])
            length_key = "generation_length" if _counter_value(counters, "generation_length") is not None else "generated_tokens"
            raw_length = _counter_value(counters, length_key)
            _require(raw_length is not None, f"{node.node_id} is missing generation_length")
            actual_length = int(raw_length)
            _require(
                actual_length == expected_length,
                f"{node.node_id} {length_key}={actual_length}, expected generation_length {expected_length}",
            )
            checked[length_key] = actual_length

    return checked


def _counter_value(counters: dict[str, Any], key: str) -> Any:
    if counters.get(key) is not None:
        return counters[key]
    ranks = counters.get("ranks")
    if not isinstance(ranks, (list, tuple)):
        return None
    values = []
    for rank in ranks:
        if not isinstance(rank, dict):
            return None
        rank_counters = rank.get("counters")
        if not isinstance(rank_counters, dict) or rank_counters.get(key) is None:
            return None
        values.append(rank_counters[key])
    if values and all(value == values[0] for value in values[1:]):
        return values[0]
    return None


def _verify_generation_counters(
    compiled: CompiledRuntimePlan,
    node: RuntimeNodeSpec,
    counters: dict[str, Any],
) -> dict[str, int | str | bool]:
    if node.vertex_id not in {"g_pi", "g_ac"}:
        return {}
    expected_backend = str(getattr(compiled.run_config.algorithm, "generation_backend", "full_sequence"))
    observed_backend = _counter_value(counters, "generation_backend")
    if observed_backend is None:
        _require(expected_backend == "full_sequence", f"{node.node_id} is missing generation_backend")
        return {}
    observed_backend = str(observed_backend)
    _require(
        observed_backend == expected_backend,
        f"{node.node_id} generation_backend={observed_backend}, expected {expected_backend}",
    )
    expected_length = _expected_generation_length(compiled, node)
    checked: dict[str, int | str | bool] = {"generation_backend": observed_backend}
    is_split_phase = node.exec_node.phase_steps is not None
    phase_decode_steps = int(node.exec_node.phase_steps or 0)
    is_prefill_phase = bool(node.exec_node.is_prefill_phase)
    if expected_backend == "static_kv":
        raw_enabled = _counter_value(counters, "kv_cache_enabled")
        _require(raw_enabled is not None, f"{node.node_id} is missing kv_cache_enabled")
        kv_enabled = _as_bool_counter(raw_enabled)
        _require(kv_enabled is True, f"{node.node_id} kv_cache_enabled={kv_enabled}, expected True")
        checked["kv_cache_enabled"] = kv_enabled
        expectations = {
            "prefill_forward_steps": 1 if expected_length > 0 and (not is_split_phase or is_prefill_phase) else 0,
            "decode_forward_steps": phase_decode_steps if is_split_phase else max(0, expected_length - 1),
            "generation_forward_steps": (
                phase_decode_steps + (1 if is_prefill_phase else 0)
                if is_split_phase
                else expected_length
            ),
            "old_logp_forward_steps": 0,
        }
        minimums = {
            "kv_cache_pool_entries": 1 if expected_length > 0 else 0,
            "kv_cache_pool_hits": 0,
            "kv_cache_pool_misses": 1 if expected_length > 0 else 0,
            "kv_cache_pool_evictions": 0,
            "kv_cache_max_batch_size": 1 if expected_length > 0 else 0,
            "kv_cache_max_sequence_length": expected_length if expected_length > 0 else 0,
        }
        for key, expected_min in minimums.items():
            raw = _counter_value(counters, key)
            _require(raw is not None, f"{node.node_id} is missing {key}")
            actual = int(raw)
            _require(actual >= expected_min, f"{node.node_id} {key}={actual}, expected >= {expected_min}")
            checked[key] = actual
        raw_reused = _counter_value(counters, "kv_cache_reused")
        if raw_reused is not None:
            checked["kv_cache_reused"] = _as_bool_counter(raw_reused)
        raw_old_logps = _counter_value(counters, "old_logps_from_generation")
        _require(raw_old_logps is not None, f"{node.node_id} is missing old_logps_from_generation")
        old_logps_from_generation = _as_bool_counter(raw_old_logps)
        _require(
            old_logps_from_generation is True,
            f"{node.node_id} old_logps_from_generation={old_logps_from_generation}, expected True",
        )
        checked["old_logps_from_generation"] = old_logps_from_generation
    else:
        raw_enabled = _counter_value(counters, "kv_cache_enabled")
        if raw_enabled is not None:
            kv_enabled = _as_bool_counter(raw_enabled)
            _require(kv_enabled is False, f"{node.node_id} kv_cache_enabled={kv_enabled}, expected False")
            checked["kv_cache_enabled"] = kv_enabled
        expectations = {
            "decode_forward_steps": (
                phase_decode_steps + (1 if is_prefill_phase else 0)
                if is_split_phase
                else expected_length
            ),
            "generation_forward_steps": (
                phase_decode_steps + (1 if is_prefill_phase else 0)
                if is_split_phase
                else expected_length
            ),
        }
    for key, expected in expectations.items():
        raw = _counter_value(counters, key)
        _require(raw is not None, f"{node.node_id} is missing {key}")
        actual = int(raw)
        _require(actual == expected, f"{node.node_id} {key}={actual}, expected {expected}")
        checked[key] = actual
    raw_prefill = _counter_value(counters, "prefill_forward_steps")
    if expected_backend != "static_kv" and raw_prefill is not None:
        actual_prefill = int(raw_prefill)
        _require(actual_prefill == 0, f"{node.node_id} prefill_forward_steps={actual_prefill}, expected 0")
        checked["prefill_forward_steps"] = actual_prefill
    if is_split_phase:
        raw_forced = _counter_value(counters, "phase_forced_completions")
        if raw_forced is not None:
            forced = int(raw_forced)
            _require(
                forced == 0,
                f"{node.node_id} forced {forced} quantile completions; quantile phases must preserve EOS semantics",
            )
            checked["phase_forced_completions"] = forced
    return checked


def _expected_generation_length(compiled: CompiledRuntimePlan, node: RuntimeNodeSpec) -> int:
    part = _split_part_index(node.node_id)
    if part is not None and compiled.run_config.workload.gen_quantiles:
        return int(compiled.run_config.workload.gen_quantiles[part][1])
    return int(compiled.run_config.workload.generation_length)


def _as_bool_counter(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _verify_weight_counters(
    compiled: CompiledRuntimePlan,
    node: RuntimeNodeSpec,
    event: dict[str, Any],
) -> dict[str, int]:
    counters = _event_counters(event)
    step = event.get("step")
    if step is None:
        step = graph_node_step(node)
    checked: dict[str, int] = {}

    if "weight_version" in counters:
        model = _model_for_version(node, counters)
        _require(model is not None, f"{node.node_id} reports weight_version without a model")
        expected = consumer_weight_version(compiled, model, int(step))
        actual = int(counters["weight_version"])
        _require(actual == expected, f"{node.node_id} consumed {model} v{actual}, expected v{expected}")
        _require_shadow_flag(
            compiled,
            node_id=node.node_id,
            model=model,
            raw_shadow=counters.get("shadow"),
            counter_name="weight_version",
        )
        checked[f"consumed:{model}"] = actual

    loaded = counters.get("loaded_weight")
    if isinstance(loaded, dict):
        model = str(loaded.get("model"))
        actual = int(loaded.get("version"))
        expected = consumer_weight_version(compiled, model, int(step))
        _require(actual == expected, f"{node.node_id} loaded {model} v{actual}, expected v{expected}")
        _require_shadow_flag(
            compiled,
            node_id=node.node_id,
            model=model,
            raw_shadow=loaded.get("shadow"),
            counter_name="loaded_weight",
        )
        checked[f"loaded:{model}"] = actual

    produced = counters.get("produced_weight")
    if isinstance(produced, dict):
        model = str(produced.get("model"))
        actual = int(produced.get("version"))
        expected = int(step)
        _require(actual == expected, f"{node.node_id} produced {model} v{actual}, expected v{expected}")
        _require_shadow_flag(
            compiled,
            node_id=node.node_id,
            model=model,
            raw_shadow=produced.get("shadow"),
            counter_name="produced_weight",
        )
        checked[f"produced:{model}"] = actual
    elif "produced_weight_version" in counters:
        model = _model_for_version(node, counters)
        _require(model is not None, f"{node.node_id} reports produced_weight_version without a model")
        actual = int(counters["produced_weight_version"])
        expected = int(step)
        _require(actual == expected, f"{node.node_id} produced {model} v{actual}, expected v{expected}")
        _require_shadow_flag(
            compiled,
            node_id=node.node_id,
            model=model,
            raw_shadow=counters.get("shadow"),
            counter_name="produced_weight_version",
        )
        checked[f"produced:{model}"] = actual

    transfer = counters.get("weight_transfer")
    if isinstance(transfer, dict):
        model = str(transfer.get("model"))
        _require_shadow_flag(
            compiled,
            node_id=node.node_id,
            model=model,
            raw_shadow=transfer.get("shadow"),
            counter_name="weight_transfer",
        )

    if node.role == "TRAIN" and node.model in {"actor", "critic", "ac_share"}:
        _require(
            any(key.startswith("produced:") for key in checked),
            f"{node.node_id} is a train node but did not report a produced weight version",
        )
    return checked


def _expected_transfer_model(node: RuntimeNodeSpec) -> str | None:
    if node.label in _RPEG_WEIGHT_LABEL_TO_MODEL:
        return _RPEG_WEIGHT_LABEL_TO_MODEL[str(node.label)]
    return node.model


def _verify_rank_summary(node_id: str, name: str, summary: dict[str, Any]) -> int:
    _require(summary.get("rank_count") is not None, f"{node_id} transfer {name} is missing rank_count")
    rank_count = int(summary["rank_count"])
    _require(rank_count >= 0, f"{node_id} transfer {name} has negative rank_count")
    ranks = summary.get("ranks") or ()
    _require(isinstance(ranks, (list, tuple)), f"{node_id} transfer {name} ranks must be a list")
    _require(len(ranks) == rank_count, f"{node_id} transfer {name} rank_count={rank_count} but has {len(ranks)} ranks")
    total_rank_bytes = 0
    saw_rank_bytes = False
    for rank in ranks:
        _require(isinstance(rank, dict), f"{node_id} transfer {name} rank entry must be a mapping")
        _require(rank.get("rank") is not None, f"{node_id} transfer {name} rank entry is missing rank")
        if rank.get("bytes") is not None:
            saw_rank_bytes = True
            rank_bytes = int(rank["bytes"])
            _require(rank_bytes >= 0, f"{node_id} transfer {name} rank {rank.get('rank')} has negative bytes")
            total_rank_bytes += rank_bytes
    if summary.get("bytes") is not None:
        total_bytes = int(summary["bytes"])
        _require(total_bytes >= 0, f"{node_id} transfer {name} has negative bytes")
        if saw_rank_bytes:
            _require(
                total_bytes == total_rank_bytes,
                f"{node_id} transfer {name} bytes={total_bytes}, expected rank byte sum {total_rank_bytes}",
            )
        return total_bytes
    return total_rank_bytes


def _verify_activation_transfer(node: RuntimeNodeSpec, counters: dict[str, Any]) -> dict[str, int]:
    transfer = counters.get("activation_transfer")
    if not isinstance(transfer, dict):
        return {}
    _require(node.label is None or transfer.get("label") == node.label, f"{node.node_id} activation transfer label mismatch")
    expected_step = graph_node_step(node)
    if expected_step is not None:
        _require(
            int(transfer.get("destination_step")) == expected_step,
            f"{node.node_id} activation destination_step={transfer.get('destination_step')}, expected {expected_step}",
        )
    artifacts = transfer.get("artifacts") or ()
    _require(isinstance(artifacts, (list, tuple)) and artifacts, f"{node.node_id} activation transfer has no artifacts")
    semantic = transfer.get("semantic_transfers")
    _require(isinstance(semantic, dict) and semantic, f"{node.node_id} activation transfer has no semantic summaries")
    missing = sorted(str(artifact) for artifact in artifacts if str(artifact) not in semantic)
    _require(not missing, f"{node.node_id} activation transfer missing semantic summaries for {missing}")
    checked: dict[str, int] = {}
    for artifact, summary in semantic.items():
        _require(isinstance(summary, dict), f"{node.node_id} activation transfer {artifact} summary must be a mapping")
        checked[str(artifact)] = _verify_rank_summary(node.node_id, str(artifact), summary)
        if summary.get("source_shards") is not None:
            source_shards = summary["source_shards"]
            _require(isinstance(source_shards, (list, tuple)), f"{node.node_id} transfer {artifact} source_shards must be a list")
            for shard in source_shards:
                _require(isinstance(shard, dict), f"{node.node_id} transfer {artifact} source shard must be a mapping")
                if shard.get("sha256") is not None:
                    sha = str(shard["sha256"])
                    _require(len(sha) == 64, f"{node.node_id} transfer {artifact} has invalid sha256 length")
                if shard.get("bytes") is not None:
                    _require(int(shard["bytes"]) >= 0, f"{node.node_id} transfer {artifact} source shard has negative bytes")
    return checked


def _verify_weight_transfer(node: RuntimeNodeSpec, counters: dict[str, Any]) -> dict[str, int]:
    transfer = counters.get("weight_transfer")
    if not isinstance(transfer, dict):
        return {}
    expected_model = _expected_transfer_model(node)
    if expected_model is not None:
        _require(transfer.get("model") == expected_model, f"{node.node_id} weight transfer model mismatch")
    if transfer.get("version") is not None:
        _require(int(transfer["version"]) >= 0, f"{node.node_id} weight transfer has negative version")
    checked = {"weight_transfer": _verify_rank_summary(node.node_id, "weight_transfer", transfer)}
    ranks = transfer.get("ranks") or ()
    _require(isinstance(ranks, (list, tuple)), f"{node.node_id} weight transfer ranks must be a list")
    in_place_copy_modes = {
        "in_place_routed_host_reshard",
        "in_place_routed_device_reshard",
        "in_place_dp_to_pp_local_extract",
        "in_place_pp_to_dp_host_staged_broadcast",
    }
    in_place_reshard = transfer.get("copy_mode") in in_place_copy_modes
    if in_place_reshard:
        _require(node.kind == ExecNodeType.RESHARD, f"{node.node_id} in-place layout switch must lower from RESHARD")
        _require(transfer.get("shadow") is False, f"{node.node_id} in-place layout switch cannot target a shadow")
        worker_resources = tuple(resource for resource in node.resource_keys if resource.startswith("worker:"))
        _require(
            len(worker_resources) == 1,
            f"{node.node_id} in-place layout switch must have exactly one worker owner",
        )
    expected_source_parallelism: tuple[int, int, int] | None = None
    expected_destination_parallelism: tuple[int, int, int] | None = None
    for rank in ranks:
        _require(isinstance(rank, dict), f"{node.node_id} weight transfer rank entry must be a mapping")
        if in_place_reshard:
            _require(
                rank.get("copy_mode") == transfer.get("copy_mode"),
                f"{node.node_id} in-place layout switch rank {rank.get('rank')} has wrong copy mode",
            )
            rank_counters = rank.get("counters")
            _require(
                isinstance(rank_counters, dict),
                f"{node.node_id} in-place layout switch rank {rank.get('rank')} is missing counters",
            )
            _require(
                rank_counters.get("switched") is True,
                f"{node.node_id} in-place layout switch rank {rank.get('rank')} did not switch",
            )
            source_parallelism = tuple(int(value) for value in rank_counters.get("source_parallelism") or ())
            destination_parallelism = tuple(
                int(value) for value in rank_counters.get("destination_parallelism") or ()
            )
            _require(
                len(source_parallelism) == 3 and all(value > 0 for value in source_parallelism),
                f"{node.node_id} in-place layout switch rank {rank.get('rank')} has invalid source parallelism",
            )
            _require(
                len(destination_parallelism) == 3 and all(value > 0 for value in destination_parallelism),
                f"{node.node_id} in-place layout switch rank {rank.get('rank')} has invalid destination parallelism",
            )
            _require(
                source_parallelism != destination_parallelism,
                f"{node.node_id} in-place layout switch rank {rank.get('rank')} did not change layout",
            )
            _require(
                int(rank_counters.get("source_rank_bytes", 0)) > 0
                and int(rank_counters.get("destination_rank_bytes", 0)) > 0,
                f"{node.node_id} in-place layout switch rank {rank.get('rank')} has no routed tensor bytes",
            )
            if expected_source_parallelism is None:
                expected_source_parallelism = source_parallelism
                expected_destination_parallelism = destination_parallelism
            _require(
                source_parallelism == expected_source_parallelism
                and destination_parallelism == expected_destination_parallelism,
                f"{node.node_id} in-place layout switch ranks disagree on layouts",
            )
            continue
        sha = rank.get("sha256")
        _require(sha is not None, f"{node.node_id} weight transfer rank {rank.get('rank')} is missing sha256")
        _require(len(str(sha)) == 64, f"{node.node_id} weight transfer rank {rank.get('rank')} has invalid sha256 length")
        if rank.get("source_sha256") is not None:
            _require(
                len(str(rank["source_sha256"])) == 64,
                f"{node.node_id} weight transfer rank {rank.get('rank')} has invalid source_sha256 length",
            )
    return checked


def verify_runtime_trace(
    compiled: CompiledRuntimePlan,
    graph: RuntimeGraph,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = Counter(graph.nodes.keys())
    replays = [event for event in events if event.get("event") == "replay"]
    last_replay_idx = max((idx for idx, event in enumerate(events) if event.get("event") == "replay"), default=-1)
    final_events = events[last_replay_idx + 1 :]
    recoveries = [event for event in final_events if event.get("event") == "recover"]
    _require(not recoveries or replays, "checkpoint recovery event requires a replay attempt")
    recovered_step = max((_recovery_step(event) for event in recoveries), default=0)
    recovered_nodes = {
        node_id
        for node_id, node in graph.nodes.items()
        if recovered_step > 0 and not _checkpoint_replays_node(node, recovered_step)
    }
    expected_final = Counter(node_id for node_id in graph.nodes if node_id not in recovered_nodes)
    terminal_node_ids = {
        event.get("node_id")
        for event in final_events
        if event.get("event") in {"start", "finish"}
    }
    unknown_node_ids = sorted(
        str(node_id)
        for node_id in terminal_node_ids
        if node_id not in graph.nodes and not is_runtime_lifecycle_node(node_id)
    )
    _require(not unknown_node_ids, f"trace references unknown nodes {unknown_node_ids}")
    starts = Counter(
        event.get("node_id")
        for event in final_events
        if event.get("event") == "start" and event.get("node_id") in graph.nodes
    )
    finishes = Counter(
        event.get("node_id")
        for event in final_events
        if event.get("event") == "finish" and event.get("node_id") in graph.nodes
    )
    _require(starts == expected_final, "trace start events do not match runtime nodes")
    _require(finishes == expected_final, "trace finish events do not match runtime nodes")
    lifecycle_starts = Counter(
        event.get("node_id")
        for event in final_events
        if event.get("event") == "start" and is_runtime_lifecycle_node(event.get("node_id"))
    )
    lifecycle_finishes = Counter(
        event.get("node_id")
        for event in final_events
        if event.get("event") == "finish" and is_runtime_lifecycle_node(event.get("node_id"))
    )
    _require(lifecycle_starts == lifecycle_finishes, "runtime lifecycle start/finish events do not match")
    _require(
        all(count == 1 for count in lifecycle_starts.values()),
        "runtime lifecycle nodes must execute exactly once per final attempt",
    )

    start_by_node: dict[str, dict[str, Any]] = {}
    finish_by_node: dict[str, dict[str, Any]] = {}
    for event in events:
        node_id = event.get("node_id")
        _require(event.get("run_id") == compiled.run_config.run_id, f"{node_id} has wrong run_id")
        if event.get("event") in {"start", "finish"} and event not in final_events:
            continue
        if event.get("event") == "start":
            _require(event.get("status") == "ok", f"{node_id} has non-ok status {event.get('status')!r}")
            if is_runtime_lifecycle_node(node_id):
                continue
            start_by_node[str(node_id)] = event
        elif event.get("event") == "finish":
            _require(event.get("status") == "ok", f"{node_id} has non-ok status {event.get('status')!r}")
            if is_runtime_lifecycle_node(node_id):
                continue
            finish_by_node[str(node_id)] = event
        elif event.get("event") == "replay":
            _require(node_id == "runtime.graph_replay", f"replay event has wrong node_id {node_id!r}")
            _require(event.get("status") == "retry", f"replay event has wrong status {event.get('status')!r}")
            counters = _event_counters(event)
            _require(counters.get("attempt") is not None, "replay event is missing attempt counter")
            _require(int(counters["attempt"]) > 0, "replay attempt counter must be positive")
        elif event.get("event") == "recover":
            _require(node_id == "runtime.checkpoint_recovery", f"recovery event has wrong node_id {node_id!r}")
            _require(event.get("status") == "ok", f"recovery event has wrong status {event.get('status')!r}")
            counters = _event_counters(event)
            _require(int(counters.get("attempt", 0)) > 0, "recovery event is missing positive attempt")
            _require(int(counters.get("step", 0)) > 0, "recovery event is missing positive step")
            _require(counters.get("path") is not None, "recovery event is missing checkpoint path")
            _require(int(counters.get("skipped_nodes", 0)) == len(recovered_nodes), "recovery skipped_nodes mismatch")
            if counters.get("restored_weight_files") is not None:
                restored_files = int(counters["restored_weight_files"])
                _require(restored_files >= 0, "recovery restored_weight_files is negative")
                if counters.get("checkpoint_weight_files") is not None:
                    checkpoint_files = int(counters["checkpoint_weight_files"])
                    _require(checkpoint_files >= 0, "recovery checkpoint_weight_files is negative")
                    _require(
                        restored_files == checkpoint_files,
                        "recovery restored_weight_files does not match checkpoint_weight_files",
                    )
            if counters.get("restored_weight_bytes") is not None:
                restored_bytes = int(counters["restored_weight_bytes"])
                _require(restored_bytes >= 0, "recovery restored_weight_bytes is negative")
                if counters.get("checkpoint_weight_bytes") is not None:
                    checkpoint_bytes = int(counters["checkpoint_weight_bytes"])
                    _require(checkpoint_bytes >= 0, "recovery checkpoint_weight_bytes is negative")
                    _require(
                        restored_bytes == checkpoint_bytes,
                        "recovery restored_weight_bytes does not match checkpoint_weight_bytes",
                    )
            for counter_name in (
                "checkpoint_weight_data_files",
                "checkpoint_weight_metadata_files",
                "checkpoint_optimizer_state_rank_files",
                "checkpoint_portable_train_versions",
                "checkpoint_portable_train_rank_files",
            ):
                if counters.get(counter_name) is not None:
                    _require(int(counters[counter_name]) >= 0, f"recovery {counter_name} is negative")
            if (
                counters.get("checkpoint_portable_train_rank_files") is not None
                and counters.get("checkpoint_optimizer_state_rank_files") is not None
            ):
                portable_rank_files = int(counters["checkpoint_portable_train_rank_files"])
                optimizer_rank_files = int(counters["checkpoint_optimizer_state_rank_files"])
                _require(
                    portable_rank_files <= optimizer_rank_files,
                    "recovery checkpoint_portable_train_rank_files exceeds checkpoint_optimizer_state_rank_files",
                )
        else:
            _fail(f"{node_id} has unknown event type {event.get('event')!r}")

    intervals: dict[str, tuple[float, float, tuple[str, ...]]] = {}
    batch_by_node: dict[str, frozenset[str]] = {}
    weight_checks: dict[str, int] = {}
    split_checks: dict[str, dict[str, int]] = {}
    generation_checks: dict[str, dict[str, int | str | bool]] = {}
    semantic_transfer_checks: dict[str, dict[str, int]] = {}
    transfer_checks = 0
    for node_id, node in graph.nodes.items():
        if node_id in recovered_nodes:
            continue
        start = start_by_node[node_id]
        finish = finish_by_node[node_id]
        start_ts = float(start["timestamp"])
        finish_ts = float(finish["timestamp"])
        _require(finish_ts >= start_ts, f"{node_id} finish timestamp precedes start timestamp")
        expected_step = graph_node_step(node)
        for event in (start, finish):
            if expected_step is not None and event.get("step") is not None:
                _require(int(event["step"]) == expected_step, f"{node_id} has wrong logical step")
        resources = tuple(_event_counters(start).get("resources") or node.resource_keys)
        intervals[node_id] = (start_ts, finish_ts, tuple(str(resource) for resource in resources))
        scheduled_batch = _event_counters(start).get("batch_scheduled_nodes")
        if isinstance(scheduled_batch, (list, tuple)) and len(scheduled_batch) > 1:
            batch_members = frozenset(str(member) for member in scheduled_batch)
            _require(node_id in batch_members, f"{node_id} start batch_scheduled_nodes does not include itself")
            missing_members = sorted(batch_members - set(graph.nodes))
            _require(not missing_members, f"{node_id} batch_scheduled_nodes references unknown nodes {missing_members}")
            for member in batch_members:
                member_start = start_by_node.get(member)
                if member_start is None:
                    continue
                member_batch = _event_counters(member_start).get("batch_scheduled_nodes")
                if isinstance(member_batch, (list, tuple)) and len(member_batch) > 1:
                    _require(
                        frozenset(str(item) for item in member_batch) == batch_members,
                        f"{node_id} batch_scheduled_nodes disagrees with {member}",
                    )
            batch_by_node[node_id] = batch_members

        counters = _event_counters(finish)
        if node.kind in {ExecNodeType.COMM, ExecNodeType.RESHARD}:
            _require(
                int(counters.get("payload_in_bytes", node.payload_in_bytes)) == node.payload_in_bytes,
                f"{node_id} payload counter does not match lowered payload",
            )
            transfer_checks += 1
        node_split_checks = _verify_split_counters(compiled, node, finish)
        if node_split_checks:
            split_checks[node_id] = node_split_checks
        node_generation_checks = _verify_generation_counters(compiled, node, counters)
        if node_generation_checks:
            generation_checks[node_id] = node_generation_checks
        node_transfer_checks = {
            **_verify_activation_transfer(node, counters),
            **_verify_weight_transfer(node, counters),
        }
        if node_transfer_checks:
            semantic_transfer_checks[node_id] = node_transfer_checks
        for key, value in _verify_weight_counters(compiled, node, finish).items():
            weight_checks[f"{node_id}:{key}"] = value

    # Natural quantile cardinalities are dynamic, but every consumer of a rollout phase must observe
    # exactly the cardinality emitted by generation.
    quantile_counts: dict[tuple[int, int], dict[str, int]] = defaultdict(dict)
    for node_id, checks in split_checks.items():
        node = graph.nodes[node_id]
        part = _split_part_index(node_id)
        step = graph_node_step(node)
        if part is None or step is None:
            continue
        quantile_counts[(int(step), int(part))][node.vertex_id] = int(checks["partial_n_seqs"])
    generated_by_step: dict[int, int] = defaultdict(int)
    for (step, part), counts in quantile_counts.items():
        generation = {
            vertex: count
            for vertex, count in counts.items()
            if vertex in {"g_pi", "g_ac"}
        }
        if not generation:
            continue
        generation_count = next(iter(generation.values()))
        for vertex, count in counts.items():
            _require(
                count == generation_count,
                f"step {step} quantile part {part} cardinality mismatch: "
                f"generation={generation_count}, {vertex}={count}",
            )
        generated_by_step[step] += generation_count
    expected_rollouts = (
        int(compiled.run_config.workload.batch_size)
        * int(compiled.run_config.workload.samples_per_prompt)
    )
    for step, count in generated_by_step.items():
        _require(
            count == expected_rollouts,
            f"step {step} natural quantile phases emitted {count} sequences, "
            f"expected complete rollout batch {expected_rollouts}",
        )

    dependency_checks = 0
    recovered_dependency_edges = 0
    for src, dst in graph.edges:
        if src in recovered_nodes or dst in recovered_nodes:
            recovered_dependency_edges += 1
            continue
        _require(
            intervals[src][1] <= intervals[dst][0] + 1e-9,
            f"dependency timing violation: {dst} started before {src} finished",
        )
        dependency_checks += 1

    by_resource: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for node_id, (start_ts, finish_ts, resources) in intervals.items():
        for resource in resources:
            by_resource[resource].append((start_ts, finish_ts, node_id))
    for resource, spans in by_resource.items():
        spans.sort()
        for previous, current in zip(spans, spans[1:]):
            if (
                batch_by_node.get(previous[2]) is not None
                and batch_by_node.get(previous[2]) == batch_by_node.get(current[2])
            ):
                continue
            _require(
                previous[1] <= current[0] + 1e-9,
                f"resource overlap on {resource}: {previous[2]} overlaps {current[2]}",
            )

    return {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "events": len(events),
        "lifecycle_nodes_checked": sum(lifecycle_starts.values()),
        "replay_attempts": len(replays),
        "recovery_events": len(recoveries),
        "recovered_step": recovered_step,
        "recovered_nodes": len(recovered_nodes),
        "dependency_edges_checked": dependency_checks,
        "recovered_dependency_edges": recovered_dependency_edges,
        "resource_keys_checked": len(by_resource),
        "resource_batch_nodes_checked": len(batch_by_node),
        "transfer_nodes_checked": transfer_checks,
        "semantic_transfer_checks": semantic_transfer_checks,
        "split_node_checks": split_checks,
        "generation_checks": generation_checks,
        "weight_version_checks": weight_checks,
    }
