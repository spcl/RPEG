# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import json
import math
import re

from html import escape
from pathlib import Path
from typing import Any


def load_trace_events(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def compare_artifact_numeric_fingerprints(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-5,
) -> dict[str, Any]:
    """
    Compare run-level artifact moments independently of serialization.
    """
    left_items = left.get("artifact_numeric_fingerprints") or {}
    right_items = right.get("artifact_numeric_fingerprints") or {}
    if not isinstance(left_items, dict) or not isinstance(right_items, dict):
        raise ValueError("artifact_numeric_fingerprints must be mappings")
    comparisons: dict[str, Any] = {}
    mismatch_count = 0
    missing_count = 0
    max_absolute_difference = 0.0
    max_relative_difference = 0.0
    for key in sorted(set(left_items) | set(right_items)):
        left_fingerprint = left_items.get(key)
        right_fingerprint = right_items.get(key)
        if not isinstance(left_fingerprint, dict) or not isinstance(right_fingerprint, dict):
            missing_count += 1
            comparisons[key] = {
                "status": "missing",
                "left_present": isinstance(left_fingerprint, dict),
                "right_present": isinstance(right_fingerprint, dict),
            }
            continue
        left_tensors = left_fingerprint.get("tensors") or {}
        right_tensors = right_fingerprint.get("tensors") or {}
        tensor_comparisons: dict[str, Any] = {}
        artifact_matches = True
        for path in sorted(set(left_tensors) | set(right_tensors)):
            left_tensor = left_tensors.get(path)
            right_tensor = right_tensors.get(path)
            if not isinstance(left_tensor, dict) or not isinstance(right_tensor, dict):
                artifact_matches = False
                tensor_comparisons[path] = {
                    "status": "missing",
                    "left_present": isinstance(left_tensor, dict),
                    "right_present": isinstance(right_tensor, dict),
                }
                continue
            structural_fields = (
                "dtypes",
                "numel",
                "finite_count",
                "nan_count",
                "posinf_count",
                "neginf_count",
            )
            structural_matches = all(
                left_tensor.get(field) == right_tensor.get(field)
                for field in structural_fields
            )
            dtypes = tuple(str(dtype) for dtype in left_tensor.get("dtypes") or ())
            exact_numeric = bool(dtypes) and all(
                re.fullmatch(r"torch\.(?:u?int\d+|bool)", dtype) is not None
                for dtype in dtypes
            )
            numeric: dict[str, Any] = {}
            numeric_matches = True
            for field in ("sum", "sum_abs", "sum_squares", "min", "max"):
                left_value = left_tensor.get(field)
                right_value = right_tensor.get(field)
                if left_value is None or right_value is None:
                    close = left_value is None and right_value is None
                    absolute_difference = None
                    relative_difference = None
                else:
                    left_number = float(left_value)
                    right_number = float(right_value)
                    absolute_difference = abs(left_number - right_number)
                    scale = max(abs(left_number), abs(right_number))
                    relative_difference = absolute_difference / scale if scale > 0.0 else 0.0
                    close = (
                        absolute_difference == 0.0
                        if exact_numeric
                        else absolute_difference <= float(atol) + float(rtol) * scale
                    )
                    max_absolute_difference = max(max_absolute_difference, absolute_difference)
                    max_relative_difference = max(max_relative_difference, relative_difference)
                numeric_matches = numeric_matches and close
                numeric[field] = {
                    "left": left_value,
                    "right": right_value,
                    "absolute_difference": absolute_difference,
                    "relative_difference": relative_difference,
                    "close": close,
                }
            matches = structural_matches and numeric_matches
            artifact_matches = artifact_matches and matches
            tensor_comparisons[path] = {
                "status": "match" if matches else "mismatch",
                "structural_match": structural_matches,
                "exact_numeric": exact_numeric,
                "numeric": numeric,
            }
        if not artifact_matches:
            mismatch_count += 1
        comparisons[key] = {
            "status": "match" if artifact_matches else "mismatch",
            "tensors": tensor_comparisons,
        }
    return {
        "schema_version": "rpeg.artifact_numeric_comparison.v1",
        "status": "match" if mismatch_count == 0 and missing_count == 0 else "mismatch",
        "atol": float(atol),
        "rtol": float(rtol),
        "artifact_count": len(comparisons),
        "mismatch_count": mismatch_count,
        "missing_count": missing_count,
        "max_absolute_difference": max_absolute_difference,
        "max_relative_difference": max_relative_difference,
        "artifacts": comparisons,
    }


def summarize_trace(events: list[dict[str, Any]], *, run_id: str | None = None, trace_path: str | None = None) -> dict[str, Any]:
    starts = [event for event in events if event.get("event") == "start"]
    finishes = [event for event in events if event.get("event") == "finish"]
    replays = [event for event in events if event.get("event") == "replay"]
    status_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    failed_nodes: list[dict[str, Any]] = []
    total_duration_s = 0.0
    transfer_bytes = 0
    transfer_count = 0
    transfer_rank_duration_s = 0.0
    transfer_dp_repartition_count = 0
    transfer_transport_counts: dict[str, int] = {}
    transfer_wire_format_counts: dict[str, int] = {}
    transfer_transport_bytes: dict[str, int] = {}
    transfer_transport_rank_duration_s: dict[str, float] = {}
    semantic_transfer_ucx_direct_cached_endpoint_count = 0
    weight_transfer_bytes = 0
    weight_transfer_network_bytes = 0
    weight_transfer_local_ipc_bytes = 0
    weight_transfer_local_endpoint_count = 0
    weight_transfer_count = 0
    weight_transfer_rank_duration_s = 0.0
    weight_transfer_transport_counts: dict[str, int] = {}
    weight_transfer_transport_bytes: dict[str, int] = {}
    weight_transfer_transport_rank_duration_s: dict[str, float] = {}
    weight_transfer_ucx_direct_cached_endpoint_count = 0
    weight_transfer_installed_active_rank_count = 0
    weight_transfer_adopted_parameter_bytes = 0
    weight_transfer_discarded_released_state_rank_count = 0
    weight_transfer_discarded_released_state_bytes = 0
    weight_transfer_destination_dp_relay_count = 0
    weight_transfer_relay_direct_bytes = 0
    weight_transfer_relay_direct_network_bytes = 0
    weight_transfer_relay_broadcast_bytes = 0
    layout_switch_count = 0
    layout_switch_bytes = 0
    layout_switch_wall_duration_s = 0.0
    layout_switch_transport_counts: dict[str, int] = {}
    layout_switch_tensor_native_count = 0
    layout_switch_optimizer_persistent_count = 0
    layout_switch_optimizer_rebound_count = 0
    operational_weight_materialize_count = 0
    operational_weight_materialized_bytes = 0
    operational_weight_release_count = 0
    operational_weight_released_bytes = 0
    operational_weight_optimizer_persistent_release_count = 0
    checkpoint_recovery_count = 0
    checkpoint_weight_files = 0
    checkpoint_weight_bytes = 0
    checkpoint_weight_data_files = 0
    checkpoint_weight_metadata_files = 0
    checkpoint_optimizer_state_rank_files = 0
    checkpoint_portable_train_versions = 0
    checkpoint_portable_train_rank_files = 0
    checkpoint_restored_weight_files = 0
    checkpoint_restored_weight_bytes = 0
    zero2_overlap_train_count = 0
    zero2_reduce_calls = 0
    zero2_reduced_bytes = 0
    fusion_sync_backend_counts: dict[str, int] = {}
    fused_pipeline_optimizer_counts: dict[str, int] = {}
    fused_pipeline_optimizer_candidates_max = 0
    fused_pipeline_makespan_units_min: float | None = None
    fused_pipeline_baseline_makespan_units_min: float | None = None
    fusion_cuda_ipc_event_exports = 0
    fusion_cuda_ipc_event_imports = 0
    fusion_cuda_ipc_event_waits = 0
    fusion_cuda_ipc_event_records = 0
    fusion_cpu_handle_polls = 0
    fusion_cpu_completion_polls = 0
    fusion_pipeline_async_sends = 0
    fusion_pipeline_async_send_waits = 0
    fusion_pipeline_async_recvs = 0
    fusion_pipeline_async_recv_waits = 0
    generation_count = 0
    generation_backend_counts: dict[str, int] = {}
    kv_cache_generation_count = 0
    generation_tokens = 0
    generation_prompt_tokens = 0
    generation_sequences = 0
    generation_node_duration_s = 0.0
    generation_prefill_forward_steps = 0
    generation_decode_forward_steps = 0
    generation_forward_steps = 0
    generation_old_logps_from_generation_count = 0
    generation_old_logp_forward_steps = 0
    dynamic_generation_batch_count = 0
    dynamic_generation_batched_request_count = 0
    dynamic_generation_batched_sequence_count = 0
    dynamic_generation_batched_token_count = 0
    generation_quantile_phase_count = 0
    generation_quantile_planned_sequences = 0
    generation_quantile_actual_sequences = 0
    generation_quantile_natural_volume_mismatch_count = 0
    generation_quantile_empty_phase_count = 0
    generation_quantile_compacted_sequences = 0
    generation_quantile_forced_completions = 0
    kv_cache_pool_entries_by_worker: dict[str, int] = {}
    kv_cache_pool_hits_tracker = _MonotonicCounterTracker()
    kv_cache_pool_misses_tracker = _MonotonicCounterTracker()
    kv_cache_pool_evictions_tracker = _MonotonicCounterTracker()
    kv_cache_max_batch_size = 0
    kv_cache_max_sequence_length = 0
    latest_losses: dict[str, float] = {}
    loss_observation_count = 0
    nonfinite_loss_count = 0
    nonfinite_losses: list[str] = []
    loaded_weight_version_stats: dict[str, dict[str, Any]] = {}
    produced_weight_version_stats: dict[str, dict[str, Any]] = {}
    cuda_peak_memory_allocated_bytes = 0
    cuda_peak_memory_reserved_bytes = 0
    activation_cuda_cache_enabled_observation_count = 0
    activation_cuda_cache_bytes_current_max = 0
    activation_cuda_cache_bytes_limit_max = 0
    activation_cuda_cache_placements_max = 0
    activation_cuda_cache_cpu_fallbacks_max = 0
    megatron_process_groups_requested_max = 0
    megatron_process_groups_created_max = 0
    megatron_process_groups_reused_max = 0
    megatron_process_group_observation_count = 0
    megatron_pipeline_cta_policy_observation_count = 0
    megatron_pipeline_cta_policy_zero_observation_count = 0
    megatron_pipeline_cta_policy_max = 0
    megatron_pipeline_groups_policy_applied_max = 0
    torch_direct_process_group_cache_size_max = 0
    training_inference_cache_transition_rank_count = 0
    training_rotary_lru_entries_cleared = 0
    training_model_rotary_entries_cleared = 0
    training_static_kv_pool_entries_retained = 0
    artifact_numeric_fingerprints: dict[str, dict[str, Any]] = {}

    for event in events:
        if event.get("event") != "recover" or event.get("node_id") != "runtime.checkpoint_recovery":
            continue
        checkpoint_recovery_count += 1
        counters = dict(event.get("counters") or {})
        for memory_counters in (counters, *_rank_counter_dicts(counters)):
            cuda_peak_memory_allocated_bytes = max(
                cuda_peak_memory_allocated_bytes,
                _int_value(memory_counters.get("cuda_peak_memory_allocated_bytes")),
            )
            cuda_peak_memory_reserved_bytes = max(
                cuda_peak_memory_reserved_bytes,
                _int_value(memory_counters.get("cuda_peak_memory_reserved_bytes")),
            )
            if _bool_counter(memory_counters.get("activation_cuda_cache_enabled")):
                activation_cuda_cache_enabled_observation_count += 1
            activation_cuda_cache_bytes_current_max = max(
                activation_cuda_cache_bytes_current_max,
                _int_value(memory_counters.get("activation_cuda_cache_bytes_current")),
            )
            activation_cuda_cache_bytes_limit_max = max(
                activation_cuda_cache_bytes_limit_max,
                _int_value(memory_counters.get("activation_cuda_cache_bytes_limit")),
            )
            activation_cuda_cache_placements_max = max(
                activation_cuda_cache_placements_max,
                _int_value(memory_counters.get("activation_cuda_cache_placements")),
            )
            activation_cuda_cache_cpu_fallbacks_max = max(
                activation_cuda_cache_cpu_fallbacks_max,
                _int_value(memory_counters.get("activation_cuda_cache_cpu_fallbacks")),
            )
        checkpoint_weight_files += _int_counter(counters, "checkpoint_weight_files")
        checkpoint_weight_bytes += _int_counter(counters, "checkpoint_weight_bytes")
        checkpoint_weight_data_files += _int_counter(counters, "checkpoint_weight_data_files")
        checkpoint_weight_metadata_files += _int_counter(counters, "checkpoint_weight_metadata_files")
        checkpoint_optimizer_state_rank_files += _int_counter(counters, "checkpoint_optimizer_state_rank_files")
        checkpoint_portable_train_versions += _int_counter(counters, "checkpoint_portable_train_versions")
        checkpoint_portable_train_rank_files += _int_counter(counters, "checkpoint_portable_train_rank_files")
        checkpoint_restored_weight_files += _int_counter(counters, "restored_weight_files")
        checkpoint_restored_weight_bytes += _int_counter(counters, "restored_weight_bytes")

    for event in finishes:
        status = str(event.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        counters = dict(event.get("counters") or {})
        for memory_counters in (counters, *_rank_counter_dicts(counters)):
            cuda_peak_memory_allocated_bytes = max(
                cuda_peak_memory_allocated_bytes,
                _int_value(memory_counters.get("cuda_peak_memory_allocated_bytes")),
            )
            cuda_peak_memory_reserved_bytes = max(
                cuda_peak_memory_reserved_bytes,
                _int_value(memory_counters.get("cuda_peak_memory_reserved_bytes")),
            )
            if _bool_counter(memory_counters.get("activation_cuda_cache_enabled")):
                activation_cuda_cache_enabled_observation_count += 1
            activation_cuda_cache_bytes_current_max = max(
                activation_cuda_cache_bytes_current_max,
                _int_value(memory_counters.get("activation_cuda_cache_bytes_current")),
            )
            activation_cuda_cache_bytes_limit_max = max(
                activation_cuda_cache_bytes_limit_max,
                _int_value(memory_counters.get("activation_cuda_cache_bytes_limit")),
            )
            activation_cuda_cache_placements_max = max(
                activation_cuda_cache_placements_max,
                _int_value(memory_counters.get("activation_cuda_cache_placements")),
            )
            activation_cuda_cache_cpu_fallbacks_max = max(
                activation_cuda_cache_cpu_fallbacks_max,
                _int_value(memory_counters.get("activation_cuda_cache_cpu_fallbacks")),
            )
        for group_counters in (counters, *_rank_counter_dicts(counters)):
            torch_direct_process_group_cache_size_max = max(
                torch_direct_process_group_cache_size_max,
                _int_value(group_counters.get("torch_direct_process_group_cache_size")),
            )
            group_keys = (
                "megatron_process_groups_requested",
                "megatron_process_groups_created",
                "megatron_process_groups_reused",
            )
            if not any(key in group_counters for key in group_keys):
                continue
            megatron_process_group_observation_count += 1
            megatron_process_groups_requested_max = max(
                megatron_process_groups_requested_max,
                _int_value(group_counters.get("megatron_process_groups_requested")),
            )
            megatron_process_groups_created_max = max(
                megatron_process_groups_created_max,
                _int_value(group_counters.get("megatron_process_groups_created")),
            )
            megatron_process_groups_reused_max = max(
                megatron_process_groups_reused_max,
                _int_value(group_counters.get("megatron_process_groups_reused")),
            )
            if "megatron_process_groups_pipeline_cta_policy" in group_counters:
                megatron_pipeline_cta_policy_observation_count += 1
                megatron_pipeline_cta_policy_max = max(
                    megatron_pipeline_cta_policy_max,
                    _int_value(
                        group_counters.get(
                            "megatron_process_groups_pipeline_cta_policy"
                        )
                    ),
                )
                if _bool_counter(
                    group_counters.get(
                        "megatron_process_groups_pipeline_cta_policy_zero"
                    )
                ):
                    megatron_pipeline_cta_policy_zero_observation_count += 1
                megatron_pipeline_groups_policy_applied_max = max(
                    megatron_pipeline_groups_policy_applied_max,
                    _int_value(
                        group_counters.get(
                            "megatron_process_groups_pipeline_groups_policy_applied"
                        )
                    ),
                )
        kind = counters.get("kind")
        if kind is not None:
            kind_counts[str(kind)] = kind_counts.get(str(kind), 0) + 1
        if "duration_s" in counters:
            total_duration_s += float(counters["duration_s"])
        if status != "ok":
            failed_nodes.append({
                "node_id": event.get("node_id"),
                "worker_id": event.get("worker_id"),
                "message": event.get("message"),
            })
        event_losses: dict[str, list[float]] = {}
        for key, value, rank in _reported_losses(counters):
            loss_observation_count += 1
            location = f"{event.get('node_id')}.{key}"
            try:
                loss = float(value)
            except (TypeError, ValueError):
                loss = math.nan
            if not math.isfinite(loss):
                nonfinite_loss_count += 1
                rank_suffix = "" if rank is None else f".rank{rank}"
                nonfinite_losses.append(location + rank_suffix)
                continue
            event_losses.setdefault(key, []).append(loss)
        for key, values in event_losses.items():
            latest_losses[f"{event.get('node_id')}.{key}"] = sum(values) / len(values)
        _observe_loaded_weight_version(
            loaded_weight_version_stats,
            counters,
            step=_event_step(event),
        )
        _observe_produced_weight_version(produced_weight_version_stats, counters)
        _observe_artifact_numeric_fingerprints(
            artifact_numeric_fingerprints,
            node_id=str(event.get("node_id") or "unknown"),
            counters=counters,
        )
        operational_weights = counters.get("operational_weights")
        if isinstance(operational_weights, dict):
            materialized_ranks = int(operational_weights.get("materialized_rank_count") or 0)
            released_ranks = int(operational_weights.get("released_rank_count") or 0)
            if materialized_ranks > 0:
                operational_weight_materialize_count += 1
            operational_weight_materialized_bytes += int(
                operational_weights.get("materialized_bytes") or 0
            )
            if released_ranks > 0:
                operational_weight_release_count += 1
            operational_weight_released_bytes += int(
                operational_weights.get("released_bytes") or 0
            )
            if int(operational_weights.get("optimizer_persistent_rank_count") or 0) > 0:
                operational_weight_optimizer_persistent_release_count += 1
        if _bool_counter(_counter_value(counters, "zero2_overlap")):
            zero2_overlap_train_count += 1
            # Pipeline stages can have different parameter counts, so their bucket counts and valid
            # (unpadded) bytes legitimately differ. Report the busiest rank for this logical train
            # operation rather than dropping the whole counter when rank values are unequal.
            zero2_reduce_calls += _max_rank_counter(counters, "zero2_reduce_calls")
            zero2_reduced_bytes += _max_rank_counter(counters, "zero2_reduced_bytes")
        training_inference_cache_transition_rank_count += _sum_rank_counter(
            counters,
            "training_inference_cache_transition_count",
        )
        training_rotary_lru_entries_cleared += _sum_rank_counter(
            counters,
            "training_rotary_lru_entries_cleared",
        )
        training_model_rotary_entries_cleared += _sum_rank_counter(
            counters,
            "training_model_rotary_entries_cleared",
        )
        training_static_kv_pool_entries_retained += _sum_rank_counter(
            counters,
            "training_static_kv_pool_entries_retained",
        )
        fusion_sync_backend = _counter_value(counters, "fusion_sync_backend")
        if fusion_sync_backend is not None:
            backend = str(fusion_sync_backend)
            fusion_sync_backend_counts[backend] = fusion_sync_backend_counts.get(backend, 0) + 1
            fusion_cuda_ipc_event_exports += _sum_rank_counter(
                counters, "fusion_cuda_ipc_event_exports"
            )
            fusion_cuda_ipc_event_imports += _sum_rank_counter(
                counters, "fusion_cuda_ipc_event_imports"
            )
            fusion_cuda_ipc_event_waits += _sum_rank_counter(
                counters, "fusion_cuda_ipc_event_waits"
            )
            fusion_cuda_ipc_event_records += _sum_rank_counter(
                counters, "fusion_cuda_ipc_event_records"
            )
            fusion_cpu_handle_polls += _sum_rank_counter(counters, "fusion_cpu_handle_polls")
            fusion_cpu_completion_polls += _sum_rank_counter(
                counters, "fusion_cpu_completion_polls"
            )
            fusion_pipeline_async_sends += _sum_rank_counter(
                counters, "fusion_pipeline_async_sends"
            )
            fusion_pipeline_async_send_waits += _sum_rank_counter(
                counters, "fusion_pipeline_async_send_waits"
            )
            fusion_pipeline_async_recvs += _sum_rank_counter(
                counters, "fusion_pipeline_async_recvs"
            )
            fusion_pipeline_async_recv_waits += _sum_rank_counter(
                counters, "fusion_pipeline_async_recv_waits"
            )
        fused_optimizer = _counter_value(counters, "fused_pipeline_optimizer")
        if fused_optimizer is not None:
            optimizer_name = str(fused_optimizer)
            fused_pipeline_optimizer_counts[optimizer_name] = (
                fused_pipeline_optimizer_counts.get(optimizer_name, 0) + 1
            )
            raw_candidates = _counter_value(
                counters,
                "fused_pipeline_optimizer_candidates",
            )
            if raw_candidates is not None:
                fused_pipeline_optimizer_candidates_max = max(
                    fused_pipeline_optimizer_candidates_max,
                    int(raw_candidates),
                )
            raw_makespan = _counter_value(counters, "fused_pipeline_makespan_units")
            if raw_makespan is not None:
                makespan = float(raw_makespan)
                fused_pipeline_makespan_units_min = (
                    makespan
                    if fused_pipeline_makespan_units_min is None
                    else min(fused_pipeline_makespan_units_min, makespan)
                )
            raw_baseline = _counter_value(
                counters,
                "fused_pipeline_baseline_makespan_units",
            )
            if raw_baseline is not None:
                baseline = float(raw_baseline)
                fused_pipeline_baseline_makespan_units_min = (
                    baseline
                    if fused_pipeline_baseline_makespan_units_min is None
                    else min(fused_pipeline_baseline_makespan_units_min, baseline)
                )
        for transfer in _semantic_transfer_summaries(counters):
            if transfer.get("bytes") is not None:
                transfer_count += 1
                bytes_value = int(transfer["bytes"])
                transfer_bytes += bytes_value
                transport = str(transfer.get("transport") or "unknown")
                transfer_transport_counts[transport] = transfer_transport_counts.get(transport, 0) + 1
                transfer_transport_bytes[transport] = transfer_transport_bytes.get(transport, 0) + bytes_value
                wire_format = str(transfer.get("wire_format") or "unknown")
                transfer_wire_format_counts[wire_format] = (
                    transfer_wire_format_counts.get(wire_format, 0) + 1
                )
                if _bool_counter(transfer.get("dp_repartitioned")):
                    transfer_dp_repartition_count += 1
                semantic_transfer_ucx_direct_cached_endpoint_count += _transfer_cached_endpoint_count(transfer)
                duration_value = transfer.get("transfer_rank_duration_s")
                if duration_value is not None:
                    duration = float(duration_value)
                    transfer_rank_duration_s += duration
                    transfer_transport_rank_duration_s[transport] = (
                        transfer_transport_rank_duration_s.get(transport, 0.0) + duration
                    )
        for transfer in _weight_transfer_summaries(counters):
            if transfer.get("bytes") is not None:
                weight_transfer_count += 1
                bytes_value = int(transfer["bytes"])
                weight_transfer_bytes += bytes_value
                weight_transfer_network_bytes += int(transfer.get("network_bytes") or 0)
                weight_transfer_local_ipc_bytes += int(transfer.get("local_ipc_bytes") or 0)
                weight_transfer_local_endpoint_count += int(
                    transfer.get("local_endpoint_count") or 0
                )
                transport = str(transfer.get("transport") or "unknown")
                weight_transfer_transport_counts[transport] = weight_transfer_transport_counts.get(transport, 0) + 1
                weight_transfer_transport_bytes[transport] = (
                    weight_transfer_transport_bytes.get(transport, 0) + bytes_value
                )
                weight_transfer_ucx_direct_cached_endpoint_count += _transfer_cached_endpoint_count(transfer)
                rank_counters = _rank_counter_dicts(transfer)
                weight_transfer_installed_active_rank_count += sum(
                    _bool_counter(rank.get("installed_active_weight_version"))
                    for rank in rank_counters
                )
                weight_transfer_adopted_parameter_bytes += sum(
                    _int_value(rank.get("adopted_parameter_bytes"))
                    for rank in rank_counters
                )
                weight_transfer_discarded_released_state_rank_count += sum(
                    _bool_counter(rank.get("discarded_released_weight_state"))
                    for rank in rank_counters
                )
                weight_transfer_discarded_released_state_bytes += sum(
                    _int_value(rank.get("discarded_released_weight_state_bytes"))
                    for rank in rank_counters
                )
                if _bool_counter(transfer.get("destination_dp_relay")):
                    weight_transfer_destination_dp_relay_count += 1
                    relay_direct_bytes = _int_value(
                        transfer.get("relay_direct_received_bytes")
                    )
                    weight_transfer_relay_direct_bytes += relay_direct_bytes
                    weight_transfer_relay_direct_network_bytes += max(
                        0,
                        relay_direct_bytes - int(transfer.get("local_ipc_bytes") or 0),
                    )
                    weight_transfer_relay_broadcast_bytes += _int_value(
                        transfer.get("relay_broadcast_bytes")
                    )
                duration = _transfer_rank_duration(transfer)
                if duration is not None:
                    weight_transfer_rank_duration_s += duration
                    weight_transfer_transport_rank_duration_s[transport] = (
                        weight_transfer_transport_rank_duration_s.get(transport, 0.0) + duration
                    )
                if transfer.get("copy_mode") in {
                    "in_place_routed_host_reshard",
                    "in_place_routed_device_reshard",
                    "in_place_dp_to_pp_local_extract",
                    "in_place_pp_to_dp_host_staged_broadcast",
                }:
                    layout_switch_count += 1
                    layout_switch_bytes += bytes_value
                    layout_switch_wall_duration_s += float(counters.get("duration_s") or 0.0)
                    layout_switch_transport_counts[transport] = (
                        layout_switch_transport_counts.get(transport, 0) + 1
                    )
                    rank_counters = _rank_counter_dicts(transfer)
                    if rank_counters and all(
                        rank_counter.get("wire_format") == "tensor_native"
                        for rank_counter in rank_counters
                    ):
                        layout_switch_tensor_native_count += 1
                    if rank_counters and all(
                        _bool_counter(rank_counter.get("optimizer_persistent"))
                        for rank_counter in rank_counters
                    ):
                        layout_switch_optimizer_persistent_count += 1
                    if rank_counters and all(
                        _bool_counter(rank_counter.get("optimizer_rebound"))
                        for rank_counter in rank_counters
                    ):
                        layout_switch_optimizer_rebound_count += 1
        generation_backend = _counter_value(counters, "generation_backend")
        if generation_backend is not None:
            backend = str(generation_backend)
            generation_count += 1
            generation_backend_counts[backend] = generation_backend_counts.get(backend, 0) + 1
            if "duration_s" in counters:
                generation_node_duration_s += float(counters["duration_s"])
            if _bool_counter(_counter_value(counters, "kv_cache_enabled")):
                kv_cache_generation_count += 1
            tokens = _generation_int_counter(counters, "generated_tokens")
            if tokens is None:
                tokens = _generation_expected_tokens(counters)
            if tokens is not None:
                generation_tokens += int(tokens)
            prompt_tokens = _generation_int_counter(counters, "prompt_tokens")
            if prompt_tokens is not None:
                generation_prompt_tokens += int(prompt_tokens)
            sequences = _generation_int_counter(counters, "batch_size")
            if sequences is not None:
                generation_sequences += int(sequences)
            artifact_part = _counter_value(counters, "artifact_part")
            if artifact_part is not None:
                generation_quantile_phase_count += 1
                planned = _int_value(_counter_value(counters, "planned_n_seqs"))
                actual = _int_value(_counter_value(counters, "partial_n_seqs"))
                generation_quantile_planned_sequences += planned
                generation_quantile_actual_sequences += actual
                generation_quantile_natural_volume_mismatch_count += int(planned != actual)
                generation_quantile_empty_phase_count += int(actual == 0)
                generation_quantile_compacted_sequences += _int_value(
                    _counter_value(counters, "phase_compacted_sequences")
                )
                generation_quantile_forced_completions += _int_value(
                    _counter_value(counters, "phase_forced_completions")
                )
            generation_prefill_forward_steps += _int_value(_counter_value(counters, "prefill_forward_steps"))
            generation_decode_forward_steps += _int_value(_counter_value(counters, "decode_forward_steps"))
            generation_forward_steps += _int_value(_counter_value(counters, "generation_forward_steps"))
            if _bool_counter(_counter_value(counters, "old_logps_from_generation")):
                generation_old_logps_from_generation_count += 1
            generation_old_logp_forward_steps += _int_value(_counter_value(counters, "old_logp_forward_steps"))
            dynamic_batch_requests = _int_value(_counter_value(counters, "dynamic_generation_batch_request_count"))
            dynamic_batch_index = _int_value(_counter_value(counters, "dynamic_generation_batch_index"))
            if dynamic_batch_requests > 1 and dynamic_batch_index == 0:
                dynamic_generation_batch_count += 1
                dynamic_generation_batched_request_count += dynamic_batch_requests
                dynamic_generation_batched_sequence_count += _int_value(
                    _counter_value(counters, "dynamic_generation_batch_size")
                )
                dynamic_generation_batched_token_count += _int_value(
                    _counter_value(counters, "dynamic_generation_batch_generated_tokens")
                )
            worker_key = str(event.get("worker_id") or event.get("node_id") or "unknown")
            kv_cache_pool_entries_by_worker[worker_key] = max(
                kv_cache_pool_entries_by_worker.get(worker_key, 0),
                _generation_replica_counter(counters, "kv_cache_pool_entries"),
            )
            kv_cache_pool_hits_tracker.observe(
                worker_key,
                _generation_replica_counter(counters, "kv_cache_pool_hits"),
            )
            kv_cache_pool_misses_tracker.observe(
                worker_key,
                _generation_replica_counter(counters, "kv_cache_pool_misses"),
            )
            kv_cache_pool_evictions_tracker.observe(
                worker_key,
                _generation_replica_counter(counters, "kv_cache_pool_evictions"),
            )
            kv_cache_max_batch_size = max(
                kv_cache_max_batch_size,
                _int_value(_counter_value(counters, "kv_cache_max_batch_size")),
            )
            kv_cache_max_sequence_length = max(
                kv_cache_max_sequence_length,
                _int_value(_counter_value(counters, "kv_cache_max_sequence_length")),
            )

    timestamps = [float(event["timestamp"]) for event in events if event.get("timestamp") is not None]
    wall_time_s = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 0.0
    generation_tokens_per_s = float(generation_tokens) / wall_time_s if wall_time_s > 0.0 else 0.0
    generation_prompt_tokens_per_s = (
        float(generation_prompt_tokens) / wall_time_s if wall_time_s > 0.0 else 0.0
    )
    generation_node_tokens_per_s = (
        float(generation_tokens) / generation_node_duration_s if generation_node_duration_s > 0.0 else 0.0
    )
    generation_prompt_node_tokens_per_s = (
        float(generation_prompt_tokens) / generation_node_duration_s
        if generation_node_duration_s > 0.0
        else 0.0
    )
    transfer_rank_bytes_per_s = (
        float(transfer_bytes) / transfer_rank_duration_s if transfer_rank_duration_s > 0.0 else 0.0
    )
    transfer_transport_rank_bytes_per_s = {
        transport: (
            float(transfer_transport_bytes.get(transport, 0)) / duration if duration > 0.0 else 0.0
        )
        for transport, duration in sorted(transfer_transport_rank_duration_s.items())
    }
    weight_transfer_rank_bytes_per_s = (
        float(weight_transfer_bytes) / weight_transfer_rank_duration_s
        if weight_transfer_rank_duration_s > 0.0
        else 0.0
    )
    weight_transfer_transport_rank_bytes_per_s = {
        transport: (
            float(weight_transfer_transport_bytes.get(transport, 0)) / duration if duration > 0.0 else 0.0
        )
        for transport, duration in sorted(weight_transfer_transport_rank_duration_s.items())
    }
    summary = {
        "schema_version": "rpeg.runtime_summary.v1",
        "run_id": run_id or (str(events[0].get("run_id")) if events else None),
        "trace_path": trace_path,
        "events": len(events),
        "started_nodes": len(starts),
        "finished_nodes": len(finishes),
        "status_counts": status_counts,
        "kind_counts": kind_counts,
        "failed_nodes": failed_nodes,
        "replay_attempts": len(replays),
        "total_node_duration_s": total_duration_s,
        "wall_time_s": wall_time_s,
        "semantic_transfer_count": transfer_count,
        "semantic_transfer_bytes": transfer_bytes,
        "semantic_transfer_rank_duration_s": transfer_rank_duration_s,
        "semantic_transfer_rank_bytes_per_s": transfer_rank_bytes_per_s,
        "semantic_transfer_dp_repartition_count": transfer_dp_repartition_count,
        "semantic_transfer_transport_counts": transfer_transport_counts,
        "semantic_transfer_wire_format_counts": transfer_wire_format_counts,
        "semantic_transfer_transport_bytes": transfer_transport_bytes,
        "semantic_transfer_transport_rank_duration_s": transfer_transport_rank_duration_s,
        "semantic_transfer_transport_rank_bytes_per_s": transfer_transport_rank_bytes_per_s,
        "semantic_transfer_ucx_direct_cached_endpoint_count": semantic_transfer_ucx_direct_cached_endpoint_count,
        "weight_transfer_count": weight_transfer_count,
        "weight_transfer_bytes": weight_transfer_bytes,
        "weight_transfer_network_bytes": weight_transfer_network_bytes,
        "weight_transfer_local_ipc_bytes": weight_transfer_local_ipc_bytes,
        "weight_transfer_local_endpoint_count": weight_transfer_local_endpoint_count,
        "weight_transfer_rank_duration_s": weight_transfer_rank_duration_s,
        "weight_transfer_rank_bytes_per_s": weight_transfer_rank_bytes_per_s,
        "weight_transfer_transport_counts": weight_transfer_transport_counts,
        "weight_transfer_transport_bytes": weight_transfer_transport_bytes,
        "weight_transfer_transport_rank_duration_s": weight_transfer_transport_rank_duration_s,
        "weight_transfer_transport_rank_bytes_per_s": weight_transfer_transport_rank_bytes_per_s,
        "weight_transfer_ucx_direct_cached_endpoint_count": weight_transfer_ucx_direct_cached_endpoint_count,
        "weight_transfer_installed_active_rank_count": weight_transfer_installed_active_rank_count,
        "weight_transfer_adopted_parameter_bytes": weight_transfer_adopted_parameter_bytes,
        "weight_transfer_discarded_released_state_rank_count": (
            weight_transfer_discarded_released_state_rank_count
        ),
        "weight_transfer_discarded_released_state_bytes": (
            weight_transfer_discarded_released_state_bytes
        ),
        "weight_transfer_destination_dp_relay_count": (
            weight_transfer_destination_dp_relay_count
        ),
        "weight_transfer_relay_direct_bytes": weight_transfer_relay_direct_bytes,
        "weight_transfer_relay_direct_network_bytes": (
            weight_transfer_relay_direct_network_bytes
        ),
        "weight_transfer_relay_broadcast_bytes": weight_transfer_relay_broadcast_bytes,
        "layout_switch_count": layout_switch_count,
        "layout_switch_bytes": layout_switch_bytes,
        "layout_switch_wall_duration_s": layout_switch_wall_duration_s,
        "layout_switch_transport_counts": layout_switch_transport_counts,
        "layout_switch_tensor_native_count": layout_switch_tensor_native_count,
        "layout_switch_optimizer_persistent_count": layout_switch_optimizer_persistent_count,
        "layout_switch_optimizer_rebound_count": layout_switch_optimizer_rebound_count,
        "operational_weight_materialize_count": operational_weight_materialize_count,
        "operational_weight_materialized_bytes": operational_weight_materialized_bytes,
        "operational_weight_release_count": operational_weight_release_count,
        "operational_weight_released_bytes": operational_weight_released_bytes,
        "operational_weight_optimizer_persistent_release_count": (
            operational_weight_optimizer_persistent_release_count
        ),
        "ucx_direct_cached_endpoint_count": (
            semantic_transfer_ucx_direct_cached_endpoint_count + weight_transfer_ucx_direct_cached_endpoint_count
        ),
        "checkpoint_recovery_count": checkpoint_recovery_count,
        "checkpoint_weight_files": checkpoint_weight_files,
        "checkpoint_weight_bytes": checkpoint_weight_bytes,
        "checkpoint_weight_data_files": checkpoint_weight_data_files,
        "checkpoint_weight_metadata_files": checkpoint_weight_metadata_files,
        "checkpoint_optimizer_state_rank_files": checkpoint_optimizer_state_rank_files,
        "checkpoint_portable_train_versions": checkpoint_portable_train_versions,
        "checkpoint_portable_train_rank_files": checkpoint_portable_train_rank_files,
        "checkpoint_restored_weight_files": checkpoint_restored_weight_files,
        "checkpoint_restored_weight_bytes": checkpoint_restored_weight_bytes,
        "zero2_overlap_train_count": zero2_overlap_train_count,
        "zero2_reduce_calls": zero2_reduce_calls,
        "zero2_reduced_bytes": zero2_reduced_bytes,
        "fusion_sync_backend_counts": fusion_sync_backend_counts,
        "fused_pipeline_optimizer_counts": fused_pipeline_optimizer_counts,
        "fused_pipeline_optimizer_candidates_max": fused_pipeline_optimizer_candidates_max,
        "fused_pipeline_makespan_units_min": fused_pipeline_makespan_units_min,
        "fused_pipeline_baseline_makespan_units_min": (
            fused_pipeline_baseline_makespan_units_min
        ),
        "fused_pipeline_predicted_improvement_fraction": (
            0.0
            if fused_pipeline_makespan_units_min is None
            or fused_pipeline_baseline_makespan_units_min is None
            or fused_pipeline_baseline_makespan_units_min <= 0.0
            else (
                fused_pipeline_baseline_makespan_units_min
                - fused_pipeline_makespan_units_min
            )
            / fused_pipeline_baseline_makespan_units_min
        ),
        "fusion_cuda_ipc_event_exports": fusion_cuda_ipc_event_exports,
        "fusion_cuda_ipc_event_imports": fusion_cuda_ipc_event_imports,
        "fusion_cuda_ipc_event_waits": fusion_cuda_ipc_event_waits,
        "fusion_cuda_ipc_event_records": fusion_cuda_ipc_event_records,
        "fusion_cpu_handle_polls": fusion_cpu_handle_polls,
        "fusion_cpu_completion_polls": fusion_cpu_completion_polls,
        "fusion_pipeline_async_sends": fusion_pipeline_async_sends,
        "fusion_pipeline_async_send_waits": fusion_pipeline_async_send_waits,
        "fusion_pipeline_async_recvs": fusion_pipeline_async_recvs,
        "fusion_pipeline_async_recv_waits": fusion_pipeline_async_recv_waits,
        "generation_count": generation_count,
        "generation_backend_counts": generation_backend_counts,
        "kv_cache_generation_count": kv_cache_generation_count,
        "generation_tokens": generation_tokens,
        "generation_response_tokens": generation_tokens,
        "generation_prompt_tokens": generation_prompt_tokens,
        "generation_sequences": generation_sequences,
        "generation_node_duration_s": generation_node_duration_s,
        "generation_tokens_per_s": generation_tokens_per_s,
        "generation_response_tokens_per_s": generation_tokens_per_s,
        "generation_prompt_tokens_per_s": generation_prompt_tokens_per_s,
        "generation_node_tokens_per_s": generation_node_tokens_per_s,
        "generation_response_node_tokens_per_s": generation_node_tokens_per_s,
        "generation_prompt_node_tokens_per_s": generation_prompt_node_tokens_per_s,
        "generation_prefill_forward_steps": generation_prefill_forward_steps,
        "generation_decode_forward_steps": generation_decode_forward_steps,
        "generation_forward_steps": generation_forward_steps,
        "generation_old_logps_from_generation_count": generation_old_logps_from_generation_count,
        "generation_old_logp_forward_steps": generation_old_logp_forward_steps,
        "dynamic_generation_batch_count": dynamic_generation_batch_count,
        "dynamic_generation_batched_request_count": dynamic_generation_batched_request_count,
        "dynamic_generation_batched_sequence_count": dynamic_generation_batched_sequence_count,
        "dynamic_generation_batched_token_count": dynamic_generation_batched_token_count,
        "generation_quantile_phase_count": generation_quantile_phase_count,
        "generation_quantile_planned_sequences": generation_quantile_planned_sequences,
        "generation_quantile_actual_sequences": generation_quantile_actual_sequences,
        "generation_quantile_natural_volume_mismatch_count": (
            generation_quantile_natural_volume_mismatch_count
        ),
        "generation_quantile_empty_phase_count": generation_quantile_empty_phase_count,
        "generation_quantile_compacted_sequences": generation_quantile_compacted_sequences,
        "generation_quantile_forced_completions": generation_quantile_forced_completions,
        "kv_cache_pool_entries": sum(kv_cache_pool_entries_by_worker.values()),
        "kv_cache_pool_hits": kv_cache_pool_hits_tracker.total,
        "kv_cache_pool_misses": kv_cache_pool_misses_tracker.total,
        "kv_cache_pool_evictions": kv_cache_pool_evictions_tracker.total,
        "kv_cache_max_batch_size": kv_cache_max_batch_size,
        "kv_cache_max_sequence_length": kv_cache_max_sequence_length,
        "latest_losses": latest_losses,
        "loss_observation_count": loss_observation_count,
        "nonfinite_loss_count": nonfinite_loss_count,
        "nonfinite_losses": nonfinite_losses,
        "cuda_peak_memory_allocated_bytes": cuda_peak_memory_allocated_bytes,
        "cuda_peak_memory_reserved_bytes": cuda_peak_memory_reserved_bytes,
        "activation_cuda_cache_enabled_observation_count": (
            activation_cuda_cache_enabled_observation_count
        ),
        "activation_cuda_cache_bytes_current_max": activation_cuda_cache_bytes_current_max,
        "activation_cuda_cache_bytes_limit_max": activation_cuda_cache_bytes_limit_max,
        "activation_cuda_cache_placements_max": activation_cuda_cache_placements_max,
        "activation_cuda_cache_cpu_fallbacks_max": activation_cuda_cache_cpu_fallbacks_max,
        "megatron_process_group_observation_count": megatron_process_group_observation_count,
        "megatron_process_groups_requested_max": megatron_process_groups_requested_max,
        "megatron_process_groups_created_max": megatron_process_groups_created_max,
        "megatron_process_groups_reused_max": megatron_process_groups_reused_max,
        "megatron_pipeline_cta_policy_observation_count": (
            megatron_pipeline_cta_policy_observation_count
        ),
        "megatron_pipeline_cta_policy_zero_observation_count": (
            megatron_pipeline_cta_policy_zero_observation_count
        ),
        "megatron_pipeline_cta_policy_max": megatron_pipeline_cta_policy_max,
        "megatron_pipeline_groups_policy_applied_max": (
            megatron_pipeline_groups_policy_applied_max
        ),
        "torch_direct_process_group_cache_size_max": (
            torch_direct_process_group_cache_size_max
        ),
        "training_inference_cache_transition_rank_count": (
            training_inference_cache_transition_rank_count
        ),
        "training_rotary_lru_entries_cleared": training_rotary_lru_entries_cleared,
        "training_model_rotary_entries_cleared": training_model_rotary_entries_cleared,
        "training_static_kv_pool_entries_retained": training_static_kv_pool_entries_retained,
        "artifact_numeric_fingerprints": artifact_numeric_fingerprints,
        "loaded_weight_version_stats": loaded_weight_version_stats,
        "produced_weight_version_stats": produced_weight_version_stats,
        "loaded_weight_count_by_model": _weight_stat_field(loaded_weight_version_stats, "count"),
        "produced_weight_count_by_model": _weight_stat_field(produced_weight_version_stats, "count"),
        "nonzero_loaded_weight_count_by_model": _weight_stat_field(loaded_weight_version_stats, "nonzero_count"),
        "lagging_loaded_weight_count_by_model": _weight_stat_field(loaded_weight_version_stats, "lagging_step_count"),
        "max_loaded_weight_version_by_model": _weight_stat_field(loaded_weight_version_stats, "max"),
        "max_produced_weight_version_by_model": _weight_stat_field(produced_weight_version_stats, "max"),
    }
    return summary


def _event_step(event: dict[str, Any]) -> int | None:
    if event.get("step") is None:
        return None
    try:
        return int(event["step"])
    except (TypeError, ValueError):
        return None


def _reported_losses(counters: dict[str, Any]):
    for key, value in counters.items():
        if key == "loss" or key.endswith("loss"):
            yield str(key), value, None
    ranks = counters.get("ranks")
    if not isinstance(ranks, list):
        return
    for rank_result in ranks:
        if not isinstance(rank_result, dict):
            continue
        rank = rank_result.get("rank")
        rank_counters = rank_result.get("counters")
        if not isinstance(rank_counters, dict):
            continue
        for key, value in rank_counters.items():
            if key == "loss" or key.endswith("loss"):
                yield str(key), value, rank


def _observe_artifact_numeric_fingerprints(
    aggregate: dict[str, dict[str, Any]],
    *,
    node_id: str,
    counters: dict[str, Any],
) -> None:
    observations = []
    direct = counters.get("artifact_numeric_fingerprint")
    if isinstance(direct, dict):
        observations.append(direct)
    for rank_counters in _rank_counter_dicts(counters):
        fingerprint = rank_counters.get("artifact_numeric_fingerprint")
        if isinstance(fingerprint, dict):
            observations.append(fingerprint)
    for fingerprint in observations:
        if fingerprint.get("schema_version") != "rpeg.artifact_numeric_fingerprint.v1":
            continue
        artifact = str(fingerprint.get("artifact") or "unknown")
        key = f"{node_id}:{artifact}"
        target = aggregate.setdefault(
            key,
            {
                "schema_version": "rpeg.artifact_numeric_fingerprint_aggregate.v1",
                "node_id": node_id,
                "artifact": artifact,
                "rank_count": 0,
                "tensors": {},
            },
        )
        target["rank_count"] = int(target["rank_count"]) + 1
        tensors = fingerprint.get("tensors")
        if not isinstance(tensors, dict):
            continue
        for path, raw in tensors.items():
            if not isinstance(raw, dict):
                continue
            tensor = target["tensors"].setdefault(
                str(path),
                {
                    "dtypes": [],
                    "shapes": [],
                    "numel": 0,
                    "finite_count": 0,
                    "nan_count": 0,
                    "posinf_count": 0,
                    "neginf_count": 0,
                    "sum": 0.0,
                    "sum_abs": 0.0,
                    "sum_squares": 0.0,
                    "min": None,
                    "max": None,
                },
            )
            dtype = str(raw.get("dtype") or "unknown")
            if dtype not in tensor["dtypes"]:
                tensor["dtypes"].append(dtype)
                tensor["dtypes"].sort()
            shape = [int(size) for size in raw.get("shape", ())]
            if shape not in tensor["shapes"]:
                tensor["shapes"].append(shape)
                tensor["shapes"].sort()
            for field in (
                "numel",
                "finite_count",
                "nan_count",
                "posinf_count",
                "neginf_count",
            ):
                tensor[field] = int(tensor[field]) + _int_value(raw.get(field))
            for field in ("sum", "sum_abs", "sum_squares"):
                tensor[field] = float(tensor[field]) + float(raw.get(field) or 0.0)
            raw_min = raw.get("min")
            if raw_min is not None:
                tensor["min"] = (
                    float(raw_min)
                    if tensor["min"] is None
                    else min(float(tensor["min"]), float(raw_min))
                )
            raw_max = raw.get("max")
            if raw_max is not None:
                tensor["max"] = (
                    float(raw_max)
                    if tensor["max"] is None
                    else max(float(tensor["max"]), float(raw_max))
                )


def _observe_loaded_weight_version(
    stats: dict[str, dict[str, Any]],
    counters: dict[str, Any],
    *,
    step: int | None,
) -> None:
    loaded = counters.get("loaded_weight")
    if isinstance(loaded, dict):
        model = loaded.get("model")
        version = loaded.get("version")
        if model is not None and version is not None:
            _observe_weight_version(stats, str(model), int(version), step=step)
        return
    if counters.get("weight_version") is not None:
        model = counters.get("model")
        if model is not None:
            _observe_weight_version(stats, str(model), int(counters["weight_version"]), step=step)


def _observe_produced_weight_version(stats: dict[str, dict[str, Any]], counters: dict[str, Any]) -> None:
    produced = counters.get("produced_weight")
    if isinstance(produced, dict):
        model = produced.get("model")
        version = produced.get("version")
        if model is not None and version is not None:
            _observe_weight_version(stats, str(model), int(version), step=None)
        return
    if counters.get("produced_weight_version") is not None:
        model = counters.get("model")
        if model is not None:
            _observe_weight_version(stats, str(model), int(counters["produced_weight_version"]), step=None)


def _observe_weight_version(
    stats: dict[str, dict[str, Any]],
    model: str,
    version: int,
    *,
    step: int | None,
) -> None:
    entry = stats.setdefault(
        model,
        {
            "count": 0,
            "nonzero_count": 0,
            "lagging_step_count": 0,
            "min": None,
            "max": None,
            "versions": {},
        },
    )
    entry["count"] = int(entry.get("count") or 0) + 1
    if version > 0:
        entry["nonzero_count"] = int(entry.get("nonzero_count") or 0) + 1
    if step is not None and version < step:
        entry["lagging_step_count"] = int(entry.get("lagging_step_count") or 0) + 1
    entry["min"] = version if entry.get("min") is None else min(int(entry["min"]), version)
    entry["max"] = version if entry.get("max") is None else max(int(entry["max"]), version)
    versions = entry.setdefault("versions", {})
    versions[str(version)] = int(versions.get(str(version), 0)) + 1


def _weight_stat_field(stats: dict[str, dict[str, Any]], field: str) -> dict[str, Any]:
    return {model: values.get(field) for model, values in sorted(stats.items())}


def _semantic_transfer_summaries(counters: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    direct = counters.get("semantic_transfers") or {}
    if isinstance(direct, dict):
        summaries.extend(transfer for transfer in direct.values() if isinstance(transfer, dict))
    activation = counters.get("activation_transfer") or {}
    if isinstance(activation, dict):
        nested = activation.get("semantic_transfers") or {}
        if isinstance(nested, dict):
            summaries.extend(transfer for transfer in nested.values() if isinstance(transfer, dict))
    return summaries


def _weight_transfer_summaries(counters: dict[str, Any]) -> list[dict[str, Any]]:
    transfer = counters.get("weight_transfer")
    if isinstance(transfer, dict):
        return [transfer]
    return []


def _rank_counter_dicts(transfer: dict[str, Any]) -> list[dict[str, Any]]:
    ranks = transfer.get("ranks")
    if not isinstance(ranks, (list, tuple)):
        return []
    return [
        rank["counters"]
        for rank in ranks
        if isinstance(rank, dict) and isinstance(rank.get("counters"), dict)
    ]


def _transfer_cached_endpoint_count(transfer: dict[str, Any]) -> int:
    value = transfer.get("ucx_direct_cached_endpoint_count")
    if value is not None:
        return int(value)
    if _bool_counter(transfer.get("ucx_direct_cached_endpoint")):
        return 1
    ranks = transfer.get("ranks")
    if not isinstance(ranks, (list, tuple)):
        return 0
    total = 0
    for rank in ranks:
        if not isinstance(rank, dict):
            continue
        rank_value = rank.get("ucx_direct_cached_endpoint_count")
        if rank_value is None:
            counters = rank.get("counters")
            if isinstance(counters, dict):
                rank_value = counters.get("ucx_direct_cached_endpoint_count")
        if rank_value is not None:
            total += int(rank_value)
            continue
        rank_cached = rank.get("ucx_direct_cached_endpoint")
        if rank_cached is None:
            counters = rank.get("counters")
            if isinstance(counters, dict):
                rank_cached = counters.get("ucx_direct_cached_endpoint")
        if _bool_counter(rank_cached):
            total += 1
    return total


def _transfer_rank_duration(transfer: dict[str, Any]) -> float | None:
    if transfer.get("transfer_rank_duration_s") is not None:
        return float(transfer["transfer_rank_duration_s"])
    ranks = transfer.get("ranks")
    if not isinstance(ranks, (list, tuple)):
        return None
    total = 0.0
    found = False
    for rank in ranks:
        if not isinstance(rank, dict):
            continue
        value = rank.get("transfer_rank_duration_s")
        if value is None:
            counters = rank.get("counters")
            if isinstance(counters, dict):
                value = counters.get("transfer_rank_duration_s")
        if value is None:
            continue
        total += float(value)
        found = True
    return total if found else None


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


def _sum_rank_counter(counters: dict[str, Any], key: str) -> int:
    """
    Sum a per-rank counter, accepting a directly aggregated counter too.
    """
    if counters.get(key) is not None:
        return _int_value(counters[key])
    return sum(_int_value(rank_counters.get(key)) for rank_counters in _rank_counter_dicts(counters))


def _max_rank_counter(counters: dict[str, Any], key: str) -> int:
    """
    Return a logical-operation counter represented by its busiest rank.
    """
    if counters.get(key) is not None:
        return _int_value(counters[key])
    values = [
        _int_value(rank_counters.get(key))
        for rank_counters in _rank_counter_dicts(counters)
        if rank_counters.get(key) is not None
    ]
    return max(values, default=0)


class _MonotonicCounterTracker:
    def __init__(self) -> None:
        self.total = 0
        self._last_by_key: dict[str, int] = {}

    def observe(self, key: str, value: int) -> None:
        previous = self._last_by_key.get(key)
        if previous is None:
            self.total += max(value, 0)
        elif value >= previous:
            self.total += value - previous
        else:
            self.total += max(value, 0)
        self._last_by_key[key] = value


def _generation_int_counter(counters: dict[str, Any], key: str) -> int | None:
    if counters.get(key) is not None:
        return _int_value(counters[key])
    ranks = counters.get("ranks")
    if not isinstance(ranks, (list, tuple)):
        return None
    by_dp_rank: dict[int, list[int]] = {}
    no_dp_rank: list[int] = []
    for rank in ranks:
        if not isinstance(rank, dict):
            continue
        rank_counters = rank.get("counters")
        if not isinstance(rank_counters, dict) or rank_counters.get(key) is None:
            continue
        value = _int_value(rank_counters[key])
        if rank_counters.get("dp_rank") is None:
            no_dp_rank.append(value)
            continue
        by_dp_rank.setdefault(_int_value(rank_counters["dp_rank"]), []).append(value)
    if not by_dp_rank and not no_dp_rank:
        return None
    total = sum(no_dp_rank)
    for values in by_dp_rank.values():
        if not values:
            continue
        first = values[0]
        total += first if all(value == first for value in values[1:]) else max(values)
    return total


def _generation_replica_counter(counters: dict[str, Any], key: str) -> int:
    ranks = counters.get("ranks")
    if isinstance(ranks, (list, tuple)) and any(
        isinstance(rank, dict)
        and isinstance(rank.get("counters"), dict)
        and rank["counters"].get("dp_rank") is not None
        for rank in ranks
    ):
        return int(_generation_int_counter(counters, key) or 0)
    return _int_value(_counter_value(counters, key))


def _generation_expected_tokens(counters: dict[str, Any]) -> int | None:
    raw_length = _counter_value(counters, "generation_length")
    if raw_length is None:
        return None
    length = _int_value(raw_length)
    sequences = _generation_int_counter(counters, "partial_n_seqs")
    if sequences is None:
        sequences = _generation_int_counter(counters, "batch_size")
    if sequences is None:
        return None
    return int(length) * int(sequences)


def _int_value(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _bool_counter(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def summary_to_prometheus(summary: dict[str, Any]) -> str:
    run_id = str(summary.get("run_id") or "")
    algorithm = str(summary.get("algorithm") or "")
    base_labels = {"run_id": run_id}
    if algorithm:
        base_labels["algorithm"] = algorithm

    lines = [
        "# HELP rpeg_runtime_events_total Runtime trace events observed.",
        "# TYPE rpeg_runtime_events_total counter",
        _metric("rpeg_runtime_events_total", summary.get("events", 0), base_labels),
        "# HELP rpeg_runtime_nodes_started_total Runtime nodes that started.",
        "# TYPE rpeg_runtime_nodes_started_total counter",
        _metric("rpeg_runtime_nodes_started_total", summary.get("started_nodes", 0), base_labels),
        "# HELP rpeg_runtime_nodes_finished_total Runtime nodes that finished.",
        "# TYPE rpeg_runtime_nodes_finished_total counter",
        _metric("rpeg_runtime_nodes_finished_total", summary.get("finished_nodes", 0), base_labels),
        "# HELP rpeg_runtime_replay_attempts_total Whole-graph replay attempts.",
        "# TYPE rpeg_runtime_replay_attempts_total counter",
        _metric("rpeg_runtime_replay_attempts_total", summary.get("replay_attempts", 0), base_labels),
        "# HELP rpeg_runtime_node_duration_seconds_total Sum of runtime node durations.",
        "# TYPE rpeg_runtime_node_duration_seconds_total counter",
        _metric("rpeg_runtime_node_duration_seconds_total", summary.get("total_node_duration_s", 0.0), base_labels),
        "# HELP rpeg_runtime_wall_time_seconds Runtime trace wall-clock span.",
        "# TYPE rpeg_runtime_wall_time_seconds gauge",
        _metric("rpeg_runtime_wall_time_seconds", summary.get("wall_time_s", 0.0), base_labels),
        "# HELP rpeg_runtime_semantic_transfers_total Semantic transfer count.",
        "# TYPE rpeg_runtime_semantic_transfers_total counter",
        _metric("rpeg_runtime_semantic_transfers_total", summary.get("semantic_transfer_count", 0), base_labels),
        "# HELP rpeg_runtime_semantic_transfer_bytes_total Semantic transfer bytes.",
        "# TYPE rpeg_runtime_semantic_transfer_bytes_total counter",
        _metric("rpeg_runtime_semantic_transfer_bytes_total", summary.get("semantic_transfer_bytes", 0), base_labels),
        "# HELP rpeg_runtime_semantic_transfer_rank_duration_seconds_total Sum of semantic transfer durations across worker ranks.",
        "# TYPE rpeg_runtime_semantic_transfer_rank_duration_seconds_total counter",
        _metric(
            "rpeg_runtime_semantic_transfer_rank_duration_seconds_total",
            summary.get("semantic_transfer_rank_duration_s", 0.0),
            base_labels,
        ),
        "# HELP rpeg_runtime_semantic_transfer_rank_bytes_per_second Semantic transfer bytes per summed rank-second.",
        "# TYPE rpeg_runtime_semantic_transfer_rank_bytes_per_second gauge",
        _metric(
            "rpeg_runtime_semantic_transfer_rank_bytes_per_second",
            summary.get("semantic_transfer_rank_bytes_per_s", 0.0),
            base_labels,
        ),
        "# HELP rpeg_runtime_semantic_transfer_dp_repartition_total Semantic transfer count with DP repartitioning.",
        "# TYPE rpeg_runtime_semantic_transfer_dp_repartition_total counter",
        _metric(
            "rpeg_runtime_semantic_transfer_dp_repartition_total",
            summary.get("semantic_transfer_dp_repartition_count", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_semantic_transfer_transport_total Semantic transfer count by transport.",
        "# TYPE rpeg_runtime_semantic_transfer_transport_total counter",
    ]
    for transport, value in sorted((summary.get("semantic_transfer_transport_counts") or {}).items()):
        lines.append(
            _metric(
                "rpeg_runtime_semantic_transfer_transport_total",
                value,
                {**base_labels, "transport": str(transport)},
            )
        )
    lines.extend([
        "# HELP rpeg_runtime_semantic_transfer_wire_format_total Semantic transfer count by wire format.",
        "# TYPE rpeg_runtime_semantic_transfer_wire_format_total counter",
    ])
    for wire_format, value in sorted((summary.get("semantic_transfer_wire_format_counts") or {}).items()):
        lines.append(
            _metric(
                "rpeg_runtime_semantic_transfer_wire_format_total",
                value,
                {**base_labels, "wire_format": str(wire_format)},
            )
        )
    lines.extend([
        "# HELP rpeg_runtime_semantic_transfer_transport_bytes_total Semantic transfer bytes by transport.",
        "# TYPE rpeg_runtime_semantic_transfer_transport_bytes_total counter",
    ])
    for transport, value in sorted((summary.get("semantic_transfer_transport_bytes") or {}).items()):
        lines.append(
            _metric(
                "rpeg_runtime_semantic_transfer_transport_bytes_total",
                value,
                {**base_labels, "transport": str(transport)},
            )
        )
    lines.extend([
        "# HELP rpeg_runtime_semantic_transfer_transport_rank_duration_seconds_total Sum of semantic transfer durations across worker ranks by transport.",
        "# TYPE rpeg_runtime_semantic_transfer_transport_rank_duration_seconds_total counter",
    ])
    for transport, value in sorted((summary.get("semantic_transfer_transport_rank_duration_s") or {}).items()):
        lines.append(
            _metric(
                "rpeg_runtime_semantic_transfer_transport_rank_duration_seconds_total",
                value,
                {**base_labels, "transport": str(transport)},
            )
        )
    lines.extend([
        "# HELP rpeg_runtime_semantic_transfer_transport_rank_bytes_per_second Semantic transfer bytes per summed rank-second by transport.",
        "# TYPE rpeg_runtime_semantic_transfer_transport_rank_bytes_per_second gauge",
    ])
    for transport, value in sorted((summary.get("semantic_transfer_transport_rank_bytes_per_s") or {}).items()):
        lines.append(
            _metric(
                "rpeg_runtime_semantic_transfer_transport_rank_bytes_per_second",
                value,
                {**base_labels, "transport": str(transport)},
            )
        )
    lines.extend([
        "# HELP rpeg_runtime_weight_transfers_total Weight-version transfer count.",
        "# TYPE rpeg_runtime_weight_transfers_total counter",
        _metric("rpeg_runtime_weight_transfers_total", summary.get("weight_transfer_count", 0), base_labels),
        "# HELP rpeg_runtime_weight_transfer_bytes_total Weight-version transfer bytes.",
        "# TYPE rpeg_runtime_weight_transfer_bytes_total counter",
        _metric("rpeg_runtime_weight_transfer_bytes_total", summary.get("weight_transfer_bytes", 0), base_labels),
        "# HELP rpeg_runtime_weight_transfer_network_bytes_total Weight-version bytes sent over non-local endpoints.",
        "# TYPE rpeg_runtime_weight_transfer_network_bytes_total counter",
        _metric(
            "rpeg_runtime_weight_transfer_network_bytes_total",
            summary.get("weight_transfer_network_bytes", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_weight_transfer_local_ipc_bytes_total Weight-version bytes sent through same-device local IPC endpoints.",
        "# TYPE rpeg_runtime_weight_transfer_local_ipc_bytes_total counter",
        _metric(
            "rpeg_runtime_weight_transfer_local_ipc_bytes_total",
            summary.get("weight_transfer_local_ipc_bytes", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_weight_transfer_local_endpoints_total Same-device local weight endpoint count.",
        "# TYPE rpeg_runtime_weight_transfer_local_endpoints_total counter",
        _metric(
            "rpeg_runtime_weight_transfer_local_endpoints_total",
            summary.get("weight_transfer_local_endpoint_count", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_weight_transfer_dp_relays_total Weight transfers using one cross-worker relay per destination model-parallel shard.",
        "# TYPE rpeg_runtime_weight_transfer_dp_relays_total counter",
        _metric(
            "rpeg_runtime_weight_transfer_dp_relays_total",
            summary.get("weight_transfer_destination_dp_relay_count", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_weight_transfer_relay_direct_bytes_total Bytes received directly by destination-DP relay ranks.",
        "# TYPE rpeg_runtime_weight_transfer_relay_direct_bytes_total counter",
        _metric(
            "rpeg_runtime_weight_transfer_relay_direct_bytes_total",
            summary.get("weight_transfer_relay_direct_bytes", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_weight_transfer_relay_direct_network_bytes_total Relay-direct bytes excluding same-device local IPC.",
        "# TYPE rpeg_runtime_weight_transfer_relay_direct_network_bytes_total counter",
        _metric(
            "rpeg_runtime_weight_transfer_relay_direct_network_bytes_total",
            summary.get("weight_transfer_relay_direct_network_bytes", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_weight_transfer_relay_broadcast_bytes_total Sum of logical bytes broadcast across destination DP ranks.",
        "# TYPE rpeg_runtime_weight_transfer_relay_broadcast_bytes_total counter",
        _metric(
            "rpeg_runtime_weight_transfer_relay_broadcast_bytes_total",
            summary.get("weight_transfer_relay_broadcast_bytes", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_weight_transfer_installed_active_ranks_total Receiver ranks that installed a transferred version directly as active weights.",
        "# TYPE rpeg_runtime_weight_transfer_installed_active_ranks_total counter",
        _metric(
            "rpeg_runtime_weight_transfer_installed_active_ranks_total",
            summary.get("weight_transfer_installed_active_rank_count", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_weight_transfer_adopted_parameter_bytes_total Received parameter bytes adopted without a second destination shard allocation.",
        "# TYPE rpeg_runtime_weight_transfer_adopted_parameter_bytes_total counter",
        _metric(
            "rpeg_runtime_weight_transfer_adopted_parameter_bytes_total",
            summary.get("weight_transfer_adopted_parameter_bytes", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_weight_transfer_discarded_released_state_ranks_total Receiver ranks that discarded a superseded retained device state before transfer.",
        "# TYPE rpeg_runtime_weight_transfer_discarded_released_state_ranks_total counter",
        _metric(
            "rpeg_runtime_weight_transfer_discarded_released_state_ranks_total",
            summary.get("weight_transfer_discarded_released_state_rank_count", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_weight_transfer_discarded_released_state_bytes_total Superseded retained device-state bytes discarded before transfer.",
        "# TYPE rpeg_runtime_weight_transfer_discarded_released_state_bytes_total counter",
        _metric(
            "rpeg_runtime_weight_transfer_discarded_released_state_bytes_total",
            summary.get("weight_transfer_discarded_released_state_bytes", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_weight_transfer_rank_duration_seconds_total Sum of weight-version transfer durations across worker ranks.",
        "# TYPE rpeg_runtime_weight_transfer_rank_duration_seconds_total counter",
        _metric(
            "rpeg_runtime_weight_transfer_rank_duration_seconds_total",
            summary.get("weight_transfer_rank_duration_s", 0.0),
            base_labels,
        ),
        "# HELP rpeg_runtime_weight_transfer_rank_bytes_per_second Weight-version transfer bytes per summed rank-second.",
        "# TYPE rpeg_runtime_weight_transfer_rank_bytes_per_second gauge",
        _metric(
            "rpeg_runtime_weight_transfer_rank_bytes_per_second",
            summary.get("weight_transfer_rank_bytes_per_s", 0.0),
            base_labels,
        ),
        "# HELP rpeg_runtime_weight_transfer_transport_total Weight-version transfer count by transport.",
        "# TYPE rpeg_runtime_weight_transfer_transport_total counter",
    ])
    for transport, value in sorted((summary.get("weight_transfer_transport_counts") or {}).items()):
        lines.append(
            _metric(
                "rpeg_runtime_weight_transfer_transport_total",
                value,
                {**base_labels, "transport": str(transport)},
            )
        )
    lines.extend([
        "# HELP rpeg_runtime_weight_transfer_transport_bytes_total Weight-version transfer bytes by transport.",
        "# TYPE rpeg_runtime_weight_transfer_transport_bytes_total counter",
    ])
    for transport, value in sorted((summary.get("weight_transfer_transport_bytes") or {}).items()):
        lines.append(
            _metric(
                "rpeg_runtime_weight_transfer_transport_bytes_total",
                value,
                {**base_labels, "transport": str(transport)},
            )
        )
    lines.extend([
        "# HELP rpeg_runtime_weight_transfer_transport_rank_duration_seconds_total Sum of weight-version transfer durations across worker ranks by transport.",
        "# TYPE rpeg_runtime_weight_transfer_transport_rank_duration_seconds_total counter",
    ])
    for transport, value in sorted((summary.get("weight_transfer_transport_rank_duration_s") or {}).items()):
        lines.append(
            _metric(
                "rpeg_runtime_weight_transfer_transport_rank_duration_seconds_total",
                value,
                {**base_labels, "transport": str(transport)},
            )
        )
    lines.extend([
        "# HELP rpeg_runtime_weight_transfer_transport_rank_bytes_per_second Weight-version transfer bytes per summed rank-second by transport.",
        "# TYPE rpeg_runtime_weight_transfer_transport_rank_bytes_per_second gauge",
    ])
    for transport, value in sorted((summary.get("weight_transfer_transport_rank_bytes_per_s") or {}).items()):
        lines.append(
            _metric(
                "rpeg_runtime_weight_transfer_transport_rank_bytes_per_second",
                value,
                {**base_labels, "transport": str(transport)},
            )
        )
    lines.extend([
        "# HELP rpeg_runtime_layout_switches_total In-place RPEG model layout switch count.",
        "# TYPE rpeg_runtime_layout_switches_total counter",
        _metric("rpeg_runtime_layout_switches_total", summary.get("layout_switch_count", 0), base_labels),
        "# HELP rpeg_runtime_layout_switch_bytes_total Tensor bytes sent by in-place layout switches.",
        "# TYPE rpeg_runtime_layout_switch_bytes_total counter",
        _metric("rpeg_runtime_layout_switch_bytes_total", summary.get("layout_switch_bytes", 0), base_labels),
        "# HELP rpeg_runtime_layout_switch_wall_duration_seconds_total Wall time spent in in-place layout switches.",
        "# TYPE rpeg_runtime_layout_switch_wall_duration_seconds_total counter",
        _metric(
            "rpeg_runtime_layout_switch_wall_duration_seconds_total",
            summary.get("layout_switch_wall_duration_s", 0.0),
            base_labels,
        ),
        "# HELP rpeg_runtime_layout_switch_tensor_native_total Tensor-native in-place layout switch count.",
        "# TYPE rpeg_runtime_layout_switch_tensor_native_total counter",
        _metric(
            "rpeg_runtime_layout_switch_tensor_native_total",
            summary.get("layout_switch_tensor_native_count", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_layout_switch_optimizer_persistent_total Layout switches retaining initialized optimizer state.",
        "# TYPE rpeg_runtime_layout_switch_optimizer_persistent_total counter",
        _metric(
            "rpeg_runtime_layout_switch_optimizer_persistent_total",
            summary.get("layout_switch_optimizer_persistent_count", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_layout_switch_optimizer_rebound_total Layout switches rebinding persistent optimizer state.",
        "# TYPE rpeg_runtime_layout_switch_optimizer_rebound_total counter",
        _metric(
            "rpeg_runtime_layout_switch_optimizer_rebound_total",
            summary.get("layout_switch_optimizer_rebound_count", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_operational_weight_materializations_total Trainable BF16 operational weight materializations.",
        "# TYPE rpeg_runtime_operational_weight_materializations_total counter",
        _metric(
            "rpeg_runtime_operational_weight_materializations_total",
            summary.get("operational_weight_materialize_count", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_operational_weight_releases_total Trainable BF16 operational weight releases.",
        "# TYPE rpeg_runtime_operational_weight_releases_total counter",
        _metric(
            "rpeg_runtime_operational_weight_releases_total",
            summary.get("operational_weight_release_count", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_operational_weight_released_bytes_total BF16 parameter bytes released after operations.",
        "# TYPE rpeg_runtime_operational_weight_released_bytes_total counter",
        _metric(
            "rpeg_runtime_operational_weight_released_bytes_total",
            summary.get("operational_weight_released_bytes", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_weight_version_loads_total Runtime weight-version loads by model.",
        "# TYPE rpeg_runtime_weight_version_loads_total counter",
    ])
    for model, value in sorted((summary.get("loaded_weight_count_by_model") or {}).items()):
        lines.append(_metric("rpeg_runtime_weight_version_loads_total", value, {**base_labels, "model": str(model)}))
    lines.extend([
        "# HELP rpeg_runtime_weight_version_productions_total Runtime weight-version productions by model.",
        "# TYPE rpeg_runtime_weight_version_productions_total counter",
    ])
    for model, value in sorted((summary.get("produced_weight_count_by_model") or {}).items()):
        lines.append(
            _metric("rpeg_runtime_weight_version_productions_total", value, {**base_labels, "model": str(model)})
        )
    lines.extend([
        "# HELP rpeg_runtime_weight_version_nonzero_loads_total Runtime nonzero weight-version loads by model.",
        "# TYPE rpeg_runtime_weight_version_nonzero_loads_total counter",
    ])
    for model, value in sorted((summary.get("nonzero_loaded_weight_count_by_model") or {}).items()):
        lines.append(
            _metric("rpeg_runtime_weight_version_nonzero_loads_total", value, {**base_labels, "model": str(model)})
        )
    lines.extend([
        "# HELP rpeg_runtime_weight_version_lagging_loads_total Runtime loads whose weight version is behind the logical step.",
        "# TYPE rpeg_runtime_weight_version_lagging_loads_total counter",
    ])
    for model, value in sorted((summary.get("lagging_loaded_weight_count_by_model") or {}).items()):
        lines.append(
            _metric("rpeg_runtime_weight_version_lagging_loads_total", value, {**base_labels, "model": str(model)})
        )
    lines.extend([
        "# HELP rpeg_runtime_weight_version_loaded_max Maximum loaded weight version by model.",
        "# TYPE rpeg_runtime_weight_version_loaded_max gauge",
    ])
    for model, value in sorted((summary.get("max_loaded_weight_version_by_model") or {}).items()):
        lines.append(_metric("rpeg_runtime_weight_version_loaded_max", value, {**base_labels, "model": str(model)}))
    lines.extend([
        "# HELP rpeg_runtime_weight_version_produced_max Maximum produced weight version by model.",
        "# TYPE rpeg_runtime_weight_version_produced_max gauge",
    ])
    for model, value in sorted((summary.get("max_produced_weight_version_by_model") or {}).items()):
        lines.append(_metric("rpeg_runtime_weight_version_produced_max", value, {**base_labels, "model": str(model)}))
    lines.extend([
        "# HELP rpeg_runtime_ucx_direct_cached_endpoints_total UCX direct transfers that used cached prewarmed endpoints.",
        "# TYPE rpeg_runtime_ucx_direct_cached_endpoints_total counter",
        _metric(
            "rpeg_runtime_ucx_direct_cached_endpoints_total",
            summary.get("ucx_direct_cached_endpoint_count", 0),
            {**base_labels, "scope": "all"},
        ),
        _metric(
            "rpeg_runtime_ucx_direct_cached_endpoints_total",
            summary.get("semantic_transfer_ucx_direct_cached_endpoint_count", 0),
            {**base_labels, "scope": "semantic"},
        ),
        _metric(
            "rpeg_runtime_ucx_direct_cached_endpoints_total",
            summary.get("weight_transfer_ucx_direct_cached_endpoint_count", 0),
            {**base_labels, "scope": "weight"},
        ),
        "# HELP rpeg_runtime_checkpoint_recoveries_total Checkpoint recovery events.",
        "# TYPE rpeg_runtime_checkpoint_recoveries_total counter",
        _metric("rpeg_runtime_checkpoint_recoveries_total", summary.get("checkpoint_recovery_count", 0), base_labels),
        "# HELP rpeg_runtime_checkpoint_weight_files_total Checkpoint-local weight-version files available for recovery.",
        "# TYPE rpeg_runtime_checkpoint_weight_files_total counter",
        _metric("rpeg_runtime_checkpoint_weight_files_total", summary.get("checkpoint_weight_files", 0), base_labels),
        "# HELP rpeg_runtime_checkpoint_weight_bytes_total Checkpoint-local weight-version bytes available for recovery.",
        "# TYPE rpeg_runtime_checkpoint_weight_bytes_total counter",
        _metric("rpeg_runtime_checkpoint_weight_bytes_total", summary.get("checkpoint_weight_bytes", 0), base_labels),
        "# HELP rpeg_runtime_checkpoint_files_on_disk Retained checkpoint files on disk.",
        "# TYPE rpeg_runtime_checkpoint_files_on_disk gauge",
        _metric(
            "rpeg_runtime_checkpoint_files_on_disk",
            summary.get("checkpoint_files_on_disk", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_checkpoint_bytes_on_disk Retained checkpoint bytes on disk.",
        "# TYPE rpeg_runtime_checkpoint_bytes_on_disk gauge",
        _metric(
            "rpeg_runtime_checkpoint_bytes_on_disk",
            summary.get("checkpoint_bytes_on_disk", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_checkpoint_weight_data_files_total Checkpoint-local weight-version tensor files available for recovery.",
        "# TYPE rpeg_runtime_checkpoint_weight_data_files_total counter",
        _metric(
            "rpeg_runtime_checkpoint_weight_data_files_total",
            summary.get("checkpoint_weight_data_files", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_checkpoint_weight_metadata_files_total Checkpoint-local weight-version metadata sidecars available for recovery.",
        "# TYPE rpeg_runtime_checkpoint_weight_metadata_files_total counter",
        _metric(
            "rpeg_runtime_checkpoint_weight_metadata_files_total",
            summary.get("checkpoint_weight_metadata_files", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_checkpoint_optimizer_state_rank_files_total Optimizer-bearing checkpoint-local train rank sidecars.",
        "# TYPE rpeg_runtime_checkpoint_optimizer_state_rank_files_total counter",
        _metric(
            "rpeg_runtime_checkpoint_optimizer_state_rank_files_total",
            summary.get("checkpoint_optimizer_state_rank_files", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_checkpoint_portable_train_versions_total Complete optimizer-bearing portable train weight versions available for recovery.",
        "# TYPE rpeg_runtime_checkpoint_portable_train_versions_total counter",
        _metric(
            "rpeg_runtime_checkpoint_portable_train_versions_total",
            summary.get("checkpoint_portable_train_versions", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_checkpoint_portable_train_rank_files_total Rank files covered by complete optimizer-bearing portable train versions.",
        "# TYPE rpeg_runtime_checkpoint_portable_train_rank_files_total counter",
        _metric(
            "rpeg_runtime_checkpoint_portable_train_rank_files_total",
            summary.get("checkpoint_portable_train_rank_files", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_checkpoint_restored_weight_files_total Weight-version files restored before suffix replay.",
        "# TYPE rpeg_runtime_checkpoint_restored_weight_files_total counter",
        _metric(
            "rpeg_runtime_checkpoint_restored_weight_files_total",
            summary.get("checkpoint_restored_weight_files", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_checkpoint_restored_weight_bytes_total Weight-version bytes restored before suffix replay.",
        "# TYPE rpeg_runtime_checkpoint_restored_weight_bytes_total counter",
        _metric(
            "rpeg_runtime_checkpoint_restored_weight_bytes_total",
            summary.get("checkpoint_restored_weight_bytes", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_failed_nodes Runtime failed node count.",
        "# TYPE rpeg_runtime_failed_nodes gauge",
        _metric("rpeg_runtime_failed_nodes", len(summary.get("failed_nodes") or []), base_labels),
        "# HELP rpeg_runtime_status_total Runtime finished nodes by status.",
        "# TYPE rpeg_runtime_status_total counter",
    ])
    for status, value in sorted((summary.get("status_counts") or {}).items()):
        lines.append(_metric("rpeg_runtime_status_total", value, {**base_labels, "status": str(status)}))

    lines.extend([
        "# HELP rpeg_runtime_fused_pipeline_optimizer_total Fused training nodes by offline schedule optimizer.",
        "# TYPE rpeg_runtime_fused_pipeline_optimizer_total counter",
    ])
    fused_optimizer_counts = summary.get("fused_pipeline_optimizer_counts") or {}
    if isinstance(fused_optimizer_counts, dict):
        for optimizer, value in sorted(fused_optimizer_counts.items()):
            lines.append(_metric(
                "rpeg_runtime_fused_pipeline_optimizer_total",
                value,
                {**base_labels, "optimizer": str(optimizer)},
            ))

    lines.extend([
        "# HELP rpeg_runtime_node_kind_total Runtime finished nodes by lowered node kind.",
        "# TYPE rpeg_runtime_node_kind_total counter",
    ])
    for kind, value in sorted((summary.get("kind_counts") or {}).items()):
        lines.append(_metric("rpeg_runtime_node_kind_total", value, {**base_labels, "kind": str(kind)}))

    lines.extend([
        "# HELP rpeg_runtime_generation_nodes_total Generation nodes by backend.",
        "# TYPE rpeg_runtime_generation_nodes_total counter",
    ])
    generation_backend_counts = summary.get("generation_backend_counts") or {}
    if isinstance(generation_backend_counts, dict):
        for backend, value in sorted(generation_backend_counts.items()):
            labels = {**base_labels, "backend": str(backend)}
            lines.append(_metric("rpeg_runtime_generation_nodes_total", value, labels))
    lines.extend([
        "# HELP rpeg_runtime_kv_cache_generation_nodes_total Generation nodes that used KV-cache decode.",
        "# TYPE rpeg_runtime_kv_cache_generation_nodes_total counter",
        _metric("rpeg_runtime_kv_cache_generation_nodes_total", summary.get("kv_cache_generation_count", 0), base_labels),
        "# HELP rpeg_runtime_kv_cache_pool_entries KV-cache serving pool active-entry high-water mark across generation workers.",
        "# TYPE rpeg_runtime_kv_cache_pool_entries gauge",
        _metric("rpeg_runtime_kv_cache_pool_entries", summary.get("kv_cache_pool_entries", 0), base_labels),
        "# HELP rpeg_runtime_kv_cache_pool_hits_total KV-cache serving pool hits observed on generation nodes.",
        "# TYPE rpeg_runtime_kv_cache_pool_hits_total counter",
        _metric("rpeg_runtime_kv_cache_pool_hits_total", summary.get("kv_cache_pool_hits", 0), base_labels),
        "# HELP rpeg_runtime_kv_cache_pool_misses_total KV-cache serving pool misses observed on generation nodes.",
        "# TYPE rpeg_runtime_kv_cache_pool_misses_total counter",
        _metric("rpeg_runtime_kv_cache_pool_misses_total", summary.get("kv_cache_pool_misses", 0), base_labels),
        "# HELP rpeg_runtime_kv_cache_pool_evictions_total KV-cache serving pool evictions observed on generation nodes.",
        "# TYPE rpeg_runtime_kv_cache_pool_evictions_total counter",
        _metric("rpeg_runtime_kv_cache_pool_evictions_total", summary.get("kv_cache_pool_evictions", 0), base_labels),
        "# HELP rpeg_runtime_kv_cache_max_batch_size Maximum KV-cache serving batch capacity observed.",
        "# TYPE rpeg_runtime_kv_cache_max_batch_size gauge",
        _metric("rpeg_runtime_kv_cache_max_batch_size", summary.get("kv_cache_max_batch_size", 0), base_labels),
        "# HELP rpeg_runtime_kv_cache_max_sequence_length Maximum KV-cache serving sequence capacity observed.",
        "# TYPE rpeg_runtime_kv_cache_max_sequence_length gauge",
        _metric("rpeg_runtime_kv_cache_max_sequence_length", summary.get("kv_cache_max_sequence_length", 0), base_labels),
        "# HELP rpeg_runtime_generation_tokens_total Generated response tokens observed in rollout nodes.",
        "# TYPE rpeg_runtime_generation_tokens_total counter",
        _metric("rpeg_runtime_generation_tokens_total", summary.get("generation_tokens", 0), base_labels),
        "# HELP rpeg_runtime_generation_prompt_tokens_total Prompt tokens processed by rollout prefill.",
        "# TYPE rpeg_runtime_generation_prompt_tokens_total counter",
        _metric(
            "rpeg_runtime_generation_prompt_tokens_total",
            summary.get("generation_prompt_tokens", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_generation_sequences_total Generated response sequences observed in rollout nodes.",
        "# TYPE rpeg_runtime_generation_sequences_total counter",
        _metric("rpeg_runtime_generation_sequences_total", summary.get("generation_sequences", 0), base_labels),
        "# HELP rpeg_runtime_generation_node_duration_seconds_total Sum of generation-node durations.",
        "# TYPE rpeg_runtime_generation_node_duration_seconds_total counter",
        _metric(
            "rpeg_runtime_generation_node_duration_seconds_total",
            summary.get("generation_node_duration_s", 0.0),
            base_labels,
        ),
        "# HELP rpeg_runtime_generation_tokens_per_second Generated response tokens per wall-clock second.",
        "# TYPE rpeg_runtime_generation_tokens_per_second gauge",
        _metric("rpeg_runtime_generation_tokens_per_second", summary.get("generation_tokens_per_s", 0.0), base_labels),
        "# HELP rpeg_runtime_generation_prompt_tokens_per_second Prompt tokens per end-to-end wall-clock second.",
        "# TYPE rpeg_runtime_generation_prompt_tokens_per_second gauge",
        _metric(
            "rpeg_runtime_generation_prompt_tokens_per_second",
            summary.get("generation_prompt_tokens_per_s", 0.0),
            base_labels,
        ),
        "# HELP rpeg_runtime_generation_node_tokens_per_second Generated response tokens per generation-node second.",
        "# TYPE rpeg_runtime_generation_node_tokens_per_second gauge",
        _metric(
            "rpeg_runtime_generation_node_tokens_per_second",
            summary.get("generation_node_tokens_per_s", 0.0),
            base_labels,
        ),
        "# HELP rpeg_runtime_generation_prompt_node_tokens_per_second Prompt tokens per generation-node second.",
        "# TYPE rpeg_runtime_generation_prompt_node_tokens_per_second gauge",
        _metric(
            "rpeg_runtime_generation_prompt_node_tokens_per_second",
            summary.get("generation_prompt_node_tokens_per_s", 0.0),
            base_labels,
        ),
        "# HELP rpeg_runtime_cuda_peak_memory_allocated_bytes Maximum per-rank CUDA allocation peak observed.",
        "# TYPE rpeg_runtime_cuda_peak_memory_allocated_bytes gauge",
        _metric(
            "rpeg_runtime_cuda_peak_memory_allocated_bytes",
            summary.get("cuda_peak_memory_allocated_bytes", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_cuda_peak_memory_reserved_bytes Maximum per-rank CUDA reservation peak observed.",
        "# TYPE rpeg_runtime_cuda_peak_memory_reserved_bytes gauge",
        _metric(
            "rpeg_runtime_cuda_peak_memory_reserved_bytes",
            summary.get("cuda_peak_memory_reserved_bytes", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_activation_cuda_cache_bytes Maximum live semantic activation bytes observed on one rank.",
        "# TYPE rpeg_runtime_activation_cuda_cache_bytes gauge",
        _metric(
            "rpeg_runtime_activation_cuda_cache_bytes",
            summary.get("activation_cuda_cache_bytes_current_max", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_activation_cuda_cache_limit_bytes Configured per-worker semantic activation HBM ceiling.",
        "# TYPE rpeg_runtime_activation_cuda_cache_limit_bytes gauge",
        _metric(
            "rpeg_runtime_activation_cuda_cache_limit_bytes",
            summary.get("activation_cuda_cache_bytes_limit_max", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_activation_cuda_cache_placements_max Maximum successful CUDA semantic-cache placements observed on one rank.",
        "# TYPE rpeg_runtime_activation_cuda_cache_placements_max gauge",
        _metric(
            "rpeg_runtime_activation_cuda_cache_placements_max",
            summary.get("activation_cuda_cache_placements_max", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_activation_cuda_cache_cpu_fallbacks_max Maximum capacity-triggered CPU fallbacks observed on one rank.",
        "# TYPE rpeg_runtime_activation_cuda_cache_cpu_fallbacks_max gauge",
        _metric(
            "rpeg_runtime_activation_cuda_cache_cpu_fallbacks_max",
            summary.get("activation_cuda_cache_cpu_fallbacks_max", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_fusion_cuda_ipc_event_exports_total Interprocess CUDA completion events exported by fused actor/critic ranks.",
        "# TYPE rpeg_runtime_fusion_cuda_ipc_event_exports_total counter",
        _metric(
            "rpeg_runtime_fusion_cuda_ipc_event_exports_total",
            summary.get("fusion_cuda_ipc_event_exports", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_fusion_cuda_ipc_event_imports_total Interprocess CUDA completion events imported by fused actor/critic ranks.",
        "# TYPE rpeg_runtime_fusion_cuda_ipc_event_imports_total counter",
        _metric(
            "rpeg_runtime_fusion_cuda_ipc_event_imports_total",
            summary.get("fusion_cuda_ipc_event_imports", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_fusion_cuda_ipc_event_waits_total GPU stream waits issued on fused actor/critic completion events.",
        "# TYPE rpeg_runtime_fusion_cuda_ipc_event_waits_total counter",
        _metric(
            "rpeg_runtime_fusion_cuda_ipc_event_waits_total",
            summary.get("fusion_cuda_ipc_event_waits", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_fusion_cuda_ipc_event_records_total GPU completion events recorded by fused actor/critic ranks.",
        "# TYPE rpeg_runtime_fusion_cuda_ipc_event_records_total counter",
        _metric(
            "rpeg_runtime_fusion_cuda_ipc_event_records_total",
            summary.get("fusion_cuda_ipc_event_records", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_fusion_cpu_handle_polls_total CPU polls waiting only for CUDA IPC handle publication.",
        "# TYPE rpeg_runtime_fusion_cpu_handle_polls_total counter",
        _metric(
            "rpeg_runtime_fusion_cpu_handle_polls_total",
            summary.get("fusion_cpu_handle_polls", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_fusion_cpu_completion_polls_total CPU polls waiting for fused GPU task completion; zero is required for the CUDA IPC backend.",
        "# TYPE rpeg_runtime_fusion_cpu_completion_polls_total counter",
        _metric(
            "rpeg_runtime_fusion_cpu_completion_polls_total",
            summary.get("fusion_cpu_completion_polls", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_fusion_pipeline_async_sends_total Nonblocking NCCL pipeline sends posted by fused actor/critic ranks.",
        "# TYPE rpeg_runtime_fusion_pipeline_async_sends_total counter",
        _metric(
            "rpeg_runtime_fusion_pipeline_async_sends_total",
            summary.get("fusion_pipeline_async_sends", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_fusion_pipeline_async_send_waits_total Nonblocking NCCL pipeline send requests completed by fused actor/critic ranks.",
        "# TYPE rpeg_runtime_fusion_pipeline_async_send_waits_total counter",
        _metric(
            "rpeg_runtime_fusion_pipeline_async_send_waits_total",
            summary.get("fusion_pipeline_async_send_waits", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_fusion_pipeline_async_recvs_total NCCL pipeline receives posted before fused device-lane dependency waits.",
        "# TYPE rpeg_runtime_fusion_pipeline_async_recvs_total counter",
        _metric(
            "rpeg_runtime_fusion_pipeline_async_recvs_total",
            summary.get("fusion_pipeline_async_recvs", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_fusion_pipeline_async_recv_waits_total Preposted NCCL pipeline receive requests completed after fused device-lane dependency waits.",
        "# TYPE rpeg_runtime_fusion_pipeline_async_recv_waits_total counter",
        _metric(
            "rpeg_runtime_fusion_pipeline_async_recv_waits_total",
            summary.get("fusion_pipeline_async_recv_waits", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_fused_pipeline_optimizer_candidates Maximum offline fused-schedule candidates evaluated for one training pair.",
        "# TYPE rpeg_runtime_fused_pipeline_optimizer_candidates gauge",
        _metric(
            "rpeg_runtime_fused_pipeline_optimizer_candidates",
            summary.get("fused_pipeline_optimizer_candidates_max", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_fused_pipeline_makespan_units Predicted makespan of the selected fused microtask order.",
        "# TYPE rpeg_runtime_fused_pipeline_makespan_units gauge",
        _metric(
            "rpeg_runtime_fused_pipeline_makespan_units",
            summary.get("fused_pipeline_makespan_units_min", 0) or 0,
            base_labels,
        ),
        "# HELP rpeg_runtime_fused_pipeline_baseline_makespan_units Predicted makespan of the previous greedy fused microtask order.",
        "# TYPE rpeg_runtime_fused_pipeline_baseline_makespan_units gauge",
        _metric(
            "rpeg_runtime_fused_pipeline_baseline_makespan_units",
            summary.get("fused_pipeline_baseline_makespan_units_min", 0) or 0,
            base_labels,
        ),
        "# HELP rpeg_runtime_fused_pipeline_predicted_improvement_fraction Predicted fractional makespan reduction from offline fused-schedule selection.",
        "# TYPE rpeg_runtime_fused_pipeline_predicted_improvement_fraction gauge",
        _metric(
            "rpeg_runtime_fused_pipeline_predicted_improvement_fraction",
            summary.get("fused_pipeline_predicted_improvement_fraction", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_megatron_process_group_observations_total Semantic rank results reporting Megatron process-group pooling counters.",
        "# TYPE rpeg_runtime_megatron_process_group_observations_total counter",
        _metric(
            "rpeg_runtime_megatron_process_group_observations_total",
            summary.get("megatron_process_group_observation_count", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_megatron_process_groups_requested_max Maximum process-group creation requests observed on one Megatron rank.",
        "# TYPE rpeg_runtime_megatron_process_groups_requested_max gauge",
        _metric(
            "rpeg_runtime_megatron_process_groups_requested_max",
            summary.get("megatron_process_groups_requested_max", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_megatron_process_groups_created_max Maximum unique process groups created on one Megatron rank.",
        "# TYPE rpeg_runtime_megatron_process_groups_created_max gauge",
        _metric(
            "rpeg_runtime_megatron_process_groups_created_max",
            summary.get("megatron_process_groups_created_max", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_megatron_process_groups_reused_max Maximum exact-compatible process-group aliases reused on one Megatron rank.",
        "# TYPE rpeg_runtime_megatron_process_groups_reused_max gauge",
        _metric(
            "rpeg_runtime_megatron_process_groups_reused_max",
            summary.get("megatron_process_groups_reused_max", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_megatron_pipeline_cta_policy_observations_total Semantic rank results reporting an explicit PP NCCL CTA policy.",
        "# TYPE rpeg_runtime_megatron_pipeline_cta_policy_observations_total counter",
        _metric(
            "rpeg_runtime_megatron_pipeline_cta_policy_observations_total",
            summary.get("megatron_pipeline_cta_policy_observation_count", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_megatron_pipeline_cta_policy_zero_observations_total Semantic rank results reporting NCCL zero-CTA policy.",
        "# TYPE rpeg_runtime_megatron_pipeline_cta_policy_zero_observations_total counter",
        _metric(
            "rpeg_runtime_megatron_pipeline_cta_policy_zero_observations_total",
            summary.get("megatron_pipeline_cta_policy_zero_observation_count", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_megatron_pipeline_cta_policy_max Maximum explicit PP NCCL CTA policy value observed.",
        "# TYPE rpeg_runtime_megatron_pipeline_cta_policy_max gauge",
        _metric(
            "rpeg_runtime_megatron_pipeline_cta_policy_max",
            summary.get("megatron_pipeline_cta_policy_max", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_megatron_pipeline_groups_policy_applied_max Maximum PP groups receiving the configured CTA policy on one rank.",
        "# TYPE rpeg_runtime_megatron_pipeline_groups_policy_applied_max gauge",
        _metric(
            "rpeg_runtime_megatron_pipeline_groups_policy_applied_max",
            summary.get("megatron_pipeline_groups_policy_applied_max", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_torch_direct_process_group_cache_size_max Maximum live semantic torch.distributed process groups on one rank.",
        "# TYPE rpeg_runtime_torch_direct_process_group_cache_size_max gauge",
        _metric(
            "rpeg_runtime_torch_direct_process_group_cache_size_max",
            summary.get("torch_direct_process_group_cache_size_max", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_training_inference_cache_transition_ranks_total Training-rank transitions that invalidated inference tensor caches.",
        "# TYPE rpeg_runtime_training_inference_cache_transition_ranks_total counter",
        _metric(
            "rpeg_runtime_training_inference_cache_transition_ranks_total",
            summary.get("training_inference_cache_transition_rank_count", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_training_rotary_lru_entries_cleared_total Megatron RoPE LRU entries invalidated before autograd.",
        "# TYPE rpeg_runtime_training_rotary_lru_entries_cleared_total counter",
        _metric(
            "rpeg_runtime_training_rotary_lru_entries_cleared_total",
            summary.get("training_rotary_lru_entries_cleared", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_training_model_rotary_entries_cleared_total Megatron model RoPE dictionary entries invalidated before autograd.",
        "# TYPE rpeg_runtime_training_model_rotary_entries_cleared_total counter",
        _metric(
            "rpeg_runtime_training_model_rotary_entries_cleared_total",
            summary.get("training_model_rotary_entries_cleared", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_training_static_kv_pool_entries_retained_total Static-KV allocation-pool entries retained across inference-to-training transitions.",
        "# TYPE rpeg_runtime_training_static_kv_pool_entries_retained_total counter",
        _metric(
            "rpeg_runtime_training_static_kv_pool_entries_retained_total",
            summary.get("training_static_kv_pool_entries_retained", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_generation_prefill_forward_steps_total Generation prefill forward steps.",
        "# TYPE rpeg_runtime_generation_prefill_forward_steps_total counter",
        _metric(
            "rpeg_runtime_generation_prefill_forward_steps_total",
            summary.get("generation_prefill_forward_steps", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_generation_decode_forward_steps_total Generation decode forward steps.",
        "# TYPE rpeg_runtime_generation_decode_forward_steps_total counter",
        _metric(
            "rpeg_runtime_generation_decode_forward_steps_total",
            summary.get("generation_decode_forward_steps", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_generation_forward_steps_total Generation forward steps.",
        "# TYPE rpeg_runtime_generation_forward_steps_total counter",
        _metric("rpeg_runtime_generation_forward_steps_total", summary.get("generation_forward_steps", 0), base_labels),
        "# HELP rpeg_runtime_generation_old_logps_from_generation_total Generation nodes that reused generation logits for old log-probs.",
        "# TYPE rpeg_runtime_generation_old_logps_from_generation_total counter",
        _metric(
            "rpeg_runtime_generation_old_logps_from_generation_total",
            summary.get("generation_old_logps_from_generation_count", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_generation_old_logp_forward_steps_total Extra completed-sequence forward steps used for old log-probs.",
        "# TYPE rpeg_runtime_generation_old_logp_forward_steps_total counter",
        _metric(
            "rpeg_runtime_generation_old_logp_forward_steps_total",
            summary.get("generation_old_logp_forward_steps", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_dynamic_generation_batches_total Physical dynamic generation batches issued by semantic workers.",
        "# TYPE rpeg_runtime_dynamic_generation_batches_total counter",
        _metric(
            "rpeg_runtime_dynamic_generation_batches_total",
            summary.get("dynamic_generation_batch_count", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_dynamic_generation_batched_requests_total Logical generation requests served inside dynamic generation batches.",
        "# TYPE rpeg_runtime_dynamic_generation_batched_requests_total counter",
        _metric(
            "rpeg_runtime_dynamic_generation_batched_requests_total",
            summary.get("dynamic_generation_batched_request_count", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_dynamic_generation_batched_sequences_total Sequence batch size served by physical dynamic generation batches.",
        "# TYPE rpeg_runtime_dynamic_generation_batched_sequences_total counter",
        _metric(
            "rpeg_runtime_dynamic_generation_batched_sequences_total",
            summary.get("dynamic_generation_batched_sequence_count", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_dynamic_generation_batched_tokens_total Response tokens served by physical dynamic generation batches.",
        "# TYPE rpeg_runtime_dynamic_generation_batched_tokens_total counter",
        _metric(
            "rpeg_runtime_dynamic_generation_batched_tokens_total",
            summary.get("dynamic_generation_batched_token_count", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_generation_quantile_phases_total Executed response-length generation phases.",
        "# TYPE rpeg_runtime_generation_quantile_phases_total counter",
        _metric(
            "rpeg_runtime_generation_quantile_phases_total",
            summary.get("generation_quantile_phase_count", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_generation_quantile_planned_sequences_total Planned quantile completion quota across phases.",
        "# TYPE rpeg_runtime_generation_quantile_planned_sequences_total counter",
        _metric(
            "rpeg_runtime_generation_quantile_planned_sequences_total",
            summary.get("generation_quantile_planned_sequences", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_generation_quantile_actual_sequences_total Naturally completed sequences emitted across phases.",
        "# TYPE rpeg_runtime_generation_quantile_actual_sequences_total counter",
        _metric(
            "rpeg_runtime_generation_quantile_actual_sequences_total",
            summary.get("generation_quantile_actual_sequences", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_generation_quantile_natural_volume_mismatches_total Phases whose natural completion volume differed from the planned quota.",
        "# TYPE rpeg_runtime_generation_quantile_natural_volume_mismatches_total counter",
        _metric(
            "rpeg_runtime_generation_quantile_natural_volume_mismatches_total",
            summary.get("generation_quantile_natural_volume_mismatch_count", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_generation_quantile_compacted_sequences_total Naturally finished sequences removed from continuing KV batches.",
        "# TYPE rpeg_runtime_generation_quantile_compacted_sequences_total counter",
        _metric(
            "rpeg_runtime_generation_quantile_compacted_sequences_total",
            summary.get("generation_quantile_compacted_sequences", 0),
            base_labels,
        ),
        "# HELP rpeg_runtime_generation_quantile_forced_completions_total Sequences forced to match planned quotas; zero preserves generation semantics.",
        "# TYPE rpeg_runtime_generation_quantile_forced_completions_total counter",
        _metric(
            "rpeg_runtime_generation_quantile_forced_completions_total",
            summary.get("generation_quantile_forced_completions", 0),
            base_labels,
        ),
    ])

    lines.extend([
        "# HELP rpeg_runtime_loss_observations_total Distributed training loss observations reported by worker ranks.",
        "# TYPE rpeg_runtime_loss_observations_total counter",
        _metric("rpeg_runtime_loss_observations_total", summary.get("loss_observation_count", 0), base_labels),
        "# HELP rpeg_runtime_nonfinite_losses_total Non-finite or malformed distributed loss observations.",
        "# TYPE rpeg_runtime_nonfinite_losses_total counter",
        _metric("rpeg_runtime_nonfinite_losses_total", summary.get("nonfinite_loss_count", 0), base_labels),
        "# HELP rpeg_runtime_latest_loss Latest reported loss values by runtime node and counter.",
        "# TYPE rpeg_runtime_latest_loss gauge",
    ])
    for name, value in sorted((summary.get("latest_losses") or {}).items()):
        node_id, _, loss_name = str(name).partition(".")
        labels = {**base_labels, "node_id": node_id, "loss": loss_name or "loss"}
        lines.append(_metric("rpeg_runtime_latest_loss", value, labels))

    alerts = summary.get("alerts") or []
    lines.extend([
        "# HELP rpeg_runtime_alerts Runtime alert state by rule and severity.",
        "# TYPE rpeg_runtime_alerts gauge",
    ])
    if isinstance(alerts, list):
        for alert in alerts:
            if not isinstance(alert, dict):
                continue
            labels = {
                **base_labels,
                "rule": str(alert.get("rule") or "unknown"),
                "severity": str(alert.get("severity") or "warning"),
            }
            lines.append(_metric("rpeg_runtime_alerts", 1, labels))

    for key in ("steps", "last_loss", "last_actor_loss", "last_value_loss", "attempt"):
        if key in summary and isinstance(summary[key], (int, float)) and not isinstance(summary[key], bool):
            lines.append(f"# HELP rpeg_runtime_{_metric_suffix(key)} Run-level numeric field {key}.")
            lines.append(f"# TYPE rpeg_runtime_{_metric_suffix(key)} gauge")
            lines.append(_metric(f"rpeg_runtime_{_metric_suffix(key)}", summary[key], base_labels))

    return "\n".join(lines) + "\n"


def evaluate_runtime_alerts(summary: dict[str, Any], rules: dict[str, Any] | None = None) -> dict[str, Any]:
    rules = dict(rules or {})
    alerts: list[dict[str, Any]] = []
    run_id = summary.get("run_id")
    algorithm = summary.get("algorithm")

    max_failed = _optional_int_rule(rules, "max_failed_nodes", 0)
    failed_count = len(summary.get("failed_nodes") or [])
    if max_failed is not None and failed_count > max_failed:
        alerts.append(
            _alert(
                "failed_nodes",
                rules,
                default_severity="critical",
                message=f"{failed_count} runtime nodes failed",
                observed=failed_count,
                threshold=max_failed,
            )
        )

    if _bool_rule(rules, "require_all_started_finished", True):
        started = int(summary.get("started_nodes") or 0)
        finished = int(summary.get("finished_nodes") or 0)
        if finished < started:
            alerts.append(
                _alert(
                    "incomplete_nodes",
                    rules,
                    default_severity="critical",
                    message=f"{started - finished} started runtime nodes did not finish",
                    observed={"started": started, "finished": finished},
                    threshold="finished_nodes >= started_nodes",
                )
            )

    if _bool_rule(rules, "require_finite_losses", True):
        nonfinite_count = int(summary.get("nonfinite_loss_count") or 0)
        if nonfinite_count:
            alerts.append(
                _alert(
                    "nonfinite_losses",
                    rules,
                    default_severity="critical",
                    message=f"{nonfinite_count} non-finite or malformed loss observations were reported",
                    observed=summary.get("nonfinite_losses") or [],
                    threshold=0,
                )
            )

    max_replays = _optional_int_rule(
        rules,
        "max_replay_attempts",
        None if _bool_rule(rules, "allow_replay_attempts", False) else 0,
    )
    replay_attempts = int(summary.get("replay_attempts") or 0)
    if max_replays is not None and replay_attempts > max_replays:
        alerts.append(
            _alert(
                "replay_attempts",
                rules,
                default_severity="warning",
                message=f"{replay_attempts} graph replay attempts occurred",
                observed=replay_attempts,
                threshold=max_replays,
            )
        )

    if _bool_rule(rules, "require_checkpoint_restore_match", True):
        recoveries = int(summary.get("checkpoint_recovery_count") or 0)
        checkpoint_files = int(summary.get("checkpoint_weight_files") or 0)
        restored_files = int(summary.get("checkpoint_restored_weight_files") or 0)
        checkpoint_bytes = int(summary.get("checkpoint_weight_bytes") or 0)
        restored_bytes = int(summary.get("checkpoint_restored_weight_bytes") or 0)
        if recoveries > 0 and (checkpoint_files != restored_files or checkpoint_bytes != restored_bytes):
            alerts.append(
                _alert(
                    "checkpoint_restore_mismatch",
                    rules,
                    default_severity="critical",
                    message="checkpoint-local weight-version restore totals do not match manifest totals",
                    observed={
                        "checkpoint_weight_files": checkpoint_files,
                        "restored_weight_files": restored_files,
                        "checkpoint_weight_bytes": checkpoint_bytes,
                        "restored_weight_bytes": restored_bytes,
                    },
                    threshold="restored totals match checkpoint totals",
                )
            )

    if _bool_rule(rules, "require_kv_cache_generation", False):
        generation_count = int(summary.get("generation_count") or 0)
        kv_generation_count = int(summary.get("kv_cache_generation_count") or 0)
        required = max(1, generation_count)
        if kv_generation_count < required:
            alerts.append(
                _alert(
                    "kv_cache_generation_missing",
                    rules,
                    default_severity="critical",
                    message="KV-cache generation was required but not observed for all generation nodes",
                    observed={
                        "generation_count": generation_count,
                        "kv_cache_generation_count": kv_generation_count,
                    },
                    threshold={"kv_cache_generation_count": required},
                )
            )

    if _bool_rule(rules, "require_kv_cache_reuse", False):
        hits = int(summary.get("kv_cache_pool_hits") or 0)
        if hits <= 0:
            alerts.append(
                _alert(
                    "kv_cache_reuse_missing",
                    rules,
                    default_severity="critical",
                    message="KV-cache pool reuse was required but no pool hit was observed",
                    observed={
                        "kv_cache_pool_entries": int(summary.get("kv_cache_pool_entries") or 0),
                        "kv_cache_pool_hits": hits,
                        "kv_cache_pool_misses": int(summary.get("kv_cache_pool_misses") or 0),
                    },
                    threshold={"kv_cache_pool_hits": 1},
                )
            )

    required_transport = rules.get("require_semantic_transfer_transport")
    if required_transport is not None:
        required_transport = str(required_transport)
        transport_counts = summary.get("semantic_transfer_transport_counts") or {}
        observed_count = int(transport_counts.get(required_transport, 0)) if isinstance(transport_counts, dict) else 0
        if observed_count <= 0:
            alerts.append(
                _alert(
                    "semantic_transfer_transport_missing",
                    rules,
                    default_severity="critical",
                    message=f"semantic transfer transport {required_transport!r} was required but not observed",
                    observed=transport_counts,
                    threshold={"required_transport": required_transport, "min_count": 1},
                )
            )

    required_wire_format = rules.get("require_semantic_transfer_wire_format")
    if required_wire_format is not None:
        required_wire_format = str(required_wire_format)
        format_counts = summary.get("semantic_transfer_wire_format_counts") or {}
        observed_count = (
            int(format_counts.get(required_wire_format, 0))
            if isinstance(format_counts, dict)
            else 0
        )
        if observed_count <= 0:
            alerts.append(
                _alert(
                    "semantic_transfer_wire_format_missing",
                    rules,
                    default_severity="critical",
                    message=(
                        f"semantic transfer wire format {required_wire_format!r} "
                        "was required but not observed"
                    ),
                    observed=format_counts,
                    threshold={"required_wire_format": required_wire_format, "min_count": 1},
                )
            )

    required_weight_transport = rules.get("require_weight_transfer_transport")
    if required_weight_transport is not None:
        required_weight_transport = str(required_weight_transport)
        transport_counts = summary.get("weight_transfer_transport_counts") or {}
        observed_count = int(transport_counts.get(required_weight_transport, 0)) if isinstance(transport_counts, dict) else 0
        if observed_count <= 0:
            alerts.append(
                _alert(
                    "weight_transfer_transport_missing",
                    rules,
                    default_severity="critical",
                    message=f"weight transfer transport {required_weight_transport!r} was required but not observed",
                    observed=transport_counts,
                    threshold={"required_transport": required_weight_transport, "min_count": 1},
                )
            )

    required_layout_switch_transport = rules.get("require_layout_switch_transport")
    if required_layout_switch_transport is not None:
        required_layout_switch_transport = str(required_layout_switch_transport)
        transport_counts = summary.get("layout_switch_transport_counts") or {}
        observed_count = (
            int(transport_counts.get(required_layout_switch_transport, 0))
            if isinstance(transport_counts, dict)
            else 0
        )
        if observed_count <= 0:
            alerts.append(
                _alert(
                    "layout_switch_transport_missing",
                    rules,
                    default_severity="critical",
                    message=(
                        f"layout switch transport {required_layout_switch_transport!r} "
                        "was required but not observed"
                    ),
                    observed=transport_counts,
                    threshold={
                        "required_transport": required_layout_switch_transport,
                        "min_count": 1,
                    },
                )
            )

    required_fused_optimizer = rules.get("require_fused_pipeline_optimizer")
    if required_fused_optimizer is not None:
        required_fused_optimizer = str(required_fused_optimizer)
        optimizer_counts = summary.get("fused_pipeline_optimizer_counts") or {}
        observed_count = (
            int(optimizer_counts.get(required_fused_optimizer, 0))
            if isinstance(optimizer_counts, dict)
            else 0
        )
        unexpected_optimizers = (
            sorted(
                str(name)
                for name, count in optimizer_counts.items()
                if str(name) != required_fused_optimizer and int(count) > 0
            )
            if isinstance(optimizer_counts, dict)
            else []
        )
        if observed_count <= 0 or unexpected_optimizers:
            alerts.append(
                _alert(
                    "fused_pipeline_optimizer_mismatch",
                    rules,
                    default_severity="critical",
                    message=(
                        f"fused pipeline optimizer {required_fused_optimizer!r} "
                        "was required exclusively but observations did not match"
                    ),
                    observed=optimizer_counts,
                    threshold={
                        "required_optimizer": required_fused_optimizer,
                        "min_count": 1,
                        "unexpected_optimizers": unexpected_optimizers,
                    },
                )
            )

    for rule_name, field in (
        ("min_steps", "steps"),
        ("min_loss_count", "loss_observation_count"),
        ("min_finished_nodes", "finished_nodes"),
        ("min_semantic_transfer_count", "semantic_transfer_count"),
        ("min_semantic_transfer_bytes", "semantic_transfer_bytes"),
        ("min_semantic_transfer_dp_repartition_count", "semantic_transfer_dp_repartition_count"),
        ("min_weight_transfer_count", "weight_transfer_count"),
        ("min_weight_transfer_bytes", "weight_transfer_bytes"),
        ("min_weight_transfer_installed_active_rank_count", "weight_transfer_installed_active_rank_count"),
        ("min_weight_transfer_adopted_parameter_bytes", "weight_transfer_adopted_parameter_bytes"),
        (
            "min_weight_transfer_discarded_released_state_rank_count",
            "weight_transfer_discarded_released_state_rank_count",
        ),
        (
            "min_weight_transfer_discarded_released_state_bytes",
            "weight_transfer_discarded_released_state_bytes",
        ),
        ("min_layout_switch_count", "layout_switch_count"),
        ("min_layout_switch_bytes", "layout_switch_bytes"),
        ("min_layout_switch_tensor_native_count", "layout_switch_tensor_native_count"),
        ("min_layout_switch_optimizer_persistent_count", "layout_switch_optimizer_persistent_count"),
        ("min_layout_switch_optimizer_rebound_count", "layout_switch_optimizer_rebound_count"),
        ("min_operational_weight_materialize_count", "operational_weight_materialize_count"),
        ("min_operational_weight_release_count", "operational_weight_release_count"),
        ("min_operational_weight_released_bytes", "operational_weight_released_bytes"),
        (
            "min_operational_weight_optimizer_persistent_release_count",
            "operational_weight_optimizer_persistent_release_count",
        ),
        ("min_checkpoint_weight_files", "checkpoint_weight_files"),
        ("min_checkpoint_weight_bytes", "checkpoint_weight_bytes"),
        ("min_checkpoint_weight_data_files", "checkpoint_weight_data_files"),
        ("min_checkpoint_weight_metadata_files", "checkpoint_weight_metadata_files"),
        ("min_checkpoint_optimizer_state_rank_files", "checkpoint_optimizer_state_rank_files"),
        ("min_checkpoint_portable_train_versions", "checkpoint_portable_train_versions"),
        ("min_checkpoint_portable_train_rank_files", "checkpoint_portable_train_rank_files"),
        ("min_zero2_overlap_train_count", "zero2_overlap_train_count"),
        ("min_zero2_reduce_calls", "zero2_reduce_calls"),
        ("min_zero2_reduced_bytes", "zero2_reduced_bytes"),
        ("min_fusion_cuda_ipc_event_exports", "fusion_cuda_ipc_event_exports"),
        ("min_fusion_cuda_ipc_event_imports", "fusion_cuda_ipc_event_imports"),
        ("min_fusion_cuda_ipc_event_waits", "fusion_cuda_ipc_event_waits"),
        ("min_fusion_cuda_ipc_event_records", "fusion_cuda_ipc_event_records"),
        ("min_generation_count", "generation_count"),
        ("min_kv_cache_generation_count", "kv_cache_generation_count"),
        ("min_kv_cache_pool_hits", "kv_cache_pool_hits"),
        ("min_kv_cache_pool_evictions", "kv_cache_pool_evictions"),
        ("min_dynamic_generation_batch_count", "dynamic_generation_batch_count"),
        ("min_dynamic_generation_batched_request_count", "dynamic_generation_batched_request_count"),
        ("min_dynamic_generation_batched_sequence_count", "dynamic_generation_batched_sequence_count"),
        ("min_dynamic_generation_batched_token_count", "dynamic_generation_batched_token_count"),
        ("min_generation_quantile_phase_count", "generation_quantile_phase_count"),
        (
            "min_generation_quantile_natural_volume_mismatch_count",
            "generation_quantile_natural_volume_mismatch_count",
        ),
        (
            "min_generation_quantile_compacted_sequences",
            "generation_quantile_compacted_sequences",
        ),
        ("min_generation_tokens", "generation_tokens"),
        (
            "min_activation_cuda_cache_enabled_observation_count",
            "activation_cuda_cache_enabled_observation_count",
        ),
        ("min_activation_cuda_cache_placements", "activation_cuda_cache_placements_max"),
        ("min_generation_tokens_per_s", "generation_tokens_per_s"),
        ("min_generation_node_tokens_per_s", "generation_node_tokens_per_s"),
        ("min_ucx_direct_prewarm_pairs", "ucx_direct_prewarm_pairs"),
        ("min_ucx_direct_prewarm_connections", "ucx_direct_prewarm_connections"),
        ("min_ucx_direct_prewarm_rank_duration_s", "ucx_direct_prewarm_rank_duration_s"),
        ("min_ucx_direct_cached_endpoint_count", "ucx_direct_cached_endpoint_count"),
        (
            "min_megatron_process_group_observation_count",
            "megatron_process_group_observation_count",
        ),
        ("min_megatron_process_groups_reused", "megatron_process_groups_reused_max"),
        (
            "min_megatron_pipeline_groups_policy_applied",
            "megatron_pipeline_groups_policy_applied_max",
        ),
        (
            "min_training_inference_cache_transition_rank_count",
            "training_inference_cache_transition_rank_count",
        ),
        (
            "min_training_rotary_lru_entries_cleared",
            "training_rotary_lru_entries_cleared",
        ),
        (
            "min_training_static_kv_pool_entries_retained",
            "training_static_kv_pool_entries_retained",
        ),
    ):
        threshold = _optional_float_rule(rules, rule_name, None)
        if threshold is None:
            continue
        observed = float(summary.get(field) or 0.0)
        if observed < threshold:
            alerts.append(
                _alert(
                    rule_name,
                    rules,
                    default_severity="critical",
                    message=f"{field} below configured minimum",
                    observed=observed,
                    threshold=threshold,
                )
            )

    for model in ("actor", "critic", "ac_share", "reference", "reward"):
        for rule_name, stats_name, field in (
            (f"min_{model}_loaded_weight_count", "loaded_weight_version_stats", "count"),
            (f"min_{model}_produced_weight_count", "produced_weight_version_stats", "count"),
            (f"min_{model}_nonzero_loaded_weight_count", "loaded_weight_version_stats", "nonzero_count"),
            (f"min_{model}_lagging_loaded_weight_count", "loaded_weight_version_stats", "lagging_step_count"),
            (f"min_{model}_loaded_weight_version", "loaded_weight_version_stats", "max"),
            (f"min_{model}_produced_weight_version", "produced_weight_version_stats", "max"),
        ):
            threshold = _optional_float_rule(rules, rule_name, None)
            if threshold is None:
                continue
            observed = _weight_stat_value(summary, stats_name, model, field)
            if observed < threshold:
                alerts.append(
                    _alert(
                        rule_name,
                        rules,
                        default_severity="critical",
                        message=f"{stats_name}.{model}.{field} below configured minimum",
                        observed=observed,
                        threshold=threshold,
                    )
                )

    for rule_name, field in (
        ("max_wall_time_s", "wall_time_s"),
        ("max_total_node_duration_s", "total_node_duration_s"),
        ("max_semantic_transfer_bytes", "semantic_transfer_bytes"),
        ("max_weight_transfer_bytes", "weight_transfer_bytes"),
        ("max_fusion_cpu_completion_polls", "fusion_cpu_completion_polls"),
        ("max_activation_cuda_cache_cpu_fallbacks", "activation_cuda_cache_cpu_fallbacks_max"),
        ("max_generation_quantile_forced_completions", "generation_quantile_forced_completions"),
    ):
        threshold = _optional_float_rule(rules, rule_name, None)
        if threshold is None:
            continue
        observed = float(summary.get(field) or 0.0)
        if observed > threshold:
            alerts.append(
                _alert(
                    rule_name,
                    rules,
                    default_severity="warning",
                    message=f"{field} exceeded configured threshold",
                    observed=observed,
                    threshold=threshold,
                )
            )

    critical_count = sum(1 for alert in alerts if alert.get("severity") == "critical")
    warning_count = sum(1 for alert in alerts if alert.get("severity") == "warning")
    return {
        "schema_version": "rpeg.runtime_alerts.v1",
        "run_id": run_id,
        "algorithm": algorithm,
        "trace_path": summary.get("trace_path"),
        "status": "critical" if critical_count else ("warning" if warning_count else "ok"),
        "alert_count": len(alerts),
        "critical_count": critical_count,
        "warning_count": warning_count,
        "alerts": alerts,
    }


def _weight_stat_value(summary: dict[str, Any], stats_name: str, model: str, field: str) -> float:
    stats = summary.get(stats_name) or {}
    if not isinstance(stats, dict):
        return 0.0
    model_stats = stats.get(model) or {}
    if not isinstance(model_stats, dict):
        return 0.0
    value = model_stats.get(field)
    if value is None:
        return 0.0
    return float(value)


def write_runtime_alerts(
    summary: dict[str, Any],
    alert_path: str | Path,
    *,
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    alert_report = evaluate_runtime_alerts(summary, rules=rules)
    _write_text_atomic(Path(alert_path), json.dumps(alert_report, indent=2, sort_keys=True) + "\n")
    return alert_report


def summary_to_dashboard_html(summary: dict[str, Any]) -> str:
    alerts = [alert for alert in (summary.get("alerts") or []) if isinstance(alert, dict)]
    status = str(summary.get("alert_status") or ("critical" if alerts else "ok"))
    title = f"RPEG Runtime {summary.get('run_id') or ''}".strip()
    sections = [
        _dashboard_section(
            "Run",
            _kv_table(
                {
                    "run_id": summary.get("run_id"),
                    "algorithm": summary.get("algorithm"),
                    "trace_path": summary.get("trace_path"),
                    "alert_status": status,
                    "events": summary.get("events"),
                    "started_nodes": summary.get("started_nodes"),
                    "finished_nodes": summary.get("finished_nodes"),
                    "replay_attempts": summary.get("replay_attempts"),
                    "wall_time_s": summary.get("wall_time_s"),
                    "steps": summary.get("steps"),
                    "output_dir": summary.get("output_dir"),
                }
            ),
        ),
        _dashboard_section("Alerts", _alerts_table(alerts)),
        _dashboard_section(
            "Fused Training Synchronization",
            _kv_table(
                {
                    "fusion_sync_backend_counts": summary.get("fusion_sync_backend_counts"),
                    "fusion_cuda_ipc_event_exports": summary.get("fusion_cuda_ipc_event_exports"),
                    "fusion_cuda_ipc_event_imports": summary.get("fusion_cuda_ipc_event_imports"),
                    "fusion_cuda_ipc_event_waits": summary.get("fusion_cuda_ipc_event_waits"),
                    "fusion_cuda_ipc_event_records": summary.get("fusion_cuda_ipc_event_records"),
                    "fusion_cpu_handle_polls": summary.get("fusion_cpu_handle_polls"),
                    "fusion_cpu_completion_polls": summary.get("fusion_cpu_completion_polls"),
                    "fusion_pipeline_async_sends": summary.get(
                        "fusion_pipeline_async_sends"
                    ),
                    "fusion_pipeline_async_send_waits": summary.get(
                        "fusion_pipeline_async_send_waits"
                    ),
                    "fusion_pipeline_async_recvs": summary.get(
                        "fusion_pipeline_async_recvs"
                    ),
                    "fusion_pipeline_async_recv_waits": summary.get(
                        "fusion_pipeline_async_recv_waits"
                    ),
                    "fused_pipeline_optimizer_counts": summary.get(
                        "fused_pipeline_optimizer_counts"
                    ),
                    "fused_pipeline_optimizer_candidates_max": summary.get(
                        "fused_pipeline_optimizer_candidates_max"
                    ),
                    "fused_pipeline_makespan_units_min": summary.get(
                        "fused_pipeline_makespan_units_min"
                    ),
                    "fused_pipeline_baseline_makespan_units_min": summary.get(
                        "fused_pipeline_baseline_makespan_units_min"
                    ),
                    "fused_pipeline_predicted_improvement_fraction": summary.get(
                        "fused_pipeline_predicted_improvement_fraction"
                    ),
                }
            ),
        ),
        _dashboard_section(
            "Generation",
            _kv_table(
                {
                    "generation_count": summary.get("generation_count"),
                    "generation_backend_counts": summary.get("generation_backend_counts"),
                    "kv_cache_generation_count": summary.get("kv_cache_generation_count"),
                    "kv_cache_pool_entries": summary.get("kv_cache_pool_entries"),
                    "kv_cache_pool_hits": summary.get("kv_cache_pool_hits"),
                    "kv_cache_pool_misses": summary.get("kv_cache_pool_misses"),
                    "kv_cache_pool_evictions": summary.get("kv_cache_pool_evictions"),
                    "kv_cache_max_batch_size": summary.get("kv_cache_max_batch_size"),
                    "kv_cache_max_sequence_length": summary.get("kv_cache_max_sequence_length"),
                    "generation_tokens": summary.get("generation_tokens"),
                    "generation_sequences": summary.get("generation_sequences"),
                    "generation_node_duration_s": summary.get("generation_node_duration_s"),
                    "generation_tokens_per_s": summary.get("generation_tokens_per_s"),
                    "generation_node_tokens_per_s": summary.get("generation_node_tokens_per_s"),
                    "generation_prefill_forward_steps": summary.get("generation_prefill_forward_steps"),
                    "generation_decode_forward_steps": summary.get("generation_decode_forward_steps"),
                    "generation_forward_steps": summary.get("generation_forward_steps"),
                    "generation_old_logps_from_generation_count": summary.get("generation_old_logps_from_generation_count"),
                    "generation_old_logp_forward_steps": summary.get("generation_old_logp_forward_steps"),
                    "dynamic_generation_batch_count": summary.get("dynamic_generation_batch_count"),
                    "dynamic_generation_batched_request_count": summary.get("dynamic_generation_batched_request_count"),
                    "dynamic_generation_batched_sequence_count": summary.get("dynamic_generation_batched_sequence_count"),
                    "dynamic_generation_batched_token_count": summary.get("dynamic_generation_batched_token_count"),
                }
            ),
        ),
        _dashboard_section(
            "Transfers",
            _kv_table(
                {
                    "semantic_transfer_count": summary.get("semantic_transfer_count"),
                    "semantic_transfer_bytes": summary.get("semantic_transfer_bytes"),
                    "semantic_transfer_rank_duration_s": summary.get("semantic_transfer_rank_duration_s"),
                    "semantic_transfer_rank_bytes_per_s": summary.get("semantic_transfer_rank_bytes_per_s"),
                    "semantic_transfer_dp_repartition_count": summary.get("semantic_transfer_dp_repartition_count"),
                    "semantic_transfer_transport_counts": summary.get("semantic_transfer_transport_counts"),
                    "semantic_transfer_wire_format_counts": summary.get("semantic_transfer_wire_format_counts"),
                    "semantic_transfer_transport_bytes": summary.get("semantic_transfer_transport_bytes"),
                    "semantic_transfer_transport_rank_duration_s": summary.get(
                        "semantic_transfer_transport_rank_duration_s"
                    ),
                    "semantic_transfer_transport_rank_bytes_per_s": summary.get(
                        "semantic_transfer_transport_rank_bytes_per_s"
                    ),
                    "weight_transfer_count": summary.get("weight_transfer_count"),
                    "weight_transfer_bytes": summary.get("weight_transfer_bytes"),
                    "weight_transfer_rank_duration_s": summary.get("weight_transfer_rank_duration_s"),
                    "weight_transfer_rank_bytes_per_s": summary.get("weight_transfer_rank_bytes_per_s"),
                    "weight_transfer_destination_dp_relay_count": summary.get(
                        "weight_transfer_destination_dp_relay_count"
                    ),
                    "weight_transfer_relay_direct_bytes": summary.get(
                        "weight_transfer_relay_direct_bytes"
                    ),
                    "weight_transfer_relay_direct_network_bytes": summary.get(
                        "weight_transfer_relay_direct_network_bytes"
                    ),
                    "weight_transfer_relay_broadcast_bytes": summary.get(
                        "weight_transfer_relay_broadcast_bytes"
                    ),
                    "weight_transfer_transport_counts": summary.get("weight_transfer_transport_counts"),
                    "weight_transfer_transport_bytes": summary.get("weight_transfer_transport_bytes"),
                    "weight_transfer_transport_rank_duration_s": summary.get(
                        "weight_transfer_transport_rank_duration_s"
                    ),
                    "weight_transfer_transport_rank_bytes_per_s": summary.get(
                        "weight_transfer_transport_rank_bytes_per_s"
                    ),
                    "loaded_weight_count_by_model": summary.get("loaded_weight_count_by_model"),
                    "produced_weight_count_by_model": summary.get("produced_weight_count_by_model"),
                    "nonzero_loaded_weight_count_by_model": summary.get("nonzero_loaded_weight_count_by_model"),
                    "lagging_loaded_weight_count_by_model": summary.get("lagging_loaded_weight_count_by_model"),
                    "max_loaded_weight_version_by_model": summary.get("max_loaded_weight_version_by_model"),
                    "max_produced_weight_version_by_model": summary.get("max_produced_weight_version_by_model"),
                    "checkpoint_recovery_count": summary.get("checkpoint_recovery_count"),
                    "checkpoint_weight_files": summary.get("checkpoint_weight_files"),
                    "checkpoint_weight_bytes": summary.get("checkpoint_weight_bytes"),
                    "checkpoint_weight_data_files": summary.get("checkpoint_weight_data_files"),
                    "checkpoint_weight_metadata_files": summary.get("checkpoint_weight_metadata_files"),
                    "checkpoint_optimizer_state_rank_files": summary.get("checkpoint_optimizer_state_rank_files"),
                    "checkpoint_portable_train_versions": summary.get("checkpoint_portable_train_versions"),
                    "checkpoint_portable_train_rank_files": summary.get("checkpoint_portable_train_rank_files"),
                    "checkpoint_restored_weight_files": summary.get("checkpoint_restored_weight_files"),
                    "checkpoint_restored_weight_bytes": summary.get("checkpoint_restored_weight_bytes"),
                }
            ),
        ),
        _dashboard_section("Node Status", _mapping_table(summary.get("status_counts") or {}, "status", "count")),
        _dashboard_section("Node Kinds", _mapping_table(summary.get("kind_counts") or {}, "kind", "count")),
        _dashboard_section("Latest Losses", _mapping_table(summary.get("latest_losses") or {}, "metric", "value")),
        _dashboard_section("Failed Nodes", _failed_nodes_table(summary.get("failed_nodes") or [])),
    ]
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{escape(title)}</title>",
            "<style>",
            _DASHBOARD_CSS,
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            f"<h1>{escape(title)}</h1>",
            f'<p class="status status-{escape(_css_token(status))}">alert status: {escape(status)}</p>',
            *sections,
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )


def write_runtime_dashboard(summary: dict[str, Any], dashboard_path: str | Path) -> None:
    _write_text_atomic(Path(dashboard_path), summary_to_dashboard_html(summary))


def write_prometheus_metrics(summary: dict[str, Any], metrics_path: str | Path) -> None:
    _write_text_atomic(Path(metrics_path), summary_to_prometheus(summary))


def write_runtime_summary(
    trace_path: str | Path,
    summary_path: str | Path,
    *,
    run_id: str | None = None,
    extra: dict[str, Any] | None = None,
    metrics_path: str | Path | None = None,
    alert_path: str | Path | None = None,
    alert_rules: dict[str, Any] | None = None,
    dashboard_path: str | Path | None = None,
) -> dict[str, Any]:
    trace_path = Path(trace_path)
    summary = summarize_trace(load_trace_events(trace_path), run_id=run_id, trace_path=str(trace_path))
    if extra:
        summary.update(extra)
    if alert_path is not None or dashboard_path is not None:
        alert_report = evaluate_runtime_alerts(summary, rules=alert_rules)
        if alert_path is not None:
            _write_text_atomic(Path(alert_path), json.dumps(alert_report, indent=2, sort_keys=True) + "\n")
        summary["alert_status"] = alert_report["status"]
        summary["alerts"] = alert_report["alerts"]
    out = Path(summary_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(out, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if metrics_path is not None:
        write_prometheus_metrics(summary, metrics_path)
    if dashboard_path is not None:
        write_runtime_dashboard(summary, dashboard_path)
    return summary


def write_runtime_summary_for_run(run_config: Any, *, extra: dict[str, Any] | None = None) -> dict[str, Any] | None:
    monitoring = getattr(run_config, "monitoring", {}) or {}
    summary_path = monitoring.get("summary_path")
    metrics_path = monitoring.get("metrics_path")
    alert_path = monitoring.get("alert_path")
    alert_rules = monitoring.get("alert_rules")
    dashboard_path = monitoring.get("dashboard_path")
    if not summary_path and not metrics_path and not alert_path and not dashboard_path:
        return None
    trace_path = getattr(run_config.runtime, "trace_path")
    summary = summarize_trace(load_trace_events(trace_path), run_id=getattr(run_config, "run_id", None), trace_path=str(trace_path))
    if extra:
        summary.update(extra)
    if alert_path or dashboard_path:
        alert_report = evaluate_runtime_alerts(summary, rules=alert_rules)
        if alert_path:
            _write_text_atomic(Path(alert_path), json.dumps(alert_report, indent=2, sort_keys=True) + "\n")
        summary["alert_status"] = alert_report["status"]
        summary["alerts"] = alert_report["alerts"]
    if summary_path:
        _write_text_atomic(Path(summary_path), json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if metrics_path:
        write_prometheus_metrics(summary, metrics_path)
    if dashboard_path:
        write_runtime_dashboard(summary, dashboard_path)
    return summary


def _int_counter(counters: dict[str, Any], key: str) -> int:
    value = counters.get(key)
    if value is None:
        return 0
    return int(value)


def _metric(name: str, value: Any, labels: dict[str, str]) -> str:
    return f"{name}{_labels(labels)} {_number(value)}"


def _labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    rendered = ",".join(f'{key}="{_escape_label(str(value))}"' for key, value in sorted(labels.items()))
    return "{" + rendered + "}"


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _number(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


def _metric_suffix(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", value).lower()


def _alert(
    rule: str,
    rules: dict[str, Any],
    *,
    default_severity: str,
    message: str,
    observed: Any,
    threshold: Any,
) -> dict[str, Any]:
    severity_rules = rules.get("severity") if isinstance(rules.get("severity"), dict) else {}
    severity = str(severity_rules.get(rule, default_severity)).lower()
    if severity not in {"warning", "critical"}:
        severity = default_severity
    return {
        "rule": rule,
        "severity": severity,
        "message": message,
        "observed": observed,
        "threshold": threshold,
    }


def _optional_int_rule(rules: dict[str, Any], key: str, default: int | None) -> int | None:
    value = rules.get(key, default)
    if value is None:
        return None
    return int(value)


def _optional_float_rule(rules: dict[str, Any], key: str, default: float | None) -> float | None:
    value = rules.get(key, default)
    if value is None:
        return None
    return float(value)


def _bool_rule(rules: dict[str, Any], key: str, default: bool) -> bool:
    value = rules.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _dashboard_section(title: str, body: str) -> str:
    return f"<section><h2>{escape(title)}</h2>{body}</section>"


def _kv_table(values: dict[str, Any]) -> str:
    rows = [
        f"<tr><th>{escape(str(key))}</th><td>{escape(_display_value(value))}</td></tr>"
        for key, value in values.items()
        if value is not None
    ]
    return _table(("<th>field</th><th>value</th>",), rows)


def _mapping_table(values: dict[str, Any], key_name: str, value_name: str) -> str:
    rows = [
        f"<tr><td>{escape(str(key))}</td><td>{escape(_display_value(value))}</td></tr>"
        for key, value in sorted(values.items())
    ]
    return _table((f"<th>{escape(key_name)}</th><th>{escape(value_name)}</th>",), rows)


def _alerts_table(alerts: list[dict[str, Any]]) -> str:
    if not alerts:
        return '<p class="empty">No runtime alerts.</p>'
    rows = []
    for alert in alerts:
        rows.append(
            "<tr>"
            f"<td>{escape(str(alert.get('severity') or 'warning'))}</td>"
            f"<td>{escape(str(alert.get('rule') or 'unknown'))}</td>"
            f"<td>{escape(str(alert.get('message') or ''))}</td>"
            f"<td>{escape(_display_value(alert.get('observed')))}</td>"
            f"<td>{escape(_display_value(alert.get('threshold')))}</td>"
            "</tr>"
        )
    return _table(("<th>severity</th><th>rule</th><th>message</th><th>observed</th><th>threshold</th>",), rows)


def _failed_nodes_table(nodes: list[Any]) -> str:
    rows = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{escape(str(node.get('node_id') or ''))}</td>"
            f"<td>{escape(str(node.get('worker_id') or ''))}</td>"
            f"<td>{escape(str(node.get('message') or ''))}</td>"
            "</tr>"
        )
    if not rows:
        return '<p class="empty">No failed runtime nodes.</p>'
    return _table(("<th>node_id</th><th>worker_id</th><th>message</th>",), rows)


def _table(headers: tuple[str, ...], rows: list[str]) -> str:
    if not rows:
        return '<p class="empty">No data.</p>'
    header = "".join(headers)
    return "<table><thead><tr>" + header + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _display_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return "" if value is None else str(value)


def _css_token(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "-", value).lower()


_DASHBOARD_CSS = """
:root { color-scheme: light dark; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
body { margin: 0; background: #f6f7f9; color: #17202a; }
main { max-width: 1120px; margin: 0 auto; padding: 28px 20px 48px; }
h1 { margin: 0 0 8px; font-size: 28px; font-weight: 650; }
h2 { margin: 0 0 12px; font-size: 17px; font-weight: 650; }
section { margin-top: 18px; padding-top: 4px; }
table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d8dee6; }
th, td { padding: 8px 10px; border-bottom: 1px solid #e7ebf0; text-align: left; vertical-align: top; font-size: 13px; }
th { background: #eef2f6; font-weight: 600; }
.status { display: inline-block; margin: 4px 0 8px; padding: 4px 8px; border: 1px solid #99a3ad; background: #fff; font-size: 13px; }
.status-ok { border-color: #2e7d32; color: #1b5e20; }
.status-warning { border-color: #ad6f00; color: #8a5400; }
.status-critical { border-color: #b3261e; color: #8c1d18; }
.empty { margin: 0; color: #66717d; font-size: 13px; }
@media (prefers-color-scheme: dark) {
  body { background: #101418; color: #e7eaee; }
  table { background: #171d23; border-color: #303942; }
  th, td { border-bottom-color: #2b343d; }
  th { background: #202832; }
  .status { background: #171d23; }
}
""".strip()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)
