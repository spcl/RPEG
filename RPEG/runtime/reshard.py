# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

from typing import Any

from runtime.errors import RuntimeConfigError


def tp_shard_dim(name: str, tensor_ndim: int) -> int | None:
    dim0_suffixes = (
        "embedding.word_embeddings.weight",
        "output_layer.weight",
        "self_attention.linear_qkv.weight",
        "self_attention.linear_qkv.bias",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc1.bias",
    )
    dim1_suffixes = (
        "self_attention.linear_proj.weight",
        "mlp.linear_fc2.weight",
    )
    if name.endswith(dim0_suffixes):
        return 0
    if name.endswith(dim1_suffixes):
        if tensor_ndim < 2:
            raise RuntimeConfigError(f"tensor {name!r} cannot be TP-sharded on dim=1 with rank {tensor_ndim}")
        return 1
    return None


def rank_to_dp_tp(rank: int, tp: int) -> tuple[int, int]:
    if tp <= 0:
        raise RuntimeConfigError(f"invalid tensor-parallel size {tp}")
    return rank // tp, rank % tp


def dp_tp_to_rank(dp_rank: int, tp_rank: int, tp: int) -> int:
    return dp_rank * tp + tp_rank


def rank_to_dp_pp_tp(rank: int, pp: int, tp: int) -> tuple[int, int, int]:
    if pp <= 0 or tp <= 0:
        raise RuntimeConfigError(f"invalid model parallel sizes pp={pp}, tp={tp}")
    dp_rank = rank // (pp * tp)
    rem = rank % (pp * tp)
    return dp_rank, rem // tp, rem % tp


def dp_pp_tp_to_rank(dp_rank: int, pp_rank: int, tp_rank: int, pp: int, tp: int) -> int:
    return (dp_rank * pp + pp_rank) * tp + tp_rank


def reshard_state_dict(
    source_states: list[dict[str, Any]],
    *,
    source_parallelism: tuple[int, int, int],
    destination_parallelism: tuple[int, int, int],
    destination_rank: int,
    source_name_maps: list[dict[str, str]] | None = None,
    destination_name_map: dict[str, str] | None = None,
    source_shard_specs: list[dict[str, dict[str, Any]]] | None = None,
    destination_shard_specs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    src_dp, src_pp, src_tp = source_parallelism
    dst_dp, dst_pp, dst_tp = destination_parallelism
    expected_source_world = src_dp * src_pp * src_tp
    expected_destination_world = dst_dp * dst_pp * dst_tp
    if len(source_states) != expected_source_world:
        raise RuntimeConfigError(
            f"source state count {len(source_states)} does not match source world size {expected_source_world}"
        )
    if destination_rank < 0 or destination_rank >= expected_destination_world:
        raise RuntimeConfigError(
            f"destination rank {destination_rank} outside destination world size {expected_destination_world}"
        )

    _dst_dp_rank, dst_pp_rank, dst_tp_rank = rank_to_dp_pp_tp(destination_rank, dst_pp, dst_tp)
    source_name_maps = _normalize_name_maps(source_states, source_name_maps)
    source_shard_specs = _normalize_shard_specs(source_states, source_shard_specs)
    destination_shard_specs = destination_shard_specs or {}
    destination_name_map = destination_name_map or {name: name for name in _destination_reference_names(source_states, source_name_maps, dst_pp_rank)}
    output = {}
    for destination_name, global_name in destination_name_map.items():
        source_shards = _source_values_for_global_name(
            source_states,
            source_name_maps,
            global_name,
            source_pp=src_pp,
            source_tp=src_tp,
            source_dp=src_dp,
        )
        first_tensor = source_shards[0]
        source_specs = _source_specs_for_global_name(
            source_shard_specs,
            source_name_maps,
            global_name,
            source_pp=src_pp,
            source_tp=src_tp,
        )
        source_spec = _consistent_shard_spec(global_name, source_specs)
        destination_spec = dict(destination_shard_specs.get(destination_name) or {})
        source_is_tp = bool(source_spec.get("is_tp", False))
        destination_is_tp = bool(destination_spec.get("is_tp", False))
        source_dim = source_spec.get("dim")
        destination_dim = destination_spec.get("dim")
        shard_dim = destination_dim if destination_is_tp else source_dim if source_is_tp else None
        if shard_dim is None:
            shard_dim = tp_shard_dim(global_name, int(getattr(first_tensor, "ndim", 0)))
            source_is_tp = source_is_tp or (shard_dim is not None and src_tp > 1)
            destination_is_tp = destination_is_tp or (shard_dim is not None and dst_tp > 1)
        if source_is_tp and destination_is_tp and source_dim is not None and destination_dim is not None:
            if int(source_dim) != int(destination_dim):
                raise RuntimeConfigError(
                    f"tensor {global_name!r} changes TP partition dimension from {source_dim} to {destination_dim}"
                )
        if not source_is_tp and not destination_is_tp:
            if not _all_equal(source_shards):
                raise RuntimeConfigError(
                    f"tensor {global_name!r} is not a known TP-sharded tensor and differs across TP ranks"
                )
            output[destination_name] = _clone_state_value(source_shards[0])
            continue
        if shard_dim is None:
            raise RuntimeConfigError(f"tensor {global_name!r} is TP-sharded but has no partition dimension")
        if source_is_tp:
            full = _gather_strided_tp(
                source_shards,
                dim=int(shard_dim),
                stride=int(source_spec.get("stride", 1)),
                name=global_name,
            )
        else:
            if not _all_equal(source_shards):
                raise RuntimeConfigError(f"replicated tensor {global_name!r} differs across source TP ranks")
            full = _clone_state_value(source_shards[0])
        if destination_is_tp:
            output[destination_name] = _shard_strided_tp(
                full,
                world_size=dst_tp,
                rank=dst_tp_rank,
                dim=int(shard_dim),
                stride=int(destination_spec.get("stride", 1)),
                name=global_name,
            )
        else:
            output[destination_name] = full.detach().clone()
    return output


def routed_weight_source_tp_ranks(
    source_parallelism: tuple[int, int, int],
    destination_parallelism: tuple[int, int, int],
    destination_rank: int,
) -> tuple[int, ...]:
    """
    TP source ranks whose shards intersect one destination rank.
    """
    src_dp, src_pp, src_tp = source_parallelism
    dst_dp, dst_pp, dst_tp = destination_parallelism
    if min(src_dp, src_pp, src_tp, dst_dp, dst_pp, dst_tp) <= 0:
        raise RuntimeConfigError(
            f"invalid routed weight parallelism {source_parallelism}->{destination_parallelism}"
        )
    if destination_rank < 0 or destination_rank >= dst_dp * dst_pp * dst_tp:
        raise RuntimeConfigError(
            f"destination rank {destination_rank} is outside parallelism {destination_parallelism}"
        )
    _dst_dp_rank, _dst_pp_rank, dst_tp_rank = rank_to_dp_pp_tp(
        destination_rank,
        dst_pp,
        dst_tp,
    )
    if src_tp % dst_tp == 0:
        width = src_tp // dst_tp
        return tuple(range(dst_tp_rank * width, (dst_tp_rank + 1) * width))
    if dst_tp % src_tp == 0:
        return (dst_tp_rank // (dst_tp // src_tp),)
    return tuple(range(src_tp))


def routed_weight_replicated_source_tp_rank(
    source_parallelism: tuple[int, int, int],
    destination_parallelism: tuple[int, int, int],
    destination_rank: int,
) -> int:
    """
    Choose an intersecting TP source for replicated tensors.
    """
    return routed_weight_source_tp_ranks(
        source_parallelism,
        destination_parallelism,
        destination_rank,
    )[0]


def route_state_dict_fragment(
    source_state: dict[str, Any],
    *,
    source_parallelism: tuple[int, int, int],
    destination_parallelism: tuple[int, int, int],
    source_rank: int,
    destination_rank: int,
    source_name_map: dict[str, str] | None = None,
    source_shard_specs: dict[str, dict[str, Any]] | None = None,
    num_layers: int | None = None,
    replicated_source_tp_rank: int = 0,
    active_source_dp_rank: int = 0,
    clone_values: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Build the source-rank contribution needed by one destination rank.

    The fragment is keyed by global parameter name. Only the selected source DP replica contributes.
    Compatible TP degree changes route only the source shards that intersect the destination;
    non-divisible changes forward all TP shards for exact reconstruction.
    """
    src_dp, src_pp, src_tp = source_parallelism
    dst_dp, dst_pp, dst_tp = destination_parallelism
    if source_rank < 0 or source_rank >= src_dp * src_pp * src_tp:
        raise RuntimeConfigError(f"source rank {source_rank} is outside parallelism {source_parallelism}")
    if destination_rank < 0 or destination_rank >= dst_dp * dst_pp * dst_tp:
        raise RuntimeConfigError(
            f"destination rank {destination_rank} is outside parallelism {destination_parallelism}"
        )
    src_dp_rank, _src_pp_rank, src_tp_rank = rank_to_dp_pp_tp(source_rank, src_pp, src_tp)
    _dst_dp_rank, dst_pp_rank, dst_tp_rank = rank_to_dp_pp_tp(destination_rank, dst_pp, dst_tp)
    if not 0 <= int(active_source_dp_rank) < src_dp:
        raise RuntimeConfigError(
            f"active source DP rank {active_source_dp_rank} is outside DP size {src_dp}"
        )
    if src_dp_rank != int(active_source_dp_rank):
        return {}
    if not 0 <= int(replicated_source_tp_rank) < src_tp:
        raise RuntimeConfigError(
            f"replicated source TP rank {replicated_source_tp_rank} is outside TP size {src_tp}"
        )
    source_name_map = source_name_map or {name: name for name in source_state}
    source_shard_specs = source_shard_specs or {}
    out: dict[str, dict[str, Any]] = {}
    for local_name, value in source_state.items():
        global_name = source_name_map.get(local_name, local_name)
        if not _global_tensor_on_pipeline_stage(global_name, dst_pp_rank, dst_pp, num_layers):
            continue
        spec = dict(source_shard_specs.get(local_name) or {})
        source_is_tp = bool(spec.get("is_tp", False))
        shard_dim = spec.get("dim")
        if shard_dim is None:
            shard_dim = tp_shard_dim(global_name, int(getattr(value, "ndim", 0)))
        source_is_tp = source_is_tp or (shard_dim is not None and src_tp > 1)
        destination_is_tp = shard_dim is not None and dst_tp > 1
        stride = int(spec.get("stride", 1))

        if not source_is_tp:
            if src_tp_rank != int(replicated_source_tp_rank):
                continue
            if destination_is_tp:
                routed_value = _shard_strided_tp(
                    value,
                    world_size=dst_tp,
                    rank=dst_tp_rank,
                    dim=int(shard_dim),
                    stride=stride,
                    name=global_name,
                )
            else:
                routed_value = _maybe_clone_state_value(value, clone=clone_values)
            out[global_name] = {
                "value": routed_value,
                "mode": "complete",
                "order": 0,
                "stride": stride,
                "dim": shard_dim,
            }
            continue

        if not destination_is_tp:
            out[global_name] = {
                "value": _maybe_clone_state_value(value, clone=clone_values),
                "mode": "gather",
                "order": src_tp_rank,
                "stride": stride,
                "dim": shard_dim,
            }
            continue

        if src_tp % dst_tp == 0:
            source_ranks_per_destination = src_tp // dst_tp
            if src_tp_rank // source_ranks_per_destination != dst_tp_rank:
                continue
            out[global_name] = {
                "value": _maybe_clone_state_value(value, clone=clone_values),
                "mode": "complete" if source_ranks_per_destination == 1 else "gather",
                "order": src_tp_rank % source_ranks_per_destination,
                "stride": stride,
                "dim": shard_dim,
            }
            continue

        if dst_tp % src_tp == 0:
            destinations_per_source = dst_tp // src_tp
            first_destination = src_tp_rank * destinations_per_source
            if not first_destination <= dst_tp_rank < first_destination + destinations_per_source:
                continue
            out[global_name] = {
                "value": _shard_strided_tp(
                    value,
                    world_size=destinations_per_source,
                    rank=dst_tp_rank - first_destination,
                    dim=int(shard_dim),
                    stride=stride,
                    name=global_name,
                ),
                "mode": "complete",
                "order": 0,
                "stride": stride,
                "dim": shard_dim,
            }
            continue

        out[global_name] = {
            "value": _maybe_clone_state_value(value, clone=clone_values),
            "mode": "fallback",
            "order": src_tp_rank,
            "stride": stride,
            "dim": shard_dim,
        }
    return out


def _global_tensor_on_pipeline_stage(
    global_name: str,
    destination_pp_rank: int,
    destination_pp_size: int,
    num_layers: int | None,
) -> bool:
    if destination_pp_size <= 1 or num_layers is None or num_layers <= 0:
        return True
    marker = ".layers."
    if marker in global_name:
        suffix = global_name.split(marker, 1)[1]
        layer_text = suffix.split(".", 1)[0]
        if layer_text.isdigit():
            layer = int(layer_text)
            return min(destination_pp_size - 1, layer * destination_pp_size // num_layers) == destination_pp_rank
    if global_name.startswith("embedding.") or ".embedding." in global_name:
        return destination_pp_rank == 0
    if (
        global_name.startswith("output_layer.")
        or ".output_layer." in global_name
        or global_name.startswith("decoder.final_layernorm.")
        or global_name.startswith("rpeg_")
    ):
        return destination_pp_rank == destination_pp_size - 1
    return True


def merge_state_dict_fragments(
    fragments: list[dict[str, dict[str, Any]]],
    *,
    destination_name_map: dict[str, str],
    destination_shard_specs: dict[str, dict[str, Any]],
    source_parallelism: tuple[int, int, int],
    destination_parallelism: tuple[int, int, int],
    destination_rank: int,
    clone_complete_values: bool = True,
    consume_inputs: bool = False,
) -> dict[str, Any]:
    """
    Merge routed source contributions into one destination state dict.

    consume_inputs transfers ownership from temporary routed fragments to the returned state.
    Clearing each contribution immediately after its destination tensor is assembled bounds GPU
    memory at roughly one shard instead of retaining a full source-fragment shard alongside a
    growing destination shard. Callers must not reuse consumed fragments.
    """
    src_tp = source_parallelism[2]
    dst_pp, dst_tp = destination_parallelism[1], destination_parallelism[2]
    _dst_dp_rank, _dst_pp_rank, dst_tp_rank = rank_to_dp_pp_tp(destination_rank, dst_pp, dst_tp)
    by_global_name: dict[str, list[dict[str, Any]]] = {}
    for fragment in fragments:
        for global_name, contribution in fragment.items():
            by_global_name.setdefault(global_name, []).append(contribution)
    output: dict[str, Any] = {}
    for destination_name, global_name in destination_name_map.items():
        contributions = by_global_name.get(global_name, [])
        if not contributions:
            raise RuntimeConfigError(f"routed reshard is missing tensor {global_name!r}")
        modes = {str(item["mode"]) for item in contributions}
        if modes == {"complete"}:
            values = [item["value"] for item in contributions]
            if not _all_equal(values):
                raise RuntimeConfigError(f"complete routed tensor {global_name!r} differs across contributions")
            output[destination_name] = _maybe_clone_state_value(
                values[0],
                clone=clone_complete_values,
            )
            if consume_inputs:
                _consume_fragment_contributions(contributions)
            continue
        if modes == {"gather"}:
            ordered = sorted(contributions, key=lambda item: int(item["order"]))
            first = ordered[0]
            output[destination_name] = _gather_strided_tp(
                [item["value"] for item in ordered],
                dim=int(first["dim"]),
                stride=int(first.get("stride", 1)),
                name=global_name,
            )
            if consume_inputs:
                _consume_fragment_contributions(contributions)
            continue
        if modes == {"fallback"}:
            ordered = sorted(contributions, key=lambda item: int(item["order"]))
            first = ordered[0]
            full = _gather_strided_tp(
                [item["value"] for item in ordered],
                dim=int(first["dim"]),
                stride=int(first.get("stride", 1)),
                name=global_name,
            )
            destination_spec = dict(destination_shard_specs.get(destination_name) or {})
            destination_is_tp = bool(destination_spec.get("is_tp", False)) or dst_tp > 1
            output[destination_name] = (
                full.detach()
                if consume_inputs and dst_tp == 1
                else _shard_strided_tp(
                    full,
                    world_size=dst_tp,
                    rank=dst_tp_rank,
                    dim=int(destination_spec.get("dim", first["dim"])),
                    stride=int(destination_spec.get("stride", first.get("stride", 1))),
                    name=global_name,
                )
                if destination_is_tp
                else full
            )
            if consume_inputs:
                _consume_fragment_contributions(contributions)
            continue
        raise RuntimeConfigError(f"routed tensor {global_name!r} has incompatible contribution modes {modes}")
    return output


def _consume_fragment_contributions(contributions: list[dict[str, Any]]) -> None:
    """
    Release tensor references after ownership has moved to merged output.
    """
    for contribution in contributions:
        contribution["value"] = None


def reshard_optimizer_state_by_name(
    source_optimizer_states: list[dict[str, Any]],
    *,
    source_parallelism: tuple[int, int, int],
    destination_parallelism: tuple[int, int, int],
    destination_rank: int,
    source_name_maps: list[dict[str, str]] | None = None,
    destination_name_map: dict[str, str] | None = None,
    source_shard_specs: list[dict[str, dict[str, Any]]] | None = None,
    destination_shard_specs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    src_dp, src_pp, src_tp = source_parallelism
    dst_dp, dst_pp, dst_tp = destination_parallelism
    expected_source_world = src_dp * src_pp * src_tp
    expected_destination_world = dst_dp * dst_pp * dst_tp
    if len(source_optimizer_states) != expected_source_world:
        raise RuntimeConfigError(
            f"source optimizer state count {len(source_optimizer_states)} does not match source world size {expected_source_world}"
        )
    if destination_rank < 0 or destination_rank >= expected_destination_world:
        raise RuntimeConfigError(
            f"destination rank {destination_rank} outside destination world size {expected_destination_world}"
        )
    _dst_dp_rank, dst_pp_rank, dst_tp_rank = rank_to_dp_pp_tp(destination_rank, dst_pp, dst_tp)
    source_name_maps = _normalize_name_maps(source_optimizer_states, source_name_maps)
    source_shard_specs = _normalize_shard_specs(source_optimizer_states, source_shard_specs)
    destination_shard_specs = destination_shard_specs or {}
    destination_name_map = destination_name_map or {
        name: name for name in _destination_optimizer_reference_names(source_name_maps, dst_pp_rank)
    }
    output = {}
    for destination_name, global_name in destination_name_map.items():
        source_entries = _source_optimizer_entries_for_global_name(
            source_optimizer_states,
            source_name_maps,
            global_name,
            source_dp=src_dp,
            source_pp=src_pp,
            source_tp=src_tp,
        )
        source_specs = _source_optimizer_specs_for_global_name(
            source_shard_specs,
            source_name_maps,
            global_name,
            source_dp=src_dp,
            source_pp=src_pp,
            source_tp=src_tp,
        )
        output[destination_name] = _reshard_optimizer_entry(
            global_name,
            source_entries,
            source_tp=src_tp,
            destination_tp=dst_tp,
            destination_tp_rank=dst_tp_rank,
            source_spec=_consistent_shard_spec(global_name, source_specs),
            destination_spec=dict(destination_shard_specs.get(destination_name) or {}),
        )
    return output


def _normalize_name_maps(states: list[dict[str, Any]], maps: list[dict[str, str]] | None) -> list[dict[str, str]]:
    if maps is None:
        return [{name: name for name in state} for state in states]
    if len(maps) != len(states):
        raise RuntimeConfigError(f"name map count {len(maps)} does not match state count {len(states)}")
    return [dict(item) for item in maps]


def _normalize_shard_specs(
    states: list[dict[str, Any]],
    specs: list[dict[str, dict[str, Any]]] | None,
) -> list[dict[str, dict[str, Any]]]:
    if specs is None:
        return [{} for _ in states]
    if len(specs) != len(states):
        raise RuntimeConfigError(f"shard spec count {len(specs)} does not match state count {len(states)}")
    return [{name: dict(spec) for name, spec in rank_specs.items()} for rank_specs in specs]


def _destination_reference_names(
    source_states: list[dict[str, Any]],
    source_name_maps: list[dict[str, str]],
    destination_pp_rank: int,
) -> tuple[str, ...]:
    del destination_pp_rank
    if not source_states:
        return ()
    return tuple(source_name_maps[0].values())


def _destination_optimizer_reference_names(
    source_name_maps: list[dict[str, str]],
    destination_pp_rank: int,
) -> tuple[str, ...]:
    del destination_pp_rank
    seen = set()
    out = []
    for name_map in source_name_maps:
        for global_name in name_map.values():
            if global_name in seen:
                continue
            seen.add(global_name)
            out.append(global_name)
    return tuple(out)


def _source_values_for_global_name(
    source_states: list[dict[str, Any]],
    source_name_maps: list[dict[str, str]],
    global_name: str,
    *,
    source_dp: int,
    source_pp: int,
    source_tp: int,
) -> list[Any]:
    source_pp_rank = _source_pp_rank_for_global_name(source_name_maps, global_name, source_pp, source_tp)
    values = []
    for source_tp_rank in range(source_tp):
        rank = dp_pp_tp_to_rank(0, source_pp_rank, source_tp_rank, source_pp, source_tp)
        local_name = _local_name_for_global_name(source_name_maps[rank], global_name)
        value = source_states[rank][local_name]
        for source_dp_rank in range(1, source_dp):
            replica_rank = dp_pp_tp_to_rank(source_dp_rank, source_pp_rank, source_tp_rank, source_pp, source_tp)
            replica_name = _local_name_for_global_name(source_name_maps[replica_rank], global_name)
            if not _all_equal((value, source_states[replica_rank][replica_name])):
                raise RuntimeConfigError(
                    f"model tensor {global_name!r} differs across source DP replicas "
                    f"at PP={source_pp_rank}, TP={source_tp_rank}"
                )
        values.append(value)
    return values


def _source_specs_for_global_name(
    source_shard_specs: list[dict[str, dict[str, Any]]],
    source_name_maps: list[dict[str, str]],
    global_name: str,
    *,
    source_pp: int,
    source_tp: int,
) -> list[dict[str, Any]]:
    source_pp_rank = _source_pp_rank_for_global_name(source_name_maps, global_name, source_pp, source_tp)
    out = []
    for source_tp_rank in range(source_tp):
        rank = dp_pp_tp_to_rank(0, source_pp_rank, source_tp_rank, source_pp, source_tp)
        local_name = _local_name_for_global_name(source_name_maps[rank], global_name)
        out.append(dict(source_shard_specs[rank].get(local_name) or {}))
    return out


def _source_optimizer_specs_for_global_name(
    source_shard_specs: list[dict[str, dict[str, Any]]],
    source_name_maps: list[dict[str, str]],
    global_name: str,
    *,
    source_dp: int,
    source_pp: int,
    source_tp: int,
) -> list[dict[str, Any]]:
    source_pp_rank = _source_optimizer_pp_rank_for_global_name(
        source_name_maps,
        global_name,
        source_dp,
        source_pp,
        source_tp,
    )
    out = []
    for source_tp_rank in range(source_tp):
        spec: dict[str, Any] = {}
        for source_dp_rank in range(source_dp):
            rank = dp_pp_tp_to_rank(source_dp_rank, source_pp_rank, source_tp_rank, source_pp, source_tp)
            try:
                local_name = _local_name_for_global_name(source_name_maps[rank], global_name)
            except RuntimeConfigError:
                continue
            spec = dict(source_shard_specs[rank].get(local_name) or {})
            break
        out.append(spec)
    return out


def _consistent_shard_spec(name: str, specs: list[dict[str, Any]]) -> dict[str, Any]:
    populated = [spec for spec in specs if spec]
    if not populated:
        return {}
    first = populated[0]
    comparable_keys = {"is_tp", "dim", "stride", "global_shape"}
    first_comparable = {key: first.get(key) for key in comparable_keys if key in first}
    for spec in populated[1:]:
        comparable = {key: spec.get(key) for key in comparable_keys if key in spec}
        if comparable != first_comparable:
            raise RuntimeConfigError(f"tensor {name!r} has inconsistent TP shard metadata across source ranks")
    return dict(first)


def _source_optimizer_entries_for_global_name(
    source_optimizer_states: list[dict[str, Any]],
    source_name_maps: list[dict[str, str]],
    global_name: str,
    *,
    source_dp: int,
    source_pp: int,
    source_tp: int,
) -> list[Any]:
    source_pp_rank = _source_optimizer_pp_rank_for_global_name(source_name_maps, global_name, source_dp, source_pp, source_tp)
    values = []
    for source_tp_rank in range(source_tp):
        candidates = []
        for source_dp_rank in range(source_dp):
            rank = dp_pp_tp_to_rank(source_dp_rank, source_pp_rank, source_tp_rank, source_pp, source_tp)
            try:
                local_name = _local_name_for_global_name(source_name_maps[rank], global_name)
            except RuntimeConfigError:
                continue
            if local_name in source_optimizer_states[rank]:
                candidates.append(source_optimizer_states[rank][local_name])
        if not candidates:
            raise RuntimeConfigError(
                f"optimizer state for global tensor {global_name!r} is missing across source DP ranks "
                f"at PP={source_pp_rank}, TP={source_tp_rank}"
            )
        values.append(
            _merge_flat_optimizer_entries(
                candidates,
                name=global_name,
                pp_rank=source_pp_rank,
                tp_rank=source_tp_rank,
            )
        )
    return values


def _merge_flat_optimizer_entries(
    candidates: list[dict[str, Any]],
    *,
    name: str,
    pp_rank: int,
    tp_rank: int,
) -> dict[str, Any]:
    import torch

    flat = [dict(candidate.get("_rpeg_flat_shard") or {}) for candidate in candidates]
    if not any(flat):
        first = candidates[0]
        if any(not _optimizer_entries_equal(first, candidate) for candidate in candidates[1:]):
            raise RuntimeConfigError(
                f"optimizer state for global tensor {name!r} differs across source DP replicas "
                f"at PP={pp_rank}, TP={tp_rank}"
            )
        return first
    if not all(flat):
        raise RuntimeConfigError(
            f"optimizer state for global tensor {name!r} mixes flat-sharded and replicated entries"
        )
    numel = int(flat[0]["numel"])
    shape = tuple(int(x) for x in flat[0].get("shape", (numel,)))
    if any(int(meta.get("numel", -1)) != numel or tuple(int(x) for x in meta.get("shape", ())) != shape for meta in flat):
        raise RuntimeConfigError(f"flat optimizer shard metadata differs for global tensor {name!r}")
    out: dict[str, Any] = {}
    tensor_keys = {
        key
        for candidate in candidates
        for key, value in candidate.items()
        if key != "_rpeg_flat_shard" and _is_tensor(value) and int(getattr(value, "ndim", 0)) > 0
    }
    for key in tensor_keys:
        template = next(candidate[key] for candidate in candidates if key in candidate)
        merged = torch.empty(numel, dtype=template.dtype, device=template.device)
        covered = torch.zeros(numel, dtype=torch.bool)
        for candidate, metadata in zip(candidates, flat, strict=True):
            start = int(metadata["start"])
            end = int(metadata["end"])
            if start < 0 or end <= start or end > numel:
                raise RuntimeConfigError(
                    f"invalid flat optimizer range [{start},{end}) for {name!r} with numel={numel}"
                )
            value = candidate[key].reshape(-1)
            if int(value.numel()) != end - start:
                raise RuntimeConfigError(
                    f"flat optimizer tensor {name!r}.{key!r} has {value.numel()} values "
                    f"for declared range [{start},{end})"
                )
            if covered[start:end].any():
                raise RuntimeConfigError(f"overlapping flat optimizer shards for {name!r}.{key!r}")
            merged[start:end].copy_(value)
            covered[start:end] = True
        if not covered.all():
            missing = int((~covered).sum().item())
            raise RuntimeConfigError(f"flat optimizer shards for {name!r}.{key!r} miss {missing} elements")
        out[key] = merged.view(shape)
    scalar_keys = set(candidates[0]) - tensor_keys - {"_rpeg_flat_shard"}
    for key in scalar_keys:
        values = [candidate[key] for candidate in candidates]
        first = values[0]
        if _is_tensor(first):
            if not _all_equal(values):
                raise RuntimeConfigError(f"flat optimizer scalar {name!r}.{key!r} differs across DP shards")
            out[key] = first.detach().clone()
        else:
            if any(value != first for value in values[1:]):
                raise RuntimeConfigError(f"flat optimizer value {name!r}.{key!r} differs across DP shards")
            out[key] = first
    return out


def _source_pp_rank_for_global_name(
    source_name_maps: list[dict[str, str]],
    global_name: str,
    source_pp: int,
    source_tp: int,
) -> int:
    matches = []
    for pp_rank in range(source_pp):
        rank = dp_pp_tp_to_rank(0, pp_rank, 0, source_pp, source_tp)
        if global_name in set(source_name_maps[rank].values()):
            matches.append(pp_rank)
    if not matches:
        raise RuntimeConfigError(f"global tensor {global_name!r} is not present in source PP layout")
    if len(matches) > 1:
        raise RuntimeConfigError(f"global tensor {global_name!r} is present on multiple source PP stages: {matches}")
    return matches[0]


def _source_optimizer_pp_rank_for_global_name(
    source_name_maps: list[dict[str, str]],
    global_name: str,
    source_dp: int,
    source_pp: int,
    source_tp: int,
) -> int:
    matches = []
    for pp_rank in range(source_pp):
        present = False
        for dp_rank in range(source_dp):
            for tp_rank in range(source_tp):
                rank = dp_pp_tp_to_rank(dp_rank, pp_rank, tp_rank, source_pp, source_tp)
                if global_name in set(source_name_maps[rank].values()):
                    present = True
                    break
            if present:
                break
        if present:
            matches.append(pp_rank)
    if not matches:
        raise RuntimeConfigError(f"global optimizer state {global_name!r} is not present in source PP layout")
    if len(matches) > 1:
        raise RuntimeConfigError(f"global optimizer state {global_name!r} is present on multiple source PP stages: {matches}")
    return matches[0]


def _local_name_for_global_name(name_map: dict[str, str], global_name: str) -> str:
    matches = [local for local, global_ in name_map.items() if global_ == global_name]
    if not matches:
        raise RuntimeConfigError(f"global tensor {global_name!r} is missing from source rank")
    if len(matches) > 1:
        raise RuntimeConfigError(f"global tensor {global_name!r} maps to multiple local names: {matches}")
    return matches[0]


def _reshard_optimizer_entry(
    name: str,
    source_entries: list[dict[str, Any]],
    *,
    source_tp: int,
    destination_tp: int,
    destination_tp_rank: int,
    source_spec: dict[str, Any],
    destination_spec: dict[str, Any],
) -> dict[str, Any]:
    keys = set(source_entries[0])
    if any(set(entry) != keys for entry in source_entries[1:]):
        raise RuntimeConfigError(f"optimizer state entry keys differ across source TP ranks for {name!r}")
    out = {}
    for key in keys:
        values = [entry[key] for entry in source_entries]
        first = values[0]
        if _is_tensor(first):
            source_is_tp = bool(source_spec.get("is_tp", False))
            destination_is_tp = bool(destination_spec.get("is_tp", False))
            source_dim = source_spec.get("dim")
            destination_dim = destination_spec.get("dim")
            shard_dim = destination_dim if destination_is_tp else source_dim if source_is_tp else None
            if shard_dim is None:
                shard_dim = tp_shard_dim(name, int(getattr(first, "ndim", 0)))
                source_is_tp = source_is_tp or (shard_dim is not None and source_tp > 1)
                destination_is_tp = destination_is_tp or (shard_dim is not None and destination_tp > 1)
            if int(getattr(first, "ndim", 0)) == 0 or (not source_is_tp and not destination_is_tp):
                if not _all_equal(values):
                    raise RuntimeConfigError(
                        f"optimizer state {name!r}.{key!r} is not TP-sharded and differs across TP ranks"
                    )
                out[key] = first.detach().clone()
                continue
            if shard_dim is None:
                raise RuntimeConfigError(f"optimizer state {name!r}.{key!r} is TP-sharded without a dimension")
            if source_is_tp:
                full = _gather_strided_tp(
                    values,
                    dim=int(shard_dim),
                    stride=int(source_spec.get("stride", 1)),
                    name=f"{name}.{key}",
                )
            else:
                if not _all_equal(values):
                    raise RuntimeConfigError(
                        f"replicated optimizer state {name!r}.{key!r} differs across source TP ranks"
                    )
                full = first.detach().clone()
            if destination_is_tp:
                out[key] = _shard_strided_tp(
                    full,
                    world_size=destination_tp,
                    rank=destination_tp_rank,
                    dim=int(shard_dim),
                    stride=int(destination_spec.get("stride", 1)),
                    name=f"{name}.{key}",
                )
            else:
                out[key] = full.detach().clone()
            continue
        if values[1:] and any(value != first for value in values[1:]):
            raise RuntimeConfigError(f"optimizer state {name!r}.{key!r} differs across source TP ranks")
        out[key] = first
    return out


def _gather_strided_tp(tensors, *, dim: int, stride: int, name: str):
    if stride <= 0:
        raise RuntimeConfigError(f"tensor {name!r} has invalid TP partition stride {stride}")
    if not tensors:
        raise RuntimeConfigError(f"tensor {name!r} has no source TP shards")
    pieces_by_rank = []
    for tensor in tensors:
        if tensor.shape[dim] % stride != 0:
            raise RuntimeConfigError(
                f"tensor {name!r} local dimension {dim} size {tensor.shape[dim]} "
                f"is not divisible by source stride={stride}"
            )
        pieces_by_rank.append(_chunk(tensor, chunks=stride, dim=dim))
    ordered = [pieces_by_rank[rank][part] for part in range(stride) for rank in range(len(tensors))]
    return _cat(ordered, dim=dim)


def _shard_strided_tp(tensor, *, world_size: int, rank: int, dim: int, stride: int, name: str):
    if world_size <= 0 or stride <= 0:
        raise RuntimeConfigError(
            f"tensor {name!r} has invalid destination TP={world_size}, partition stride={stride}"
        )
    chunks = world_size * stride
    if tensor.shape[dim] % chunks != 0:
        raise RuntimeConfigError(
            f"tensor {name!r} dimension {dim} size {tensor.shape[dim]} "
            f"is not divisible by destination TP={world_size} with stride={stride}"
        )
    pieces = _chunk(tensor, chunks=chunks, dim=dim)
    # torch.cat already returns independent contiguous storage. Cloning it a second time doubles the
    # peak routing workspace for the largest tensor; this is especially costly while one source fans
    # a version out to several serving replicas.
    return _cat(pieces[rank::world_size], dim=dim).detach()


def _cat(tensors, *, dim: int):
    import torch

    return torch.cat(tuple(tensors), dim=dim)


def _chunk(tensor, *, chunks: int, dim: int):
    import torch

    return torch.chunk(tensor, chunks, dim=dim)


def _clone_state_value(value):
    if hasattr(value, "detach"):
        return value.detach().clone()
    return value


def _maybe_clone_state_value(value, *, clone: bool):
    if clone:
        return _clone_state_value(value)
    if hasattr(value, "detach"):
        return value.detach()
    return value


def _all_equal(tensors) -> bool:
    import torch

    first = tensors[0]
    out = True
    for tensor in tensors[1:]:
        if isinstance(first, torch.Tensor) or isinstance(tensor, torch.Tensor):
            out = out and isinstance(first, torch.Tensor) and isinstance(tensor, torch.Tensor) and torch.equal(first, tensor)
        else:
            out = out and first == tensor
    return out


def _is_tensor(value) -> bool:
    import torch

    return isinstance(value, torch.Tensor)


def _optimizer_entries_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    import torch

    if set(left) != set(right):
        return False
    for key, left_value in left.items():
        right_value = right[key]
        if isinstance(left_value, torch.Tensor) or isinstance(right_value, torch.Tensor):
            if not (isinstance(left_value, torch.Tensor) and isinstance(right_value, torch.Tensor)):
                return False
            if not torch.equal(left_value, right_value):
                return False
            continue
        if left_value != right_value:
            return False
    return True
