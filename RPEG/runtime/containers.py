# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ShardingSpec:
    """
    Logical placement of a runtime object.

    The first implementation records layout metadata rather than moving tensors. Backends can use
    the same object to attach Megatron, DTensor, or storage specific sharding details later.
    """
    kind: str
    placement: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContainerRef:
    """
    Reference to an activation/data container produced by an RPEG node.
    """
    name: str
    step: int | None
    producer_node_id: str
    logical_shape: tuple[int, ...] | None = None
    dtype: str | None = None
    sharding: ShardingSpec | None = None
    storage_uri: str | None = None
    owner_worker_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WeightVersion:
    """
    Versioned model weights published through RPEG publication edges.
    """
    model: str
    version: int
    produced_by: str | None
    step: int | None
    layout: ShardingSpec | None = None
    dtype: str | None = None
    optimizer_state_owner: str | None = None
    parameter_storage: str | None = None
    shadow: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ContainerStore:
    """
    Small metadata store for runtime containers and weight versions.

    The store is intentionally metadata-only today. File-backed tensor storage, NCCL handles, and
    checkpoint references can be attached through storage_uri/parameter_storage without changing
    executor semantics.
    """
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = None if root is None else Path(root)
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
        self.containers: dict[str, ContainerRef] = {}
        self.weights: dict[tuple[str, int], WeightVersion] = {}
        self.latest_weight_version: dict[str, int] = {}

    def put_container(self, ref: ContainerRef) -> ContainerRef:
        self.containers[ref.name] = ref
        return ref

    def get_container(self, name: str) -> ContainerRef:
        return self.containers[name]

    def publish_weight(self, version: WeightVersion) -> WeightVersion:
        key = (version.model, version.version)
        self.weights[key] = version
        current = self.latest_weight_version.get(version.model, -1)
        if version.version >= current:
            self.latest_weight_version[version.model] = version.version
        return version

    def latest_weight(self, model: str) -> WeightVersion | None:
        version = self.latest_weight_version.get(model)
        if version is None:
            return None
        return self.weights[(model, version)]
