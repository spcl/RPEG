# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt

import hashlib
import json

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from core.types import Allocation, Container
from ir.plan import ExecutionPlan
from ir.rpeg import RPEG, TemplateEdge, TemplateVertex
from schedule.trace import Trace


@dataclass
class EvalCache:
    """
    Data class for the evaluation cache of different traces.
    """
    _store: Dict[str, Tuple[float, Trace, bool]] = field(default_factory=dict)  # hash -> (score, trace, feasible)
    hits:   int = 0
    misses: int = 0 # misses == unique evaluations computed (not the number of times a key was not found)

    def get(self, key: str) -> Optional[Tuple[float, Trace, bool]]:
        """
        Retrieve a value for key from the cache.

        :param key: Key for the retrieval.
        :type key: str
        :return: Trace for the key or None if the key is not found inside the cache.
        :rtype: Optional[Tuple[float, Trace, bool]]
        """
        if key in self._store:
            self.hits += 1
            return self._store[key]
        return None

    def put(self, key: str, score: float, trace: Trace, feasible: bool) -> None:
        """
        Add a key-value pair to the cache.

        :param key: Key to store.
        :type key: str
        :param score: Score of the trace.
        :type score: float
        :param trace: Trace to store.
        :type trace: Trace
        :param feasible: Flag, whether the trace is feasible.
        :type feasible: bool
        """
        self._store[key] = (score, trace, feasible)
        self.misses += 1

    def __contains__(self, key: str) -> bool:
        """
        Return whether a key is present in the cache.

        :param key: Key to test.
        :type key: str
        :return: Flag to indicate whether the key is present in the cache.
        :rtype: bool
        """
        return key in self._store

    def size(self) -> int:
        """
        Return the number of entries in the cache.

        :return: Number of entries in the cache.
        :rtype: int
        """
        return len(self._store)

    def stats(self) -> str:
        """
        Return statistics about the cache use as a string.

        :return: Statistics about the cache use.
        :rtype: str
        """
        total    = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return (f"hits={self.hits}  misses={self.misses}  "
                f"total={total}  hit_rate={hit_rate:.1%}")


# serializer helpers (module-level, before config_hash)
def _serialize_container(c: Container) -> dict:
    """
    Helper function to serialize a container into a dictionary.

    :param c: Container to serialize.
    :type c: Container
    :return: Serialized dictionary for the container.
    :rtype: dict
    """
    return {
         "name": c.name,
         "count": c.count,
         "elem": c.elem_size_bytes,
    }


def _serialize_vertex(v: TemplateVertex) -> dict:
    """
    Helper function to serialize a vertex into a dictionary.

    :param v: Vertex to serialize.
    :type v: TemplateVertex
    :return: Serialized dictionary for the vertex.
    :rtype: dict
    """
    return {
        "id": v.id, 
        "role": v.kind.role.value,
        "model": v.kind.model.value, 
        "read": sorted([_serialize_container(c) for c in v.kind.read], key=lambda x: x["name"]), # sort read/write container by name for canonical order 
        "write": sorted([_serialize_container(c) for c in v.kind.write], key=lambda x: x["name"])
    }


def _serialize_edge(e: TemplateEdge) -> dict:
    """
    Helper function to serialize an edge into a dictionary.

    :param e: Edge to serialize.
    :type e: TemplateEdge
    :return: Serialized dictionary for the edge.
    :rtype: dict
    """
    return {
        "src": e.src.id,
        "dst": e.dst.id,
        "container": _serialize_container(e.container),
    }


def _serialize_alloc(alloc: Allocation) -> dict:
    """
    Helper function to serialize an allocation into a dictionary.

    :param alloc: Allocation to serialize.
    :type alloc: Allocation
    :return: Serialized dictionary for the allocation.
    :rtype: dict
    """
    return {
        "origin": list(alloc.origin), 
        "extent": list(alloc.extent)
    }


# canonical hash of a configuration C = (G, P) (for cache key)
def config_hash(rpeg: RPEG, plan: ExecutionPlan) -> str:
    """
    Canonical hash of a configuration C = (G, P) (as a cache key).

    :param rpeg: RLM Pipeline Execution Graph.
    :type rpeg: RPEG
    :param plan: Execution plan.
    :type plan: ExecutionPlan
    :return: Canonical hash.
    :rtype: str
    """
    payload = {
        # RPEG
        "vertices": sorted([_serialize_vertex(v) for v in rpeg.vertices], key=lambda x: x["id"]),
        "data_edges": sorted([_serialize_edge(e) for e in rpeg.data_edges], key=lambda x: (x["src"], x["dst"])),
        "pub_edges": sorted([_serialize_edge(e) for e in rpeg.pub_edges], key=lambda x: (x["src"], x["dst"])),
        "tau": {m.value: t for m, t in sorted(rpeg.tau.items(), key=lambda x: x[0].value)},
        # ExecutionPlan
        "sigma": {v.id: _serialize_alloc(a) for v, a in sorted(plan.sigma.items(), key=lambda x: x[0].id)},
        "pi": {v.id: list(pi) for v, pi in sorted(plan.pi.items(), key=lambda x: x[0].id)},
        "mb": {v.id: mb for v, mb in sorted(plan.mb.items(), key=lambda x: x[0].id)}
    }
    serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    return hash
