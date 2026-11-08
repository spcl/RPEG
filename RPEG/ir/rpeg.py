# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Authors: Leonard Schmidt
#          Pierre Pang

from dataclasses import dataclass
from typing import Dict, Set, Tuple

from core.types import Container, Edgekind, Model, Role, TemplateEdge, TemplateVertex


class RPEGError(ValueError):
    """
    RPEG-specific error class.
    """
    pass


@dataclass(frozen=True)
class RPEG:
    """
    Data class for the RLM Pipeline Execution Graph (RPEG).
    """
    vertices: Tuple[TemplateVertex, ...] # assume that vertices[0] is the ROOT node 
    data_edges: Tuple[TemplateEdge, ...] # transient data 
    pub_edges: Tuple[TemplateEdge, ...] # persistent data 
    containers: Tuple[Container, ...] # both produced and consumed containers
    in_containers: Tuple[Container, ...] # consumed-only containers
    out_containers: Tuple[Container, ...] # produced-only containers
    tau: Dict[Model, int] # per-model staleness-budget 
    trainable: Set[Model]

    def well_formed(self, verbose: bool) -> None:
        """
        RPEG G is well-formed iff (1)-(2) hold:
            (1) Interface consistency: for all edges u->v with container a:
                (i)  a in Write(u)
                (ii) a in Read(v)
            (2) Boundary consistency:
                (i)  if a container a is read but not produced => a in A_in
                (ii) if a container a is written but not consumed => a in A_out
            (3) Data-edge acyclicity:
                the within-step dependency graph induced by data_edges is a DAG.

        :param verbose: Verbose flag for additional output.
        :type verbose: bool
        """
        # Sanity check for duplicate vertex identifiers
        vid_seen = set() 
        for v in self.vertices: 
            if v.id in vid_seen: 
                raise RPEGError(f"Duplicate vertex id: {v.id}")
            vid_seen.add(v.id)
        
        # Sanity check: edge endpoints + containers
        vset = set(self.vertices) # vertex set
        cset = set(self.containers) # container set
        all_edges = self.data_edges + self.pub_edges
        for e in all_edges:
            # Check if edge endpoints are in the vertex set
            if e.src not in vset:
                raise RPEGError(f"Edge src vertex {e.src.id} not in RPEG.vertices")
            if e.dst not in vset:
                raise RPEGError(f"Edge dst vertex {e.dst.id} not in RPEG.vertices")
            if e.container not in cset:
                raise RPEGError(f"Undeclared container {e.container} on edge {e.src.id}->{e.dst.id}")

        # Publication edges carry persistent model weights between iterations. Keeping these checks
        # here is important: unrolling cannot infer a correct publisher if the cyclic template is
        # ambiguous or malformed.
        if set(self.tau) != set(self.trainable):
            missing = set(self.trainable) - set(self.tau)
            extra = set(self.tau) - set(self.trainable)
            raise RPEGError(
                "tau/trainable mismatch: "
                f"missing={[m.value for m in sorted(missing, key=lambda x: x.value)]}, "
                f"extra={[m.value for m in sorted(extra, key=lambda x: x.value)]}"
            )
        for model, budget in self.tau.items():
            if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
                raise RPEGError(
                    f"tau[{model.value}] must be a nonnegative integer, got {budget!r}"
                )
        published_models = set()
        for e in self.data_edges:
            if e.kind != EdgeKind.DATA:
                raise RPEGError(
                    f"Data edge {e.src.id}->{e.dst.id} must have kind DATA, got {e.kind.value}"
                )
        for e in self.pub_edges:
            if e.kind != EdgeKind.PUB:
                raise RPEGError(
                    f"Publication edge {e.src.id}->{e.dst.id} must have kind PUB, got {e.kind.value}"
                )
            if e.src.kind.role != Role.TRAINING:
                raise RPEGError(
                    f"Publication edge {e.src.id}->{e.dst.id} must originate at a TRAIN vertex"
                )
            if e.src.kind.model not in self.trainable:
                raise RPEGError(
                    f"Publication edge {e.src.id}->{e.dst.id} publishes frozen model "
                    f"{e.src.kind.model.value}"
                )
            if e.dst.kind.model != e.src.kind.model:
                raise RPEGError(
                    f"Publication edge {e.src.id}->{e.dst.id} crosses model identity "
                    f"{e.src.kind.model.value}->{e.dst.kind.model.value}"
                )
            if e.container not in e.src.kind.read or e.container not in e.src.kind.write:
                raise RPEGError(
                    f"Publication edge {e.src.id}->{e.dst.id} container {e.container.name} "
                    "must be the source TRAIN vertex's read/write weight container"
                )
            published_models.add(e.src.kind.model)
        missing_publications = set(self.trainable) - published_models
        if missing_publications:
            raise RPEGError(
                "trainable models without publication edges: "
                f"{[m.value for m in sorted(missing_publications, key=lambda x: x.value)]}"
            )

        # Checking well-formedness constraint (1)
        for e in all_edges:
            u = e.src
            v = e.dst
            a = e.container
            if a not in u.kind.write:
                raise RPEGError(f"Interface violation: {a.name} on edge {u.id}->{v.id} not in Write({u.id})")
            if a not in v.kind.read:
                raise RPEGError(f"Interface violation: {a.name} on edge {u.id}->{v.id} not in Read({v.id})")

        # Checking well-formedness constraint (2)
        all_reads = set()
        all_writes = set() 
        for v in self.vertices: 
            all_reads.update(v.kind.read)
            all_writes.update(v.kind.write)
        
        implied_in = all_reads - all_writes 
        implied_out = all_writes - all_reads 

        declared_in = set(self.in_containers)
        declared_out = set(self.out_containers)

        if implied_in != declared_in : 
            missing = implied_in - declared_in
            extra = declared_in - implied_in
            raise RPEGError(
                "A_in mismatch: "
                f"missing={[c.name for c in missing]}, extra={[c.name for c in extra]}"
            )
        
        if implied_out != set(self.out_containers): 
            missing = implied_out - declared_out
            extra = declared_out - implied_out
            raise RPEGError("A_out mismatch: "
                f"missing={[c.name for c in missing]}, extra={[c.name for c in extra]}"
            )

        # Checking well-formedness constraint (3): publication edges may close cycles across steps,
        # but within-step data dependencies must be a DAG.
        indeg = {v: 0 for v in self.vertices}
        succ = {v: [] for v in self.vertices}
        for e in self.data_edges:
            succ[e.src].append(e.dst)
            indeg[e.dst] += 1

        ready = [v for v in self.vertices if indeg[v] == 0]
        seen = 0
        while ready:
            v = ready.pop()
            seen += 1
            for w in succ[v]:
                indeg[w] -= 1
                if indeg[w] == 0:
                    ready.append(w)
        if seen != len(self.vertices):
            cyclic = sorted(v.id for v, degree in indeg.items() if degree > 0)
            raise RPEGError(f"Data-edge graph must be acyclic; cycle involves {cyclic}")

        if verbose: 
            print("RPEG is well-formed.")
