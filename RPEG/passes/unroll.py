# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt
#
# Contributions: Pierre Pang

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from core.types import Container, EdgeKind, Model, TemplateVertex
from ir.rpeg import RPEG


@dataclass(frozen=True)
class UnrolledNode:
    """
    Data class for a node in the unrolled graph.

    template_vertex: Node in the original template graph.
    k: Iteration/step number.
    i: Substep when nodes are split for generation. Defaults to None.
    """
    template_vertex: TemplateVertex
    k: int 
    i: Optional[int] = None 


@dataclass(frozen=True)
class UnrolledEdge:
    """
    Data class for an edge in the unrolled graph.

    src: Origin vertex.
    dst: Target vertex.
    container: Data container associated with the edge.
    kind: Edge type.
    """
    src: UnrolledNode
    dst: UnrolledNode
    container: Container
    kind: EdgeKind


@dataclass(frozen=True)
class UnrolledRPEG:
    """
    Data class for the unrolled graph.

    nodes: Vertices in the unrolled graph.
    data_edges: (Data dependency) edges within an iteration/step.
    pub_edges: Publication edges between interations/steps.
    """
    nodes: Tuple[UnrolledNode, ...]
    data_edges: Tuple[UnrolledEdge, ...]
    pub_edges: Tuple[UnrolledEdge, ...]


def publisher_selection(model: Model, tau: Dict[Model, int], k: int) -> int:
    """
    Deterministic publisher selection function.
    (could also be randomized to some value in [...])

    :param model: Model for which its training iteration is selected.
    :type model: Model
    :param tau: Dictionary for model-specific taus.
    :type tau: Dict[Model, int]
    :param k: Current iteration/step of the unrolled graph.
    :type k: int
    :return: Iteration/step, where the model was last trained.
    :rtype: int
    :raises ValueError: If either the staleness budget is missing or has no valid value.
    """
    if model not in tau:
        raise ValueError(f"Missing staleness budget for trainable model {model.value}")
    tau_m = tau[model]
    if isinstance(tau_m, bool) or not isinstance(tau_m, int) or tau_m < 0:
        raise ValueError(f"Staleness budget for {model.value} must be a non-negative integer")
    return max(0, k - (tau_m + 1)) 


def unroll(rpeg: RPEG, K_eval: int, verbose: bool = False) -> UnrolledRPEG:
    """
    Unrolls a given (cyclic) RPEG by K_eval iterations. The returned unrolled RPEG is acyclic.

    :param rpeg: Input RPEG to unroll.
    :type rpeg: RPEG
    :param K_eval: Number of iterations to unroll.
    :type K_eval: int
    :param verbose: Verbose flag for additional output. Defaults to False.
    :type verbose: bool
    :return: Unrolled RPEG.
    :rtype: UnrolledRPEG
    :raises ValueError: If K_eval is a negative integer or zero.
    """
    if K_eval <= 0: 
        raise ValueError(f"K_eval must be >= 1, got {K_eval}.")
    rpeg.well_formed(verbose=False)

    # Unroll vertices: V_step = V x [K]
    node_lookup = {} # directory for fast TemplateVertex -> UnrolledNode mapping
    nodes = [] # list of unrolled nodes 
    for k in range(K_eval): 
        for v in rpeg.vertices:
            new_node = UnrolledNode(v, k)
            node_lookup[(v, k)] = new_node 
            nodes.append(new_node)
    
    # Unroll within-step data edges
    data_edges = []
    for k in range(K_eval): 
        for e in rpeg.data_edges:
            u = node_lookup[(e.src, k)] 
            v = node_lookup[(e.dst, k)]
            new_edge = UnrolledEdge(src=u, dst=v, container=e.container, kind=EdgeKind.DATA)
            data_edges.append(new_edge)

    # Unroll cross-step publication edges
    pub_edges = []
    for k in range(1, K_eval): # k=0 has no predecessor step, so no publication edges
        for e in rpeg.pub_edges:
            tau_m = rpeg.tau[e.src.kind.model]
            # Steps k <= tau_m use the initial weights θ^0 — no training predecessor exists, so no
            # publication edge is needed.
            if k <= tau_m:
                continue
            _k = publisher_selection(model=e.src.kind.model, tau=rpeg.tau, k=k)
            u = node_lookup[(e.src, _k)]
            v = node_lookup[(e.dst, k)]
            new_edge = UnrolledEdge(src=u, dst=v, container=e.container, kind=EdgeKind.PUB)
            pub_edges.append(new_edge)

    # Sanity check: assert that pubplication edges go forward in time
    for e in pub_edges:
        assert e.src.k < e.dst.k

    unrolled_rpeg = UnrolledRPEG(
        nodes=tuple(nodes), 
        data_edges=tuple(data_edges), 
        pub_edges=tuple(pub_edges)
    )

    if verbose:
        print(f"RPEG unrolled successfully for {K_eval} steps.")
        print(f"    Unrolled nodes: {len(unrolled_rpeg.nodes)}.")
        print(f"    Unrolled data edges: {len(unrolled_rpeg.data_edges)}.")
        print(f"    Unrolled publication edges: {len(unrolled_rpeg.pub_edges)}.")

    return unrolled_rpeg
