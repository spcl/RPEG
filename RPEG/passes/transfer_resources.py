# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

from collections.abc import Mapping, Set
from typing import Tuple

from core.types import Allocation, TransferResourcePolicy
from passes.lower import ExecNode, ExecNodeType


def is_weight_transfer(node: ExecNode) -> bool:
    """
    Return, whether it is a weight transfer based on the vertex label.

    :param node: Vertex in the executable RPEG.
    :type node: ExecNode
    :return: True, if the vertex label starts with "W_". False otherwise.
    :rtype: bool
    """
    return str(node.label or "").startswith("W_")


def transfer_preserves_data_parallelism(
    node: ExecNode,
    *,
    predecessors: Mapping[str, Set[str]],
    successors: Mapping[str, Set[str]],
    by_id: Mapping[str, ExecNode],
) -> bool:
    """
    Return whether the source and destination operators use the same DP strategy.

    Uses the actual adjacent lowered operators rather than a persistent worker's initial layout: one
    worker may own several vertices with distinct parallel strategies.

    :param node: Vertex in the executable RPEG.
    :type node: ExecNode
    :param predecessors: Look up data structure for the predecessors of a vertex.
    :type predecessors: Mapping[str, Set[str]]
    :param successors: Look up data structure for the successors of a vertex.
    :type successors: Mapping[str, Set[str]]
    :param by_id: Look up data structure from vertex IDs to their actual vertices in the graph.
    :type by_id: Mapping[str, ExecNode]
    :return: True, if both operators use the same DP strategy. False otherwise.
    :rtype: bool
    """
    adjacent_ops: list[ExecNode] = []
    for adjacent_ids in (predecessors[node.id], successors[node.id]):
        op = next(
            (
                by_id[adjacent_id]
                for adjacent_id in sorted(adjacent_ids)
                if adjacent_id in by_id
                and by_id[adjacent_id].kind == ExecNodeType.OP
                and by_id[adjacent_id].pi is not None
            ),
            None,
        )
        if op is None:
            return False
        adjacent_ops.append(op)
    return int(adjacent_ops[0].pi[0]) == int(adjacent_ops[1].pi[0])


def transfer_reserves_destination_only(
    node: ExecNode,
    *,
    policy: TransferResourcePolicy,
    predecessors: Mapping[str, Set[str]],
    successors: Mapping[str, Set[str]],
    by_id: Mapping[str, ExecNode],
) -> bool:
    """
    Return whether the transfer only reserves the destination vertex (True) or if both endpoints
    need to be reserved (False).

    :param node: Vertex in the executable RPEG.
    :type node: ExecNode
    :param policy: Policy for transfers.
    :type policy: TransferResourcePolicy
    :param predecessors: Look up data structure for the predecessors of a vertex.
    :type predecessors: Mapping[str, Set[str]]
    :param successors: Look up data structure for the successors of a vertex.
    :type successors: Mapping[str, Set[str]]
    :param by_id: Look up data structure from vertex IDs to their actual vertices in the graph.
    :type by_id: Mapping[str, ExecNode]
    :return: True, if the transfer only reserves the destination vertex. False otherwise.
    :rtype: bool
    """
    if node.kind not in {ExecNodeType.COMM, ExecNodeType.RESHARD} or len(node.allocs) < 2:
        return False
    if policy.for_weight(is_weight_transfer(node)) != "destination_only":
        return False
    if is_weight_transfer(node):
        return True
    return transfer_preserves_data_parallelism(
        node,
        predecessors=predecessors,
        successors=successors,
        by_id=by_id,
    )


def reserved_allocations(
    node: ExecNode,
    *,
    policy: TransferResourcePolicy,
    predecessors: Mapping[str, Set[str]],
    successors: Mapping[str, Set[str]],
    by_id: Mapping[str, ExecNode],
) -> Tuple[Allocation, ...]:
    """
    Select the allocation for the transfer.

    :param node: Vertex in the executable RPEG.
    :type node: ExecNode
    :param policy: Policy for transfers.
    :type policy: TransferResourcePolicy
    :param predecessors: Look up data structure for the predecessors of a vertex.
    :type predecessors: Mapping[str, Set[str]]
    :param successors: Look up data structure for the successors of a vertex.
    :type successors: Mapping[str, Set[str]]
    :param by_id: Look up data structure from vertex IDs to their actual vertices in the graph.
    :type by_id: Mapping[str, ExecNode]
    :return: Transfer allocation.
    :rtype: Tuple[Allocation, ...]
    """
    if transfer_reserves_destination_only(
        node,
        policy=policy,
        predecessors=predecessors,
        successors=successors,
        by_id=by_id,
    ):
        selected = node.allocs[-1:]
    else:
        selected = node.allocs
    # A colocated transfer can encode the same allocation as both its source and destination. It is
    # one physical resource reservation, not two mutually conflicting reservations by the same
    # communication node.
    return tuple(dict.fromkeys(selected))
