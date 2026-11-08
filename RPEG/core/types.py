# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt
#
# Contributions: Pierre Pang

from dataclasses import dataclass # implements __init__/__eq__/__repr__ automatically, frozen=True makes members immutable
from enum import Enum
from typing import Optional, Tuple


def packing_geometry(dp: int, pp: int, tp: int, G: int) -> Optional[Tuple[int, int]]:
    """
    Compute (g_ext, nodes_needed) for a given strategy and GPUs-per-node.

    Supports intra-node DP: when the entire pipeline fits within one node (nodes_per_rep == 1),
    multiple DP replicas can share a single node.

    Returns (g_ext, nodes_needed) or None if the configuration requires fractional node usage
    (e.g., dp=3 with reps_per_node=2).

    :param dp: Data parallelism configuration.
    :type dp: int
    :param pp: Pipeline parallelism configuration.
    :type pp: int
    :param tp: Tensor parallelism configuration.
    :type tp: int
    :param G: Number of GPUs per node.
    :type G: int
    :return: Tuple of the necessary number of GPUs and nodes or None in case of fractional nodes.
    :rtype: Optional[Tuple[int, int]]
    """
    s = min(pp, G // tp)
    if s == 0 or pp % s != 0:
        return None
    gpus_per_rep = s * tp           # GPUs one replica occupies per node-slice
    nodes_per_rep = pp // s         # nodes one replica spans

    if nodes_per_rep == 1:
        # Pipeline fits in one node — intra-node DP possible
        reps_per_node = G // gpus_per_rep
        if dp <= reps_per_node:
            # Case A: all replicas fit in one node
            return (dp * gpus_per_rep, 1)
        elif dp % reps_per_node == 0:
            # Case B: exact multiple of full nodes
            return (G, dp // reps_per_node)
        else:
            # Fractional node — reject
            return None
    else:
        # Pipeline spans multiple nodes — DP always across nodes
        return (G, dp * nodes_per_rep)


def rectangular_node_layouts(nodes_needed: int, R: int, N: int) -> Tuple[Tuple[int, int], ...]:
    """
    Enumerate exact rectangular rack-by-node embeddings. Please note that this functionality is
    specific to the targeted cluster for our experimental evaluation.

    Alps allocations are exposed as a logical rack/node/GPU mesh. Requiring every cross-rack
    allocation to consume complete logical racks excludes useful rectangles such as 2 x 42 nodes for
    an 84-node pipeline. Exact factor rectangles retain the allocation model's contiguity and never
    reserve padding nodes.

    :param nodes_needed: Number of requested compute nodes. Should be a non-negative integer.
    :type nodes_needed: int
    :param R: Number of available racks. Should be a non-negative integer.
    :type R: int
    :param N: Number of compute nodes within one rack. Should be a non-negative integer.
    :type N: int
    :return: Tuples of tuples of rack counts and number of compute nodes within one rack.
    :rtype: Tuple[Tuple[int, int], ...]
    """
    if nodes_needed <= 0 or R <= 0 or N <= 0:
        return ()
    return tuple(
        (r_ext, nodes_needed // r_ext)
        for r_ext in range(1, min(R, nodes_needed) + 1)
        if nodes_needed % r_ext == 0 and nodes_needed // r_ext <= N
    )


class Role(Enum):
    """
    Enumeration class for the different model roles.
    """
    GENERATION = "GEN"
    INFERENCE = "INF"
    TRAINING = "TRAIN"


class Model(Enum):
    """
    Enumeration class for model abbrevations.
    """
    ACTOR = "Pi"
    CRITIC = "V"
    REFERENCE = "Ref"
    REWARD = "R"
    AC_SHARE = "AC"


class Collective(Enum):
    """
    Enumeration class for communication collectives.
    """
    ALLREDUCE = "AllReduce"
    ALLGATHER = "AllGather" 
    REDUCESCATTER = "ReduceScatter"
    ALLTOALL = "AllToAll"
    P2P = "PointToPoint"


class EdgeKind(Enum):
    """
    Enumeration class for edge types, which distinguish between edges indicating data communication
    and publication edges indicating dependencies across iterations.
    """
    DATA = "data"
    PUB = "pub"


@dataclass(frozen=True)
class TransferResourcePolicy:
    """
    Device reservations used when scheduling explicit data movement.

    The RPEG analytical model permits a source GPU to become available after posting a send. Real
    transports can opt out per payload class, when their source process, CUDA work, or live send
    buffer must remain exclusive for the duration of the transfer.

    Potential values: "destination_only", "both_endpoints"

    weight = option for the transfer of weights
    activation = option for the transfer of activations
    """
    weight: str = "destination_only"
    activation: str = "destination_only"

    def __post_init__(self) -> None:
        """
        Consistency check after the initialization.

        :raises ValueError: If either field contains an invalid value.
        """
        allowed = {"destination_only", "both_endpoints"}
        if self.weight not in allowed:
            raise ValueError(f"unsupported weight transfer resource policy {self.weight!r}")
        if self.activation not in allowed:
            raise ValueError(f"unsupported activation transfer resource policy {self.activation!r}")

    def for_weight(self, is_weight: bool) -> str:
        """
        Return value for the weight transfer (if is_weight is True) or the activation transfer (if
        is_weight is False).

        :param is_weight: Flag to indicate which value to return.
        :type is_weight: bool
        :return: Class value for the requested transfer.
        :rtype: str
        """
        return self.weight if is_weight else self.activation

    def to_dict(self) -> dict[str, str]:
        """
        Return class configuration as a dictionary.

        :return: Dictionary containing the class configuration.
        :rtype: dict[str, str]
        """
        return {"weight": self.weight, "activation": self.activation}

    @classmethod
    def from_dict(cls, value: object) -> "TransferResourcePolicy":
        """
        Initialize the class from dictionary.

        :param cls: Class to be initialized.
        :param value: Object used for initialization. Should be a dictionary.
        :type value: object
        :returns: Initialized class.
        :rtype: "TransferResourcePolicy"
        :raises ValueError: If value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise ValueError("transfer_resource_policy must be a mapping")
        return cls(
            weight=str(value.get("weight", "destination_only")),
            activation=str(value.get("activation", "destination_only")),
        )


# Set default configuration
RPEG_TRANSFER_RESOURCE_POLICY = TransferResourcePolicy()
CONSERVATIVE_TRANSFER_RESOURCE_POLICY = TransferResourcePolicy(
    weight="both_endpoints",
    activation="both_endpoints",
)


@dataclass(frozen=True)
class DeviceSpace:
    """
    Data class representing the device space.

    R = number of racks
    N = number of compute nodes per rack
    G = number of GPUs per node
    """
    R: int
    N: int 
    G: int 

    def num_devices(self) -> int:
        """
        Returns the number of available GPUs.

        :return: Number of available GPUs.
        :rtype: int
        """
        return self.R * self.N * self.G 
    
    def device_id(self, r: int, n: int, g: int) -> int:
        """
        Returns an unique GPU ID based on the rack, node and GPU number.

        :param r: Rack number.
        :type r: int
        :param n: Node number.
        :type n: int
        :param g: GPU number.
        :type g: int
        :return: Unique GPU ID.
        :rtype: int
        """
        return (r * self.N * self.G) + (n * self.G) + g 
    

@dataclass(frozen=True)
class Allocation:
    """
    Data class representing an allocation.

    origin = Start point of the allocation in the device space.
    extent = Tuple of the number of used elements in each dimension (racks, nodes and GPUs).
    """
    origin: Tuple[int, int, int]
    extent: Tuple[int, int, int]

    def size(self) -> int: 
        """
        Returns number of GPUs in the allocation.

        :return: Number of allocated GPUs.
        :rtype: int
        """
        r, n, g = self.extent 
        return r * n * g 


@dataclass(frozen=True)
class Container:
    """
    Data class representing a data container.

    container a = (|a|, s(a))
    |a| := symbolic element count
    s(a) := size of a single element in Bytes

    name = container name
    count = number of elements
    elem_size_bytes = element size in Bytes
    """
    name: str
    count: int
    elem_size_bytes: int

    def bytes(self) -> int:
        """
        Returns the size of the data container in Bytes.

        :return: Number of Bytes in the data container.
        :rtype: int
        """
        return self.count * self.elem_size_bytes


@dataclass(frozen=True)
class OperatorKind:
    """
    Data class representing an operator.

    operator kind o := (r, m)
    has read/write interfaces Read(o), Write(o) \subset A

    role = role of the operator
    model = model type
    read = input container interface
    write = output container interface
    """
    role: Role
    model: Model  
    read: Tuple[Container, ...]
    write: Tuple[Container, ...]


@dataclass(frozen=True)
class TemplateVertex:
    """
    Data class representing a vertex in the RPEG.

    Each node is annotated with role, model, read/write sets.

    id = node ID
    kind = operator type
    """
    id: str 
    kind: OperatorKind


@dataclass(frozen=True)
class TemplateEdge:
    """
    Data class representing a directed edge in the RPEG.

    Each edge carries a single container from src to dst.

    src = origin vertex of the edge
    dst = target vertex of the edge
    container = associated data container
    kind = edge type
    """
    src: TemplateVertex
    dst: TemplateVertex
    container: Container
    kind: EdgeKind
