# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Authors: Leonard Schmidt
#          Pierre Pang

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from core.types import Allocation
from passes.lower import ExecNode, ExecNodeType


@dataclass(frozen=True)
class Trace:
    """
    Data class for execution traces including the computation of various metrics.

    start: Start time of the node execution.
    finish: End time of the node execution.
    reserved_allocations: Allocations reserved for the graph node.
    """
    start: Dict[ExecNode, float]
    finish: Dict[ExecNode, float]
    reserved_allocations: Optional[Dict[ExecNode, Tuple[Allocation, ...]]] = None

    def makespan(self) -> float:
        """
        Compute the makespan of the execution trace.

        :return: Makespan of the execution trace in seconds.
        :rtype: float
        """
        return max(self.finish.values())

    def throughput(self, W_step: float, K: int) -> float:
        """
        Compute the throughput of the execution trace.

        :param W_step: Work per step in FLOP.
        :type: float
        :param K: Number of iterations/steps.
        :type K: int
        :return: Throughput of the execution trace in FLOP/s.
        :rtype: float
        """
        return K * W_step / self.makespan()

    def utilization(self) -> float:
        """
        Mean GPU utilization = mean(busy_time(d) / makespan) across all active devices.

        busy_time(d) = sum of (finish - start) for all nodes scheduled on device d.
        Idle time on device d = makespan - busy_time(d).

        :return: Mean GPU utilization as a fraction in [0, 1].
        :rtype: float
        """
        ms = self.makespan()
        if ms <= 0:
            return 0.0

        intervals: Dict[Tuple[int,int,int], list[tuple[float, float]]] = defaultdict(list)
        for node, s in self.start.items():
            # Use the exact reservation selected by the scheduler.
            # Legacy support: Older traces retain the original destination-only transfer convention.
            allocs = (
                self.reserved_allocations[node]
                if self.reserved_allocations is not None
                else node.allocs[-1:]
                if node.kind in {ExecNodeType.COMM, ExecNodeType.RESHARD}
                else node.allocs
            )
            for alloc in allocs:
                o, e = alloc.origin, alloc.extent
                for r in range(o[0], o[0] + e[0]):
                    for n in range(o[1], o[1] + e[1]):
                        for g in range(o[2], o[2] + e[2]):
                            intervals[(r, n, g)].append((s, self.finish[node]))

        if not intervals:
            return 0.0
        busy: Dict[Tuple[int, int, int], float] = {}
        for device, spans in intervals.items():
            merged = 0.0
            current_start = current_end = None
            for start, end in sorted(spans):
                if current_end is None or start > current_end:
                    if current_end is not None:
                        merged += current_end - current_start
                    current_start, current_end = start, end
                else:
                    current_end = max(current_end, end)
            if current_end is not None:
                merged += current_end - current_start
            busy[device] = merged
        return sum(value / ms for value in busy.values()) / len(busy)

    def n_active(self) -> int:
        """
        Compute the number of GPUs with non-zero busy time.

        :return: Number of GPUs with non-zero busy time.
        :rtype: int
        """
        devices = set()
        for node in self.start:
            for alloc in node.allocs:
                o, e = alloc.origin, alloc.extent
                for r in range(o[0], o[0] + e[0]):
                    for n in range(o[1], o[1] + e[1]):
                        for g in range(o[2], o[2] + e[2]):
                            devices.add((r, n, g))
        return len(devices)
