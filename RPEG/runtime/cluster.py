# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import os
import subprocess

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.types import Allocation, DeviceSpace
from runtime.errors import RuntimeConfigError
from runtime.io import load_mapping


@dataclass(frozen=True)
class GPURef:
    rack: int
    node: int
    gpu: int
    hostname: str
    local_cuda_index: int
    pci_bus_id: str | None = None


@dataclass(frozen=True)
class ClusterInventory:
    device_space: DeviceSpace
    gpus: dict[tuple[int, int, int], GPURef]

    def resolve_allocation(self, allocation: Allocation) -> tuple[GPURef, ...]:
        refs = []
        r0, n0, g0 = allocation.origin
        r_ext, n_ext, g_ext = allocation.extent
        for r in range(r0, r0 + r_ext):
            for n in range(n0, n0 + n_ext):
                for g in range(g0, g0 + g_ext):
                    key = (r, n, g)
                    if key not in self.gpus:
                        raise RuntimeConfigError(f"Allocation references missing device {key}")
                    refs.append(self.gpus[key])
        return tuple(refs)


def local_inventory(device_space: DeviceSpace, hostname: str = "localhost") -> ClusterInventory:
    gpus = {}
    for r in range(device_space.R):
        for n in range(device_space.N):
            node_name = hostname if device_space.N == 1 else f"{hostname}-{n}"
            for g in range(device_space.G):
                gpus[(r, n, g)] = GPURef(r, n, g, node_name, g)
    return ClusterInventory(device_space=device_space, gpus=gpus)


def _parse_gpu_ids(value: str | None, fallback_count: int) -> list[int]:
    if not value:
        return list(range(fallback_count))
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out or list(range(fallback_count))


def query_nvidia_smi_bus_ids() -> list[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"],
            text=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def slurm_inventory_from_env(device_space: DeviceSpace) -> ClusterInventory:
    hostnames = slurm_allocated_hostnames()
    if not hostnames:
        hostnames = [os.uname().nodename]
    bus_ids = query_nvidia_smi_bus_ids()
    job_gpu_ids = _parse_gpu_ids(os.environ.get("SLURM_JOB_GPUS"), device_space.G)
    gpus: dict[tuple[int, int, int], GPURef] = {}
    for r in range(device_space.R):
        for n in range(device_space.N):
            hostname = hostnames[n % len(hostnames)]
            for g in range(device_space.G):
                local_idx = job_gpu_ids[g % len(job_gpu_ids)]
                gpus[(r, n, g)] = GPURef(
                    rack=r,
                    node=n,
                    gpu=g,
                    hostname=hostname,
                    local_cuda_index=local_idx,
                    pci_bus_id=bus_ids[local_idx] if local_idx < len(bus_ids) else None,
                )
    return ClusterInventory(device_space=device_space, gpus=gpus)


def load_inventory(path: str | None, device_space: DeviceSpace, scheduler: str = "local") -> ClusterInventory:
    if path is None:
        if scheduler == "slurm":
            return slurm_inventory_from_env(device_space)
        return local_inventory(device_space)
    p = Path(path)
    if not p.exists():
        raise RuntimeConfigError(f"Cluster inventory not found: {p}")
    data = load_mapping(p)
    racks = data.get("racks")
    if not isinstance(racks, list):
        raise RuntimeConfigError("cluster inventory requires a racks list")
    gpus: dict[tuple[int, int, int], GPURef] = {}
    for r_idx, rack in enumerate(racks):
        nodes = rack.get("nodes") if isinstance(rack, dict) else None
        if not isinstance(nodes, list):
            raise RuntimeConfigError(f"inventory rack {r_idx} requires nodes list")
        for n_idx, node in enumerate(nodes):
            name = str(node.get("name", f"node{n_idx}"))
            gpu_ids = node.get("gpus")
            if not isinstance(gpu_ids, list):
                raise RuntimeConfigError(f"inventory node {name} requires gpus list")
            for logical_g, local_idx in enumerate(gpu_ids):
                gpus[(r_idx, n_idx, logical_g)] = GPURef(
                    rack=r_idx,
                    node=n_idx,
                    gpu=logical_g,
                    hostname=name,
                    local_cuda_index=int(local_idx),
                    pci_bus_id=None,
                )
    inv = ClusterInventory(device_space=device_space, gpus=gpus)
    expected = device_space.num_devices()
    if len(inv.gpus) < expected:
        raise RuntimeConfigError(f"inventory has {len(inv.gpus)} GPUs, plan requires {expected}")
    return inv


def slurm_allocated_hostnames() -> list[str]:
    nodelist = os.environ.get("SLURM_NODELIST")
    if not nodelist:
        return []
    try:
        out = subprocess.check_output(["scontrol", "show", "hostnames", nodelist], text=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]
