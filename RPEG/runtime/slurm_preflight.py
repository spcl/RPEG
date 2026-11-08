# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime.errors import RuntimeConfigError


@dataclass(frozen=True)
class GPUHealth:
    hostname: str
    cuda_visible_devices: str | None
    torch_cuda_available: bool
    torch_device_count: int
    nvidia_smi_device_count: int
    ok: bool
    message: str | None = None
    device_names: tuple[str, ...] = ()
    total_memory_bytes: tuple[int, ...] = ()
    driver_version: str | None = None
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    nccl_version: str | None = None
    wall_time_ns: int | None = None
    infiniband_firmware: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "hostname": self.hostname,
            "cuda_visible_devices": self.cuda_visible_devices,
            "torch_cuda_available": self.torch_cuda_available,
            "torch_device_count": self.torch_device_count,
            "nvidia_smi_device_count": self.nvidia_smi_device_count,
            "ok": self.ok,
            "message": self.message,
            "device_names": list(self.device_names),
            "total_memory_bytes": list(self.total_memory_bytes),
            "min_total_memory_bytes": min(self.total_memory_bytes) if self.total_memory_bytes else None,
            "driver_version": self.driver_version,
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
            "nccl_version": self.nccl_version,
            "wall_time_ns": self.wall_time_ns,
            "infiniband_firmware": list(self.infiniband_firmware),
        }


def check_local_gpu_health(expected_gpus: int) -> GPUHealth:
    hostname = os.uname().nodename
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    torch_available = False
    torch_count = 0
    device_names: tuple[str, ...] = ()
    total_memory_bytes: tuple[int, ...] = ()
    torch_version = None
    torch_cuda_version = None
    nccl_version = None
    try:
        import torch

        torch_version = str(getattr(torch, "__version__", "unknown"))
        torch_cuda_version = str(getattr(getattr(torch, "version", None), "cuda", "unknown"))
        try:
            nccl_version = ".".join(str(part) for part in torch.cuda.nccl.version())
        except Exception:
            nccl_version = None
        torch_available = bool(torch.cuda.is_available())
        torch_count = int(torch.cuda.device_count())
        if torch_available:
            device_names = tuple(str(torch.cuda.get_device_name(index)) for index in range(torch_count))
            total_memory_bytes = tuple(
                int(torch.cuda.get_device_properties(index).total_memory)
                for index in range(torch_count)
            )
    except Exception as exc:  # noqa: BLE001 - preflight must report exact environment failures.
        return GPUHealth(
            hostname=hostname,
            cuda_visible_devices=cuda_visible_devices,
            torch_cuda_available=False,
            torch_device_count=0,
            nvidia_smi_device_count=0,
            ok=False,
            message=f"torch CUDA probe failed: {exc}",
        )

    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        return GPUHealth(
            hostname=hostname,
            cuda_visible_devices=cuda_visible_devices,
            torch_cuda_available=torch_available,
            torch_device_count=torch_count,
            nvidia_smi_device_count=0,
            ok=False,
            message=f"nvidia-smi not found: {exc}",
        )
    smi_count = sum(1 for line in proc.stdout.splitlines() if line.startswith("GPU "))
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or f"nvidia-smi exited {proc.returncode}"
        return GPUHealth(
            hostname=hostname,
            cuda_visible_devices=cuda_visible_devices,
            torch_cuda_available=torch_available,
            torch_device_count=torch_count,
            nvidia_smi_device_count=smi_count,
            ok=False,
            message=message,
            device_names=device_names,
            total_memory_bytes=total_memory_bytes,
        )

    driver_version = None
    try:
        driver_proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        versions = sorted({line.strip() for line in driver_proc.stdout.splitlines() if line.strip()})
        if driver_proc.returncode == 0 and versions:
            driver_version = ",".join(versions)
    except FileNotFoundError:
        pass
    firmware = []
    for fw_path in sorted(Path("/sys/class/infiniband").glob("*/fw_ver")):
        try:
            firmware.append(f"{fw_path.parent.name}:{fw_path.read_text().strip()}")
        except OSError:
            continue

    capacity_complete = (
        len(device_names) >= expected_gpus
        and len(total_memory_bytes) >= expected_gpus
        and all(value > 0 for value in total_memory_bytes[:expected_gpus])
    )
    ok = torch_available and torch_count >= expected_gpus and smi_count >= expected_gpus and capacity_complete
    message = None
    if not ok:
        message = (
            f"expected at least {expected_gpus} GPUs, "
            f"torch reports available={torch_available} count={torch_count}, "
            f"nvidia-smi count={smi_count}, capacity records={len(total_memory_bytes)}"
        )
    return GPUHealth(
        hostname=hostname,
        cuda_visible_devices=cuda_visible_devices,
        torch_cuda_available=torch_available,
        torch_device_count=torch_count,
        nvidia_smi_device_count=smi_count,
        ok=ok,
        message=message,
        device_names=device_names,
        total_memory_bytes=total_memory_bytes,
        driver_version=driver_version,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        nccl_version=nccl_version,
        wall_time_ns=time.time_ns(),
        infiniband_firmware=tuple(firmware),
    )


def preflight_cluster_contract(
    path: str | Path,
    *,
    max_clock_skew_s: float = 1.0,
) -> dict[str, Any]:
    """
    Summarize whether preflight rows are comparable for performance work.
    """
    rows = [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    drivers = sorted({str(row["driver_version"]) for row in rows if row.get("driver_version")})
    cuda_versions = sorted({str(row["torch_cuda_version"]) for row in rows if row.get("torch_cuda_version")})
    nccl_versions = sorted({str(row["nccl_version"]) for row in rows if row.get("nccl_version")})
    firmware = sorted({
        tuple(str(value) for value in row.get("infiniband_firmware", ()))
        for row in rows
    })
    clocks = [int(row["wall_time_ns"]) for row in rows if row.get("wall_time_ns") is not None]
    clock_skew_s = (max(clocks) - min(clocks)) / 1e9 if len(clocks) >= 2 else 0.0
    reasons = []
    if len(drivers) > 1:
        reasons.append("heterogeneous NVIDIA driver versions")
    if len(cuda_versions) > 1:
        reasons.append("heterogeneous PyTorch CUDA versions")
    if len(nccl_versions) > 1:
        reasons.append("heterogeneous NCCL versions")
    if len(firmware) > 1:
        reasons.append("heterogeneous InfiniBand firmware")
    if clock_skew_s > float(max_clock_skew_s):
        reasons.append(
            f"node clock skew {clock_skew_s:.6f}s exceeds {float(max_clock_skew_s):.6f}s"
        )
    return {
        "schema_version": "rpeg.cluster_contract.v1",
        "nodes": len(rows),
        "hostnames": sorted(str(row.get("hostname") or "") for row in rows),
        "driver_versions": drivers,
        "torch_cuda_versions": cuda_versions,
        "nccl_versions": nccl_versions,
        "infiniband_firmware_sets": [list(values) for values in firmware],
        "clock_skew_s": clock_skew_s,
        "max_clock_skew_s": float(max_clock_skew_s),
        "performance_comparable": not reasons,
        "non_comparable_reasons": reasons,
    }


def resolve_preflight_hardware(
    path: str | Path,
    *,
    expected_nodes: int,
    explicit_memory_bytes: int | None = None,
    explicit_gpu_type: str | None = None,
) -> tuple[str, int]:
    preflight_path = Path(path)
    if preflight_path.exists():
        rows = [
            json.loads(line)
            for line in preflight_path.read_text().splitlines()
            if line.strip()
        ]
        capacities = [
            int(row["min_total_memory_bytes"])
            for row in rows
            if row.get("min_total_memory_bytes")
        ]
        if len(capacities) != int(expected_nodes):
            raise RuntimeConfigError(
                f"GPU preflight reported memory capacity for {len(capacities)} "
                f"of {expected_nodes} nodes"
            )
        names = sorted({
            str(name)
            for row in rows
            for name in row.get("device_names", [])
            if name
        })
        return ",".join(names) or "unknown", min(capacities)
    if explicit_memory_bytes is None:
        raise RuntimeConfigError(
            "GPU capacity is unknown; enable preflight or provide explicit memory "
            "when GPU preflight is disabled"
        )
    if int(explicit_memory_bytes) <= 0:
        raise RuntimeConfigError("explicit GPU memory capacity must be positive")
    return explicit_gpu_type or "explicit", int(explicit_memory_bytes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate local GPU health inside a Slurm node allocation")
    parser.add_argument("--expected-gpus", type=int)
    parser.add_argument("--contract-from")
    parser.add_argument("--contract-output")
    parser.add_argument("--max-clock-skew-s", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.contract_from:
        contract = preflight_cluster_contract(
            args.contract_from,
            max_clock_skew_s=args.max_clock_skew_s,
        )
        encoded = json.dumps(contract, indent=2, sort_keys=True) + "\n"
        if args.contract_output:
            Path(args.contract_output).write_text(encoded)
        print(encoded, end="", flush=True)
        return 0
    if args.expected_gpus is None:
        parser.error("--expected-gpus is required unless --contract-from is used")
    health = check_local_gpu_health(args.expected_gpus)
    print(json.dumps(health.to_dict(), sort_keys=True), flush=True)
    return 0 if health.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
