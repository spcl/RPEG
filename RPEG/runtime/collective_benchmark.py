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
import socket
import time

from pathlib import Path


def _rank_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    return rank, world_size, local_rank


def _parse_sizes(value: str) -> tuple[int, ...]:
    suffixes = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}
    sizes: list[int] = []
    for raw in value.split(","):
        text = raw.strip().lower()
        if not text:
            continue
        multiplier = suffixes.get(text[-1], 1)
        number = text[:-1] if multiplier != 1 else text
        size = int(float(number) * multiplier)
        if size < 4:
            raise argparse.ArgumentTypeError("collective message sizes must be at least four bytes")
        sizes.append(size)
    if not sizes:
        raise argparse.ArgumentTypeError("at least one collective message size is required")
    return tuple(sizes)


def benchmark_all_reduce(
    *,
    sizes: tuple[int, ...],
    warmups: int,
    iterations: int,
) -> list[dict[str, float | int]]:
    import torch
    import torch.distributed as dist

    rank, world_size, local_rank = _rank_context()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the collective benchmark")
    visible_devices = torch.cuda.device_count()
    device_index = 0 if visible_devices == 1 else local_rank
    if device_index >= visible_devices:
        raise RuntimeError(
            f"local rank {local_rank} cannot select from {visible_devices} visible CUDA devices"
        )
    torch.cuda.set_device(device_index)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    device = torch.device("cuda", device_index)
    results: list[dict[str, float | int]] = []
    try:
        for size_bytes in sizes:
            elements = (size_bytes + 3) // 4
            tensor = torch.ones(elements, dtype=torch.float32, device=device)
            for _ in range(warmups):
                dist.all_reduce(tensor)
                tensor.fill_(1.0)
            torch.cuda.synchronize(device)
            dist.barrier()

            samples: list[float] = []
            for _ in range(iterations):
                dist.barrier()
                start = time.perf_counter()
                dist.all_reduce(tensor)
                torch.cuda.synchronize(device)
                samples.append(time.perf_counter() - start)
                tensor.fill_(1.0)

            local = torch.tensor(
                [min(samples), sum(samples) / len(samples), max(samples)],
                dtype=torch.float64,
                device=device,
            )
            global_min = local.clone()
            global_mean = local.clone()
            global_max = local.clone()
            dist.all_reduce(global_min, op=dist.ReduceOp.MIN)
            dist.all_reduce(global_mean, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
            global_mean /= world_size

            duration_s = float(global_max[2].item())
            algorithmic_gbps = size_bytes / duration_s / 1e9
            bus_factor = 2.0 * (world_size - 1) / world_size if world_size > 1 else 1.0
            results.append(
                {
                    "size_bytes": size_bytes,
                    "iterations": iterations,
                    "duration_min_s": float(global_min[0].item()),
                    "duration_mean_s": float(global_mean[1].item()),
                    "duration_max_s": duration_s,
                    "algorithmic_bandwidth_gbps": algorithmic_gbps,
                    "bus_bandwidth_gbps": algorithmic_gbps * bus_factor,
                }
            )
    finally:
        dist.destroy_process_group()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark NCCL all-reduce inside an existing Slurm allocation")
    parser.add_argument("--sizes", type=_parse_sizes, default=_parse_sizes("1M,16M,256M,1G"))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-large-message-gbps", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.iterations < 1:
        parser.error("--warmups must be >=0 and --iterations must be >=1")

    rank, world_size, local_rank = _rank_context()
    results = benchmark_all_reduce(sizes=args.sizes, warmups=args.warmups, iterations=args.iterations)
    if rank != 0:
        return 0

    report = {
        "schema_version": "rpeg.collective_benchmark.v1",
        "world_size": world_size,
        "gpus_per_node": int(os.environ.get("SLURM_NTASKS_PER_NODE", "4").split("(")[0]),
        "hostname": socket.gethostname(),
        "torch_version": None,
        "cuda_version": None,
        "nccl_version": None,
        "results": results,
    }
    import torch

    report["torch_version"] = torch.__version__
    report["cuda_version"] = torch.version.cuda
    report["nccl_version"] = torch.cuda.nccl.version()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)

    observed = float(results[-1]["bus_bandwidth_gbps"])
    if args.min_large_message_gbps > 0 and observed < args.min_large_message_gbps:
        raise SystemExit(
            f"large-message bus bandwidth {observed:.3f} GB/s is below "
            f"{args.min_large_message_gbps:.3f} GB/s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
