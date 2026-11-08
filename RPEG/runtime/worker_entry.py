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
import signal
import time

from pathlib import Path

from runtime.control import Heartbeat, HeartbeatStore


def _rank_context() -> tuple[int | None, int | None, int | None]:
    world_size = int(os.environ.get("RPEG_WORLD_SIZE") or os.environ.get("WORLD_SIZE") or "1")
    rank_offset = int(os.environ.get("RPEG_RANK_OFFSET", "0"))
    if os.environ.get("RANK") is not None:
        rank = int(os.environ["RANK"])
    else:
        rank = rank_offset + int(
            os.environ.get("SLURM_PROCID")
            or os.environ.get("OMPI_COMM_WORLD_RANK")
            or "0"
        )
    assignments_raw = os.environ.get("RPEG_RANK_ASSIGNMENTS_JSON")
    if assignments_raw:
        assignments = json.loads(assignments_raw)
        selected = next((a for a in assignments if int(a["rank"]) == rank), None)
        if selected is not None:
            os.environ["RANK"] = str(selected["rank"])
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["RPEG_PHYSICAL_LOCAL_RANK"] = str(selected["local_rank"])
            os.environ["CUDA_VISIBLE_DEVICES"] = str(selected["local_cuda_index"])
            # After narrowing CUDA_VISIBLE_DEVICES to one physical GPU, the process-local CUDA
            # ordinal is always 0. Keep the physical local rank separately for logs and scheduler
            # diagnostics.
            os.environ["LOCAL_RANK"] = "0"
            return int(selected["rank"]), world_size, int(selected["local_rank"])
    local_rank = int(
        os.environ.get("LOCAL_RANK")
        or os.environ.get("SLURM_LOCALID")
        or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
        or "0"
    )
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    return rank, world_size, local_rank


def _write(
    store: HeartbeatStore,
    args,
    state: str,
    rank: int | None,
    world_size: int | None,
    local_rank: int | None,
    message: str | None = None,
) -> None:
    store.write(
        Heartbeat(
            worker_id=args.worker_id,
            run_id=args.run_id,
            state=state,
            timestamp=time.time(),
            pid=os.getpid(),
            rank=rank if world_size and world_size > 1 else None,
            world_size=world_size,
            local_rank=local_rank,
            message=message,
            metadata={
                "vertex_id": args.vertex_id,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "master_addr": os.environ.get("MASTER_ADDR", ""),
                "master_port": os.environ.get("MASTER_PORT", ""),
                "hostname": os.uname().nodename,
            },
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RPEG Phase 2 worker-group heartbeat process")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--vertex-id", required=True)
    parser.add_argument("--heartbeat-path", required=True)
    parser.add_argument("--interval-s", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    store = HeartbeatStore(Path(args.heartbeat_path).parent)
    rank, world_size, local_rank = _rank_context()
    stopping = False

    def _stop(signum, _frame) -> None:
        nonlocal stopping
        stopping = True
        _write(store, args, "stopping", rank, world_size, local_rank, f"received signal {signum}")

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    print(
        f"worker_id={args.worker_id} vertex_id={args.vertex_id} "
        f"rank={rank} world_size={world_size} local_rank={local_rank} "
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        flush=True,
    )
    _write(store, args, "ready", rank, world_size, local_rank)
    if args.once:
        _write(store, args, "finished", rank, world_size, local_rank)
        return 0
    while not stopping:
        _write(store, args, "ready", rank, world_size, local_rank)
        time.sleep(args.interval_s)
    _write(store, args, "stopped", rank, world_size, local_rank)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
