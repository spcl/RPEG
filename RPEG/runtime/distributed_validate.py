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
import time

from pathlib import Path

from runtime.control import Heartbeat, HeartbeatStore
from runtime.worker_entry import _rank_context


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Distributed CUDA/NCCL validation worker")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--vertex-id", required=True)
    parser.add_argument("--heartbeat-path", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    rank, world_size, local_rank = _rank_context()
    store = HeartbeatStore(Path(args.heartbeat_path).parent)

    def heartbeat(state: str, message: str | None = None) -> None:
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
                    "hostname": os.uname().nodename,
                },
            )
        )

    heartbeat("starting")
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            backend = "nccl"
            value = torch.tensor([float(rank)], device="cuda")
        else:
            backend = "gloo"
            value = torch.tensor([float(rank)])
        torch.distributed.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
        torch.distributed.all_reduce(value)
        expected = float(world_size * (world_size - 1) // 2)
        if float(value.cpu()[0]) != expected:
            raise RuntimeError(f"all_reduce result {float(value.cpu()[0])} != {expected}")
        print(json.dumps({"rank": rank, "world_size": world_size, "all_reduce": float(value.cpu()[0])}), flush=True)
        torch.distributed.destroy_process_group()
    except Exception as exc:
        heartbeat("error", str(exc))
        raise
    heartbeat("finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
