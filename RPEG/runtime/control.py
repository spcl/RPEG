# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import errno
import json
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime.errors import RuntimeExecutionError


@dataclass(frozen=True)
class Heartbeat:
    worker_id: str
    run_id: str
    state: str
    timestamp: float
    pid: int | None = None
    rank: int | None = None
    world_size: int | None = None
    local_rank: int | None = None
    message: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "run_id": self.run_id,
            "state": self.state,
            "timestamp": self.timestamp,
            "pid": self.pid,
            "rank": self.rank,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "message": self.message,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Heartbeat":
        return cls(
            worker_id=str(d["worker_id"]),
            run_id=str(d["run_id"]),
            state=str(d["state"]),
            timestamp=float(d["timestamp"]),
            pid=None if d.get("pid") is None else int(d["pid"]),
            rank=None if d.get("rank") is None else int(d["rank"]),
            world_size=None if d.get("world_size") is None else int(d["world_size"]),
            local_rank=None if d.get("local_rank") is None else int(d["local_rank"]),
            message=d.get("message"),
            metadata=d.get("metadata") or {},
        )


class HeartbeatStore:
    """
    File-backed heartbeat store shared by launcher and worker processes.
    """
    _TRANSIENT_READ_ERRNOS = {errno.EAGAIN, errno.EINTR, errno.EIO, errno.ESTALE, errno.ETIMEDOUT}
    _READ_ATTEMPTS = 5
    _READ_RETRY_S = 0.02

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, worker_id: str, rank: int | None = None) -> Path:
        if rank is not None:
            return self.root / f"{worker_id}.rank{rank}.heartbeat.json"
        return self.root / f"{worker_id}.heartbeat.json"

    def write(self, heartbeat: Heartbeat) -> None:
        rank = heartbeat.rank if heartbeat.world_size and heartbeat.world_size > 1 else None
        path = self.path_for(heartbeat.worker_id, rank=rank)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(heartbeat.to_dict(), sort_keys=True) + "\n")
        tmp.replace(path)

    def read(self, worker_id: str, rank: int | None = None) -> Heartbeat:
        path = self.path_for(worker_id, rank=rank)
        heartbeat = self._read_optional_path(path)
        if heartbeat is None:
            raise RuntimeExecutionError(f"missing heartbeat for worker {worker_id}: {path}")
        return heartbeat

    def read_optional(self, worker_id: str, rank: int | None = None) -> Heartbeat | None:
        path = self.path_for(worker_id, rank=rank)
        return self._read_optional_path(path)

    def _read_optional_path(self, path: Path) -> Heartbeat | None:
        for attempt in range(self._READ_ATTEMPTS):
            try:
                return Heartbeat.from_dict(json.loads(path.read_text()))
            except FileNotFoundError:
                return None
            except OSError as exc:
                if exc.errno not in self._TRANSIENT_READ_ERRNOS or attempt + 1 == self._READ_ATTEMPTS:
                    raise RuntimeExecutionError(f"failed to read heartbeat {path}: {exc}") from exc
                time.sleep(self._READ_RETRY_S)
        raise AssertionError("unreachable")

    def clear(self, worker_id: str) -> None:
        paths = [self.path_for(worker_id)]
        paths.extend(self.root.glob(f"{worker_id}.rank*.heartbeat.json"))
        for path in paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


class ControlPlane:
    """
    Minimal control plane.
    """
    def __init__(self, run_id: str, log_dir: str | Path) -> None:
        self.run_id = run_id
        self.log_dir = Path(log_dir)
        self.heartbeat_store = HeartbeatStore(self.log_dir / "heartbeats")

    def heartbeat_path(self, worker_id: str, rank: int | None = None) -> Path:
        return self.heartbeat_store.path_for(worker_id, rank=rank)

    def clear_worker_heartbeats(self, worker_id: str) -> None:
        self.heartbeat_store.clear(worker_id)

    def record_launcher_state(self, worker_id: str, state: str, message: str | None = None) -> None:
        self.heartbeat_store.write(
            Heartbeat(
                worker_id=worker_id,
                run_id=self.run_id,
                state=state,
                timestamp=time.time(),
                message=message,
            )
        )

    def wait_for_state(
        self,
        worker_id: str,
        states: set[str],
        timeout_s: float,
        rank: int | None = None,
        fail_states: set[str] | None = None,
    ) -> Heartbeat:
        deadline = time.monotonic() + timeout_s
        last_error: RuntimeExecutionError | None = None
        terminal = (fail_states or {"error", "stopping", "stopped"}) - states
        while time.monotonic() < deadline:
            try:
                heartbeat = self.heartbeat_store.read(worker_id, rank=rank)
            except RuntimeExecutionError as exc:
                last_error = exc
            else:
                if heartbeat.state in states:
                    return heartbeat
                if heartbeat.state in terminal:
                    rank_label = "" if rank is None else f" rank {rank}"
                    message = "" if not heartbeat.message else f": {heartbeat.message}"
                    raise RuntimeExecutionError(
                        f"worker {worker_id}{rank_label} reached terminal state {heartbeat.state!r}{message}"
                    )
            time.sleep(0.05)
        if last_error is not None:
            raise last_error
        raise RuntimeExecutionError(
            f"worker {worker_id} did not reach one of {sorted(states)} within {timeout_s:.1f}s"
        )

    def wait_for_all_ranks(
        self,
        worker_id: str,
        world_size: int,
        states: set[str],
        timeout_s: float,
    ) -> tuple[Heartbeat, ...]:
        return tuple(self.wait_for_state(worker_id, states, timeout_s, rank=rank) for rank in range(world_size))

    def rank_snapshot(self, worker_id: str, world_size: int) -> tuple[Heartbeat | None, ...]:
        if world_size <= 1:
            heartbeat = self.heartbeat_store.read_optional(worker_id)
            if heartbeat is None:
                heartbeat = self.heartbeat_store.read_optional(worker_id, rank=0)
            return (heartbeat,)
        return tuple(self.heartbeat_store.read_optional(worker_id, rank=rank) for rank in range(world_size))
