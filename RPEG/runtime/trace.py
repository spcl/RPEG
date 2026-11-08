# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import json
import time

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class RuntimeTraceEvent:
    run_id: str
    event: str
    node_id: str
    timestamp: float
    status: str = "ok"
    worker_id: str | None = None
    step: int | None = None
    counters: dict[str, Any] = field(default_factory=dict)
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "run_id": self.run_id,
            "event": self.event,
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "status": self.status,
            "counters": self.counters,
        }
        if self.worker_id is not None:
            d["worker_id"] = self.worker_id
        if self.step is not None:
            d["step"] = self.step
        if self.message is not None:
            d["message"] = self.message
        return d


class RuntimeTraceWriter:
    """
    JSONL trace writer keyed by lowered runtime node IDs.
    """
    def __init__(self, path: str | Path, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        event: str,
        node_id: str,
        *,
        status: str = "ok",
        worker_id: str | None = None,
        step: int | None = None,
        counters: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> None:
        trace_event = RuntimeTraceEvent(
            run_id=self.run_id,
            event=event,
            node_id=node_id,
            timestamp=time.time(),
            status=status,
            worker_id=worker_id,
            step=step,
            counters=counters or {},
            message=message,
        )
        with self.path.open("a") as f:
            f.write(json.dumps(trace_event.to_dict(), sort_keys=True) + "\n")

    @contextmanager
    def node(
        self,
        node_id: str,
        *,
        worker_id: str | None = None,
        step: int | None = None,
        counters: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        start = time.perf_counter()
        dynamic_counters: dict[str, Any] = dict(counters or {})
        self.append("start", node_id, worker_id=worker_id, step=step, counters=dynamic_counters)
        try:
            yield dynamic_counters
        except Exception as exc:
            dynamic_counters["duration_s"] = time.perf_counter() - start
            self.append(
                "finish",
                node_id,
                status="error",
                worker_id=worker_id,
                step=step,
                counters=dynamic_counters,
                message=str(exc),
            )
            raise
        dynamic_counters["duration_s"] = time.perf_counter() - start
        self.append("finish", node_id, worker_id=worker_id, step=step, counters=dynamic_counters)
