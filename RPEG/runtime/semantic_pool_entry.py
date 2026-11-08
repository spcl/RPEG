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
from typing import Any

from runtime.errors import RuntimeConfigError


def _redirect(path: str, fd: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    opened = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o640)
    try:
        os.dup2(opened, fd)
    finally:
        os.close(opened)


def _idle(stop_path: Path) -> int:
    stopping = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping and not stop_path.exists():
        time.sleep(0.1)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RPEG shared Slurm-step task dispatcher")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text())
    actual_host = os.uname().nodename.split(".", 1)[0]
    local_id = int(os.environ.get("SLURM_LOCALID", "0"))
    host_tasks = (manifest.get("host_tasks") or {}).get(actual_host)
    if not isinstance(host_tasks, list) or not 0 <= local_id < len(host_tasks):
        raise RuntimeConfigError(
            f"shared Slurm-step has no local task {local_id} for host {actual_host}"
        )
    task_id = str(host_tasks[local_id])
    task = (manifest.get("tasks") or {}).get(task_id)
    if not isinstance(task, dict):
        raise RuntimeConfigError(f"shared Slurm-step manifest has no task {task_id}")
    expected_host = str(task.get("hostname") or "").split(".", 1)[0]
    if expected_host and actual_host != expected_host:
        raise RuntimeConfigError(
            f"shared Slurm-step task {task_id} expected host {expected_host}, got {actual_host}"
        )
    stop_path = Path(str(manifest["stop_path"]))
    if bool(task.get("idle", False)):
        return _idle(stop_path)
    env = task.get("env")
    command = task.get("argv")
    if not isinstance(env, dict) or not isinstance(command, list):
        raise RuntimeConfigError(f"malformed shared Slurm-step task {task_id}")
    os.environ.update({str(key): str(value) for key, value in env.items()})
    _redirect(str(task["stdout_path"]), 1)
    _redirect(str(task["stderr_path"]), 2)
    from runtime.semantic_worker import main as semantic_worker_main

    return semantic_worker_main([str(value) for value in command])


if __name__ == "__main__":
    raise SystemExit(main())
