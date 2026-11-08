# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys

from dataclasses import dataclass
from pathlib import Path

from runtime.compiler import WorkerSpec
from runtime.control import ControlPlane
from runtime.errors import RuntimeConfigError, RuntimeExecutionError
from runtime.schema import LauncherConfig


@dataclass(frozen=True)
class WorkerLaunch:
    worker_id: str
    command: tuple[str, ...]
    env: dict[str, str]
    stdout_path: Path
    stderr_path: Path
    heartbeat_path: Path
    process: subprocess.Popen | None = None


@dataclass(frozen=True)
class RankAssignment:
    rank: int
    local_rank: int
    hostname: str
    local_cuda_index: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "hostname": self.hostname,
            "local_cuda_index": self.local_cuda_index,
        }


def cuda_visible_devices(worker: WorkerSpec) -> str:
    return ",".join(str(device.local_cuda_index) for device in worker.devices)


def worker_log_paths(log_dir: str | Path, worker_id: str) -> tuple[Path, Path]:
    root = Path(log_dir) / "workers"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{worker_id}.out", root / f"{worker_id}.err"


def rank_assignments(worker: WorkerSpec) -> tuple[RankAssignment, ...]:
    local_counts: dict[str, int] = {}
    assignments: list[RankAssignment] = []
    for rank, device in enumerate(worker.devices):
        local_rank = local_counts.get(device.hostname, 0)
        local_counts[device.hostname] = local_rank + 1
        assignments.append(
            RankAssignment(
                rank=rank,
                local_rank=local_rank,
                hostname=device.hostname,
                local_cuda_index=device.local_cuda_index,
            )
        )
    return tuple(assignments)


def _master_addr(assignments: tuple[RankAssignment, ...]) -> str:
    if not assignments:
        raise RuntimeConfigError("worker has no rank assignments")
    return assignments[0].hostname


def _worker_python(launcher_env: dict[str, str] | None = None) -> str:
    launcher_env = launcher_env or {}
    explicit = launcher_env.get("RPEG_WORKER_PYTHON") or os.environ.get(
        "RPEG_WORKER_PYTHON"
    )
    if explicit:
        return str(explicit)
    # A venv's bin/python may resolve sys.executable to the base uenv binary. Reusing that resolved
    # path in a fresh srun loses the venv site-packages, including Megatron and Transformer Engine.
    # Preserve the invocation path.
    venv = launcher_env.get("RPEG_VENV") or os.environ.get("RPEG_VENV")
    if venv:
        return str(Path(venv) / "bin" / "python")
    return str(sys.executable)


def _rank_env(worker: WorkerSpec, assignments: tuple[RankAssignment, ...]) -> dict[str, str]:
    per_host: dict[str, int] = {}
    for assignment in assignments:
        per_host[assignment.hostname] = per_host.get(assignment.hostname, 0) + 1
    local_world_sizes = set(per_host.values())
    return {
        "MASTER_ADDR": _master_addr(assignments),
        "MASTER_PORT": str(worker.rendezvous_port),
        "WORLD_SIZE": str(len(assignments)),
        "RPEG_WORLD_SIZE": str(len(assignments)),
        "RPEG_LOCAL_WORLD_SIZE": str(next(iter(local_world_sizes))) if len(local_world_sizes) == 1 else "1",
        "RPEG_RANK_ASSIGNMENTS_JSON": json.dumps([a.to_dict() for a in assignments], sort_keys=True),
    }


class LocalWorkerGroupLauncher:
    """
    Launch one worker group on the current node for local/smoke execution.
    """
    def __init__(self, run_id: str, log_dir: str | Path, launcher_config: LauncherConfig | None = None) -> None:
        self.run_id = run_id
        self.log_dir = Path(log_dir)
        self.launcher_config = launcher_config or LauncherConfig()
        self.control = ControlPlane(run_id, self.log_dir)

    def build_env(self, worker: WorkerSpec) -> dict[str, str]:
        assignments = rank_assignments(worker)
        if len(assignments) != 1:
            raise RuntimeConfigError("local launcher supports exactly one rank; use Slurm for distributed workers")
        env = os.environ.copy()
        env.update(self.launcher_config.env)
        env.update(_rank_env(worker, assignments))
        env["RPEG_RUN_ID"] = self.run_id
        env["RPEG_WORKER_ID"] = worker.worker_id
        env["RPEG_VERTEX_ID"] = worker.vertex_id
        env["RANK"] = "0"
        env["LOCAL_RANK"] = "0"
        env["CUDA_VISIBLE_DEVICES"] = str(assignments[0].local_cuda_index)
        return env

    def command(
        self,
        worker: WorkerSpec,
        *,
        once: bool = False,
        entry_module: str = "runtime.worker_entry",
        extra_args: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        cmd = [
            _worker_python(self.launcher_config.env),
            "-m",
            entry_module,
            "--run-id",
            self.run_id,
            "--worker-id",
            worker.worker_id,
            "--vertex-id",
            worker.vertex_id,
            "--heartbeat-path",
            str(self.control.heartbeat_path(worker.worker_id)),
        ]
        if once:
            cmd.append("--once")
        cmd.extend(extra_args)
        return tuple(cmd)

    def launch(
        self,
        worker: WorkerSpec,
        *,
        once: bool = False,
        entry_module: str = "runtime.worker_entry",
        extra_args: tuple[str, ...] = (),
    ) -> WorkerLaunch:
        if any(device.hostname not in {"localhost", os.uname().nodename} for device in worker.devices):
            raise RuntimeConfigError(
                f"local launcher cannot launch {worker.worker_id} on remote devices "
                f"{[device.hostname for device in worker.devices]}"
            )
        stdout_path, stderr_path = worker_log_paths(self.log_dir, worker.worker_id)
        env = self.build_env(worker)
        cmd = self.command(worker, once=once, entry_module=entry_module, extra_args=extra_args)
        self.control.clear_worker_heartbeats(worker.worker_id)
        stdout = stdout_path.open("wb")
        stderr = stderr_path.open("wb")
        try:
            process = subprocess.Popen(
                cmd,
                stdout=stdout,
                stderr=stderr,
                env=env,
                start_new_session=True,
            )
        except Exception:
            stdout.close()
            stderr.close()
            raise
        stdout.close()
        stderr.close()
        return WorkerLaunch(
            worker_id=worker.worker_id,
            command=cmd,
            env=env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            heartbeat_path=self.control.heartbeat_path(worker.worker_id),
            process=process,
        )


class SlurmWorkerGroupLauncher:
    """
    Build and launch per-vertex worker groups inside one Slurm allocation.
    """
    def __init__(self, run_id: str, log_dir: str | Path, launcher_config: LauncherConfig) -> None:
        if launcher_config.type != "slurm":
            raise RuntimeConfigError("SlurmWorkerGroupLauncher requires launcher.type=slurm")
        self.run_id = run_id
        self.log_dir = Path(log_dir)
        self.launcher_config = launcher_config
        self.control = ControlPlane(run_id, self.log_dir)
        self._pool_process: subprocess.Popen | None = None
        self._pool_stop_path: Path | None = None

    def build_env(self, worker: WorkerSpec) -> dict[str, str]:
        assignments = rank_assignments(worker)
        env = os.environ.copy()
        env.update(self.launcher_config.env)
        env.update(_rank_env(worker, assignments))
        env["RPEG_RUN_ID"] = self.run_id
        env["RPEG_WORKER_ID"] = worker.worker_id
        env["RPEG_VERTEX_ID"] = worker.vertex_id
        return env

    def command(
        self,
        worker: WorkerSpec,
        *,
        once: bool = False,
        entry_module: str = "runtime.worker_entry",
        extra_args: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        assignments = rank_assignments(worker)
        hostnames, counts = _host_rank_counts(assignments)
        if not hostnames:
            raise RuntimeConfigError(f"worker {worker.worker_id} has no assigned devices")
        per_node_counts = set(counts.values())
        worker_cmd = [
            _worker_python(self.launcher_config.env),
            "-m",
            entry_module,
            "--run-id",
            self.run_id,
            "--worker-id",
            worker.worker_id,
            "--vertex-id",
            worker.vertex_id,
            "--heartbeat-path",
            str(self.control.heartbeat_path(worker.worker_id)),
        ]
        if once:
            worker_cmd.append("--once")
        worker_cmd.extend(extra_args)
        if len(per_node_counts) != 1:
            return self._heterogeneous_command(hostnames, counts, worker_cmd)
        cmd = self._srun_prefix(
            hostnames=hostnames,
            ntasks=len(assignments),
            ntasks_per_node=next(iter(per_node_counts)),
        )
        cmd.extend(worker_cmd)
        return tuple(cmd)

    def _srun_prefix(self, *, hostnames: list[str], ntasks: int, ntasks_per_node: int) -> list[str]:
        cmd: list[str] = [
            "srun",
            "--overlap",
            "--nodes",
            str(len(hostnames)),
            "--ntasks",
            str(ntasks),
            "--ntasks-per-node",
            str(ntasks_per_node),
            "--nodelist",
            ",".join(hostnames),
            "--distribution",
            "block:block",
            "--export",
            "ALL",
            "--mpi",
            self.launcher_config.mpi,
        ]
        if self.launcher_config.partition:
            cmd.extend(["--partition", self.launcher_config.partition])
        if self.launcher_config.account:
            cmd.extend(["--account", self.launcher_config.account])
        if self.launcher_config.time_limit:
            cmd.extend(["--time", self.launcher_config.time_limit])
        if self.launcher_config.container:
            cmd.extend(["--container-image", self.launcher_config.container])
        return cmd

    def _heterogeneous_command(self, hostnames: list[str], counts: dict[str, int], worker_cmd: list[str]) -> tuple[str, ...]:
        lines = ["set -e", "pids=()", "cleanup() { for pid in \"${pids[@]:-}\"; do kill \"$pid\" 2>/dev/null || true; done; }", "trap cleanup INT TERM"]
        rank_offset = 0
        for hostname in hostnames:
            count = counts[hostname]
            srun_cmd = self._srun_prefix(
                hostnames=[hostname],
                ntasks=count,
                ntasks_per_node=count,
            )
            line = f"RPEG_RANK_OFFSET={rank_offset} {shlex.join(srun_cmd + worker_cmd)} &"
            lines.append(line)
            lines.append("pids+=(\"$!\")")
            rank_offset += count
        lines.append("status=0")
        lines.append("for pid in \"${pids[@]}\"; do wait \"$pid\" || status=$?; done")
        lines.append("exit \"$status\"")
        return ("/usr/bin/bash", "-lc", "\n".join(lines))

    def launch(
        self,
        worker: WorkerSpec,
        *,
        once: bool = False,
        entry_module: str = "runtime.worker_entry",
        extra_args: tuple[str, ...] = (),
    ) -> WorkerLaunch:
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeExecutionError("Slurm launcher must run inside an existing Slurm allocation")
        stdout_path, stderr_path = worker_log_paths(self.log_dir, worker.worker_id)
        env = self.build_env(worker)
        cmd = self.command(worker, once=once, entry_module=entry_module, extra_args=extra_args)
        self.control.clear_worker_heartbeats(worker.worker_id)
        stdout = stdout_path.open("wb")
        stderr = stderr_path.open("wb")
        try:
            process = subprocess.Popen(
                cmd,
                stdout=stdout,
                stderr=stderr,
                env=env,
                start_new_session=True,
            )
        except Exception:
            stdout.close()
            stderr.close()
            raise
        stdout.close()
        stderr.close()
        return WorkerLaunch(
            worker_id=worker.worker_id,
            command=cmd,
            env=env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            heartbeat_path=self.control.heartbeat_path(worker.worker_id),
            process=process,
        )
    def launch_pool(
        self,
        workers: tuple[WorkerSpec, ...],
        *,
        entry_module: str = "runtime.semantic_worker",
        extra_args: tuple[str, ...] = (),
    ) -> dict[str, WorkerLaunch]:
        """
        Launch every semantic rank in one Slurm step.

        HPE Slingshot security credentials are scoped to a Slurm step when job_vni is disabled. A
        single step is therefore required for reliable NCCL communication between independently
        placed RPEG roles.
        """
        if not self.launcher_config.shared_step:
            raise RuntimeConfigError("launch_pool requires launcher.shared_step=true")
        if entry_module != "runtime.semantic_worker":
            raise RuntimeConfigError("shared Slurm-step launcher supports runtime.semantic_worker only")
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeExecutionError("Slurm launcher must run inside an existing Slurm allocation")
        if self._pool_process is not None and self._pool_process.poll() is None:
            raise RuntimeExecutionError("shared semantic Slurm step is already running")
        if not workers:
            return {}

        hostnames: list[str] = []
        per_host: dict[str, list[tuple[WorkerSpec, RankAssignment]]] = {}
        worker_assignments: dict[str, tuple[RankAssignment, ...]] = {}
        for worker in workers:
            assignments = rank_assignments(worker)
            if not assignments:
                raise RuntimeConfigError(f"worker {worker.worker_id} has no assigned devices")
            worker_assignments[worker.worker_id] = assignments
            self.control.clear_worker_heartbeats(worker.worker_id)
            for assignment in assignments:
                if assignment.hostname not in per_host:
                    hostnames.append(assignment.hostname)
                    per_host[assignment.hostname] = []
                per_host[assignment.hostname].append((worker, assignment))

        slots_per_host = max(len(per_host[hostname]) for hostname in hostnames)
        pool_root = self.log_dir / "semantic_pool"
        pool_root.mkdir(parents=True, exist_ok=True)
        stop_path = pool_root / "stop"
        stop_path.unlink(missing_ok=True)
        manifest_path = pool_root / "launch_manifest.json"
        tasks: dict[str, dict[str, object]] = {}
        host_task_ids: dict[str, list[str]] = {hostname: [] for hostname in hostnames}
        worker_task_envs: dict[str, dict[str, str]] = {}
        for host_index, hostname in enumerate(hostnames):
            host_tasks = per_host[hostname]
            for slot in range(slots_per_host):
                task_id = str(host_index * slots_per_host + slot)
                host_task_ids[hostname].append(task_id)
                if slot >= len(host_tasks):
                    tasks[task_id] = {"idle": True, "hostname": hostname}
                    continue
                worker, assignment = host_tasks[slot]
                assignments = worker_assignments[worker.worker_id]
                task_env = _rank_env(worker, assignments)
                task_env.update(
                    {
                        "RPEG_RUN_ID": self.run_id,
                        "RPEG_WORKER_ID": worker.worker_id,
                        "RPEG_VERTEX_ID": worker.vertex_id,
                        "RANK": str(assignment.rank),
                        "LOCAL_RANK": "0",
                    }
                )
                stdout_path, stderr_path = worker_log_paths(self.log_dir, worker.worker_id)
                task_argv = [
                    "--run-id",
                    self.run_id,
                    "--worker-id",
                    worker.worker_id,
                    "--vertex-id",
                    worker.vertex_id,
                    "--heartbeat-path",
                    str(self.control.heartbeat_path(worker.worker_id)),
                    *extra_args,
                ]
                tasks[task_id] = {
                    "idle": False,
                    "hostname": hostname,
                    "worker_id": worker.worker_id,
                    "worker_rank": assignment.rank,
                    "env": task_env,
                    "argv": task_argv,
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                }
                worker_task_envs.setdefault(worker.worker_id, task_env)
        manifest = {
            "schema_version": "rpeg.semantic_pool.v1",
            "run_id": self.run_id,
            "stop_path": str(stop_path),
            "host_tasks": host_task_ids,
            "hostnames": hostnames,
            "slots_per_host": slots_per_host,
            "tasks": tasks,
        }
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary_manifest.replace(manifest_path)

        command = self._srun_prefix(
            hostnames=hostnames,
            ntasks=len(hostnames) * slots_per_host,
            ntasks_per_node=slots_per_host,
        )
        command.extend(
            [
                _worker_python(self.launcher_config.env),
                "-m",
                "runtime.semantic_pool_entry",
                "--manifest",
                str(manifest_path),
            ]
        )
        cmd = tuple(command)
        pool_env = os.environ.copy()
        pool_env.update(self.launcher_config.env)
        pool_stdout_path = pool_root / "semantic_pool.out"
        pool_stderr_path = pool_root / "semantic_pool.err"
        stdout = pool_stdout_path.open("wb")
        stderr = pool_stderr_path.open("wb")
        try:
            process = subprocess.Popen(
                cmd,
                stdout=stdout,
                stderr=stderr,
                env=pool_env,
                start_new_session=True,
            )
        except Exception:
            stdout.close()
            stderr.close()
            raise
        stdout.close()
        stderr.close()
        self._pool_process = process
        self._pool_stop_path = stop_path
        launches: dict[str, WorkerLaunch] = {}
        for worker in workers:
            stdout_path, stderr_path = worker_log_paths(self.log_dir, worker.worker_id)
            launches[worker.worker_id] = WorkerLaunch(
                worker_id=worker.worker_id,
                command=cmd,
                env={**pool_env, **worker_task_envs[worker.worker_id]},
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                heartbeat_path=self.control.heartbeat_path(worker.worker_id),
                process=process,
            )
        return launches

    def stop_pool(self) -> None:
        if self._pool_stop_path is not None:
            self._pool_stop_path.parent.mkdir(parents=True, exist_ok=True)
            self._pool_stop_path.touch()


def _host_rank_counts(assignments: tuple[RankAssignment, ...]) -> tuple[list[str], dict[str, int]]:
    hostnames = []
    counts: dict[str, int] = {}
    for assignment in assignments:
        if assignment.hostname not in counts:
            hostnames.append(assignment.hostname)
        counts[assignment.hostname] = counts.get(assignment.hostname, 0) + 1
    return hostnames, counts
