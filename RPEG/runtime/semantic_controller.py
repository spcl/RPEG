# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import time
import copy
import os
import signal

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from runtime.compiler import WorkerSpec
from runtime.control import Heartbeat
from runtime.errors import RuntimeExecutionError
from runtime.launcher import LocalWorkerGroupLauncher, SlurmWorkerGroupLauncher, WorkerLaunch
from runtime.semantic import (
    SemanticOperationNames,
    SemanticResult,
    SemanticWorkerGroupClient,
    SocketSemanticWorkerGroupClient,
)


@dataclass
class ManagedSemanticWorker:
    worker: WorkerSpec
    launch: WorkerLaunch
    client: SemanticWorkerGroupClient | SocketSemanticWorkerGroupClient


@dataclass(frozen=True)
class _ReplayRequest:
    operation: str
    node_id: str
    step: int
    payload: dict[str, Any]
    timeout_s: float | None


class SemanticWorkerGroupRuntime:
    """
    Controller for persistent semantic worker groups.

    The runtime keeps worker process lifetime separate from RPEG graph scheduling. The executor
    decides which semantic operation should run next; this controller only ensures that all ranks in
    a worker group receive the same command and that every rank returns a result or times out.
    """
    def __init__(
        self,
        *,
        run_id: str,
        launcher: LocalWorkerGroupLauncher | SlurmWorkerGroupLauncher,
        command_root: str | Path,
        state_root: str | Path | None = None,
        backend: str,
        control_transport: str = "tcp",
        ready_timeout_s: float = 120.0,
        operation_timeout_s: float = 600.0,
        restart_retries: int = 0,
        client_factory: Callable[
            [WorkerSpec, Path, float],
            SemanticWorkerGroupClient | SocketSemanticWorkerGroupClient,
        ]
        | None = None,
    ) -> None:
        self.run_id = run_id
        self.launcher = launcher
        self.command_root = Path(command_root)
        self.state_root = None if state_root is None else Path(state_root)
        self.backend = backend
        self.control_transport = str(control_transport).strip().lower()
        if self.control_transport not in {"tcp", "file"}:
            raise RuntimeExecutionError(f"unsupported semantic control transport {control_transport!r}")
        self.ready_timeout_s = ready_timeout_s
        self.operation_timeout_s = operation_timeout_s
        self.restart_retries = max(0, int(restart_retries))
        self.client_factory = client_factory
        self._workers: dict[str, ManagedSemanticWorker] = {}
        self._restart_attempts: dict[str, int] = {}
        self._restart_safe: dict[str, bool] = {}
        self._replay_journal: dict[str, list[_ReplayRequest]] = {}
        self._rank_host_cache: dict[str, tuple[str, ...]] = {}

    def start_worker(self, worker: WorkerSpec) -> ManagedSemanticWorker:
        if worker.worker_id in self._workers:
            return self._workers[worker.worker_id]
        return self._launch_worker(worker)

    def _launch_worker(self, worker: WorkerSpec) -> ManagedSemanticWorker:
        extra_args = (
            "--command-root",
            str(self.command_root),
            "--backend",
            self.backend,
            "--control-transport",
            self.control_transport,
        )
        if self.state_root is not None:
            extra_args = extra_args + ("--state-root", str(self.state_root))
        launch = self.launcher.launch(
            worker,
            entry_module="runtime.semantic_worker",
            extra_args=extra_args,
        )
        world_size = len(worker.devices)
        try:
            if world_size == 1:
                ready_snapshot = (
                    self.launcher.control.wait_for_state(
                        worker.worker_id,
                        {"ready"},
                        timeout_s=self.ready_timeout_s,
                    ),
                )
            else:
                ready_snapshot = self.launcher.control.wait_for_all_ranks(
                    worker.worker_id,
                    world_size,
                    {"ready"},
                    timeout_s=self.ready_timeout_s,
                )
        except RuntimeExecutionError as exc:
            raise RuntimeExecutionError(
                f"semantic worker {worker.worker_id} failed to become ready: {exc}; "
                f"{self._worker_diagnostics(worker, launch)}"
            ) from exc
        client = self._client_for_worker(worker, snapshot=ready_snapshot)
        managed = ManagedSemanticWorker(worker=worker, launch=launch, client=client)
        self._workers[worker.worker_id] = managed
        self._restart_safe.setdefault(worker.worker_id, True)
        self._replay_journal.setdefault(worker.worker_id, [])
        return managed

    def start_all(self, workers: tuple[WorkerSpec, ...]) -> tuple[ManagedSemanticWorker, ...]:
        if isinstance(self.launcher, SlurmWorkerGroupLauncher) and self.launcher.launcher_config.shared_step:
            return self._start_shared_step(workers)
        return tuple(self.start_worker(worker) for worker in workers)

    def _start_shared_step(self, workers: tuple[WorkerSpec, ...]) -> tuple[ManagedSemanticWorker, ...]:
        worker_ids = [worker.worker_id for worker in workers]
        if len(set(worker_ids)) != len(worker_ids):
            raise RuntimeExecutionError("shared semantic Slurm step contains duplicate worker IDs")
        if all(worker_id in self._workers for worker_id in worker_ids):
            return tuple(self._workers[worker_id] for worker_id in worker_ids)
        if self._workers:
            raise RuntimeExecutionError("shared semantic Slurm step must start all workers atomically")
        extra_args = (
            "--command-root",
            str(self.command_root),
            "--backend",
            self.backend,
            "--control-transport",
            self.control_transport,
        )
        if self.state_root is not None:
            extra_args = extra_args + ("--state-root", str(self.state_root))
        launches = self.launcher.launch_pool(
            workers,
            entry_module="runtime.semantic_worker",
            extra_args=extra_args,
        )
        process = next(iter(launches.values())).process if launches else None
        ready_snapshots = {}
        try:
            for worker in workers:
                world_size = len(worker.devices)
                if world_size == 1:
                    ready_snapshots[worker.worker_id] = (
                        self.launcher.control.wait_for_state(
                            worker.worker_id,
                            {"ready"},
                            timeout_s=self.ready_timeout_s,
                        ),
                    )
                else:
                    ready_snapshots[worker.worker_id] = self.launcher.control.wait_for_all_ranks(
                        worker.worker_id,
                        world_size,
                        {"ready"},
                        timeout_s=self.ready_timeout_s,
                    )
            managed_workers = tuple(
                ManagedSemanticWorker(
                    worker=worker,
                    launch=launches[worker.worker_id],
                    client=self._client_for_worker(
                        worker,
                        snapshot=ready_snapshots[worker.worker_id],
                    ),
                )
                for worker in workers
            )
        except Exception as exc:
            self.launcher.stop_pool()
            if process is not None and process.poll() is None:
                self._terminate_process_group(process)
            diagnostics = "; ".join(
                self._worker_diagnostics(worker, launches[worker.worker_id])
                for worker in workers
                if worker.worker_id in launches
            )
            raise RuntimeExecutionError(
                f"shared semantic Slurm step failed to become ready: {exc}; {diagnostics}"
            ) from exc
        for managed in managed_workers:
            worker_id = managed.worker.worker_id
            self._workers[worker_id] = managed
            self._restart_safe.setdefault(worker_id, True)
            self._replay_journal.setdefault(worker_id, [])
        return managed_workers
    def client(self, worker_id: str) -> SemanticWorkerGroupClient | SocketSemanticWorkerGroupClient:
        try:
            return self._workers[worker_id].client
        except KeyError as exc:
            raise RuntimeExecutionError(f"semantic worker {worker_id} has not been started") from exc

    def request(
        self,
        worker_id: str,
        operation: str,
        *,
        node_id: str,
        step: int,
        payload: dict | None = None,
        timeout_s: float | None = None,
    ) -> tuple[SemanticResult, ...]:
        managed = self._managed(worker_id)
        managed = self._ensure_healthy_or_restart(managed, before=f"before {operation}")
        try:
            results = managed.client.request(
                operation,
                node_id=node_id,
                step=step,
                payload=payload,
                timeout_s=timeout_s,
            )
        except RuntimeExecutionError as exc:
            diagnostics = self._worker_diagnostics(managed.worker, managed.launch)
            self._abort_worker_group(managed)
            raise RuntimeExecutionError(
                f"semantic operation {operation} failed for {worker_id}: {exc}; "
                f"{diagnostics}; worker group aborted"
            ) from exc
        self._record_success(
            worker_id,
            _ReplayRequest(
                operation=operation,
                node_id=node_id,
                step=step,
                payload=copy.deepcopy(payload or {}),
                timeout_s=timeout_s,
            ),
        )
        return results

    def request_many(
        self,
        requests: tuple[tuple[str, str, str, int, dict[str, Any], float | None], ...],
    ) -> dict[str, tuple[SemanticResult, ...]]:
        """
        Run semantic requests for multiple worker groups concurrently.

        Each request tuple is (worker_id, operation, node_id, step, payload, timeout_s). This is
        intentionally small and synchronous: graph handlers can use it for paired source/destination
        transfers without changing executor scheduling semantics.
        """
        if not requests:
            return {}
        # Complete health inspection for every participant before dispatching either half of a
        # paired transfer.
        prepared: list[
            tuple[ManagedSemanticWorker, str, str, int, dict[str, Any], float | None]
        ] = []
        for worker_id, operation, node_id, step, payload, timeout_s in requests:
            managed = self._managed(worker_id)
            managed = self._ensure_healthy_or_restart(managed, before=f"before paired {operation}")
            prepared.append((managed, operation, node_id, step, payload, timeout_s))

        out: dict[str, tuple[SemanticResult, ...]] = {}
        with ThreadPoolExecutor(max_workers=len(requests)) as pool:
            futures = {
                pool.submit(
                    managed.client.request,
                    operation,
                    node_id=node_id,
                    step=step,
                    payload=payload,
                    timeout_s=timeout_s,
                ): (managed, operation, node_id, step, payload, timeout_s)
                for managed, operation, node_id, step, payload, timeout_s in prepared
            }
            for future in as_completed(futures):
                managed, operation, node_id, step, payload, timeout_s = futures[future]
                worker_id = managed.worker.worker_id
                try:
                    out[worker_id] = future.result()
                    self._record_success(
                        worker_id,
                        _ReplayRequest(
                            operation=operation,
                            node_id=node_id,
                            step=step,
                            payload=copy.deepcopy(payload),
                            timeout_s=timeout_s,
                        ),
                    )
                except Exception as exc:
                    for pending in futures:
                        pending.cancel()
                    # A peer may already be blocked inside NCCL/Gloo/UCX. Abort every participant,
                    # so executor shutdown cannot wait for a collective timeout.
                    for participant, *_rest in prepared:
                        if participant.worker.worker_id in self._workers:
                            self._abort_worker_group(participant)
                    raise RuntimeExecutionError(
                        f"paired semantic operation {operation} failed for {worker_id}: {exc}"
                    ) from exc
        return out

    def request_batch(
        self,
        worker_id: str,
        requests: tuple[tuple[str, str, int, dict[str, Any], float | None], ...],
    ) -> dict[str, tuple[SemanticResult, ...]]:
        """
        Publish multiple operations to one worker group before waiting.

        Each request tuple is (operation, node_id, step, payload, timeout_s). This is used for
        same-worker graph-node batches such as RPEG rollout quantile parts that should enter the
        semantic worker queue together.
        """
        if not requests:
            return {}
        managed = self._managed(worker_id)
        managed = self._ensure_healthy_or_restart(managed, before="before batched semantic request")
        try:
            results = managed.client.request_many(requests)
        except RuntimeExecutionError as exc:
            diagnostics = self._worker_diagnostics(managed.worker, managed.launch)
            self._abort_worker_group(managed)
            raise RuntimeExecutionError(
                f"batched semantic request failed for {worker_id}: {exc}; "
                f"{diagnostics}; worker group aborted"
            ) from exc
        for operation, node_id, step, payload, timeout_s in requests:
            self._record_success(
                worker_id,
                _ReplayRequest(
                    operation=operation,
                    node_id=node_id,
                    step=step,
                    payload=copy.deepcopy(payload or {}),
                    timeout_s=timeout_s,
                ),
            )
        return results

    def rank_hosts(self, worker_id: str) -> tuple[str, ...]:
        managed = self._managed(worker_id)
        cached = self._rank_host_cache.get(worker_id)
        if cached is not None:
            if len(cached) != len(managed.worker.devices):
                raise RuntimeExecutionError(
                    f"cached rank host count for {worker_id} changed from "
                    f"{len(cached)} to {len(managed.worker.devices)}"
                )
            return cached
        snapshot = self.launcher.control.rank_snapshot(worker_id, len(managed.worker.devices))
        hosts: list[str] = []
        for rank, heartbeat in enumerate(snapshot):
            if heartbeat is None:
                raise RuntimeExecutionError(f"missing heartbeat for {worker_id} rank {rank}")
            host = str((heartbeat.metadata or {}).get("hostname") or "")
            if not host:
                raise RuntimeExecutionError(f"missing hostname heartbeat metadata for {worker_id} rank {rank}")
            hosts.append(host)
        resolved = tuple(hosts)
        self._rank_host_cache[worker_id] = resolved
        return resolved

    def shutdown(self, timeout_s: float = 60.0) -> None:
        managed_workers = list(self._workers.values())
        for managed in managed_workers:
            try:
                managed.client.request(
                    SemanticOperationNames.SHUTDOWN,
                    node_id="runtime.shutdown",
                    step=0,
                    timeout_s=timeout_s,
                )
            except RuntimeExecutionError:
                pass
        if isinstance(self.launcher, SlurmWorkerGroupLauncher) and self.launcher.launcher_config.shared_step:
            self.launcher.stop_pool()
        deadline = time.monotonic() + timeout_s
        processes: dict[int, Any] = {}
        for managed in managed_workers:
            process = managed.launch.process
            if process is not None:
                processes.setdefault(id(process), process)
        for process in processes.values():
            remaining = max(0.1, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except Exception:
                self._terminate_process_group(process)
        for managed in managed_workers:
            close = getattr(managed.client, "close", None)
            if close is not None:
                close()
        self._workers.clear()
        self._rank_host_cache.clear()

    def _client_for_worker(self, worker: WorkerSpec, *, snapshot=None):
        if self.client_factory is not None:
            return self.client_factory(worker, self.command_root, self.operation_timeout_s)
        if self.control_transport == "file":
            return SemanticWorkerGroupClient(worker, self.command_root, default_timeout_s=self.operation_timeout_s)
        if snapshot is None:
            snapshot = self.launcher.control.rank_snapshot(worker.worker_id, len(worker.devices))
        if len(snapshot) != len(worker.devices):
            raise RuntimeExecutionError(
                f"TCP control snapshot for {worker.worker_id} has {len(snapshot)} ranks, "
                f"expected {len(worker.devices)}"
            )
        endpoints: list[tuple[str, int]] = []
        hosts: list[str] = []
        for rank, heartbeat in enumerate(snapshot):
            if heartbeat is None:
                raise RuntimeExecutionError(f"missing TCP control heartbeat for {worker.worker_id} rank {rank}")
            metadata = heartbeat.metadata or {}
            host = str(metadata.get("hostname") or "")
            port = int(metadata.get("semantic_control_port") or 0)
            if not host or port <= 0:
                raise RuntimeExecutionError(
                    f"missing TCP control endpoint for {worker.worker_id} rank {rank}: {metadata}"
                )
            endpoints.append((host, port))
            hosts.append(host)
        self._rank_host_cache[worker.worker_id] = tuple(hosts)
        return SocketSemanticWorkerGroupClient(
            worker,
            tuple(endpoints),
            default_timeout_s=self.operation_timeout_s,
        )

    def _managed(self, worker_id: str) -> ManagedSemanticWorker:
        try:
            return self._workers[worker_id]
        except KeyError as exc:
            raise RuntimeExecutionError(f"semantic worker {worker_id} has not been started") from exc

    def _assert_healthy(
        self,
        managed: ManagedSemanticWorker,
        *,
        before: str,
        original: RuntimeExecutionError | None = None,
    ) -> None:
        launch = managed.launch
        process = launch.process
        if process is not None:
            return_code = process.poll()
            if return_code is not None:
                prefix = f"semantic worker {managed.worker.worker_id} exited with code {return_code} {before}"
                if original is not None:
                    prefix += f" while handling prior error: {original}"
                raise RuntimeExecutionError(f"{prefix}; {self._worker_diagnostics(managed.worker, launch)}")
        if self.control_transport == "tcp":
            # Persistent rank sockets are the authoritative hot-path liveness check. Re-reading
            # every rank heartbeat from a parallel filesystem before every operation adds
            # O(world_size) metadata I/O and can dominate millisecond-scale transfers.
            return
        bad_states = {"error", "stopping", "stopped"}
        snapshot = self.launcher.control.rank_snapshot(managed.worker.worker_id, len(managed.worker.devices))
        bad = [hb for hb in snapshot if hb is not None and hb.state in bad_states]
        if bad:
            details = ", ".join(_format_heartbeat(hb) for hb in bad)
            prefix = f"semantic worker {managed.worker.worker_id} unhealthy {before}: {details}"
            if original is not None:
                prefix += f"; prior error: {original}"
            raise RuntimeExecutionError(f"{prefix}; {self._worker_diagnostics(managed.worker, launch, snapshot=snapshot)}")

    def _ensure_healthy_or_restart(self, managed: ManagedSemanticWorker, *, before: str) -> ManagedSemanticWorker:
        try:
            self._assert_healthy(managed, before=before)
            return managed
        except RuntimeExecutionError as exc:
            if not self._can_restart(managed.worker.worker_id):
                raise RuntimeExecutionError(
                    f"{exc}; automatic restart is unsafe or exhausted for {managed.worker.worker_id}"
                ) from exc
            return self._restart_worker(managed, reason=str(exc))

    def _can_restart(self, worker_id: str) -> bool:
        if isinstance(self.launcher, SlurmWorkerGroupLauncher) and self.launcher.launcher_config.shared_step:
            return False
        if self.restart_retries <= 0:
            return False
        if self._restart_attempts.get(worker_id, 0) >= self.restart_retries:
            return False
        return self._restart_safe.get(worker_id, True)

    def _restart_worker(self, managed: ManagedSemanticWorker, *, reason: str) -> ManagedSemanticWorker:
        worker_id = managed.worker.worker_id
        close = getattr(managed.client, "close", None)
        if close is not None:
            close()
        process = managed.launch.process
        if process is not None and process.poll() is None:
            self._terminate_process_group(process)
        self._workers.pop(worker_id, None)
        self._restart_attempts[worker_id] = self._restart_attempts.get(worker_id, 0) + 1
        restarted = self._launch_worker(managed.worker)
        for request in self._replay_journal.get(worker_id, ()):
            try:
                restarted.client.request(
                    request.operation,
                    node_id=request.node_id,
                    step=request.step,
                    payload=copy.deepcopy(request.payload),
                    timeout_s=request.timeout_s,
                )
            except RuntimeExecutionError as exc:
                raise RuntimeExecutionError(
                    f"semantic worker {worker_id} restart replay failed after {reason}: {exc}; "
                    f"{self._worker_diagnostics(restarted.worker, restarted.launch)}"
                ) from exc
        return restarted

    def _abort_worker_group(self, managed: ManagedSemanticWorker) -> None:
        process = managed.launch.process
        peers = [peer for peer in self._workers.values() if peer.launch.process is process]
        if isinstance(self.launcher, SlurmWorkerGroupLauncher) and self.launcher.launcher_config.shared_step:
            self.launcher.stop_pool()
        if process is not None and process.poll() is None:
            self._terminate_process_group(process)
        for peer in peers:
            close = getattr(peer.client, "close", None)
            if close is not None:
                close()
        for peer in peers:
            self._workers.pop(peer.worker.worker_id, None)
            self._rank_host_cache.pop(peer.worker.worker_id, None)

    @staticmethod
    def _terminate_process_group(process, *, grace_s: float = 5.0) -> None:
        """
        Terminate an srun/local worker and every rank it spawned.
        """
        try:
            pid = int(process.pid)
        except (AttributeError, TypeError, ValueError):
            pid = 0
        try:
            if pid > 0:
                os.killpg(pid, signal.SIGTERM)
            else:
                process.terminate()
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.terminate()
            except Exception:
                return
        try:
            process.wait(timeout=max(0.1, float(grace_s)))
            return
        except Exception:
            pass
        try:
            if pid > 0:
                os.killpg(pid, signal.SIGKILL)
            else:
                process.kill()
        except (ProcessLookupError, PermissionError, OSError, AttributeError):
            try:
                process.kill()
            except Exception:
                pass
        try:
            process.wait(timeout=max(0.1, float(grace_s)))
        except Exception:
            pass

    def _record_success(self, worker_id: str, request: _ReplayRequest) -> None:
        if request.operation in _REPLAYABLE_STATE_OPS:
            journal = self._replay_journal.setdefault(worker_id, [])
            journal.append(request)
            self._restart_safe[worker_id] = True
            return
        if _operation_mutates_model(request.operation):
            self._restart_safe[worker_id] = False

    def _worker_diagnostics(
        self,
        worker: WorkerSpec,
        launch: WorkerLaunch,
        *,
        snapshot: tuple[Heartbeat | None, ...] | None = None,
    ) -> str:
        if snapshot is None:
            snapshot = self.launcher.control.rank_snapshot(worker.worker_id, len(worker.devices))
        ranks = ", ".join(
            f"rank {idx}: missing" if heartbeat is None else _format_heartbeat(heartbeat)
            for idx, heartbeat in enumerate(snapshot)
        )
        return (
            f"heartbeats=[{ranks}], stdout={launch.stdout_path}, "
            f"stderr={launch.stderr_path}, command={' '.join(launch.command)}"
        )


def _format_heartbeat(heartbeat: Heartbeat) -> str:
    rank = "group" if heartbeat.rank is None else f"rank {heartbeat.rank}"
    message = "" if not heartbeat.message else f" message={heartbeat.message!r}"
    return f"{rank} state={heartbeat.state}{message}"


_REPLAYABLE_STATE_OPS = {
    SemanticOperationNames.BUILD_MODEL,
    SemanticOperationNames.LOAD_HF_CHECKPOINT,
    SemanticOperationNames.LOAD_DISTRIBUTED_CHECKPOINT,
    SemanticOperationNames.LOAD_WEIGHT_VERSION,
}


def _operation_mutates_model(operation: str) -> bool:
    return operation == SemanticOperationNames.LM_TRAINING_STEP or operation.endswith("_train_step")
