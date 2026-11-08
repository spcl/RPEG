# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import json
import select
import socket
import struct
import threading
import time
import uuid

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from runtime.compiler import WorkerSpec
from runtime.errors import RuntimeConfigError, RuntimeExecutionError


@dataclass(frozen=True)
class SemanticCapability:
    operation: str
    supported: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "supported": self.supported,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SemanticCapability":
        return cls(
            operation=str(d["operation"]),
            supported=bool(d["supported"]),
            reason=str(d.get("reason") or ""),
        )


@dataclass(frozen=True)
class SemanticRequest:
    request_id: str
    worker_id: str
    vertex_id: str
    node_id: str
    operation: str
    step: int
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rpeg.semantic_request.v1",
            "request_id": self.request_id,
            "worker_id": self.worker_id,
            "vertex_id": self.vertex_id,
            "node_id": self.node_id,
            "operation": self.operation,
            "step": self.step,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SemanticRequest":
        version = str(d.get("schema_version", ""))
        if version != "rpeg.semantic_request.v1":
            raise RuntimeConfigError(f"unsupported semantic request schema_version {version!r}")
        return cls(
            request_id=str(d["request_id"]),
            worker_id=str(d["worker_id"]),
            vertex_id=str(d["vertex_id"]),
            node_id=str(d["node_id"]),
            operation=str(d["operation"]),
            step=int(d["step"]),
            payload=dict(d.get("payload") or {}),
        )


@dataclass(frozen=True)
class SemanticResult:
    request_id: str
    worker_id: str
    rank: int
    world_size: int
    status: str
    counters: dict[str, Any] = field(default_factory=dict)
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rpeg.semantic_result.v1",
            "request_id": self.request_id,
            "worker_id": self.worker_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "status": self.status,
            "counters": self.counters,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SemanticResult":
        version = str(d.get("schema_version", ""))
        if version != "rpeg.semantic_result.v1":
            raise RuntimeConfigError(f"unsupported semantic result schema_version {version!r}")
        return cls(
            request_id=str(d["request_id"]),
            worker_id=str(d["worker_id"]),
            rank=int(d["rank"]),
            world_size=int(d["world_size"]),
            status=str(d["status"]),
            counters=dict(d.get("counters") or {}),
            message=d.get("message"),
        )


class SemanticBackend(Protocol):
    def capabilities(self) -> dict[str, SemanticCapability]:
        ...

    def execute(self, request: SemanticRequest) -> SemanticResult:
        ...

    def shutdown(self) -> None:
        ...


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, sort_keys=True) + "\n")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


class FileSemanticTransport:
    def __init__(self, root: str | Path, worker: WorkerSpec) -> None:
        self.root = Path(root)
        self.worker = worker
        self.worker_root = self.root / worker.worker_id
        self.command_dir = self.worker_root / "commands"
        self.result_dir = self.worker_root / "results"
        self.command_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def command_path(self, request_id: str) -> Path:
        return self.command_dir / f"{request_id}.json"

    def result_path(self, request_id: str, rank: int) -> Path:
        return self.result_dir / f"{request_id}.rank{rank}.json"

    def publish(self, request: SemanticRequest) -> None:
        write_json_atomic(self.command_path(request.request_id), request.to_dict())

    def load_request(self, path: Path) -> SemanticRequest:
        return SemanticRequest.from_dict(read_json(path))

    def write_result(self, result: SemanticResult) -> None:
        write_json_atomic(self.result_path(result.request_id, result.rank), result.to_dict())

    def wait_results(self, request: SemanticRequest, *, timeout_s: float) -> tuple[SemanticResult, ...]:
        deadline = time.monotonic() + timeout_s
        expected_world = len(self.worker.devices)
        seen: dict[int, SemanticResult] = {}
        while time.monotonic() < deadline:
            for rank in range(expected_world):
                if rank in seen:
                    continue
                path = self.result_path(request.request_id, rank)
                if path.exists():
                    result = SemanticResult.from_dict(read_json(path))
                    if result.request_id != request.request_id:
                        raise RuntimeExecutionError(
                            f"semantic result request mismatch: {result.request_id} != {request.request_id}"
                        )
                    seen[rank] = result
            if len(seen) == expected_world:
                results = tuple(seen[rank] for rank in range(expected_world))
                failed = [r for r in results if r.status != "ok"]
                if failed:
                    details = "; ".join(f"rank {r.rank}: {r.message or r.status}" for r in failed)
                    raise RuntimeExecutionError(
                        f"semantic operation {request.operation} failed on {request.worker_id}: {details}"
                    )
                return results
            time.sleep(0.05)
        missing = sorted(set(range(expected_world)) - set(seen))
        raise RuntimeExecutionError(
            f"semantic operation {request.operation} timed out after {timeout_s:.1f}s "
            f"for {request.worker_id}; missing ranks={missing}"
        )


class SemanticWorkerGroupClient:
    def __init__(self, worker: WorkerSpec, command_root: str | Path, *, default_timeout_s: float = 600.0) -> None:
        self.worker = worker
        self.transport = FileSemanticTransport(command_root, worker)
        self.default_timeout_s = default_timeout_s

    def request(
        self,
        operation: str,
        *,
        node_id: str,
        step: int,
        payload: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> tuple[SemanticResult, ...]:
        request = SemanticRequest(
            request_id=f"{step:08d}-{self.worker.worker_id}-{operation}-{uuid.uuid4().hex}",
            worker_id=self.worker.worker_id,
            vertex_id=self.worker.vertex_id,
            node_id=node_id,
            operation=operation,
            step=step,
            payload=payload or {},
        )
        self.transport.publish(request)
        return self.transport.wait_results(request, timeout_s=timeout_s or self.default_timeout_s)

    def request_many(
        self,
        requests: tuple[tuple[str, str, int, dict[str, Any] | None, float | None], ...],
    ) -> dict[str, tuple[SemanticResult, ...]]:
        """
        Publish several requests to one worker group before waiting.

        Each tuple is (operation, node_id, step, payload, timeout_s). The return value is keyed by
        graph node_id to distinguish concurrent requests with the same worker and operation.
        """
        semantic_requests: list[tuple[str, SemanticRequest, float | None]] = []
        for operation, node_id, step, payload, timeout_s in requests:
            request = SemanticRequest(
                request_id=f"{step:08d}-{self.worker.worker_id}-{operation}-{uuid.uuid4().hex}",
                worker_id=self.worker.worker_id,
                vertex_id=self.worker.vertex_id,
                node_id=node_id,
                operation=operation,
                step=step,
                payload=payload or {},
            )
            semantic_requests.append((node_id, request, timeout_s))
        for _node_id, request, _timeout_s in semantic_requests:
            self.transport.publish(request)
        out: dict[str, tuple[SemanticResult, ...]] = {}
        for node_id, request, timeout_s in semantic_requests:
            out[node_id] = self.transport.wait_results(request, timeout_s=timeout_s or self.default_timeout_s)
        return out


_FRAME_HEADER = struct.Struct("!Q")


def send_json_frame(sock: socket.socket, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sock.sendall(_FRAME_HEADER.pack(len(encoded)))
    sock.sendall(encoded)


def recv_json_frame(sock: socket.socket) -> dict[str, Any] | None:
    header = _recv_exact(sock, _FRAME_HEADER.size)
    if header is None:
        return None
    size = int(_FRAME_HEADER.unpack(header)[0])
    if size <= 0 or size > 1024 * 1024 * 1024:
        raise RuntimeExecutionError(f"invalid semantic control frame size {size}")
    body = _recv_exact(sock, size)
    if body is None:
        raise RuntimeExecutionError("semantic control connection closed inside a frame")
    decoded = json.loads(body.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeExecutionError("semantic control frame must contain a JSON object")
    return decoded


def _recv_exact(sock: socket.socket, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            if not chunks:
                return None
            received = size - remaining
            raise RuntimeExecutionError(
                f"semantic control connection closed after {received} of {size} frame bytes"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class SocketSemanticWorkerGroupClient:
    def __init__(
        self,
        worker: WorkerSpec,
        endpoints: tuple[tuple[str, int], ...],
        *,
        default_timeout_s: float = 600.0,
    ) -> None:
        if len(endpoints) != len(worker.devices):
            raise RuntimeConfigError(
                f"semantic socket endpoint count {len(endpoints)} does not match worker world size {len(worker.devices)}"
            )
        self.worker = worker
        self.endpoints = endpoints
        self.default_timeout_s = float(default_timeout_s)
        self._sockets: list[socket.socket] = []
        self._lock = threading.Lock()

    def request(
        self,
        operation: str,
        *,
        node_id: str,
        step: int,
        payload: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> tuple[SemanticResult, ...]:
        request = self._request(operation, node_id=node_id, step=step, payload=payload)
        return self._roundtrip((request,), timeout_s=timeout_s)[request.request_id]

    def request_many(
        self,
        requests: tuple[tuple[str, str, int, dict[str, Any] | None, float | None], ...],
    ) -> dict[str, tuple[SemanticResult, ...]]:
        semantic_requests = tuple(
            self._request(operation, node_id=node_id, step=step, payload=payload)
            for operation, node_id, step, payload, _timeout_s in requests
        )
        requested_timeouts = [float(timeout) for *_prefix, timeout in requests if timeout is not None]
        timeout_s = max(requested_timeouts, default=self.default_timeout_s)
        by_request = self._roundtrip(semantic_requests, timeout_s=timeout_s)
        return {
            request.node_id: by_request[request.request_id]
            for request in semantic_requests
        }

    def close(self) -> None:
        with self._lock:
            self._close_sockets()

    def _request(
        self,
        operation: str,
        *,
        node_id: str,
        step: int,
        payload: dict[str, Any] | None,
    ) -> SemanticRequest:
        return SemanticRequest(
            request_id=f"{step:08d}-{self.worker.worker_id}-{operation}-{uuid.uuid4().hex}",
            worker_id=self.worker.worker_id,
            vertex_id=self.worker.vertex_id,
            node_id=node_id,
            operation=operation,
            step=step,
            payload=payload or {},
        )

    def _roundtrip(
        self,
        requests: tuple[SemanticRequest, ...],
        *,
        timeout_s: float | None,
    ) -> dict[str, tuple[SemanticResult, ...]]:
        if not requests:
            return {}
        timeout = float(timeout_s or self.default_timeout_s)
        envelope = {
            "schema_version": "rpeg.semantic_control.v1",
            "requests": [request.to_dict() for request in requests],
        }
        with self._lock:
            try:
                sockets = self._connected_sockets(timeout)
                for sock in sockets:
                    sock.settimeout(timeout)
                    send_json_frame(sock, envelope)
                responses = _recv_rank_responses(sockets, timeout_s=timeout)
            except (OSError, RuntimeExecutionError, json.JSONDecodeError, ValueError) as exc:
                self._close_sockets()
                raise RuntimeExecutionError(
                    f"semantic socket request failed for {self.worker.worker_id}: {exc}"
                ) from exc
        collected: dict[str, dict[int, SemanticResult]] = {request.request_id: {} for request in requests}
        for rank, response in enumerate(responses):
            if response is None or response.get("schema_version") != "rpeg.semantic_control_result.v1":
                raise RuntimeExecutionError(f"invalid semantic socket response from rank {rank}")
            for raw_result in response.get("results") or []:
                result = SemanticResult.from_dict(raw_result)
                if result.rank != rank:
                    raise RuntimeExecutionError(
                        f"semantic socket response rank mismatch: connection rank {rank}, result rank {result.rank}"
                    )
                if result.request_id not in collected:
                    raise RuntimeExecutionError(f"unexpected semantic socket result {result.request_id}")
                collected[result.request_id][rank] = result
        out: dict[str, tuple[SemanticResult, ...]] = {}
        expected_ranks = set(range(len(self.endpoints)))
        for request in requests:
            by_rank = collected[request.request_id]
            if set(by_rank) != expected_ranks:
                raise RuntimeExecutionError(
                    f"semantic socket response for {request.request_id} missing ranks "
                    f"{sorted(expected_ranks - set(by_rank))}"
                )
            results = tuple(by_rank[rank] for rank in range(len(self.endpoints)))
            failed = [result for result in results if result.status != "ok"]
            if failed:
                details = "; ".join(f"rank {r.rank}: {r.message or r.status}" for r in failed)
                raise RuntimeExecutionError(
                    f"semantic operation {request.operation} failed on {request.worker_id}: {details}"
                )
            out[request.request_id] = results
        return out

    def _connected_sockets(self, timeout_s: float) -> list[socket.socket]:
        if len(self._sockets) == len(self.endpoints):
            return self._sockets
        self._close_sockets()
        try:
            self._sockets = [
                socket.create_connection(endpoint, timeout=timeout_s)
                for endpoint in self.endpoints
            ]
        except OSError:
            self._close_sockets()
            raise
        return self._sockets

    def _close_sockets(self) -> None:
        for sock in self._sockets:
            try:
                sock.close()
            except OSError:
                pass
        self._sockets.clear()


def _recv_rank_responses(
    sockets: list[socket.socket],
    *,
    timeout_s: float,
) -> list[dict[str, Any] | None]:
    """
    Collect rank responses in readiness order and stop on the first rank error.
    """
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    rank_by_socket = {sock: rank for rank, sock in enumerate(sockets)}
    pending = dict(rank_by_socket)
    responses: list[dict[str, Any] | None] = [None] * len(sockets)
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise RuntimeExecutionError(
                f"semantic socket response timed out after {timeout_s:.1f}s; "
                f"missing ranks={sorted(pending.values())}"
            )
        readable, _, _ = select.select(tuple(pending), (), (), remaining)
        if not readable:
            continue
        for sock in readable:
            rank = pending.pop(sock)
            sock.settimeout(max(0.1, deadline - time.monotonic()))
            response = recv_json_frame(sock)
            if response is None or response.get("schema_version") != "rpeg.semantic_control_result.v1":
                raise RuntimeExecutionError(f"invalid semantic socket response from rank {rank}")
            failed = [
                SemanticResult.from_dict(raw)
                for raw in response.get("results") or []
                if str(raw.get("status")) != "ok"
            ]
            if failed:
                details = "; ".join(
                    f"rank {result.rank}: {result.message or result.status}"
                    for result in failed
                )
                raise RuntimeExecutionError(f"semantic rank failed before its peers completed: {details}")
            responses[rank] = response
    return responses


class SemanticOperationNames:
    CAPABILITIES = "capabilities"
    BUILD_MODEL = "build_model"
    LOAD_HF_CHECKPOINT = "load_hf_checkpoint"
    LM_INFERENCE_STEP = "lm_inference_step"
    LM_TRAINING_STEP = "lm_training_step"
    SAVE_DISTRIBUTED_CHECKPOINT = "save_distributed_checkpoint"
    LOAD_DISTRIBUTED_CHECKPOINT = "load_distributed_checkpoint"
    SAVE_WEIGHT_VERSION = "save_weight_version"
    LOAD_WEIGHT_VERSION = "load_weight_version"
    PRUNE_WEIGHT_VERSIONS = "prune_weight_versions"
    RESHARD_WEIGHT_VERSION = "reshard_weight_version"
    SWITCH_MODEL_LAYOUT = "switch_model_layout"
    ENSURE_OPERATIONAL_WEIGHTS = "ensure_operational_weights"
    RELEASE_OPERATIONAL_WEIGHTS = "release_operational_weights"
    SEND_WEIGHT_VERSION = "send_weight_version"
    RECV_WEIGHT_VERSION = "recv_weight_version"
    TRANSFER_ACTIVATION = "transfer_activation"
    REPARTITION_LOCAL_ACTIVATION = "repartition_local_activation"
    SEND_ACTIVATION = "send_activation"
    RECV_ACTIVATION = "recv_activation"
    RELEASE_ACTIVATIONS = "release_activations"
    SEND_DIRECT_PREWARM = "send_direct_prewarm"
    RECV_DIRECT_PREWARM = "recv_direct_prewarm"
    SHUTDOWN = "shutdown"

    PPO_ACTOR_GENERATE = "actor_generate_with_old_logprobs"
    PPO_REFERENCE_LOGPROBS = "reference_token_logprobs"
    PPO_REWARD_SCORE = "reward_score"
    PPO_CRITIC_VALUES = "critic_token_values"
    PPO_ACTOR_TRAIN = "ppo_actor_train_step"
    PPO_CRITIC_TRAIN = "ppo_critic_train_step"
    PPO_AC_TRAIN = "ppo_actor_critic_train_step"
    PPO_FUSED_ACTOR_CRITIC_TRAIN = "ppo_fused_actor_critic_train_step"

    GRPO_ACTOR_GENERATE = "grpo_actor_generate_with_old_logprobs"
    GRPO_REFERENCE_LOGPROBS = "grpo_reference_token_logprobs"
    GRPO_REWARD_SCORE = "grpo_reward_score"
    GRPO_ACTOR_TRAIN = "grpo_actor_train_step"

    DPO_ACTOR_LOGPROBS = "dpo_actor_logprobs"
    DPO_REFERENCE_LOGPROBS = "dpo_reference_logprobs"
    DPO_ACTOR_TRAIN = "dpo_actor_train_step"
