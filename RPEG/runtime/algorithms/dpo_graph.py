# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from passes.lower import ExecNodeType
from runtime.algorithms.dpo import (
    DPOTrainResult,
    _concat,
    _iter_preferences,
    _require_torch_transformers,
    _sequence_logps,
)
from runtime.compiler import CompiledRuntimePlan
from runtime.containers import ContainerRef, ContainerStore, ShardingSpec, WeightVersion
from runtime.data import dataset_batch_indices
from runtime.errors import RuntimeConfigError
from runtime.executor import NodeExecutionResult, RuntimeExecutionContext, RuntimeExecutor
from runtime.graph import RuntimeNodeSpec, build_runtime_graph
from runtime.hf import resolve_model_path
from runtime.metrics import write_runtime_summary_for_run
from runtime.trace import RuntimeTraceWriter
from runtime.weight_versions import consumer_weight_version, load_module_state, snapshot_module_state, uses_shadow_copies


@dataclass
class _DPOBatch:
    rows: list[dict[str, str]]
    prompts: list[str]
    chosen: list[str]
    rejected: list[str]
    prompt_lens: list[int]


class TorchDPOGraphHandler:
    """
    Actual local torch execution for the lowered DPO graph.

    This handler is intentionally strict: it only accepts plans that the local torch backend can
    execute exactly. Multi-process/multi-device DPO should be implemented as a separate worker-
    backed handler instead of silently sharing one process.
    """
    def __init__(self, compiled: CompiledRuntimePlan, *, step: int) -> None:
        self.compiled = compiled
        self.run = compiled.run_config
        self.step = step
        self.torch = None
        self.actor = None
        self.reference = None
        self.actor_inference = None
        self.tokenizer = None
        self.optimizer = None
        self.max_len = self.run.workload.context_length + self.run.workload.generation_length
        self.last_loss: float | None = None
        self.actor_versions: dict[int, dict[str, Any]] = {}
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        run = self.run
        if run.algorithm.name != "dpo":
            raise RuntimeConfigError("TorchDPOGraphHandler only supports algorithm.name=dpo")
        if not run.data.preferences:
            raise RuntimeConfigError("DPO requires data.preferences")
        if run.runtime.default_backend not in {"torch", "local_torch", "torch_dpo"}:
            raise RuntimeConfigError("Graph DPO training requires runtime.default_backend=torch/local_torch/torch_dpo")

        _validate_local_torch_exactness(self.compiled)
        torch, AutoModelForCausalLM, AutoTokenizer = _require_torch_transformers()
        self.torch = torch

        actor_cfg = run.models["actor"]
        actor_dir = resolve_model_path(
            actor_cfg.source,
            actor_cfg.repo,
            actor_cfg.checkpoint_path,
            revision=actor_cfg.revision,
        )
        ref_cfg = run.models.get("reference")
        ref_dir = None
        if ref_cfg and ref_cfg.source not in {"actor"}:
            ref_dir = resolve_model_path(
                ref_cfg.source,
                ref_cfg.repo,
                ref_cfg.checkpoint_path,
                revision=ref_cfg.revision,
            )

        tokenizer_checkpoint = run.tokenizer.checkpoint_path
        if tokenizer_checkpoint is None and run.tokenizer.source == "local":
            tokenizer_checkpoint = str(actor_dir)
        tokenizer_path = resolve_model_path(
            run.tokenizer.source,
            run.tokenizer.repo,
            tokenizer_checkpoint,
            revision=run.tokenizer.revision,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=run.tokenizer.source == "local" and actor_cfg.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = run.tokenizer.padding_side
        tokenizer.truncation_side = run.tokenizer.truncation_side
        self.tokenizer = tokenizer

        dtype = torch.bfloat16 if actor_cfg.dtype == "bf16" else torch.float32
        actor = AutoModelForCausalLM.from_pretrained(
            actor_dir,
            torch_dtype=dtype,
            trust_remote_code=actor_cfg.trust_remote_code,
        )
        if torch.cuda.is_available():
            actor = actor.cuda()
        actor.train()
        self.actor = actor
        self.actor_inference = copy.deepcopy(actor).eval()

        if ref_dir is None:
            reference = copy.deepcopy(actor).eval()
        else:
            reference = AutoModelForCausalLM.from_pretrained(
                ref_dir,
                torch_dtype=dtype,
                trust_remote_code=ref_cfg.trust_remote_code if ref_cfg else False,
            ).eval()
            if torch.cuda.is_available():
                reference = reference.cuda()
        for p in reference.parameters():
            p.requires_grad_(False)
        self.reference = reference

        self.optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=run.optimizer_runtime.learning_rate_actor,
            weight_decay=run.optimizer_runtime.weight_decay,
        )
        self.actor_versions[0] = snapshot_module_state(actor)
        self._initialized = True

    async def execute(self, node: RuntimeNodeSpec, context: RuntimeExecutionContext) -> NodeExecutionResult:
        self.initialize()
        if node.kind in {ExecNodeType.COMM, ExecNodeType.RESHARD}:
            return self._transfer_metadata(node, context)
        if node.kind != ExecNodeType.OP:
            raise RuntimeConfigError(f"Unsupported DPO runtime node kind {node.kind}")
        if node.vertex_id == "i_pi":
            return self._run_actor_inference(node, context)
        if node.vertex_id == "i_ref":
            return self._run_reference_inference(node, context)
        if node.vertex_id == "t_pi":
            return self._run_actor_training(node, context)
        raise RuntimeConfigError(f"Unsupported DPO vertex {node.vertex_id!r}")

    def save_if_needed(self, step: int | None = None) -> None:
        checkpoint_step = self.step if step is None else int(step)
        if checkpoint_step % self.run.checkpoints.interval_steps != 0:
            return
        out = Path(self.run.checkpoints.output_dir) / f"step_{checkpoint_step}"
        out.mkdir(parents=True, exist_ok=True)
        self.actor.save_pretrained(out)
        self.tokenizer.save_pretrained(out)

    def save_final(self) -> str:
        out = Path(self.run.checkpoints.output_dir) / "final"
        out.mkdir(parents=True, exist_ok=True)
        self.actor.save_pretrained(out)
        self.tokenizer.save_pretrained(out)
        return str(out)

    def _node_step(self, node: RuntimeNodeSpec) -> int:
        if node.logical_step is not None:
            return int(node.logical_step)
        if node.exec_node.op is None:
            return self.step
        return int(node.exec_node.op.k) + 1

    def _ensure_batch(self, context: RuntimeExecutionContext, step: int) -> _DPOBatch:
        state_key = f"dpo_batch:{step}"
        batch = context.state.get(state_key)
        if batch is not None:
            return batch
        assert self.tokenizer is not None
        rows = _preference_batch(
            self.run.data.preferences,
            step=step,
            batch_size=self.run.workload.batch_size,
            offset=self.run.data.preference_offset,
            shuffle=self.run.data.shuffle,
            seed=self.run.data.seed,
            shard_index=self.run.data.shard_index,
            num_shards=self.run.data.num_shards,
        )
        prompts = [r["prompt"] for r in rows]
        chosen = [_concat(r["prompt"], r["chosen"]) for r in rows]
        rejected = [_concat(r["prompt"], r["rejected"]) for r in rows]
        prompt_lens = [
            len(self.tokenizer(p, add_special_tokens=True, truncation=True, max_length=self.max_len)["input_ids"])
            for p in prompts
        ]
        batch = _DPOBatch(rows=rows, prompts=prompts, chosen=chosen, rejected=rejected, prompt_lens=prompt_lens)
        context.state[state_key] = batch
        context.store.put_container(
            ContainerRef(
                name=f"dpo_batch:{step}",
                step=step,
                producer_node_id="input",
                logical_shape=(self.run.workload.batch_size, 2, self.max_len),
                dtype="tokens",
                sharding=ShardingSpec("replicated"),
                metadata={"response_format": self.run.data.response_format},
            )
        )
        return batch

    def _run_actor_inference(self, node: RuntimeNodeSpec, context: RuntimeExecutionContext) -> NodeExecutionResult:
        torch = self.torch
        step = self._node_step(node)
        batch = self._ensure_batch(context, step)
        actor_version = consumer_weight_version(self.compiled, "actor", step)
        actor_shadow = uses_shadow_copies(self.compiled, "actor")
        actor = self._actor_for_inference(actor_version)
        with torch.no_grad():
            actor_chosen = _sequence_logps(torch, actor, self.tokenizer, batch.chosen, batch.prompt_lens, self.max_len)
            actor_rejected = _sequence_logps(torch, actor, self.tokenizer, batch.rejected, batch.prompt_lens, self.max_len)
        context.state[f"log_pi:{step}"] = (actor_chosen, actor_rejected)
        ref = context.store.put_container(
            ContainerRef(
                name=f"log_pi:{step}",
                step=step,
                producer_node_id=node.node_id,
                logical_shape=(self.run.workload.batch_size, 2),
                dtype=str(actor_chosen.dtype),
                sharding=ShardingSpec("worker", (node.worker_id or "",)),
                owner_worker_id=node.worker_id,
                metadata={"model": "actor", "weight_version": actor_version, "shadow": actor_shadow},
            )
        )
        return NodeExecutionResult(
            node_id=node.node_id,
            counters={
                "batch_size": self.run.workload.batch_size,
                "kind": "actor_logprob",
                "weight_version": actor_version,
                "shadow": actor_shadow,
            },
            containers=(ref,),
        )

    def _run_reference_inference(self, node: RuntimeNodeSpec, context: RuntimeExecutionContext) -> NodeExecutionResult:
        torch = self.torch
        step = self._node_step(node)
        batch = self._ensure_batch(context, step)
        with torch.no_grad():
            ref_chosen = _sequence_logps(torch, self.reference, self.tokenizer, batch.chosen, batch.prompt_lens, self.max_len)
            ref_rejected = _sequence_logps(torch, self.reference, self.tokenizer, batch.rejected, batch.prompt_lens, self.max_len)
        context.state[f"log_ref:{step}"] = (ref_chosen, ref_rejected)
        ref = context.store.put_container(
            ContainerRef(
                name=f"log_ref:{step}",
                step=step,
                producer_node_id=node.node_id,
                logical_shape=(self.run.workload.batch_size, 2),
                dtype=str(ref_chosen.dtype),
                sharding=ShardingSpec("worker", (node.worker_id or "",)),
                owner_worker_id=node.worker_id,
            )
        )
        return NodeExecutionResult(
            node_id=node.node_id,
            counters={"batch_size": self.run.workload.batch_size, "kind": "reference_logprob"},
            containers=(ref,),
        )

    def _run_actor_training(self, node: RuntimeNodeSpec, context: RuntimeExecutionContext) -> NodeExecutionResult:
        torch = self.torch
        step = self._node_step(node)
        log_ref_key = f"log_ref:{step}"
        if f"log_pi:{step}" not in context.state or log_ref_key not in context.state:
            raise RuntimeConfigError(f"{node.node_id} missing log_pi/log_ref inputs")
        batch = self._ensure_batch(context, step)
        actor_chosen = _sequence_logps(torch, self.actor, self.tokenizer, batch.chosen, batch.prompt_lens, self.max_len)
        actor_rejected = _sequence_logps(torch, self.actor, self.tokenizer, batch.rejected, batch.prompt_lens, self.max_len)
        ref_chosen, ref_rejected = context.state[log_ref_key]
        actor_margin = actor_chosen - actor_rejected
        ref_margin = ref_chosen - ref_rejected
        loss = -torch.nn.functional.logsigmoid(self.run.algorithm.beta * (actor_margin - ref_margin)).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.run.optimizer_runtime.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.run.optimizer_runtime.grad_clip)
        self.optimizer.step()
        self.last_loss = float(loss.detach().cpu())
        self.actor_versions[step] = snapshot_module_state(self.actor)

        weight = context.store.publish_weight(
            WeightVersion(
                model="actor",
                version=step,
                produced_by=node.node_id,
                step=step,
                layout=ShardingSpec("worker", (node.worker_id or "",)),
                dtype=str(next(self.actor.parameters()).dtype),
                optimizer_state_owner=node.worker_id,
                shadow=uses_shadow_copies(self.compiled, "actor"),
            )
        )
        return NodeExecutionResult(
            node_id=node.node_id,
            counters={
                "loss": self.last_loss,
                "kind": "dpo_train",
                "produced_weight": {"model": "actor", "version": step, "shadow": weight.shadow},
            },
            weights=(weight,),
        )

    def _actor_for_inference(self, version: int):
        if version not in self.actor_versions:
            raise RuntimeConfigError(
                f"actor weight version {version} is required but has not been published"
            )
        load_module_state(self.actor_inference, self.actor_versions[version])
        self.actor_inference.eval()
        return self.actor_inference

    def _transfer_metadata(self, node: RuntimeNodeSpec, context: RuntimeExecutionContext) -> NodeExecutionResult:
        step = _graph_node_step(node) or self.step
        ref = context.store.put_container(
            ContainerRef(
                name=f"{node.node_id}:{node.label or node.kind.value}:{step}",
                step=step,
                producer_node_id=node.node_id,
                sharding=ShardingSpec(node.kind.value),
                metadata={"payload_in_bytes": node.payload_in_bytes, "label": node.label},
            )
        )
        return NodeExecutionResult(
            node_id=node.node_id,
            counters={"kind": node.kind.value, "payload_in_bytes": node.payload_in_bytes},
            containers=(ref,),
        )


def _preference_batch(
    path: str | Path,
    *,
    step: int,
    batch_size: int,
    offset: int = 0,
    shuffle: bool = False,
    seed: int = 0,
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[dict[str, str]]:
    rows = list(_iter_preferences(path))
    if not rows:
        raise RuntimeConfigError(f"DPO preference file is empty: {path}")
    indices = dataset_batch_indices(
        len(rows),
        step=step,
        batch_size=batch_size,
        offset=offset,
        shuffle=shuffle,
        seed=seed,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    return [rows[idx] for idx in indices]


def _validate_local_torch_exactness(compiled: CompiledRuntimePlan) -> None:
    devices = {(d.hostname, d.local_cuda_index) for worker in compiled.workers for d in worker.devices}
    for worker in compiled.workers:
        if worker.parallelism != (1, 1, 1):
            raise RuntimeConfigError(
                f"Local torch DPO only supports exact dp=pp=tp=1 workers; {worker.worker_id} has {worker.parallelism}"
            )
        if len(worker.devices) != 1:
            raise RuntimeConfigError(f"Local torch DPO requires one device per worker: {worker.worker_id}")
    if len(devices) != 1:
        raise RuntimeConfigError(
            "Local torch DPO cannot execute a multi-device RPEG plan exactly; use a distributed worker handler"
        )

def _graph_node_step(node: RuntimeNodeSpec) -> int | None:
    if node.logical_step is not None:
        return int(node.logical_step)
    if node.exec_node.op is not None:
        return int(node.exec_node.op.k) + 1
    for node_id in (*node.dependencies, *node.successors):
        if "@" not in node_id:
            continue
        try:
            return int(node_id.rsplit("@", 1)[1].split(":", 1)[0]) + 1
        except ValueError:
            continue
    return None

def run_graph_dpo(compiled: CompiledRuntimePlan) -> DPOTrainResult:
    """
    Execute DPO training through the lowered RPEG graph executor.
    """
    if compiled.run_config.training.max_steps < 1:
        raise RuntimeConfigError("DPO graph training requires training.max_steps >= 1")
    if compiled.run_config.runtime.default_backend == "megatron":
        from runtime.algorithms.megatron_ppo_graph import run_megatron_graph_ppo

        return run_megatron_graph_ppo(compiled)
    graph = compiled.runtime_graph()
    trace = RuntimeTraceWriter(compiled.run_config.runtime.trace_path, compiled.run_config.run_id)
    store_root = Path(compiled.run_config.runtime.store_dir or Path(compiled.run_config.checkpoints.output_dir) / "runtime_store")
    store = ContainerStore(store_root / "containers")
    handler = TorchDPOGraphHandler(compiled, step=1)
    last_loss = None
    output_dir = ""

    executor = RuntimeExecutor(
        graph,
        run_id=compiled.run_config.run_id,
        handler=handler,
        trace=trace,
        store=store,
    )
    executor.run()
    handler.save_if_needed(compiled.run_config.training.max_steps)
    last_loss = handler.last_loss
    if compiled.run_config.training.max_steps % compiled.run_config.training.log_every_steps == 0:
        print(json.dumps({"step": compiled.run_config.training.max_steps, "loss": last_loss}))

    if compiled.run_config.checkpoints.save_final:
        output_dir = handler.save_final()
    else:
        output_dir = str(Path(compiled.run_config.checkpoints.output_dir))

    write_runtime_summary_for_run(
        compiled.run_config,
        extra={
            "algorithm": "dpo",
            "steps": compiled.run_config.training.max_steps,
            "output_dir": output_dir,
            "last_loss": last_loss,
        },
    )
    return DPOTrainResult(
        steps=compiled.run_config.training.max_steps,
        last_loss=last_loss,
        output_dir=output_dir,
    )
