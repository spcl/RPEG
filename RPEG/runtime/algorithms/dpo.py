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
from typing import Iterator

from runtime.compiler import CompiledRuntimePlan
from runtime.errors import RuntimeConfigError, RuntimeDependencyError
from runtime.hf import resolve_model_path
from runtime.trace import RuntimeTraceWriter


@dataclass(frozen=True)
class DPOTrainResult:
    steps: int
    last_loss: float | None
    output_dir: str


def _require_torch_transformers():
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeDependencyError(
            "Local DPO execution requires torch and transformers. "
            "Phase 0 dry-run works without them; install the runtime stack to train."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def _iter_preferences(path: str | Path) -> Iterator[dict[str, str]]:
    with Path(path).open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                yield {
                    "prompt": str(row["prompt"]),
                    "chosen": str(row["chosen"]),
                    "rejected": str(row["rejected"]),
                }
            except KeyError as exc:
                raise RuntimeConfigError(f"{path}:{line_no} missing DPO field {exc.args[0]!r}") from exc


def _cycle_preferences(path: str | Path) -> Iterator[dict[str, str]]:
    while True:
        yielded = False
        for row in _iter_preferences(path):
            yielded = True
            yield row
        if not yielded:
            raise RuntimeConfigError(f"DPO preference file is empty: {path}")


def _batch(iterator: Iterator[dict[str, str]], batch_size: int) -> list[dict[str, str]]:
    return [next(iterator) for _ in range(batch_size)]


def _concat(prompt: str, response: str) -> str:
    if prompt.endswith(" ") or response.startswith(" "):
        return prompt + response
    return prompt + " " + response


def _sequence_logps(torch, model, tokenizer, texts: list[str], prompt_lengths: list[int], max_length: int):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    outputs = model(**enc)
    logits = outputs.logits[:, :-1, :]
    labels = enc["input_ids"][:, 1:]
    attn = enc["attention_mask"][:, 1:].bool()
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    masks = []
    for i, prompt_len in enumerate(prompt_lengths):
        mask = attn[i].clone()
        valid_tokens = int(enc["attention_mask"][i].sum().item())
        pad_tokens = enc["attention_mask"].shape[1] - valid_tokens if tokenizer.padding_side == "left" else 0
        # Labels are shifted by one. Keep labels that predict response tokens, accounting for left
        # padding in decoder-only batches.
        mask[: max(0, pad_tokens + prompt_len - 1)] = False
        masks.append(mask)
    response_mask = torch.stack(masks, dim=0)
    return (token_logps * response_mask).sum(dim=1)


def run_local_dpo(compiled: CompiledRuntimePlan) -> DPOTrainResult:
    run = compiled.run_config
    if run.algorithm.name != "dpo":
        raise RuntimeConfigError("run_local_dpo only supports algorithm.name=dpo")
    if not run.data.preferences:
        raise RuntimeConfigError("DPO requires data.preferences")
    if run.runtime.default_backend not in {"torch", "local_torch"}:
        raise RuntimeConfigError("Phase 1 local DPO requires runtime.default_backend=torch")

    for worker in compiled.workers:
        if worker.parallelism != (1, 1, 1):
            raise RuntimeConfigError(
                f"Phase 1 local DPO only supports exact dp=pp=tp=1 workers; "
                f"{worker.worker_id} has {worker.parallelism}"
            )
        if len(worker.devices) != 1:
            raise RuntimeConfigError(f"Phase 1 local DPO requires one device per worker: {worker.worker_id}")

    torch, AutoModelForCausalLM, AutoTokenizer = _require_torch_transformers()
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

    dtype = torch.bfloat16 if actor_cfg.dtype == "bf16" else torch.float32
    actor = AutoModelForCausalLM.from_pretrained(
        actor_dir,
        torch_dtype=dtype,
        trust_remote_code=actor_cfg.trust_remote_code,
    )
    if torch.cuda.is_available():
        actor = actor.cuda()
    actor.train()

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

    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=run.optimizer_runtime.learning_rate_actor,
        weight_decay=run.optimizer_runtime.weight_decay,
    )

    pref_iter = _cycle_preferences(run.data.preferences)
    batch_size = run.workload.batch_size
    max_len = run.workload.context_length + run.workload.generation_length
    last_loss = None
    beta = run.algorithm.beta
    trace = RuntimeTraceWriter(run.runtime.trace_path, run.run_id)
    worker_by_vertex = {w.vertex_id: w.worker_id for w in compiled.workers}

    for step in range(1, run.training.max_steps + 1):
        rows = _batch(pref_iter, batch_size)
        prompts = [r["prompt"] for r in rows]
        chosen = [_concat(r["prompt"], r["chosen"]) for r in rows]
        rejected = [_concat(r["prompt"], r["rejected"]) for r in rows]
        prompt_lens = [
            len(tokenizer(p, add_special_tokens=True, truncation=True, max_length=max_len)["input_ids"])
            for p in prompts
        ]
        with trace.node("i_pi", worker_id=worker_by_vertex.get("i_pi"), step=step) as counters:
            actor_chosen = _sequence_logps(torch, actor, tokenizer, chosen, prompt_lens, max_len)
            actor_rejected = _sequence_logps(torch, actor, tokenizer, rejected, prompt_lens, max_len)
            counters["batch_size"] = batch_size
        with trace.node("i_ref", worker_id=worker_by_vertex.get("i_ref"), step=step) as counters:
            with torch.no_grad():
                ref_chosen = _sequence_logps(torch, reference, tokenizer, chosen, prompt_lens, max_len)
                ref_rejected = _sequence_logps(torch, reference, tokenizer, rejected, prompt_lens, max_len)
            counters["batch_size"] = batch_size
        with trace.node("t_pi", worker_id=worker_by_vertex.get("t_pi"), step=step) as counters:
            actor_margin = actor_chosen - actor_rejected
            ref_margin = ref_chosen - ref_rejected
            loss = -torch.nn.functional.logsigmoid(beta * (actor_margin - ref_margin)).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if run.optimizer_runtime.grad_clip:
                torch.nn.utils.clip_grad_norm_(actor.parameters(), run.optimizer_runtime.grad_clip)
            optimizer.step()
            counters["loss"] = float(loss.detach().cpu())
        last_loss = float(loss.detach().cpu())

        if step % run.training.log_every_steps == 0:
            print(json.dumps({"step": step, "loss": last_loss}))
        if step % run.checkpoints.interval_steps == 0:
            out = Path(run.checkpoints.output_dir) / f"step_{step}"
            out.mkdir(parents=True, exist_ok=True)
            actor.save_pretrained(out)
            tokenizer.save_pretrained(out)

    output_dir = Path(run.checkpoints.output_dir) / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    actor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return DPOTrainResult(steps=run.training.max_steps, last_loss=last_loss, output_dir=str(output_dir))
