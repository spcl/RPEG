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
from typing import Any, Iterator

from passes.lower import ExecNodeType
from runtime.compiler import CompiledRuntimePlan
from runtime.containers import ContainerRef, ContainerStore, ShardingSpec, WeightVersion
from runtime.data import dataset_batch_indices
from runtime.errors import RuntimeConfigError, RuntimeDependencyError
from runtime.executor import NodeExecutionResult, RuntimeExecutionContext, RuntimeExecutor
from runtime.graph import RuntimeNodeSpec, build_runtime_graph
from runtime.hf import resolve_model_path
from runtime.metrics import write_runtime_summary_for_run
from runtime.ppo_math import minibatch_indices
from runtime.trace import RuntimeTraceWriter
from runtime.weight_versions import consumer_weight_version, load_module_state, snapshot_module_state, uses_shadow_copies


@dataclass(frozen=True)
class PPOTrainResult:
    steps: int
    last_actor_loss: float | None
    last_value_loss: float | None
    output_dir: str


@dataclass
class _PPOBatch:
    prompts: list[str]
    prompt_lens: list[int]
    input_ids: object
    attention_mask: object
    completion_ids: object | None = None
    response_mask: object | None = None
    old_logps: object | None = None
    actor_weight_version: int | None = None
    ref_logps: object | None = None
    rewards: object | None = None
    values: object | None = None
    critic_weight_version: int | None = None
    advantages: object | None = None
    returns: object | None = None


class CriticWithValueHead:
    def __init__(self, torch, base_model) -> None:
        self.torch = torch
        self.base_model = base_model
        hidden_size = int(getattr(base_model.config, "hidden_size", 0) or getattr(base_model.config, "n_embd", 0))
        if hidden_size <= 0:
            raise RuntimeConfigError("critic model config must expose hidden_size or n_embd")
        self.value_head = torch.nn.Linear(hidden_size, 1)
        first_param = next(base_model.parameters(), None)
        if first_param is not None:
            self.value_head = self.value_head.to(device=first_param.device, dtype=first_param.dtype)
        elif torch.cuda.is_available():
            self.value_head = self.value_head.cuda()

    def train(self):
        self.base_model.train()
        self.value_head.train()
        return self

    def eval(self):
        self.base_model.eval()
        self.value_head.eval()
        return self

    def parameters(self):
        yield from self.base_model.parameters()
        yield from self.value_head.parameters()

    def state_dict(self):
        state = {}
        state.update({f"base_model.{k}": v for k, v in self.base_model.state_dict().items()})
        state.update({f"value_head.{k}": v for k, v in self.value_head.state_dict().items()})
        return state

    def load_state_dict(self, state, strict: bool = True):
        base_prefix = "base_model."
        value_prefix = "value_head."
        base_state = {k[len(base_prefix):]: v for k, v in state.items() if k.startswith(base_prefix)}
        value_state = {k[len(value_prefix):]: v for k, v in state.items() if k.startswith(value_prefix)}
        self.base_model.load_state_dict(base_state, strict=strict)
        self.value_head.load_state_dict(value_state, strict=strict)

    def values(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeConfigError("critic forward must return hidden_states when output_hidden_states=True")
        return self.value_head(hidden_states[-1]).squeeze(-1)

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.base_model.save_pretrained(path)
        self.torch.save(self.value_head.state_dict(), path / "value_head.pt")


class ActorCriticWithValueHead:
    def __init__(self, torch, base_model) -> None:
        self.torch = torch
        self.base_model = base_model
        self.config = base_model.config
        hidden_size = int(getattr(base_model.config, "hidden_size", 0) or getattr(base_model.config, "n_embd", 0))
        if hidden_size <= 0:
            raise RuntimeConfigError("ac_share model config must expose hidden_size or n_embd")
        self.value_head = torch.nn.Linear(hidden_size, 1)
        first_param = next(base_model.parameters(), None)
        if first_param is not None:
            self.value_head = self.value_head.to(device=first_param.device, dtype=first_param.dtype)
        elif torch.cuda.is_available():
            self.value_head = self.value_head.cuda()

    def __call__(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["torch"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.torch is None:
            import torch  # type: ignore

            self.torch = torch

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def train(self):
        self.base_model.train()
        self.value_head.train()
        return self

    def eval(self):
        self.base_model.eval()
        self.value_head.eval()
        return self

    def parameters(self):
        yield from self.base_model.parameters()
        yield from self.value_head.parameters()

    def state_dict(self):
        state = {}
        state.update({f"base_model.{k}": v for k, v in self.base_model.state_dict().items()})
        state.update({f"value_head.{k}": v for k, v in self.value_head.state_dict().items()})
        return state

    def load_state_dict(self, state, strict: bool = True):
        base_prefix = "base_model."
        value_prefix = "value_head."
        base_state = {k[len(base_prefix):]: v for k, v in state.items() if k.startswith(base_prefix)}
        value_state = {k[len(value_prefix):]: v for k, v in state.items() if k.startswith(value_prefix)}
        self.base_model.load_state_dict(base_state, strict=strict)
        self.value_head.load_state_dict(value_state, strict=strict)

    def values(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeConfigError("ac_share forward must return hidden_states when output_hidden_states=True")
        return self.value_head(hidden_states[-1]).squeeze(-1)

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.base_model.save_pretrained(path)
        self.torch.save(self.value_head.state_dict(), path / "value_head.pt")


class TorchPPOGraphHandler:
    """
    Actual local torch PPO/GRPO/AC-share execution for the lowered RPEG graph.

    This is deliberately strict and single-process. Distributed Megatron PPO and GRPO use the
    worker-backed semantic handler for full-scale execution.
    """
    def __init__(self, compiled: CompiledRuntimePlan, *, step: int) -> None:
        self.compiled = compiled
        self.run = compiled.run_config
        self.step = step
        self.torch = None
        self.actor = None
        self.actor_inference = None
        self.reference = None
        self.reward = None
        self.critic: CriticWithValueHead | None = None
        self.critic_inference: CriticWithValueHead | None = None
        self.tokenizer = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.is_ac_share = bool(compiled.optimized_plan.structure.get("ac_share", False))
        self.prompt_iter = None
        self.last_actor_loss: float | None = None
        self.last_value_loss: float | None = None
        self.actor_versions: dict[int, dict[str, Any]] = {}
        self.critic_versions: dict[int, dict[str, Any]] = {}
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        run = self.run
        if run.algorithm.name not in {"ppo", "grpo"}:
            raise RuntimeConfigError("TorchPPOGraphHandler only supports algorithm.name=ppo/grpo")
        if not run.data.train_prompts:
            raise RuntimeConfigError(f"{run.algorithm.name.upper()} requires data.train_prompts")
        allowed_backends = {"torch_ppo", "torch", "local_torch"} if run.algorithm.name == "ppo" else {"torch_grpo", "torch", "local_torch"}
        if run.runtime.default_backend not in allowed_backends:
            expected = "/".join(sorted(allowed_backends))
            raise RuntimeConfigError(
                f"Local {run.algorithm.name.upper()} graph training requires runtime.default_backend={expected}"
            )
        _validate_local_torch_exactness(self.compiled)

        torch, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer = _require_torch_transformers()
        self.torch = torch
        actor_cfg = run.models["actor"]
        actor_dir = resolve_model_path(actor_cfg.source, actor_cfg.repo, actor_cfg.checkpoint_path, revision=actor_cfg.revision)
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
        actor_base = AutoModelForCausalLM.from_pretrained(actor_dir, torch_dtype=dtype, trust_remote_code=actor_cfg.trust_remote_code)
        _set_model_pad_token_id(actor_base, tokenizer.pad_token_id)
        if torch.cuda.is_available():
            actor_base = actor_base.cuda()
        actor = ActorCriticWithValueHead(torch, actor_base) if self.is_ac_share else actor_base
        actor.train()
        self.actor = actor
        self.actor_inference = copy.deepcopy(actor).eval()

        ref_cfg = run.models.get("reference")
        if ref_cfg and ref_cfg.source not in {"actor"}:
            ref_dir = resolve_model_path(ref_cfg.source, ref_cfg.repo, ref_cfg.checkpoint_path, revision=ref_cfg.revision)
            reference = AutoModelForCausalLM.from_pretrained(
                ref_dir,
                torch_dtype=dtype,
                trust_remote_code=ref_cfg.trust_remote_code,
            )
        else:
            reference = copy.deepcopy(actor.base_model if self.is_ac_share else actor)
        _set_model_pad_token_id(reference, tokenizer.pad_token_id)
        if torch.cuda.is_available():
            reference = reference.cuda()
        reference.eval()
        for p in reference.parameters():
            p.requires_grad_(False)
        self.reference = reference

        if run.algorithm.name == "ppo" and not self.is_ac_share:
            critic_cfg = run.models.get("critic", actor_cfg)
            critic_dir = actor_dir if critic_cfg.source == "actor" else resolve_model_path(
                critic_cfg.source,
                critic_cfg.repo,
                critic_cfg.checkpoint_path,
                revision=critic_cfg.revision,
            )
            critic_base = AutoModelForCausalLM.from_pretrained(
                critic_dir,
                torch_dtype=dtype,
                trust_remote_code=critic_cfg.trust_remote_code,
            )
            _set_model_pad_token_id(critic_base, tokenizer.pad_token_id)
            if torch.cuda.is_available():
                critic_base = critic_base.cuda()
            self.critic = CriticWithValueHead(torch, critic_base).train()
            self.critic_inference = CriticWithValueHead(torch, copy.deepcopy(critic_base)).eval()

        reward_cfg = run.models.get("reward")
        if reward_cfg is None:
            raise RuntimeConfigError(f"{run.algorithm.name.upper()} requires models.reward")
        reward_dir = actor_dir if reward_cfg.source == "actor" else resolve_model_path(
            reward_cfg.source,
            reward_cfg.repo,
            reward_cfg.checkpoint_path,
            revision=reward_cfg.revision,
        )
        try:
            reward = AutoModelForSequenceClassification.from_pretrained(
                reward_dir,
                torch_dtype=dtype,
                trust_remote_code=reward_cfg.trust_remote_code,
            )
        except Exception:
            reward = AutoModelForCausalLM.from_pretrained(
                reward_dir,
                torch_dtype=dtype,
                trust_remote_code=reward_cfg.trust_remote_code,
            )
        _set_model_pad_token_id(reward, tokenizer.pad_token_id)
        if torch.cuda.is_available():
            reward = reward.cuda()
        reward.eval()
        for p in reward.parameters():
            p.requires_grad_(False)
        self.reward = reward

        self.actor_optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=run.optimizer_runtime.learning_rate_actor,
            weight_decay=run.optimizer_runtime.weight_decay,
        )
        self.actor_versions[0] = snapshot_module_state(self.actor)
        if self.critic is not None:
            self.critic_optimizer = torch.optim.AdamW(
                self.critic.parameters(),
                lr=run.optimizer_runtime.learning_rate_critic,
                weight_decay=run.optimizer_runtime.weight_decay,
            )
            self.critic_versions[0] = snapshot_module_state(self.critic)
        self._initialized = True

    async def execute(self, node: RuntimeNodeSpec, context: RuntimeExecutionContext) -> NodeExecutionResult:
        self.initialize()
        if node.kind in {ExecNodeType.COMM, ExecNodeType.RESHARD}:
            return self._transfer_metadata(node, context)
        if node.kind != ExecNodeType.OP:
            raise RuntimeConfigError(f"Unsupported {self.run.algorithm.name.upper()} runtime node kind {node.kind}")
        if node.vertex_id in {"g_pi", "g_ac"}:
            return self._run_generation(node, context)
        if node.vertex_id == "i_ref":
            return self._run_reference(node, context)
        if node.vertex_id == "i_R":
            return self._run_reward(node, context)
        if node.vertex_id in {"i_V", "i_ac"}:
            return self._run_values(node, context)
        if node.vertex_id in {"t_pi", "t_ac"}:
            return self._train_actor(node, context)
        if node.vertex_id == "t_V":
            return self._train_critic(node, context)
        raise RuntimeConfigError(f"Unsupported {self.run.algorithm.name.upper()} vertex {node.vertex_id!r}")

    def _node_step(self, node: RuntimeNodeSpec) -> int:
        if node.logical_step is not None:
            return int(node.logical_step)
        if node.exec_node.op is None:
            return self.step
        return int(node.exec_node.op.k) + 1

    def save_if_needed(self, step: int | None = None) -> None:
        checkpoint_step = self.step if step is None else int(step)
        if checkpoint_step % self.run.checkpoints.interval_steps != 0:
            return
        root = Path(self.run.checkpoints.output_dir) / f"step_{checkpoint_step}"
        self._save(root)

    def save_final(self) -> str:
        root = Path(self.run.checkpoints.output_dir) / "final"
        self._save(root)
        return str(root)

    def _save(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.actor.save_pretrained(root / ("ac_share" if self.is_ac_share else "actor"))
        if self.critic is not None:
            self.critic.save_pretrained(root / "critic")
        self.tokenizer.save_pretrained(root / "tokenizer")

    def _new_batch(self, context: RuntimeExecutionContext, step: int) -> _PPOBatch:
        state_key = f"policy_batch:{step}"
        if state_key in context.state:
            return context.state[state_key]
        prompts = _prompt_batch(
            self.run.data.train_prompts,
            step=step,
            batch_size=self.run.workload.batch_size,
            offset=self.run.data.prompt_offset,
            shuffle=self.run.data.shuffle,
            seed=self.run.data.seed,
            shard_index=self.run.data.shard_index,
            num_shards=self.run.data.num_shards,
        )
        if self.run.algorithm.name in {"ppo", "grpo"}:
            samples_per_prompt = int(self.run.workload.samples_per_prompt)
            if samples_per_prompt <= 0:
                raise RuntimeConfigError("PPO/GRPO requires workload.samples_per_prompt >= 1")
            prompts = [prompt for prompt in prompts for _ in range(samples_per_prompt)]
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.run.workload.context_length,
        )
        device = next(self.actor.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        prompt_lens = [int(enc["input_ids"].shape[1])] * int(enc["input_ids"].shape[0])
        batch = _PPOBatch(
            prompts=prompts,
            prompt_lens=prompt_lens,
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
        context.state[state_key] = batch
        context.store.put_container(
            ContainerRef(
                name=f"prompts:{step}",
                step=step,
                producer_node_id="input",
                logical_shape=tuple(enc["input_ids"].shape),
                dtype="tokens",
                sharding=ShardingSpec("replicated"),
            )
        )
        return batch

    def _run_generation(self, node: RuntimeNodeSpec, context: RuntimeExecutionContext) -> NodeExecutionResult:
        torch = self.torch
        step = self._node_step(node)
        batch = self._new_batch(context, step)
        model_name = "ac_share" if self.is_ac_share else "actor"
        actor_version = consumer_weight_version(self.compiled, model_name, step)
        actor_shadow = uses_shadow_copies(self.compiled, model_name)
        actor = self._actor_for_inference(actor_version)
        with torch.no_grad():
            generation_kwargs = {
                "input_ids": batch.input_ids,
                "attention_mask": batch.attention_mask,
                "max_new_tokens": self.run.workload.generation_length,
                "do_sample": self.run.algorithm.do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if self.run.algorithm.do_sample:
                seed = int(self.run.algorithm.seed) + step * 100_000
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                generation_kwargs.update(
                    {
                        "temperature": self.run.algorithm.temperature,
                        "top_p": self.run.algorithm.top_p,
                        "top_k": self.run.algorithm.top_k,
                    }
                )
            generated = actor.generate(**generation_kwargs)
            full_attention = torch.ones_like(generated, dtype=batch.attention_mask.dtype)
            old_logps, response_mask = _token_logps(
                torch,
                actor,
                generated,
                full_attention,
                batch.prompt_lens,
                self.run.workload.generation_length,
            )
        batch.completion_ids = generated
        batch.response_mask = response_mask
        batch.old_logps = old_logps.detach()
        batch.actor_weight_version = actor_version
        ref = context.store.put_container(
            ContainerRef(
                name=f"completions:{step}",
                step=step,
                producer_node_id=node.node_id,
                logical_shape=tuple(generated.shape),
                dtype="tokens",
                sharding=ShardingSpec("worker", (node.worker_id or "",)),
                owner_worker_id=node.worker_id,
                metadata={"model": model_name, "weight_version": actor_version, "shadow": actor_shadow},
            )
        )
        return NodeExecutionResult(
            node_id=node.node_id,
            counters={
                "batch_size": len(batch.prompts),
                "prompt_batch_size": self.run.workload.batch_size,
                "samples_per_prompt": self.run.workload.samples_per_prompt,
                "generated_tokens": int(response_mask.sum().item()),
                "weight_version": actor_version,
                "shadow": actor_shadow,
            },
            containers=(ref,),
        )

    def _run_reference(self, node: RuntimeNodeSpec, context: RuntimeExecutionContext) -> NodeExecutionResult:
        torch = self.torch
        step = self._node_step(node)
        batch = self._require_generated(context, step)
        with torch.no_grad():
            ref_logps, _ = _token_logps(
                torch,
                self.reference,
                batch.completion_ids,
                torch.ones_like(batch.completion_ids),
                batch.prompt_lens,
                self.run.workload.generation_length,
            )
        batch.ref_logps = ref_logps.detach()
        return self._container_result(context, node, "log_ref", tuple(ref_logps.shape), str(ref_logps.dtype), {"kind": "ref_logprob"})

    def _run_reward(self, node: RuntimeNodeSpec, context: RuntimeExecutionContext) -> NodeExecutionResult:
        torch = self.torch
        step = self._node_step(node)
        batch = self._require_generated(context, step)
        with torch.no_grad():
            outputs = self.reward(input_ids=batch.completion_ids, attention_mask=torch.ones_like(batch.completion_ids))
            logits = getattr(outputs, "logits", outputs)
            if logits.ndim == 1:
                rewards = logits.float()
            elif logits.shape[-1] == 1:
                rewards = logits.squeeze(-1).float()
            else:
                rewards = logits[..., -1].float()
            if rewards.ndim > 1:
                rewards = rewards.mean(dim=tuple(range(1, rewards.ndim)))
        batch.rewards = rewards.detach()
        return self._container_result(context, node, "rewards", tuple(rewards.shape), str(rewards.dtype), {"kind": "reward"})

    def _run_values(self, node: RuntimeNodeSpec, context: RuntimeExecutionContext) -> NodeExecutionResult:
        torch = self.torch
        step = self._node_step(node)
        batch = self._require_generated(context, step)
        if self.is_ac_share:
            ac_version = consumer_weight_version(self.compiled, "ac_share", step)
            ac_shadow = uses_shadow_copies(self.compiled, "ac_share")
            ac_model = self._actor_for_inference(ac_version)
            with torch.no_grad():
                values_all = ac_model.values(batch.completion_ids, torch.ones_like(batch.completion_ids))
                values = _select_response_values(torch, values_all, batch.prompt_lens, self.run.workload.generation_length)
            batch.values = values.detach()
            batch.critic_weight_version = ac_version
            return self._container_result(
                context,
                node,
                "values",
                tuple(values.shape),
                str(values.dtype),
                {"kind": "ac_share_values", "weight_version": ac_version, "shadow": ac_shadow},
                metadata={"model": "ac_share", "weight_version": ac_version, "shadow": ac_shadow},
            )
        if self.critic is None:
            raise RuntimeConfigError("critic-free graph has no separate critic value node")
        critic_version = consumer_weight_version(self.compiled, "critic", step)
        critic_shadow = uses_shadow_copies(self.compiled, "critic")
        critic = self._critic_for_inference(critic_version)
        with torch.no_grad():
            values_all = critic.values(batch.completion_ids, torch.ones_like(batch.completion_ids))
            values = _select_response_values(torch, values_all, batch.prompt_lens, self.run.workload.generation_length)
        batch.values = values.detach()
        batch.critic_weight_version = critic_version
        return self._container_result(
            context,
            node,
            "values",
            tuple(values.shape),
            str(values.dtype),
            {"kind": "critic_values", "weight_version": critic_version, "shadow": critic_shadow},
            metadata={"model": "critic", "weight_version": critic_version, "shadow": critic_shadow},
        )

    def _train_actor(self, node: RuntimeNodeSpec, context: RuntimeExecutionContext) -> NodeExecutionResult:
        torch = self.torch
        step = self._node_step(node)
        batch = self._require_rollout(context, step)
        if self.run.algorithm.name == "grpo":
            current_logps, _ = _token_logps(
                torch,
                self.actor,
                batch.completion_ids,
                torch.ones_like(batch.completion_ids),
                batch.prompt_lens,
                self.run.workload.generation_length,
            )
            advantages = _grpo_group_advantages(
                torch,
                batch.rewards,
                batch.response_mask,
                samples_per_prompt=self.run.workload.samples_per_prompt,
            )
            batch.returns = batch.rewards.view(-1, 1).expand_as(current_logps).detach()
            batch.advantages = advantages.detach()
            ratio = torch.exp(current_logps - batch.old_logps)
            clipped = torch.clamp(ratio, 1.0 - self.run.algorithm.clip_range, 1.0 + self.run.algorithm.clip_range)
            pg = -torch.minimum(ratio * advantages, clipped * advantages)
            kl = (current_logps - batch.ref_logps) * batch.response_mask
            loss = ((pg * batch.response_mask).sum() + self.run.algorithm.kl_coef * kl.sum()) / batch.response_mask.sum().clamp(min=1.0)
            value_loss = None
            self.actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.run.optimizer_runtime.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.run.optimizer_runtime.grad_clip)
            self.actor_optimizer.step()
            optimizer_updates = 1
        else:
            returns, advantages = _returns_advantages(
                batch.old_logps,
                batch.ref_logps,
                batch.rewards,
                batch.values,
                batch.response_mask,
                beta=self.run.algorithm.beta,
                gamma=self.run.algorithm.gamma,
                gae_lambda=self.run.algorithm.gae_lambda,
                normalize_advantages=self.run.algorithm.normalize_advantages,
            )
            batch.returns = returns.detach()
            batch.advantages = advantages.detach()
            configured_minibatch = self.run.algorithm.minibatch_size
            minibatch_size = int(batch.completion_ids.shape[0]) if configured_minibatch is None else configured_minibatch
            losses = []
            value_losses = []
            update_indices = minibatch_indices(
                int(batch.completion_ids.shape[0]),
                minibatch_size,
                epochs=self.run.algorithm.epochs,
                seed=self.run.algorithm.seed + step * 100_000,
            )
            for _epoch, indices_cpu in update_indices:
                indices = indices_cpu.to(device=batch.completion_ids.device)
                input_ids = batch.completion_ids.index_select(0, indices)
                prompt_lens = [batch.prompt_lens[int(i)] for i in indices_cpu.tolist()]
                response_mask = batch.response_mask.index_select(0, indices)
                old_logps = batch.old_logps.index_select(0, indices)
                mb_advantages = batch.advantages.index_select(0, indices)
                current_logps, _ = _token_logps(
                    torch,
                    self.actor,
                    input_ids,
                    torch.ones_like(input_ids),
                    prompt_lens,
                    self.run.workload.generation_length,
                )
                ratio = torch.exp(current_logps - old_logps)
                clipped = torch.clamp(ratio, 1.0 - self.run.algorithm.clip_range, 1.0 + self.run.algorithm.clip_range)
                pg = -torch.minimum(ratio * mb_advantages, clipped * mb_advantages)
                actor_loss = (pg * response_mask).sum() / response_mask.sum().clamp(min=1.0)
                if self.run.algorithm.entropy_coef:
                    entropies = _token_entropies(
                        torch,
                        self.actor,
                        input_ids,
                        torch.ones_like(input_ids),
                        prompt_lens,
                        self.run.workload.generation_length,
                    )
                    actor_loss = actor_loss - self.run.algorithm.entropy_coef * (
                        entropies * response_mask
                    ).sum() / response_mask.sum().clamp(min=1.0)
                value_loss = None
                if self.is_ac_share:
                    values_all = self.actor.values(input_ids, torch.ones_like(input_ids))
                    current_values = _select_response_values(
                        torch,
                        values_all,
                        prompt_lens,
                        self.run.workload.generation_length,
                    )
                    mb_returns = batch.returns.index_select(0, indices)
                    value_loss = (((current_values - mb_returns) ** 2) * response_mask).sum() / response_mask.sum().clamp(min=1.0)
                    loss = actor_loss + value_loss
                else:
                    loss = actor_loss
                self.actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.run.optimizer_runtime.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.run.optimizer_runtime.grad_clip)
                self.actor_optimizer.step()
                losses.append(float(loss.detach().cpu()))
                if value_loss is not None:
                    value_losses.append(float(value_loss.detach().cpu()))
            loss = torch.tensor(sum(losses) / len(losses))
            value_loss = None if not value_losses else torch.tensor(sum(value_losses) / len(value_losses))
            optimizer_updates = len(losses)
        self.last_actor_loss = float(loss.detach().cpu())
        if value_loss is not None:
            self.last_value_loss = float(value_loss.detach().cpu())
        self.actor_versions[step] = snapshot_module_state(self.actor)
        model_name = "ac_share" if self.is_ac_share else "actor"
        weight = context.store.publish_weight(
            WeightVersion(
                model=model_name,
                version=step,
                produced_by=node.node_id,
                step=step,
                layout=ShardingSpec("worker", (node.worker_id or "",)),
                dtype=str(next(self.actor.parameters()).dtype),
                optimizer_state_owner=node.worker_id,
                shadow=uses_shadow_copies(self.compiled, model_name),
            )
        )
        return NodeExecutionResult(
            node_id=node.node_id,
            counters={
                "loss": self.last_actor_loss,
                "kind": "ppo_ac_train" if self.is_ac_share else f"{self.run.algorithm.name}_actor_train",
                "optimizer_updates": optimizer_updates,
                "produced_weight": {"model": model_name, "version": step, "shadow": weight.shadow},
                **({"value_loss": self.last_value_loss} if self.is_ac_share else {}),
            },
            weights=(weight,),
        )

    def _train_critic(self, node: RuntimeNodeSpec, context: RuntimeExecutionContext) -> NodeExecutionResult:
        torch = self.torch
        step = self._node_step(node)
        batch = self._require_rollout(context, step)
        if self.critic is None or self.critic_optimizer is None:
            raise RuntimeConfigError("critic-free graph has no separate critic train node")
        if batch.returns is None:
            returns, advantages = _returns_advantages(
                batch.old_logps,
                batch.ref_logps,
                batch.rewards,
                batch.values,
                batch.response_mask,
                beta=self.run.algorithm.beta,
                gamma=self.run.algorithm.gamma,
                gae_lambda=self.run.algorithm.gae_lambda,
                normalize_advantages=self.run.algorithm.normalize_advantages,
            )
            batch.returns = returns.detach()
            batch.advantages = advantages.detach()
        configured_minibatch = self.run.algorithm.minibatch_size
        minibatch_size = int(batch.completion_ids.shape[0]) if configured_minibatch is None else configured_minibatch
        losses = []
        for _epoch, indices_cpu in minibatch_indices(
            int(batch.completion_ids.shape[0]),
            minibatch_size,
            epochs=self.run.algorithm.epochs,
            seed=self.run.algorithm.seed + step * 100_000,
        ):
            indices = indices_cpu.to(device=batch.completion_ids.device)
            input_ids = batch.completion_ids.index_select(0, indices)
            prompt_lens = [batch.prompt_lens[int(i)] for i in indices_cpu.tolist()]
            response_mask = batch.response_mask.index_select(0, indices)
            returns = batch.returns.index_select(0, indices)
            values_all = self.critic.values(input_ids, torch.ones_like(input_ids))
            values = _select_response_values(torch, values_all, prompt_lens, self.run.workload.generation_length)
            loss = (((values - returns) ** 2) * response_mask).sum() / response_mask.sum().clamp(min=1.0)
            self.critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.run.optimizer_runtime.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.run.optimizer_runtime.grad_clip)
            self.critic_optimizer.step()
            losses.append(float(loss.detach().cpu()))
        self.last_value_loss = sum(losses) / len(losses)
        self.critic_versions[step] = snapshot_module_state(self.critic)
        weight = context.store.publish_weight(
            WeightVersion(
                model="critic",
                version=step,
                produced_by=node.node_id,
                step=step,
                layout=ShardingSpec("worker", (node.worker_id or "",)),
                dtype=str(next(self.critic.base_model.parameters()).dtype),
                optimizer_state_owner=node.worker_id,
                shadow=uses_shadow_copies(self.compiled, "critic"),
            )
        )
        return NodeExecutionResult(
            node_id=node.node_id,
            counters={
                "loss": self.last_value_loss,
                "kind": "ppo_critic_train",
                "optimizer_updates": len(losses),
                "produced_weight": {"model": "critic", "version": step, "shadow": weight.shadow},
            },
            weights=(weight,),
        )

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

    def _container_result(
        self,
        context: RuntimeExecutionContext,
        node: RuntimeNodeSpec,
        name: str,
        shape: tuple[int, ...],
        dtype: str,
        counters: dict[str, object],
        metadata: dict[str, object] | None = None,
    ) -> NodeExecutionResult:
        step = self._node_step(node)
        ref = context.store.put_container(
            ContainerRef(
                name=f"{name}:{step}",
                step=step,
                producer_node_id=node.node_id,
                logical_shape=shape,
                dtype=dtype,
                sharding=ShardingSpec("worker", (node.worker_id or "",)),
                owner_worker_id=node.worker_id,
                metadata=metadata or {},
            )
        )
        return NodeExecutionResult(node_id=node.node_id, counters=counters, containers=(ref,))

    def _require_generated(self, context: RuntimeExecutionContext, step: int) -> _PPOBatch:
        batch = self._new_batch(context, step)
        if batch.completion_ids is None or batch.old_logps is None or batch.response_mask is None:
            raise RuntimeConfigError(f"{self.run.algorithm.name.upper()} node requires generated completions")
        return batch

    def _require_rollout(self, context: RuntimeExecutionContext, step: int) -> _PPOBatch:
        batch = self._require_generated(context, step)
        requirements = ("ref_logps", "rewards") if self.run.algorithm.name == "grpo" else ("ref_logps", "rewards", "values")
        missing = [
            name
            for name in requirements
            if getattr(batch, name) is None
        ]
        if missing:
            raise RuntimeConfigError(f"{self.run.algorithm.name.upper()} training node missing rollout inputs: {missing}")
        return batch

    def _actor_for_inference(self, version: int):
        if version not in self.actor_versions:
            raise RuntimeConfigError(
                f"actor weight version {version} is required but has not been published"
            )
        load_module_state(self.actor_inference, self.actor_versions[version])
        self.actor_inference.eval()
        return self.actor_inference

    def _critic_for_inference(self, version: int):
        if self.critic_inference is None:
            raise RuntimeConfigError("critic-free graph has no separate critic inference model")
        if version not in self.critic_versions:
            raise RuntimeConfigError(
                f"critic weight version {version} is required but has not been published"
            )
        load_module_state(self.critic_inference, self.critic_versions[version])
        self.critic_inference.eval()
        return self.critic_inference


def _require_torch_transformers():
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeDependencyError("PPO/GRPO/AC-share torch execution requires torch and transformers") from exc
    return torch, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


def _set_model_pad_token_id(model, pad_token_id) -> None:
    config = getattr(model, "config", None)
    if config is not None and getattr(config, "pad_token_id", None) is None and pad_token_id is not None:
        config.pad_token_id = int(pad_token_id)


def _cycle_prompts(path: str | Path) -> Iterator[str]:
    while True:
        yielded = False
        with Path(path).open() as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "prompt" not in row:
                    raise RuntimeConfigError(f"{path}:{line_no} missing prompt field")
                yielded = True
                yield str(row["prompt"])
        if not yielded:
            raise RuntimeConfigError(f"Prompt file is empty: {path}")


def _prompt_batch(
    path: str | Path,
    *,
    step: int,
    batch_size: int,
    offset: int = 0,
    shuffle: bool = False,
    seed: int = 0,
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[str]:
    prompts = _load_prompts(path)
    indices = dataset_batch_indices(
        len(prompts),
        step=step,
        batch_size=batch_size,
        offset=offset,
        shuffle=shuffle,
        seed=seed,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    return [prompts[idx] for idx in indices]


def _load_prompts(path: str | Path) -> list[str]:
    prompts: list[str] = []
    with Path(path).open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "prompt" not in row:
                raise RuntimeConfigError(f"{path}:{line_no} missing prompt field")
            prompts.append(str(row["prompt"]))
    if not prompts:
        raise RuntimeConfigError(f"Prompt file is empty: {path}")
    return prompts


def _validate_local_torch_exactness(compiled: CompiledRuntimePlan) -> None:
    devices = {(d.hostname, d.local_cuda_index) for worker in compiled.workers for d in worker.devices}
    for worker in compiled.workers:
        if worker.parallelism != (1, 1, 1):
            raise RuntimeConfigError(
                f"Local torch PPO/GRPO/AC-share only supports exact dp=pp=tp=1 workers; {worker.worker_id} has {worker.parallelism}"
            )
        if len(worker.devices) != 1:
            raise RuntimeConfigError(f"Local torch PPO/GRPO/AC-share requires one device per worker: {worker.worker_id}")
    if len(devices) != 1:
        raise RuntimeConfigError(
            "Local torch PPO/GRPO/AC-share cannot execute a multi-device RPEG plan exactly; use the Megatron worker handler"
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


def _token_logps(torch, model, input_ids, attention_mask, prompt_lens: list[int], max_new_tokens: int):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    token_logps = torch.nn.functional.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    masks = []
    selected = []
    width = token_logps.shape[1]
    for row, prompt_len in enumerate(prompt_lens):
        start = max(0, prompt_len - 1)
        end = min(width, start + max_new_tokens)
        vals = token_logps[row, start:end]
        if vals.numel() < max_new_tokens:
            vals = torch.nn.functional.pad(vals, (0, max_new_tokens - vals.numel()))
        mask = torch.zeros(max_new_tokens, device=input_ids.device, dtype=torch.float32)
        mask[: max(0, end - start)] = 1.0
        selected.append(vals[:max_new_tokens])
        masks.append(mask)
    return torch.stack(selected, dim=0), torch.stack(masks, dim=0)


def _token_entropies(torch, model, input_ids, attention_mask, prompt_lens: list[int], max_new_tokens: int):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = torch.nn.functional.log_softmax(outputs.logits[:, :-1, :].float(), dim=-1)
    token_entropies = -(log_probs.exp() * log_probs).sum(dim=-1)
    rows = []
    width = token_entropies.shape[1]
    for row, prompt_len in enumerate(prompt_lens):
        start = max(0, prompt_len - 1)
        end = min(width, start + max_new_tokens)
        values = token_entropies[row, start:end]
        if values.numel() < max_new_tokens:
            values = torch.nn.functional.pad(values, (0, max_new_tokens - values.numel()))
        rows.append(values[:max_new_tokens])
    return torch.stack(rows, dim=0)


def _select_response_values(torch, values_all, prompt_lens: list[int], max_new_tokens: int):
    rows = []
    width = values_all.shape[1]
    for row, prompt_len in enumerate(prompt_lens):
        start = max(0, prompt_len - 1)
        end = min(width, start + max_new_tokens)
        vals = values_all[row, start:end]
        if vals.numel() < max_new_tokens:
            vals = torch.nn.functional.pad(vals, (0, max_new_tokens - vals.numel()))
        rows.append(vals[:max_new_tokens])
    return torch.stack(rows, dim=0)


def _returns_advantages(
    old_logps,
    ref_logps,
    rewards,
    values,
    response_mask,
    *,
    beta: float,
    gamma: float,
    gae_lambda: float,
    normalize_advantages: bool,
):
    from runtime.ppo_math import build_ppo_targets

    _token_rewards, advantages, returns = build_ppo_targets(
        old_logps,
        ref_logps,
        rewards,
        values,
        response_mask,
        beta=beta,
        gamma=gamma,
        gae_lambda=gae_lambda,
        normalize_advantages=normalize_advantages,
    )
    return returns, advantages


def _grpo_group_advantages(torch, rewards, response_mask, *, samples_per_prompt: int):
    if samples_per_prompt <= 0:
        raise RuntimeConfigError("GRPO requires workload.samples_per_prompt >= 1")
    if rewards.numel() % samples_per_prompt != 0:
        raise RuntimeConfigError(
            f"GRPO reward batch size {rewards.numel()} is incompatible with samples_per_prompt={samples_per_prompt}"
        )
    grouped = rewards.float().view(-1, samples_per_prompt)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
    advantages = ((grouped - mean) / std).reshape(-1, 1)
    return advantages.expand_as(response_mask)


def run_graph_ppo(compiled: CompiledRuntimePlan) -> PPOTrainResult:
    if compiled.run_config.training.max_steps < 1:
        raise RuntimeConfigError("PPO graph training requires training.max_steps >= 1")
    if compiled.run_config.runtime.default_backend == "megatron":
        from runtime.algorithms.megatron_ppo_graph import run_megatron_graph_ppo

        return run_megatron_graph_ppo(compiled)
    graph = compiled.runtime_graph()
    trace = RuntimeTraceWriter(compiled.run_config.runtime.trace_path, compiled.run_config.run_id)
    store_root = Path(compiled.run_config.runtime.store_dir or Path(compiled.run_config.checkpoints.output_dir) / "runtime_store")
    store = ContainerStore(store_root / "containers")
    handler = TorchPPOGraphHandler(compiled, step=1)
    last_actor_loss = None
    last_value_loss = None
    executor = RuntimeExecutor(
        graph,
        run_id=compiled.run_config.run_id,
        handler=handler,
        trace=trace,
        store=store,
    )
    executor.run()
    handler.save_if_needed(compiled.run_config.training.max_steps)
    last_actor_loss = handler.last_actor_loss
    last_value_loss = handler.last_value_loss
    if compiled.run_config.training.max_steps % compiled.run_config.training.log_every_steps == 0:
        print(
            json.dumps(
                {
                    "step": compiled.run_config.training.max_steps,
                    "actor_loss": last_actor_loss,
                    "value_loss": last_value_loss,
                }
            )
        )
    if compiled.run_config.checkpoints.save_final:
        output_dir = handler.save_final()
    else:
        output_dir = str(Path(compiled.run_config.checkpoints.output_dir))
    write_runtime_summary_for_run(
        compiled.run_config,
        extra={
            "algorithm": "ppo",
            "steps": compiled.run_config.training.max_steps,
            "output_dir": output_dir,
            "last_actor_loss": last_actor_loss,
            "last_value_loss": last_value_loss,
        },
    )
    return PPOTrainResult(
        steps=compiled.run_config.training.max_steps,
        last_actor_loss=last_actor_loss,
        last_value_loss=last_value_loss,
        output_dir=output_dir,
    )
