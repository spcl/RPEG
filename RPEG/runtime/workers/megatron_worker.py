# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import json
import gc
import os
import tempfile
import time

from contextlib import nullcontext
from functools import partial
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from runtime.errors import RuntimeConfigError, RuntimeDependencyError
from runtime.models.base import MegatronGPTArchitecture
from runtime.models.registry import architecture_from_hf_path
from runtime.workers.base import BuiltModelInfo


def _activation(name: str):
    import torch.nn.functional as F

    normalized = name.lower()
    if normalized in {"silu", "swish"}:
        return F.silu
    if normalized == "gelu":
        return F.gelu
    raise RuntimeConfigError(f"Unsupported Megatron activation {name!r}")


def _torch_dtype(dtype: str):
    import torch

    normalized = dtype.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise RuntimeConfigError(f"Unsupported Megatron dtype {dtype!r}")


def _ordinary_training_tensor(value, *, device, dtype):
    """
    Move a semantic input to the training device as an ordinary tensor.

    Tensor.to may return its input unchanged when device and dtype already match. That is unsafe for
    a tensor produced under inference_mode: embedding and other autograd kernels may need to save it
    for backward. Enforce the contract at the last boundary before the training forward.
    """
    import torch

    is_inference = getattr(value, "is_inference", None)
    with torch.inference_mode(False):
        value = value.detach().to(device=device, dtype=dtype)
        if callable(is_inference) and value.is_inference():
            value = value.clone()
    return value


def _split_sequence_evenly(value, parts: int):
    if parts <= 1:
        return [value]
    length = len(value)
    if length % parts != 0:
        raise RuntimeConfigError(
            f"sequence length {length} must be divisible by pipeline microbatch count {parts}"
        )
    width = length // parts
    return [value[idx * width:(idx + 1) * width] for idx in range(parts)]


def _sample_next_token(logits, *, temperature: float, top_p: float, top_k: int, generator=None):
    import torch

    scores = logits / float(temperature)
    vocab = int(scores.shape[-1])
    if top_k > 0 and top_k < vocab:
        threshold = torch.topk(scores, k=top_k, dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < threshold, float("-inf"))
    if top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        sorted_probs = torch.nn.functional.softmax(sorted_scores, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        scores = torch.full_like(scores, float("-inf"))
        scores.scatter_(dim=-1, index=sorted_indices, src=sorted_scores)
    probs = torch.nn.functional.softmax(scores, dim=-1)
    if torch.isnan(probs).any() or (probs.sum(dim=-1) <= 0).any():
        raise RuntimeConfigError("generation sampling produced invalid probabilities")
    return torch.multinomial(probs, num_samples=1, generator=generator)


def _attention_backend(name: str):
    from megatron.core.transformer.enums import AttnBackend

    normalized = name.lower()
    if normalized == "auto":
        return AttnBackend.auto
    if normalized == "flash":
        return AttnBackend.flash
    if normalized == "fused":
        return AttnBackend.fused
    if normalized == "unfused":
        return AttnBackend.unfused
    if normalized == "local":
        return AttnBackend.local
    raise RuntimeConfigError(f"Unsupported Megatron attention backend {name!r}")


def _initialize_dense_model_parallel(
    parallel_state,
    *,
    tp: int,
    pp: int,
    group_sink: list[object] | None = None,
    accelerator_backend: str | None = None,
    pipeline_max_ctas: int | None = None,
    pipeline_cta_policy: int | None = None,
    nccl_options_factory=None,
) -> dict[str, int]:
    """
    Initialize Megatron while pooling equivalent process groups.

    Megatron creates separate logical groups whose memberships are identical when context and expert
    parallelism are both one. Dense RPEG workers never use expert parallelism, and distinct NCCL
    communicators for every alias can exhaust CXI endpoint resources. Reuse is limited to groups
    with identical ranks and compatible construction options.
    """
    if pipeline_max_ctas is not None:
        if isinstance(pipeline_max_ctas, bool) or int(pipeline_max_ctas) < 1:
            raise RuntimeConfigError(
                "pipeline_max_ctas must be a positive integer, got "
                f"{pipeline_max_ctas!r}"
            )
        pipeline_max_ctas = int(pipeline_max_ctas)
    if pipeline_cta_policy is not None:
        if isinstance(pipeline_cta_policy, bool) or int(pipeline_cta_policy) not in {0, 1, 2}:
            raise RuntimeConfigError(
                "pipeline_cta_policy must be default=0, efficiency=1, or "
                f"zero=2, got {pipeline_cta_policy!r}"
            )
        pipeline_cta_policy = int(pipeline_cta_policy)

    original_create_group = parallel_state.create_group
    cache: dict[tuple[object, ...], object] = {}
    counters = {"requested": 0, "created": 0, "reused": 0}
    if pipeline_max_ctas is not None:
        counters.update(
            {
                "pipeline_max_ctas": pipeline_max_ctas,
                "pipeline_groups_limited": 0,
            }
        )
    if pipeline_cta_policy is not None:
        counters.update(
            {
                "pipeline_cta_policy": pipeline_cta_policy,
                "pipeline_cta_policy_zero": int(pipeline_cta_policy == 2),
                "pipeline_groups_policy_applied": 0,
            }
        )

    def create_group(
        ranks=None,
        timeout=None,
        backend=None,
        pg_options=None,
        use_local_synchronization=False,
        group_desc=None,
    ):
        counters["requested"] += 1
        # The cluster uses a Gloo default group as a control plane, so that the finite Slingshot/CXI
        # endpoint budget is reserved for data-plane communicators. Megatron leaves most subgroup
        # backends unspecified; bind those groups explicitly to NCCL, while preserving deliberate
        # Gloo groups.
        effective_backend = accelerator_backend if backend is None else backend
        effective_pg_options = pg_options
        if (
            (pipeline_max_ctas is not None or pipeline_cta_policy is not None)
            and str(effective_backend).lower() == "nccl"
            and group_desc == "PIPELINE_MODEL_PARALLEL_GROUP"
        ):
            if effective_pg_options is not None:
                raise RuntimeConfigError(
                    "RPEG PP NCCL options cannot override Megatron PP "
                    "pg_options; configure only one PP communicator policy"
                )
            if nccl_options_factory is None:
                import torch

                effective_pg_options = torch.distributed.ProcessGroupNCCL.Options()
            else:
                effective_pg_options = nccl_options_factory()
            if pipeline_max_ctas is not None:
                effective_pg_options.config.max_ctas = pipeline_max_ctas
                counters["pipeline_groups_limited"] += 1
            if pipeline_cta_policy is not None:
                effective_pg_options.config.cta_policy = pipeline_cta_policy
                counters["pipeline_groups_policy_applied"] += 1
        cacheable = (
            ranks is not None
            and effective_pg_options is None
            and not use_local_synchronization
        )
        key = (
            tuple(int(rank) for rank in ranks) if ranks is not None else None,
            effective_backend,
            timeout,
        )
        if cacheable and key in cache:
            counters["reused"] += 1
            return cache[key]
        group = original_create_group(
            ranks=ranks,
            timeout=timeout,
            backend=effective_backend,
            pg_options=effective_pg_options,
            use_local_synchronization=use_local_synchronization,
            group_desc=group_desc,
        )
        counters["created"] += 1
        if group_sink is not None:
            group_sink.append(group)
        if cacheable:
            cache[key] = group
        return group

    parallel_state.create_group = create_group
    try:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            # Megatron's inference wrapper computes peers from this group but its v0.13
            # communication helper omits the group from P2POp. Keep the authoritative PP object
            # unambiguously on NCCL; the local compatibility patch below also passes it to every
            # P2POp.
            pipeline_model_parallel_comm_backend=(
                accelerator_backend if pp > 1 else None
            ),
            order="tp-pp-dp",
        )
    finally:
        parallel_state.create_group = original_create_group
    return counters


def _pipeline_nccl_max_ctas_from_env() -> int | None:
    """
    Read the opt-in PP communicator CTA budget.
    """
    raw = os.environ.get("RPEG_NCCL_PP_MAX_CTAS", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeConfigError(
            f"RPEG_NCCL_PP_MAX_CTAS must be a positive integer, got {raw!r}"
        ) from exc
    if value < 1:
        raise RuntimeConfigError(
            f"RPEG_NCCL_PP_MAX_CTAS must be a positive integer, got {raw!r}"
        )
    return value


def _pipeline_nccl_cta_policy_from_env() -> int | None:
    """
    Read the opt-in NCCL 2.29 PP communicator CTA policy.
    """
    raw = os.environ.get("RPEG_NCCL_PP_CTA_POLICY", "").strip().lower()
    if not raw:
        return None
    policies = {
        "default": 0,
        "0": 0,
        "efficiency": 1,
        "1": 1,
        "zero": 2,
        "2": 2,
    }
    if raw not in policies:
        raise RuntimeConfigError(
            "RPEG_NCCL_PP_CTA_POLICY must be default, efficiency, or zero, "
            f"got {raw!r}"
        )
    return policies[raw]


def _install_explicit_megatron_inference_pp_group(
    *,
    torch_module=None,
    communication_module=None,
    wrapper_module=None,
) -> bool:
    """
    Make Megatron inference P2P use its NCCL pipeline group.

    Megatron-Core v0.13 accepts pp_group in its inference send/receive helpers, uses it to calculate
    the peer, but constructs P2POp without that group. PyTorch consequently falls back to WORLD.
    RPEG deliberately keeps WORLD on Gloo as a low-volume control plane , so that fallback attempts
    to send CUDA activations with Gloo and fails with 'writev: Bad address'. Patch both the defining
    module and the names imported by abstract_model_inference_wrapper. No process group is created
    here; traffic stays on Megatron's existing full PP communicator.

    Returns True only when this call installed the patch.
    """
    if torch_module is None:
        import torch as torch_module
    if communication_module is None:
        from megatron.core.inference import communication_utils as communication_module
    if wrapper_module is None:
        from megatron.core.inference.model_inference_wrappers import (
            abstract_model_inference_wrapper as wrapper_module,
        )

    marker = "_rpeg_explicit_pipeline_process_group"
    if getattr(communication_module, marker, False):
        return False

    original_recv = communication_module.recv_from_prev_pipeline_rank_
    original_send = communication_module.send_to_next_pipeline_rank

    def recv_from_prev_pipeline_rank_(recv_buffer=None, pp_group=None):
        if pp_group is None:
            return original_recv(recv_buffer, pp_group)
        group_ranks = torch_module.distributed.get_process_group_ranks(pp_group)
        previous_rank = group_ranks[(pp_group.rank() - 1) % pp_group.size()]
        operation = torch_module.distributed.P2POp(
            torch_module.distributed.irecv,
            recv_buffer,
            previous_rank,
            group=pp_group,
        )
        requests = torch_module.distributed.batch_isend_irecv([operation])
        for request in requests:
            request.wait()
        torch_module.cuda.synchronize()

    def send_to_next_pipeline_rank(tensor=None, pp_group=None):
        if pp_group is None:
            return original_send(tensor, pp_group)
        group_ranks = torch_module.distributed.get_process_group_ranks(pp_group)
        next_rank = group_ranks[(pp_group.rank() + 1) % pp_group.size()]
        operation = torch_module.distributed.P2POp(
            torch_module.distributed.isend,
            tensor,
            next_rank,
            group=pp_group,
        )
        requests = torch_module.distributed.batch_isend_irecv([operation])
        for request in requests:
            request.wait()
        torch_module.cuda.synchronize()

    communication_module.recv_from_prev_pipeline_rank_ = recv_from_prev_pipeline_rank_
    communication_module.send_to_next_pipeline_rank = send_to_next_pipeline_rank
    # These functions were imported into the wrapper module by name, so patch its bound globals as
    # well as their defining module.
    wrapper_module.recv_from_prev_pipeline_rank_ = recv_from_prev_pipeline_rank_
    wrapper_module.send_to_next_pipeline_rank = send_to_next_pipeline_rank
    setattr(communication_module, marker, True)
    return True



def _unique_prefixes(*prefixes: str) -> tuple[str, ...]:
    seen = set()
    out = []
    for prefix in prefixes:
        normalized = str(prefix).strip().removesuffix(".")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return tuple(out)


def _hf_auxiliary_aliases(*aliases: str) -> tuple[str, ...]:
    wrappers = (
        "",
        "model",
        "transformer",
        "pretrained_model",
        "base_model",
        "base_model.model",
        "model.model",
    )
    out = []
    for alias in aliases:
        normalized = str(alias).strip().removesuffix(".")
        if not normalized:
            continue
        for wrapper in wrappers:
            out.append(normalized if not wrapper else f"{wrapper}.{normalized}")
    return tuple(out)


def _normalize_generation_backend(name: str) -> str:
    normalized = str(name or "full_sequence").strip().lower().replace("-", "_")
    if normalized in {"full", "full_sequence", "naive"}:
        return "full_sequence"
    if normalized in {"static_kv", "kv", "kv_cache"}:
        return "static_kv"
    raise RuntimeConfigError(f"unsupported generation backend {name!r}")


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return int(value)
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def _coerce_auxiliary_head_tensor(value, current_value, key: str):
    if tuple(value.shape) == tuple(current_value.shape):
        return value
    if tuple(current_value.shape) == (1,) and tuple(value.shape) == ():
        return value.reshape(1)
    if len(current_value.shape) == 2 and current_value.shape[0] == 1:
        if tuple(value.shape) == (int(current_value.shape[1]),):
            return value.reshape_as(current_value)
    raise RuntimeConfigError(
        f"HF auxiliary head tensor {key} has shape {tuple(value.shape)}, "
        f"expected {tuple(current_value.shape)}"
    )


@dataclass(frozen=True)
class MegatronBuildConfig:
    model_path: str
    parallelism: tuple[int, int, int]
    dtype: str = "bf16"
    use_transformer_engine: bool = True
    perform_initialization: bool = False
    max_sequence_length: int | None = None
    activation_checkpointing: bool = False
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    distributed_optimizer: str = "none"
    zero2_bucket_bytes: int = 64 * 1024 * 1024
    zero2_max_inflight_buckets: int = 2
    attention_backend: str = "unfused"
    microbatch: int = 1
    auxiliary_heads: tuple[str, ...] = ()
    value_head_prefixes: tuple[str, ...] = ()
    reward_head_prefixes: tuple[str, ...] = ()
    static_kv_cache_max_entries: int = 4
    static_kv_cache_batch_bucket: int = 0
    static_kv_cache_sequence_bucket: int = 0
    pipeline_nccl_max_ctas: int | None = None
    pipeline_nccl_cta_policy: int | None = None
    seed: int = 1234


@dataclass
class _PhasedGenerationState:
    generated: Any
    old_logps: Any
    prompt_lens: list[int]
    original_indices: Any
    max_new_tokens: int
    tokens_generated: int
    backend: str
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    eos_token_ids: tuple[int, ...]
    generator: Any
    static_context: Any = None
    static_wrapper: Any = None
    static_cache_hit: bool = False
    static_cache_capacity: tuple[int, int] | None = None
    pending_completions: Any = None


class MegatronWorker:
    """
    Megatron-Core worker with exact rank/world initialization.

    The worker owns one Megatron model-parallel world. It supports multi-rank construction,
    inference, rank-sharded checkpoint save/load, and scheduled training/inference paths for
    pipeline-parallel Megatron workers where the runtime has explicit schedule coverage.
    """
    def __init__(self) -> None:
        self._owns_process_group = False
        self._model_parallel_initialized = False
        self._model_parallel_group_counters: dict[str, int] = {}
        self._model_parallel_process_groups: list[object] = []
        self.model = None
        self.architecture: MegatronGPTArchitecture | None = None
        self.config: MegatronBuildConfig | None = None
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self._optimizer = None
        self._optimizer_parallelism: tuple[int, int, int] | None = None
        self._released_operational_parameter_specs: tuple[tuple[Any, tuple[int, ...], Any, Any], ...] = ()
        self.value_head = None
        self.reward_head = None
        self._last_generation_counters: dict[str, Any] = {}
        self._static_kv_serving_cache: dict[tuple[int, int], tuple[Any, Any]] = {}
        self._static_kv_serving_hits = 0
        self._static_kv_serving_misses = 0
        self._static_kv_serving_evictions = 0
        self._inference_tensor_caches_dirty = False
        self._last_training_transition_counters: dict[str, Any] = {}
        self._phased_generation_states: dict[str, _PhasedGenerationState] = {}
        self._zero2_bucket_layout: dict[int, dict[str, Any]] = {}
        self._zero2_bucket_inputs: dict[int, Any] = {}
        self._zero2_bucket_seen: dict[int, set[Any]] = {}
        self._zero2_bucket_order: list[int] = []
        self._zero2_pending_buckets: dict[int, tuple[Any, list[tuple[Any, int, int]], Any]] = {}
        self._zero2_reduced_grads: dict[Any, Any] = {}
        self._zero2_local_grad_shard = None
        self._zero2_gradient_hooks: list[Any] = []
        self._zero2_gradients_already_reduced = False
        self._last_zero2_counters: dict[str, Any] = {}
        self._last_fusion_counters: dict[str, Any] = {}

    def _require_backend(self) -> None:
        try:
            import torch  # noqa: F401
            import megatron.core  # noqa: F401
            if self._use_transformer_engine:
                import transformer_engine.pytorch  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeDependencyError(
                "Megatron worker requires megatron-core and transformer-engine[pytorch] installed"
            ) from exc

    def _init_distributed(
        self,
        dp: int,
        pp: int,
        tp: int,
        *,
        pipeline_max_ctas: int | None = None,
        pipeline_cta_policy: int | None = None,
    ) -> None:
        import torch
        from megatron.core import parallel_state

        expected_world_size = dp * pp * tp
        if expected_world_size != self.world_size:
            raise RuntimeConfigError(
                f"Megatron exactness violation: dp*pp*tp={expected_world_size} but WORLD_SIZE={self.world_size}"
            )
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            torch.cuda.set_device(0)
        if not torch.distributed.is_initialized():
            configured_control_backend = os.environ.get(
                "RPEG_MEGATRON_CONTROL_BACKEND",
                "nccl" if cuda_available else "gloo",
            ).strip().lower()
            if configured_control_backend not in {"gloo", "nccl"}:
                raise RuntimeConfigError(
                    "RPEG_MEGATRON_CONTROL_BACKEND must be gloo or nccl, got "
                    f"{configured_control_backend!r}"
                )
            if configured_control_backend == "nccl" and not cuda_available:
                raise RuntimeConfigError(
                    "RPEG_MEGATRON_CONTROL_BACKEND=nccl requires CUDA"
                )
            if self.world_size == 1:
                handle = tempfile.NamedTemporaryFile(prefix="rpeg_megatron_pg_", delete=False)
                handle.close()
                init_method = f"file://{handle.name}"
                backend = configured_control_backend
            else:
                init_method = "env://"
                backend = configured_control_backend
            torch.distributed.init_process_group(
                backend=backend,
                init_method=init_method,
                rank=self.rank,
                world_size=self.world_size,
            )
            self._owns_process_group = True

        if not parallel_state.model_parallel_is_initialized():
            self._model_parallel_group_counters = _initialize_dense_model_parallel(
                parallel_state,
                tp=tp,
                pp=pp,
                group_sink=self._model_parallel_process_groups,
                accelerator_backend="nccl" if cuda_available else None,
                pipeline_max_ctas=(
                    pipeline_max_ctas
                    if pipeline_max_ctas is not None
                    else _pipeline_nccl_max_ctas_from_env()
                    if cuda_available and pp > 1
                    else None
                ),
                pipeline_cta_policy=(
                    pipeline_cta_policy
                    if pipeline_cta_policy is not None
                    else _pipeline_nccl_cta_policy_from_env()
                    if cuda_available and pp > 1
                    else None
                ),
            )
            if cuda_available and pp > 1:
                pp_group = parallel_state.get_pipeline_model_parallel_group()
                pp_backend = str(torch.distributed.get_backend(pp_group)).lower()
                if "nccl" not in pp_backend:
                    raise RuntimeConfigError(
                        "Megatron pipeline data plane must use NCCL for CUDA tensors, "
                        f"got backend={pp_backend!r}"
                    )
                installed = _install_explicit_megatron_inference_pp_group(
                    torch_module=torch,
                )
                self._model_parallel_group_counters.update(
                    {
                        "pipeline_backend_nccl": 1,
                        "inference_explicit_pipeline_group": int(installed),
                    }
                )
            if torch.cuda.is_available():
                from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

                model_parallel_cuda_manual_seed(1234)
            self._model_parallel_initialized = True

    def build_model(self, config: MegatronBuildConfig):
        dp, pp, tp = config.parallelism
        self._released_operational_parameter_specs = ()
        if dp < 1 or pp < 1 or tp < 1:
            raise RuntimeConfigError(f"invalid Megatron parallelism dp={dp}, pp={pp}, tp={tp}")

        self._use_transformer_engine = config.use_transformer_engine
        self._require_backend()
        self._init_distributed(
            dp,
            pp,
            tp,
            pipeline_max_ctas=config.pipeline_nccl_max_ctas,
            pipeline_cta_policy=config.pipeline_nccl_cta_policy,
        )

        import torch
        from megatron.core import parallel_state
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_local_spec,
            get_gpt_layer_with_transformer_engine_spec,
        )
        from megatron.core.models.gpt.gpt_model import GPTModel
        from megatron.core.transformer.transformer_config import TransformerConfig

        arch = architecture_from_hf_path(config.model_path)
        dtype = _torch_dtype(config.dtype)
        transformer_impl = "transformer_engine" if config.use_transformer_engine else "local"
        layer_spec = (
            get_gpt_layer_with_transformer_engine_spec()
            if config.use_transformer_engine
            else get_gpt_layer_local_spec(normalization=arch.normalization)
        )

        transformer_config = TransformerConfig(
            num_layers=arch.num_layers,
            hidden_size=arch.hidden_size,
            num_attention_heads=arch.num_attention_heads,
            num_query_groups=arch.num_query_groups,
            ffn_hidden_size=arch.ffn_hidden_size,
            kv_channels=arch.kv_channels,
            normalization=arch.normalization,
            layernorm_epsilon=arch.layernorm_epsilon,
            gated_linear_unit=arch.gated_linear_unit,
            activation_func=_activation(arch.activation),
            add_bias_linear=arch.add_bias_linear,
            add_qkv_bias=arch.add_qkv_bias,
            attention_dropout=arch.attention_dropout,
            hidden_dropout=arch.hidden_dropout,
            attention_backend=_attention_backend(config.attention_backend),
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            pipeline_dtype=dtype if pp > 1 else None,
            # The CXI provider has a small per-process endpoint budget. Unbatched NCCL P2P lazily
            # creates a separate two-rank communicator for every peer; keep all pipeline traffic on
            # the full PP group.
            batch_p2p_comm=True,
            bf16=dtype is torch.bfloat16,
            fp16=dtype is torch.float16,
            params_dtype=dtype,
            enable_autocast=dtype is not torch.float32,
            autocast_dtype=dtype if dtype is not torch.float32 else None,
            perform_initialization=config.perform_initialization,
            # Checkpoint tensors are loaded immediately after construction. Building an
            # uninitialized large-model shard on CPU first adds avoidable host-memory allocation and
            # CPU-to-HBM traffic on GPU workers.
            use_cpu_initialization=not torch.cuda.is_available(),
            transformer_impl=transformer_impl,
            recompute_granularity="full" if config.activation_checkpointing else None,
            recompute_method="uniform" if config.activation_checkpointing else None,
            recompute_num_layers=1 if config.activation_checkpointing else None,
        )

        self.model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=layer_spec,
            vocab_size=arch.vocab_size,
            max_sequence_length=config.max_sequence_length or arch.max_position_embeddings,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
            position_embedding_type=arch.position_embedding_type,
            rotary_base=arch.rotary_base,
            rotary_percent=arch.rotary_percent,
            rope_scaling=arch.rope_scaling,
            rope_scaling_factor=arch.rope_scaling_factor,
            share_embeddings_and_output_weights=arch.share_embeddings_and_output_weights,
        )
        if torch.cuda.is_available():
            self.model.cuda()
        self.architecture = arch
        self.config = config
        self._build_auxiliary_heads(config.auxiliary_heads, dtype)
        self._sync_tensor_parallel_auxiliary_heads()
        self._clear_static_kv_serving_cache()
        return self.model

    def _build_auxiliary_heads(self, names: tuple[str, ...], dtype) -> None:
        import torch

        assert self.model is not None and self.architecture is not None
        if self.config is not None and self.config.parallelism[1] > 1:
            _dp_rank, pp_rank, _tp_rank = self._rank_to_dp_pp_tp(self.rank)
            if pp_rank != self.config.parallelism[1] - 1:
                self.value_head = None
                self.reward_head = None
                return
        if self.config is not None and self.config.parallelism[1] == 1:
            self.value_head = None
            self.reward_head = None
        device = next(self.model.parameters()).device
        for name in names:
            normalized = str(name).lower()
            # Auxiliary heads are created on CPU before moving to the model device. Megatron's CUDA
            # RNG tracker therefore does not seed their initialization. Use an isolated CPU
            # generator state, so identical run seeds yield identical value/reward heads without
            # perturbing unrelated worker RNG streams.
            head_seed = int(getattr(self.config, "seed", 1234)) + {
                "value": 17,
                "reward": 29,
            }.get(normalized, 0)
            if normalized == "value":
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(head_seed)
                    self.value_head = torch.nn.Linear(self.architecture.hidden_size, 1, bias=True)
                    torch.nn.init.normal_(self.value_head.weight, mean=0.0, std=1.0 / max(self.architecture.hidden_size, 1) ** 0.5)
                    torch.nn.init.zeros_(self.value_head.bias)
                self.value_head.to(device=device, dtype=dtype)
            elif normalized == "reward":
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(head_seed)
                    self.reward_head = torch.nn.Linear(self.architecture.hidden_size, 1, bias=True)
                    torch.nn.init.normal_(self.reward_head.weight, mean=0.0, std=1.0 / max(self.architecture.hidden_size, 1) ** 0.5)
                    torch.nn.init.zeros_(self.reward_head.bias)
                self.reward_head.to(device=device, dtype=dtype)
            else:
                raise RuntimeConfigError(f"unsupported auxiliary head {name!r}")

    def built_model_info(self, dtype: str | None = None) -> BuiltModelInfo:
        if self.model is None or self.architecture is None:
            raise RuntimeConfigError("No Megatron model has been built")
        assert self.config is not None
        dp, pp, tp = self.config.parallelism
        return BuiltModelInfo(
            backend="megatron",
            family=self.architecture.family,
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            dtype=dtype or self.config.dtype,
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            data_parallel_size=dp,
            rope_scaling=self.architecture.rope_scaling,
            rope_scaling_factor=self.architecture.rope_scaling_factor,
        )

    def _require_model(self):
        if self.model is None or self.config is None:
            raise RuntimeConfigError("No Megatron model has been built")
        return self.model, self.config

    def trainable_parameters(self):
        self._require_model()
        params = [p for p in self.model.parameters() if p.requires_grad]
        for head in (self.value_head, self.reward_head):
            if head is not None:
                params.extend(p for p in head.parameters() if p.requires_grad)
        return params

    def _set_training_mode(self) -> None:
        """
        Restore train mode after generation/inference switched the worker to eval.
        """
        self._require_model()
        if self._inference_tensor_caches_dirty:
            # Megatron's RotaryEmbedding.forward is functools.lru_cache'd.
            # GPTInferenceWrapper.run_one_forward_step is inference_mode'd, so the cached RoPE
            # tensor is an inference tensor. If training uses the same sequence length, activation
            # checkpointing tries to save that cached tensor for backward and PyTorch correctly
            # rejects it. Clear only tensor-producing inference caches; retain the static KV pool
            # allocation so the serving fast path remains reusable.
            rotary = getattr(self.model, "rotary_pos_emb", None)
            rotary_forward = getattr(rotary, "forward", None)
            cache_info = getattr(rotary_forward, "cache_info", None)
            cached_rotary_entries = int(cache_info().currsize) if callable(cache_info) else 0
            cache_clear = getattr(rotary_forward, "cache_clear", None)
            if callable(cache_clear):
                cache_clear()
            model_rotary_cache = getattr(self.model, "rotary_pos_emb_cache", None)
            model_rotary_entries = len(model_rotary_cache) if isinstance(model_rotary_cache, dict) else 0
            if isinstance(model_rotary_cache, dict):
                model_rotary_cache.clear()
            self._last_training_transition_counters = {
                "training_inference_cache_transition_count": 1,
                "training_rotary_lru_entries_cleared": cached_rotary_entries,
                "training_model_rotary_entries_cleared": model_rotary_entries,
                "training_static_kv_pool_entries_retained": len(self._static_kv_serving_cache),
            }
            self._inference_tensor_caches_dirty = False
        self.model.train()
        for head in (self.value_head, self.reward_head):
            if head is not None:
                head.train()

    def _average_data_parallel_gradients(self) -> None:
        import torch

        assert self.config is not None
        dp = int(self.config.parallelism[0])
        if self._distributed_optimizer_mode() == "zero2" and dp == 1:
            optimizer = self._ensure_optimizer()
            dtype = _torch_dtype(getattr(self.config, "dtype", "bf16"))
            self._zero2_local_grad_shard = optimizer.set_local_model_gradients(dtype=dtype)
            self._zero2_gradients_already_reduced = False
            self._last_zero2_counters = {
                "zero2_overlap": False,
                "zero2_reduce_calls": 0,
                "zero2_reduce_scatter_calls": 0,
                "zero2_reduced_bytes": 0,
                "zero2_reduce_scatter_padded_bytes": 0,
                "zero2_bucket_bytes": self._zero2_reduction_bucket_limit(),
                "zero2_inflight_bucket_limit": self._zero2_inflight_bucket_limit(),
                "zero2_peak_inflight_buckets": 0,
                "zero2_degenerate_dp1": True,
            }
            return
        if dp <= 1 or not torch.distributed.is_initialized():
            return
        if self._zero2_gradients_already_reduced:
            self._zero2_gradients_already_reduced = False
            return
        from megatron.core import parallel_state

        dp_group = parallel_state.get_data_parallel_group()
        if self._distributed_optimizer_mode() == "zero2":
            del dp_group
            self._reduce_scatter_zero2_gradients()
            return
        for param in self.trainable_parameters():
            if param.grad is not None:
                torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM, group=dp_group)
                param.grad.div_(dp)

    def _run_backward_with_zero2_overlap(self, backward):
        active = self._begin_zero2_gradient_overlap()
        try:
            result = backward()
        except Exception:
            if active:
                self._abort_zero2_gradient_overlap()
            raise
        if active:
            self._finish_zero2_gradient_overlap()
        return result

    def _begin_zero2_gradient_overlap(self) -> bool:
        import torch

        assert self.config is not None
        dp = int(self.config.parallelism[0])
        if self._distributed_optimizer_mode() != "zero2" or dp <= 1 or not torch.distributed.is_initialized():
            return False
        if self._zero2_gradient_hooks:
            raise RuntimeConfigError("ZeRO-2 gradient overlap is already active")
        self._initialize_zero2_reduction(overlap=True)
        for param in self.trainable_parameters():
            if not param.requires_grad:
                continue

            def post_accumulate_hook(parameter):
                grad = parameter.grad
                if grad is None:
                    return
                parameter.grad = None
                self._queue_zero2_parameter_gradient(parameter, grad.detach())

            self._zero2_gradient_hooks.append(param.register_post_accumulate_grad_hook(post_accumulate_hook))
        return True

    def _zero2_reduction_bucket_limit(self) -> int:
        configured = int(getattr(self.config, "zero2_bucket_bytes", 64 * 1024 * 1024) or 0)
        if configured <= 0:
            raise RuntimeConfigError(f"zero2_bucket_bytes must be positive, got {configured}")
        return configured

    def _zero2_inflight_bucket_limit(self) -> int:
        configured = int(getattr(self.config, "zero2_max_inflight_buckets", 2) or 0)
        if configured <= 0:
            raise RuntimeConfigError(
                f"zero2_max_inflight_buckets must be positive, got {configured}"
            )
        return configured

    def _zero2_flat_layout(self):
        from runtime.optimizer_sharding import FlatParameterLayout

        assert self.config is not None
        return FlatParameterLayout(
            self.trainable_parameters(),
            data_parallel_size=int(self.config.parallelism[0]),
            data_parallel_rank=self._data_parallel_rank(),
        )

    def _zero2_parameter_buckets(self) -> tuple[dict[str, Any], ...]:
        assert self.config is not None
        layout = self._zero2_flat_layout()
        params = layout.parameters
        if not params:
            return ()
        dtype = _torch_dtype(getattr(self.config, "dtype", str(params[0].dtype).removeprefix("torch.")))
        device = params[0].device
        element_size = int(params[0].element_size())
        bucket_numel = max(1, self._zero2_reduction_bucket_limit() // element_size)
        buckets = []
        for shard_offset in range(0, layout.shard_numel, bucket_numel):
            chunk_numel = min(bucket_numel, layout.shard_numel - shard_offset)
            entries = []
            expected_parameters = set()
            valid_numel = 0
            for owner in range(layout.data_parallel_size):
                owner_start = owner * layout.shard_numel + shard_offset
                owner_end = owner_start + chunk_numel
                for segment in layout.segments_for_global_range(owner_start, owner_end):
                    input_offset = owner * chunk_numel + segment.global_offset - owner_start
                    entries.append((
                        segment.parameter,
                        int(segment.parameter_offset),
                        int(input_offset),
                        int(segment.numel),
                    ))
                    expected_parameters.add(segment.parameter)
                    valid_numel += int(segment.numel)
            buckets.append({
                "shard_offset": int(shard_offset),
                "chunk_numel": int(chunk_numel),
                "entries": tuple(entries),
                "expected_parameters": frozenset(expected_parameters),
                "valid_numel": int(valid_numel),
                "dtype": dtype,
                "device": device,
            })
        return tuple(buckets)

    def _initialize_zero2_reduction(self, *, overlap: bool) -> None:
        import torch

        self._zero2_bucket_layout = {
            index: bucket for index, bucket in enumerate(self._zero2_parameter_buckets())
        }
        self._zero2_bucket_inputs.clear()
        self._zero2_bucket_seen.clear()
        self._zero2_bucket_order = list(self._zero2_bucket_layout)
        self._zero2_pending_buckets.clear()
        self._zero2_reduced_grads.clear()
        layout = self._zero2_flat_layout()
        dtype = next(iter(self._zero2_bucket_layout.values()))["dtype"] if self._zero2_bucket_layout else None
        device = next(iter(self._zero2_bucket_layout.values()))["device"] if self._zero2_bucket_layout else None
        self._zero2_local_grad_shard = (
            None
            if dtype is None
            else torch.zeros(layout.shard_numel, dtype=dtype, device=device)
        )
        self._zero2_gradients_already_reduced = False
        self._last_zero2_counters = {
            "zero2_overlap": bool(overlap),
            "zero2_reduce_calls": 0,
            "zero2_reduce_scatter_calls": 0,
            "zero2_reduced_bytes": 0,
            "zero2_reduce_scatter_padded_bytes": 0,
            "zero2_bucket_bytes": self._zero2_reduction_bucket_limit(),
            "zero2_inflight_bucket_limit": self._zero2_inflight_bucket_limit(),
            "zero2_peak_inflight_buckets": 0,
        }

    def _queue_zero2_parameter_gradient(self, parameter, grad) -> None:
        import torch

        flat_grad = grad.reshape(-1)
        for bucket_id, bucket in self._zero2_bucket_layout.items():
            parameter_entries = [
                entry for entry in bucket["entries"] if entry[0] is parameter
            ]
            if not parameter_entries:
                continue
            flat_input = self._zero2_bucket_inputs.get(bucket_id)
            if flat_input is None:
                assert self.config is not None
                flat_input = torch.zeros(
                    int(self.config.parallelism[0]) * int(bucket["chunk_numel"]),
                    dtype=bucket["dtype"],
                    device=bucket["device"],
                )
                self._zero2_bucket_inputs[bucket_id] = flat_input
            for _parameter, parameter_offset, input_offset, numel in parameter_entries:
                flat_input.narrow(0, input_offset, numel).add_(
                    flat_grad.narrow(0, parameter_offset, numel).to(dtype=flat_input.dtype)
                )
            seen = self._zero2_bucket_seen.setdefault(bucket_id, set())
            seen.add(parameter)
            if seen >= bucket["expected_parameters"]:
                self._flush_zero2_gradient_bucket(bucket_id)

    def _flush_zero2_gradient_bucket(self, bucket_id: int) -> None:
        import torch
        from megatron.core import parallel_state

        flat_input = self._zero2_bucket_inputs.pop(bucket_id, None)
        if flat_input is None:
            return
        self._complete_zero2_gradient_bucket(bucket_id)
        inflight_limit = self._zero2_inflight_bucket_limit()
        while len(self._zero2_pending_buckets) >= inflight_limit:
            oldest_bucket_id = next(iter(self._zero2_pending_buckets))
            self._complete_zero2_gradient_bucket(oldest_bucket_id)
        bucket = self._zero2_bucket_layout[bucket_id]
        chunk_numel = int(bucket["chunk_numel"])
        output = torch.empty(
            chunk_numel,
            dtype=bucket["dtype"],
            device=bucket["device"],
        )
        work = torch.distributed.reduce_scatter_tensor(
            output,
            flat_input,
            op=torch.distributed.ReduceOp.SUM,
            group=parallel_state.get_data_parallel_group(),
            async_op=True,
        )
        self._zero2_pending_buckets[bucket_id] = (
            output,
            [(None, int(bucket["shard_offset"]), chunk_numel)],
            work,
        )
        self._last_zero2_counters["zero2_peak_inflight_buckets"] = max(
            self._last_zero2_counters["zero2_peak_inflight_buckets"],
            len(self._zero2_pending_buckets),
        )
        valid_bytes = int(bucket["valid_numel"]) * int(output.element_size())
        self._zero2_bucket_seen[bucket_id] = set()
        padded_bytes = int(flat_input.numel() * flat_input.element_size())
        self._last_zero2_counters["zero2_reduce_calls"] += 1
        self._last_zero2_counters["zero2_reduce_scatter_calls"] += 1
        self._last_zero2_counters["zero2_reduced_bytes"] += valid_bytes
        self._last_zero2_counters["zero2_reduce_scatter_padded_bytes"] += padded_bytes

    def _complete_zero2_gradient_bucket(self, bucket_id: int) -> None:
        pending = self._zero2_pending_buckets.pop(bucket_id, None)
        if pending is None:
            return
        flat, specs, work = pending
        work.wait()
        assert self.config is not None
        flat.div_(int(self.config.parallelism[0]))
        if self._zero2_local_grad_shard is None:
            raise RuntimeConfigError("ZeRO-2 local gradient shard was not initialized")
        _unused, shard_offset, numel = specs[0]
        self._zero2_local_grad_shard.narrow(0, shard_offset, numel).add_(flat)

    def _reduce_scatter_zero2_gradients(self) -> None:
        self._initialize_zero2_reduction(overlap=False)
        for parameter in self.trainable_parameters():
            grad = parameter.grad
            if grad is None:
                continue
            parameter.grad = None
            self._queue_zero2_parameter_gradient(parameter, grad.detach())
        self._finish_zero2_gradient_overlap(remove_hooks=False)

    def _finish_zero2_gradient_overlap(self, *, remove_hooks: bool = True) -> None:
        try:
            for parameter in self.trainable_parameters():
                grad = parameter.grad
                if grad is None:
                    continue
                parameter.grad = None
                self._queue_zero2_parameter_gradient(parameter, grad.detach())
            for bucket_id in self._zero2_bucket_order:
                self._flush_zero2_gradient_bucket(bucket_id)
            for bucket_id in self._zero2_bucket_order:
                self._complete_zero2_gradient_bucket(bucket_id)
            self._zero2_gradients_already_reduced = True
        finally:
            if remove_hooks:
                self._remove_zero2_gradient_hooks()
            self._zero2_bucket_layout.clear()
            self._zero2_bucket_inputs.clear()
            self._zero2_bucket_seen.clear()
            self._zero2_bucket_order.clear()
            self._zero2_pending_buckets.clear()
            self._zero2_reduced_grads.clear()

    def _abort_zero2_gradient_overlap(self) -> None:
        self._remove_zero2_gradient_hooks()
        self._zero2_bucket_layout.clear()
        self._zero2_bucket_inputs.clear()
        self._zero2_bucket_seen.clear()
        self._zero2_bucket_order.clear()
        self._zero2_pending_buckets.clear()
        self._zero2_reduced_grads.clear()
        self._zero2_local_grad_shard = None
        self._zero2_gradients_already_reduced = False

    def _remove_zero2_gradient_hooks(self) -> None:
        for handle in self._zero2_gradient_hooks:
            handle.remove()
        self._zero2_gradient_hooks.clear()

    def last_optimizer_counters(self) -> dict[str, Any]:
        counters = dict(self._last_zero2_counters)
        counters.update(self._last_training_transition_counters)
        if self._optimizer is not None:
            counters["optimizer_fused"] = self._optimizer.fused
        return counters

    def last_fusion_counters(self) -> dict[str, Any]:
        return dict(self._last_fusion_counters)

    def _distributed_optimizer_mode(self) -> str:
        assert self.config is not None
        mode = str(getattr(self.config, "distributed_optimizer", "none") or "none").strip().lower().replace("-", "_")
        if mode in {"none", "adamw"}:
            return "none"
        if mode in {"zero1", "zero_1", "zero_stage_1", "stage1"}:
            return "zero1"
        if mode in {"zero2", "zero_2", "zero_stage_2", "stage2"}:
            return "zero2"
        raise RuntimeConfigError(f"unsupported Megatron distributed optimizer mode {mode!r}")

    def _data_parallel_rank(self) -> int:
        assert self.config is not None
        dp = int(self.config.parallelism[0])
        if dp <= 1:
            return 0
        try:
            from megatron.core import parallel_state

            return int(parallel_state.get_data_parallel_rank())
        except Exception:
            pp, tp = int(self.config.parallelism[1]), int(self.config.parallelism[2])
            return int(self.rank) // max(pp * tp, 1)

    def _data_parallel_global_rank(self, dp_rank: int) -> int:
        assert self.config is not None
        _my_dp_rank, pp_rank, tp_rank = self._rank_to_dp_pp_tp(int(self.rank))
        pp, tp = int(self.config.parallelism[1]), int(self.config.parallelism[2])
        return int(dp_rank) * pp * tp + pp_rank * tp + tp_rank

    def _rank_to_dp_pp_tp(self, rank: int) -> tuple[int, int, int]:
        assert self.config is not None
        pp, tp = int(self.config.parallelism[1]), int(self.config.parallelism[2])
        model_parallel = max(pp * tp, 1)
        dp_rank = int(rank) // model_parallel
        rem = int(rank) % model_parallel
        pp_rank = rem // max(tp, 1)
        tp_rank = rem % max(tp, 1)
        return dp_rank, pp_rank, tp_rank

    def _zero1_hidden_gradients(self) -> list[tuple[Any, Any]]:
        from runtime.optimizer_sharding import balanced_parameter_owners

        assert self.config is not None
        dp = int(self.config.parallelism[0])
        if self._distributed_optimizer_mode() not in {"zero1", "zero2"} or dp <= 1:
            return []
        if self._distributed_optimizer_mode() == "zero2":
            return []
        dp_rank = self._data_parallel_rank()
        hidden = []
        params = self.trainable_parameters()
        for owner_dp_rank, param in zip(balanced_parameter_owners(params, dp), params, strict=True):
            if owner_dp_rank != dp_rank and param.grad is not None:
                hidden.append((param, param.grad))
                param.grad = None
        return hidden

    def _restore_hidden_gradients(self, hidden: list[tuple[Any, Any]]) -> None:
        for param, grad in hidden:
            param.grad = grad

    def _sync_zero1_parameters(self) -> None:
        import torch
        from runtime.optimizer_sharding import balanced_parameter_owners

        assert self.config is not None
        dp = int(self.config.parallelism[0])
        mode = self._distributed_optimizer_mode()
        if mode == "zero2" and dp == 1:
            self._ensure_optimizer().copy_flat_master_parameters_to_model()
            self._zero2_local_grad_shard = None
            return
        if mode not in {"zero1", "zero2"} or dp <= 1 or not torch.distributed.is_initialized():
            return
        if mode == "zero2":
            self._sync_zero2_parameters()
            return
        from megatron.core import parallel_state

        dp_group = parallel_state.get_data_parallel_group()
        params = self.trainable_parameters()
        for owner_dp_rank, param in zip(balanced_parameter_owners(params, dp), params, strict=True):
            torch.distributed.broadcast(
                param.data,
                src=self._data_parallel_global_rank(owner_dp_rank),
                group=dp_group,
            )

    def _sync_zero2_parameters(self) -> None:
        import torch
        from megatron.core import parallel_state

        assert self.config is not None
        dp = int(self.config.parallelism[0])
        dp_group = parallel_state.get_data_parallel_group()
        optimizer = self._ensure_optimizer()
        if not optimizer.is_flat_sharded:
            raise RuntimeConfigError("ZeRO-2 parameter synchronization requires a flat-sharded optimizer")
        gather_calls = 0
        gathered_bytes = 0
        padded_bytes = 0
        for bucket in self._zero2_parameter_buckets():
            chunk_numel = int(bucket["chunk_numel"])
            shard_offset = int(bucket["shard_offset"])
            local_shard = optimizer.local_model_parameter_shard(dtype=bucket["dtype"])
            local = local_shard.narrow(0, shard_offset, chunk_numel).contiguous()
            gathered = torch.empty(
                dp * chunk_numel,
                dtype=bucket["dtype"],
                device=bucket["device"],
            )
            if hasattr(torch.distributed, "all_gather_into_tensor"):
                torch.distributed.all_gather_into_tensor(gathered, local, group=dp_group)
            else:
                shards = [torch.empty_like(local) for _ in range(dp)]
                torch.distributed.all_gather(shards, local, group=dp_group)
                gathered.copy_(torch.cat(shards))
            for parameter, parameter_offset, input_offset, numel in bucket["entries"]:
                parameter.data.reshape(-1).narrow(0, parameter_offset, numel).copy_(
                    gathered.narrow(0, input_offset, numel).to(dtype=parameter.dtype)
                )
            gather_calls += 1
            gathered_bytes += int(bucket["valid_numel"]) * int(gathered.element_size())
            padded_bytes += int(gathered.numel() * gathered.element_size())
        self._last_zero2_counters["zero2_all_gather_calls"] = gather_calls
        self._last_zero2_counters["zero2_gathered_bytes"] = gathered_bytes
        self._last_zero2_counters["zero2_all_gather_padded_bytes"] = padded_bytes
        self._zero2_local_grad_shard = None

    def _sync_tensor_parallel_auxiliary_heads(self) -> None:
        import torch

        if self.config is None or not torch.distributed.is_initialized():
            return
        tp = int(self.config.parallelism[2])
        if tp <= 1:
            return
        from megatron.core import parallel_state

        tp_group = parallel_state.get_tensor_model_parallel_group()
        _dp_rank, _pp_rank, tp_rank = self._rank_to_dp_pp_tp(int(self.rank))
        src_rank = int(self.rank) - int(tp_rank)
        for head in (self.value_head, self.reward_head):
            if head is None:
                continue
            for param in head.parameters():
                torch.distributed.broadcast(param.data, src=src_rank, group=tp_group)

    def rpeg_state_dict(self) -> dict[str, Any]:
        self._require_model()
        state = {
            key: value.detach().to(device="cpu", copy=True) if value is not None else None
            for key, value in self.model.state_dict().items()
        }
        if self.value_head is not None:
            state.update({
                f"rpeg_value_head.{key}": value.detach().to(device="cpu", copy=True)
                for key, value in self.value_head.state_dict().items()
            })
        if self.reward_head is not None:
            state.update({
                f"rpeg_reward_head.{key}": value.detach().to(device="cpu", copy=True)
                for key, value in self.reward_head.state_dict().items()
            })
        return state

    def rpeg_device_state_dict(self, *, clone: bool = True) -> dict[str, Any]:
        """
        Return the local model shard without staging it through host memory.

        Published weight versions clone because they must remain immutable. A synchronous in-place
        layout switch may request detached views to avoid duplicating an entire shard before its
        NCCL redistribution.
        """
        self._require_model()
        state = {
            key: (
                value.detach().clone()
                if clone and value is not None
                else value.detach()
                if value is not None
                else None
            )
            for key, value in self.model.state_dict().items()
        }
        if self.value_head is not None:
            state.update({
                f"rpeg_value_head.{key}": value.detach().clone() if clone else value.detach()
                for key, value in self.value_head.state_dict().items()
            })
        if self.reward_head is not None:
            state.update({
                f"rpeg_reward_head.{key}": value.detach().clone() if clone else value.detach()
                for key, value in self.reward_head.state_dict().items()
            })
        return state

    @property
    def operational_weights_materialized(self) -> bool:
        return not self._released_operational_parameter_specs

    def release_operational_weights(self) -> int:
        """
        Free BF16 parameter storage while retaining modules and optimizer masters.
        """
        import torch

        self._require_model()
        if self._released_operational_parameter_specs:
            return 0
        parameters = []
        seen: set[int] = set()
        for module in (self.model, self.value_head, self.reward_head):
            if module is None:
                continue
            for parameter in module.parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                parameters.append(parameter)
        specs = []
        released_bytes = 0
        for parameter in parameters:
            released_bytes += int(parameter.numel()) * int(parameter.element_size())
            specs.append((
                parameter,
                tuple(int(dim) for dim in parameter.shape),
                parameter.dtype,
                parameter.device,
            ))
            parameter.grad = None
            parameter.data = torch.empty(0, dtype=parameter.dtype, device=parameter.device)
        self._released_operational_parameter_specs = tuple(specs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return released_bytes

    def materialize_operational_weights(self) -> int:
        """
        Reallocate parameter storage without rebuilding the Megatron module graph.
        """
        import torch

        self._require_model()
        if not self._released_operational_parameter_specs:
            return 0
        materialized_bytes = 0
        for parameter, shape, dtype, device in self._released_operational_parameter_specs:
            parameter.data = torch.empty(shape, dtype=dtype, device=device)
            materialized_bytes += int(parameter.numel()) * int(parameter.element_size())
        self._released_operational_parameter_specs = ()
        return materialized_bytes

    def adopt_rpeg_state_dict_storage(self, loaded: dict[str, Any], *, strict: bool) -> int:
        """
        Install received tensors as released parameter storage without cloning.

        This path is for inference/generation owners with no optimizer. It avoids simultaneously
        allocating an empty destination shard and a complete received version before load_state_dict
        copies between them. Parameter objects remain stable; only their .data storage is replaced,
        preserving Megatron's module graph.
        """
        if self.operational_weights_materialized:
            self.load_rpeg_state_dict(loaded, strict=strict)
            return 0
        if (
            self._optimizer is not None
            and bool(getattr(self._optimizer, "model_parameters_bound", True))
        ):
            raise RuntimeConfigError(
                "cannot adopt received parameter storage while an optimizer is bound to the parameters"
            )

        released = {
            id(parameter): (shape, dtype, device)
            for parameter, shape, dtype, device in self._released_operational_parameter_specs
        }
        named: list[tuple[str, Any]] = []
        for prefix, module in (
            ("", self.model),
            ("rpeg_value_head.", self.value_head),
            ("rpeg_reward_head.", self.reward_head),
        ):
            if module is None:
                continue
            try:
                parameters = module.named_parameters(remove_duplicate=False)
            except TypeError:
                parameters = module.named_parameters()
            named.extend((f"{prefix}{name}", parameter) for name, parameter in parameters)

        assignments: list[tuple[Any, Any, Any, Any]] = []
        seen: set[int] = set()
        missing: list[str] = []
        for name, parameter in named:
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            spec = released.get(id(parameter))
            if spec is None:
                continue
            shape, dtype, device = spec
            value = loaded.get(name)
            if value is None:
                missing.append(name)
                continue
            if tuple(int(dim) for dim in value.shape) != tuple(shape):
                raise RuntimeConfigError(
                    f"received parameter {name!r} has shape {tuple(value.shape)}, expected {shape}"
                )
            assignments.append((parameter, value, dtype, device))
        if strict and missing:
            raise RuntimeConfigError(
                f"received weight version is missing released parameters: {missing[:20]}"
            )

        adopted_bytes = 0
        for parameter, value, dtype, device in assignments:
            storage = value.detach().to(device=device, dtype=dtype, non_blocking=False)
            parameter.data = storage
            adopted_bytes += int(storage.numel()) * int(storage.element_size())
        # Replace every parameter entry, including tied aliases, with the adopted storage. The
        # strict load below then validates names/buffers without allocating a second device copy of
        # the received shard.
        for name, parameter in named:
            if name in loaded:
                loaded[name] = parameter.data
        self._released_operational_parameter_specs = ()
        self.load_rpeg_state_dict(loaded, strict=strict)
        return adopted_bytes

    def load_rpeg_state_dict(self, loaded: dict[str, Any], *, strict: bool) -> None:
        self._require_model()
        model_state = {
            key: value
            for key, value in loaded.items()
            if not key.startswith("rpeg_value_head.") and not key.startswith("rpeg_reward_head.")
        }
        current = self.model.state_dict()
        for key, value in model_state.items():
            if key in current and current[key] is not None and hasattr(value, "to"):
                model_state[key] = value.to(device=current[key].device, dtype=current[key].dtype)
        self.model.load_state_dict(model_state, strict=strict)
        self._load_auxiliary_head_state(loaded, "rpeg_value_head.", self.value_head, strict=strict)
        self._load_auxiliary_head_state(loaded, "rpeg_reward_head.", self.reward_head, strict=strict)
        self._sync_tensor_parallel_auxiliary_heads()

    def _load_auxiliary_head_state(self, loaded: dict[str, Any], prefix: str, head, *, strict: bool) -> None:
        if head is None:
            return
        head_state = {key[len(prefix):]: value for key, value in loaded.items() if key.startswith(prefix)}
        if not head_state:
            return
        current = head.state_dict()
        for key, value in head_state.items():
            if key in current and hasattr(value, "to"):
                head_state[key] = value.to(device=current[key].device, dtype=current[key].dtype)
        head.load_state_dict(head_state, strict=strict)

    def _load_hf_auxiliary_heads(self, hf_state: dict[str, Any]) -> tuple[str, ...]:
        loaded = []
        value_prefixes = _unique_prefixes(
            *(() if self.config is None else getattr(self.config, "value_head_prefixes", ())),
            *_hf_auxiliary_aliases(
                "rpeg_value_head",
                "value_head",
                "critic_head",
                "score_head",
                "classifier",
                "classification_head",
                "classification_head.out_proj",
                "summary",
                "v_head.summary",
                "v_head",
            ),
        )
        reward_prefixes = _unique_prefixes(
            *(() if self.config is None else getattr(self.config, "reward_head_prefixes", ())),
            *_hf_auxiliary_aliases(
                "rpeg_reward_head",
                "reward_head",
                "reward_model.head",
                "rm_head",
                "score_head",
                "score",
                "classifier",
                "classification_head",
                "classification_head.out_proj",
                "summary",
            ),
        )
        if self.value_head is not None and self._load_hf_auxiliary_head(
            self.value_head,
            hf_state,
            prefixes=value_prefixes,
        ):
            loaded.append("value")
        if self.reward_head is not None and self._load_hf_auxiliary_head(
            self.reward_head,
            hf_state,
            prefixes=reward_prefixes,
        ):
            loaded.append("reward")
        if loaded:
            self._sync_tensor_parallel_auxiliary_heads()
        return tuple(loaded)

    def _load_hf_auxiliary_head(self, head, hf_state: dict[str, Any], *, prefixes: tuple[str, ...]) -> bool:
        current = head.state_dict()
        for prefix in prefixes:
            head_state = {}
            for name, current_value in current.items():
                key = f"{prefix}.{name}"
                if key not in hf_state:
                    continue
                value = hf_state[key]
                value = _coerce_auxiliary_head_tensor(value, current_value, key)
                if tuple(value.shape) != tuple(current_value.shape):
                    raise RuntimeConfigError(
                        f"HF auxiliary head tensor {key} has shape {tuple(value.shape)}, "
                        f"expected {tuple(current_value.shape)}"
                    )
                head_state[name] = value.to(device=current_value.device, dtype=current_value.dtype)
            if head_state:
                merged = dict(current)
                merged.update(head_state)
                head.load_state_dict(merged, strict=True)
                return True
        return False

    def _require_no_pipeline_schedule(self) -> None:
        assert self.config is not None
        if self.config.parallelism[1] != 1:
            raise RuntimeConfigError(
                "Megatron train/inference calls require pp=1 in this runtime path; "
                "pipeline-parallel models must be executed through a pipeline scheduler"
            )

    def _uses_pipeline_schedule(self) -> bool:
        assert self.config is not None
        return self.config.parallelism[1] > 1

    def _autocast_context(self):
        import torch

        assert self.config is not None
        if not torch.cuda.is_available():
            return nullcontext()
        dtype = _torch_dtype(self.config.dtype)
        if dtype is torch.float32:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _inputs(self, input_ids, labels=None, loss_mask=None):
        import torch

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device=device, dtype=torch.long)
        batch, seq = input_ids.shape
        position_ids = torch.arange(seq, device=device, dtype=torch.long).unsqueeze(0).expand(batch, seq)
        attention_mask = torch.triu(
            torch.ones((1, 1, seq, seq), device=device, dtype=torch.bool),
            diagonal=1,
        )
        if labels is None:
            labels = input_ids
        labels = labels.to(device=device, dtype=torch.long)
        if loss_mask is None:
            loss_mask = torch.ones_like(labels, dtype=torch.float32, device=device)
        else:
            loss_mask = loss_mask.to(device=device, dtype=torch.float32)
        return input_ids, position_ids, attention_mask, labels, loss_mask

    @staticmethod
    def _lm_loss(loss_mask, output_tensor, non_loss_data: bool = False):
        import torch

        if non_loss_data:
            return output_tensor.detach() if hasattr(output_tensor, "detach") else output_tensor
        losses = output_tensor.float().view(-1)
        mask = loss_mask.contiguous().view(-1).float()
        loss_sum = torch.sum(losses * mask)
        token_count = mask.sum().to(dtype=torch.int)
        reduced = loss_sum.detach() / torch.clamp(mask.sum(), min=1.0)
        return loss_sum, token_count, {"lm loss": reduced}

    def _forward_step_func(self, data_iterator, model, checkpoint_activations_microbatch=None):
        del checkpoint_activations_microbatch
        input_ids, position_ids, attention_mask, labels, loss_mask = next(data_iterator)
        with self._autocast_context():
            output = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )
        return output, partial(self._lm_loss, loss_mask)

    def _forward_backward(self, input_ids, *, labels=None, loss_mask=None, forward_only: bool = False):
        self._require_model()
        import torch
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

        input_ids, position_ids, attention_mask, labels, loss_mask = self._inputs(input_ids, labels, loss_mask)
        batch, seq = input_ids.shape
        num_microbatches, micro_batch_size = self._pipeline_microbatching(batch)
        forward_backward_func = get_forward_backward_func()
        return forward_backward_func(
            forward_step_func=self._forward_step_func,
            data_iterator=self._iter_loss_microbatches(
                input_ids,
                position_ids,
                attention_mask,
                labels,
                loss_mask,
                num_microbatches=num_microbatches,
            ),
            model=self.model,
            num_microbatches=num_microbatches,
            seq_length=seq,
            micro_batch_size=micro_batch_size,
            forward_only=forward_only,
            collect_non_loss_data=forward_only,
        )

    def inference_step(self, input_ids, *, labels=None, loss_mask=None):
        self._require_model()
        import torch

        self.model.eval()
        with torch.no_grad():
            return self._forward_backward(input_ids, labels=labels, loss_mask=loss_mask, forward_only=True)

    def _direct_inputs(self, input_ids):
        self._require_model()

        self._require_no_pipeline_schedule()
        return self._logit_inputs(input_ids)

    def _logit_inputs(self, input_ids):
        self._require_model()
        import torch

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device=device, dtype=torch.long)
        batch, seq = input_ids.shape
        position_ids = torch.arange(seq, device=device, dtype=torch.long).unsqueeze(0).expand(batch, seq)
        attention_mask = torch.triu(
            torch.ones((1, 1, seq, seq), device=device, dtype=torch.bool),
            diagonal=1,
        )
        return input_ids, position_ids, attention_mask

    @staticmethod
    def _logits_non_loss(output_tensor, non_loss_data: bool = False):
        if non_loss_data:
            return output_tensor.detach() if hasattr(output_tensor, "detach") else output_tensor
        return output_tensor

    def _forward_logits_step_func(self, data_iterator, model, checkpoint_activations_microbatch=None):
        del checkpoint_activations_microbatch
        input_ids, position_ids, attention_mask = next(data_iterator)
        with self._autocast_context():
            output = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=None,
                runtime_gather_output=True,
            )
        return output, self._logits_non_loss

    def _forward_hidden_step_func(self, data_iterator, model, checkpoint_activations_microbatch=None):
        del checkpoint_activations_microbatch
        input_ids, position_ids, attention_mask = next(data_iterator)
        with self._autocast_context():
            preproc_output = model._preprocess(
                input_ids=input_ids,
                position_ids=position_ids,
                decoder_input=None,
                inference_context=None,
                packed_seq_params=None,
                padding_mask=None,
            )
            (
                decoder_input,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                sequence_len_offset,
                padding_mask,
            ) = preproc_output[:6]
            rotary_pos_cos_sin = preproc_output[6] if len(preproc_output) == 7 else None
            output = model.decoder(
                hidden_states=decoder_input,
                attention_mask=attention_mask,
                inference_context=None,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                packed_seq_params=None,
                sequence_len_offset=sequence_len_offset,
                padding_mask=padding_mask,
            )
        return output, self._logits_non_loss

    def _normalize_logits(self, output, input_ids):
        if not hasattr(output, "shape"):
            raise RuntimeConfigError(f"Megatron logits forward returned unsupported type {type(output)!r}")
        if output.ndim != 3:
            raise RuntimeConfigError(f"Megatron logits forward returned shape {tuple(output.shape)}, expected rank 3")
        if output.shape[0] == input_ids.shape[1] and output.shape[1] == input_ids.shape[0]:
            output = output.transpose(0, 1).contiguous()
        if output.shape[0] != input_ids.shape[0] or output.shape[1] != input_ids.shape[1]:
            raise RuntimeConfigError(
                f"Megatron logits shape {tuple(output.shape)} is incompatible with input {tuple(input_ids.shape)}"
            )
        return output

    def _normalize_hidden_states(self, output, input_ids):
        assert self.architecture is not None
        if not hasattr(output, "shape"):
            raise RuntimeConfigError(f"Megatron hidden-state forward returned unsupported type {type(output)!r}")
        if output.ndim != 3:
            raise RuntimeConfigError(
                f"Megatron hidden-state forward returned shape {tuple(output.shape)}, expected rank 3"
            )
        if output.shape[0] == input_ids.shape[1] and output.shape[1] == input_ids.shape[0]:
            output = output.transpose(0, 1).contiguous()
        if (
            output.shape[0] != input_ids.shape[0]
            or output.shape[1] != input_ids.shape[1]
            or output.shape[2] != self.architecture.hidden_size
        ):
            raise RuntimeConfigError(
                f"Megatron hidden-state shape {tuple(output.shape)} is incompatible with input "
                f"{tuple(input_ids.shape)} and hidden size {self.architecture.hidden_size}"
            )
        return output

    def _first_tensor(self, value):
        if hasattr(value, "shape"):
            return value
        if isinstance(value, dict):
            for item in value.values():
                found = self._first_tensor(item)
                if found is not None:
                    return found
        if isinstance(value, (list, tuple)):
            for item in value:
                found = self._first_tensor(item)
                if found is not None:
                    return found
        return None

    def _tensor_list(self, value):
        out = []
        if hasattr(value, "shape"):
            out.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                out.extend(self._tensor_list(item))
        elif isinstance(value, (list, tuple)):
            for item in value:
                out.extend(self._tensor_list(item))
        return out

    def _pipeline_logits(self, input_ids):
        self._require_model()
        import torch
        from megatron.core import parallel_state
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

        input_ids, position_ids, attention_mask = self._logit_inputs(input_ids)
        batch, seq = input_ids.shape
        num_microbatches, micro_batch_size = self._pipeline_microbatching(batch)
        self.model.eval()
        with torch.no_grad():
            output_store = get_forward_backward_func()(
                forward_step_func=self._forward_logits_step_func,
                data_iterator=self._iter_logits_microbatches(
                    input_ids,
                    position_ids,
                    attention_mask,
                    num_microbatches=num_microbatches,
                ),
                model=self.model,
                num_microbatches=num_microbatches,
                seq_length=seq,
                micro_batch_size=micro_batch_size,
                forward_only=True,
                collect_non_loss_data=True,
            )
        outputs = self._tensor_list(output_store)
        dtype = next(self.model.parameters()).dtype
        if parallel_state.is_pipeline_last_stage():
            if len(outputs) != num_microbatches:
                raise RuntimeConfigError(
                    f"pipeline last stage returned {len(outputs)} logit shards, "
                    f"expected {num_microbatches}"
                )
            normalized = [
                self._normalize_logits(output, shard)
                for output, shard in zip(
                    outputs,
                    input_ids.chunk(num_microbatches, dim=0),
                    strict=True,
                )
            ]
            logits = torch.cat(tuple(normalized), dim=0).contiguous()
        else:
            logits = torch.empty(
                (batch, seq, self.architecture.vocab_size),
                device=input_ids.device,
                dtype=dtype,
            )
        self._pipeline_result_to_first_stage(logits)
        return logits

    def _pipeline_token_logps(self, input_ids):
        self._require_model()
        import torch
        from megatron.core import parallel_state
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

        input_ids, position_ids, attention_mask = self._logit_inputs(input_ids)
        batch, seq = input_ids.shape
        num_microbatches, micro_batch_size = self._pipeline_microbatching(batch)
        input_shards = input_ids.chunk(num_microbatches, dim=0)
        self.model.eval()
        with torch.no_grad():
            output_store = get_forward_backward_func()(
                forward_step_func=self._forward_logits_step_func,
                data_iterator=self._iter_logits_microbatches(
                    input_ids,
                    position_ids,
                    attention_mask,
                    num_microbatches=num_microbatches,
                ),
                model=self.model,
                num_microbatches=num_microbatches,
                seq_length=seq,
                micro_batch_size=micro_batch_size,
                forward_only=True,
                collect_non_loss_data=True,
            )
        if parallel_state.is_pipeline_last_stage():
            outputs = self._tensor_list(output_store)
            if len(outputs) != num_microbatches:
                raise RuntimeConfigError(
                    f"pipeline last stage returned {len(outputs)} logit shards, "
                    f"expected {num_microbatches}"
                )
            shards = []
            for output, ids in zip(outputs, input_shards, strict=True):
                logits = self._normalize_logits(output, ids)
                labels = ids[:, 1:]
                logps = torch.nn.functional.log_softmax(logits[:, :-1, :].float(), dim=-1)
                shards.append(logps.gather(-1, labels.unsqueeze(-1)).squeeze(-1))
            token_logps = torch.cat(tuple(shards), dim=0).contiguous()
        else:
            token_logps = torch.empty(
                (batch, max(seq - 1, 0)),
                device=input_ids.device,
                dtype=torch.float32,
            )
        self._pipeline_result_to_first_stage(token_logps)
        return token_logps

    def _pipeline_hidden_states(self, input_ids):
        self._require_model()
        import torch
        from megatron.core import parallel_state
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

        input_ids, position_ids, attention_mask = self._logit_inputs(input_ids)
        batch, seq = input_ids.shape
        num_microbatches, micro_batch_size = self._pipeline_microbatching(batch)
        self.model.eval()
        with torch.no_grad():
            output_store = get_forward_backward_func()(
                forward_step_func=self._forward_hidden_step_func,
                data_iterator=self._iter_logits_microbatches(
                    input_ids,
                    position_ids,
                    attention_mask,
                    num_microbatches=num_microbatches,
                ),
                model=self.model,
                num_microbatches=num_microbatches,
                seq_length=seq,
                micro_batch_size=micro_batch_size,
                forward_only=True,
                collect_non_loss_data=True,
            )
        outputs = self._tensor_list(output_store)
        dtype = next(self.model.parameters()).dtype
        if parallel_state.is_pipeline_last_stage():
            if len(outputs) != num_microbatches:
                raise RuntimeConfigError(
                    f"pipeline last stage returned {len(outputs)} hidden-state shards, "
                    f"expected {num_microbatches}"
                )
            normalized = [
                self._normalize_hidden_states(output, shard)
                for output, shard in zip(
                    outputs,
                    input_ids.chunk(num_microbatches, dim=0),
                    strict=True,
                )
            ]
            hidden_states = torch.cat(tuple(normalized), dim=0).contiguous()
        else:
            hidden_states = torch.empty(
                (batch, seq, self.architecture.hidden_size),
                device=input_ids.device,
                dtype=dtype,
            )
        self._pipeline_result_to_first_stage(hidden_states)
        return hidden_states

    def _pipeline_auxiliary_token_scores(self, input_ids, head, *, shift_for_next_token: bool = True):
        self._require_model()
        import torch
        from megatron.core import parallel_state
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

        if head is None and parallel_state.is_pipeline_last_stage():
            raise RuntimeConfigError("pipeline auxiliary score path requires a head on the last pipeline stage")
        input_ids, position_ids, attention_mask = self._logit_inputs(input_ids)
        batch, seq = input_ids.shape
        num_microbatches, micro_batch_size = self._pipeline_microbatching(batch)
        self.model.eval()
        with torch.no_grad():
            output_store = get_forward_backward_func()(
                forward_step_func=self._forward_hidden_step_func,
                data_iterator=self._iter_logits_microbatches(
                    input_ids,
                    position_ids,
                    attention_mask,
                    num_microbatches=num_microbatches,
                ),
                model=self.model,
                num_microbatches=num_microbatches,
                seq_length=seq,
                micro_batch_size=micro_batch_size,
                forward_only=True,
                collect_non_loss_data=True,
            )
        dtype = next(self.model.parameters()).dtype
        if parallel_state.is_pipeline_last_stage():
            outputs = self._tensor_list(output_store)
            if len(outputs) != num_microbatches:
                raise RuntimeConfigError(
                    f"pipeline last stage returned {len(outputs)} auxiliary hidden-state "
                    f"shards, expected {num_microbatches}"
                )
            scores = []
            head_dtype = next(head.parameters()).dtype
            for output, shard in zip(
                outputs,
                input_ids.chunk(num_microbatches, dim=0),
                strict=True,
            ):
                hidden_states = self._normalize_hidden_states(output, shard)
                selected = hidden_states[:, :-1, :] if shift_for_next_token else hidden_states
                scores.append(head(selected.to(dtype=head_dtype)).squeeze(-1))
            token_scores = torch.cat(tuple(scores), dim=0).contiguous()
        else:
            width = max(seq - 1, 0) if shift_for_next_token else seq
            token_scores = torch.empty((batch, width), device=input_ids.device, dtype=dtype)
        self._pipeline_result_to_first_stage(token_scores)
        return token_scores

    @staticmethod
    def _pipeline_result_to_first_stage(tensor) -> None:
        """
        Move a pipeline result to the stage that publishes runtime artifacts.

        Megatron's forward schedule uses point-to-point NCCL operations on the pipeline group.
        Following that schedule with an all-stage collective on the same communicator can leave
        intermediate stages waiting in NCCL for PP > 2. Only the first stage publishes semantic
        artifacts, so use one matched endpoint transfer instead of broadcasting through unused
        ranks.
        """
        import torch
        from megatron.core import parallel_state

        first_rank = parallel_state.get_pipeline_model_parallel_first_rank()
        last_rank = parallel_state.get_pipeline_model_parallel_last_rank()
        if first_rank == last_rank:
            return
        group = parallel_state.get_pipeline_model_parallel_group()
        if parallel_state.is_pipeline_first_stage():
            operation = torch.distributed.P2POp(
                torch.distributed.irecv,
                tensor,
                last_rank,
                group,
            )
        elif parallel_state.is_pipeline_last_stage():
            operation = torch.distributed.P2POp(
                torch.distributed.isend,
                tensor,
                first_rank,
                group,
            )
        else:
            return
        # Even a one-operation batch is significant: ProcessGroupNCCL then uses the existing PP
        # communicator instead of allocating a hidden endpoint-pair communicator between the first
        # and last stages.
        requests = torch.distributed.batch_isend_irecv([operation])
        for request in requests:
            request.wait()
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)

    def logits(self, input_ids, *, grad: bool = False):
        self._require_model()
        import torch

        if self._uses_pipeline_schedule():
            if grad:
                raise RuntimeConfigError("pipeline logits with gradients must be executed through a pipeline loss schedule")
            return self._pipeline_logits(input_ids)
        input_ids, position_ids, attention_mask = self._direct_inputs(input_ids)
        self.model.train(mode=grad)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            with self._autocast_context():
                output = self.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    labels=None,
                    runtime_gather_output=True,
                )
        return self._normalize_logits(output, input_ids)

    def hidden_states(self, input_ids, *, grad: bool = False):
        self._require_model()
        import torch

        if self._uses_pipeline_schedule():
            if grad:
                raise RuntimeConfigError("pipeline hidden states with gradients must be executed through a pipeline loss schedule")
            return self._pipeline_hidden_states(input_ids)
        input_ids, position_ids, attention_mask = self._direct_inputs(input_ids)
        self.model.train(mode=grad)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            with self._autocast_context():
                preproc_output = self.model._preprocess(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    decoder_input=None,
                    inference_context=None,
                    packed_seq_params=None,
                    padding_mask=None,
                )
                (
                    decoder_input,
                    rotary_pos_emb,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    sequence_len_offset,
                    padding_mask,
                ) = preproc_output[:6]
                rotary_pos_cos_sin = preproc_output[6] if len(preproc_output) == 7 else None
                output = self.model.decoder(
                    hidden_states=decoder_input,
                    attention_mask=attention_mask,
                    inference_context=None,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    rotary_pos_cos_sin=rotary_pos_cos_sin,
                    packed_seq_params=None,
                    sequence_len_offset=sequence_len_offset,
                    padding_mask=padding_mask,
                )
        return self._normalize_hidden_states(output, input_ids)

    def token_logps(self, input_ids, *, grad: bool = False):
        import torch

        if self._uses_pipeline_schedule():
            if grad:
                raise RuntimeConfigError(
                    "pipeline token log-probabilities with gradients must be executed "
                    "through a pipeline loss schedule"
                )
            return self._pipeline_token_logps(input_ids)
        logits = self.logits(input_ids, grad=grad)
        labels = input_ids.to(device=logits.device, dtype=torch.long)[:, 1:]
        logps = torch.nn.functional.log_softmax(logits[:, :-1, :].float(), dim=-1)
        return logps.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    def response_token_logps(self, input_ids, prompt_lens, max_new_tokens: int, *, grad: bool = False):
        token_logps = self.token_logps(input_ids, grad=grad)
        return self._select_response_values(token_logps, prompt_lens, max_new_tokens)

    def token_values(self, input_ids, *, grad: bool = False):
        if self._uses_pipeline_schedule():
            if grad:
                raise RuntimeConfigError("pipeline token values with gradients must be executed through a pipeline loss schedule")
            values = self._pipeline_auxiliary_token_scores(input_ids, self.value_head)
            expected = input_ids.shape[1] - 1
            if values.shape[1] != expected:
                raise RuntimeConfigError(f"value head returned width {values.shape[1]}, expected {expected}")
            return values
        if self.value_head is None:
            raise RuntimeConfigError("critic_token_values requires a value auxiliary head")
        hidden_states = self.hidden_states(input_ids, grad=grad)
        head_dtype = next(self.value_head.parameters()).dtype
        values = self.value_head(hidden_states[:, :-1, :].to(dtype=head_dtype)).squeeze(-1)
        expected = input_ids.shape[1] - 1
        if values.shape[1] != expected:
            raise RuntimeConfigError(f"value head returned width {values.shape[1]}, expected {expected}")
        return values

    def response_token_values(self, input_ids, prompt_lens, max_new_tokens: int, *, grad: bool = False):
        token_values = self.token_values(input_ids, grad=grad)
        return self._select_response_values(token_values, prompt_lens, max_new_tokens)

    def reward_scores(self, input_ids, prompt_lens, max_new_tokens: int, *, response_mask=None):
        import torch

        if self._uses_pipeline_schedule():
            token_scores = self._pipeline_auxiliary_token_scores(
                input_ids,
                self.reward_head,
                shift_for_next_token=False,
            )
        else:
            if self.reward_head is None:
                raise RuntimeConfigError("reward_score requires a reward auxiliary head")
            hidden_states = self.hidden_states(input_ids, grad=False)
            head_dtype = next(self.reward_head.parameters()).dtype
            token_scores = self.reward_head(hidden_states.to(dtype=head_dtype)).squeeze(-1)
        response_scores, selected_mask = self._select_response_hidden_states(
            token_scores,
            prompt_lens,
            max_new_tokens,
        )
        mask = selected_mask if response_mask is None else response_mask.to(
            device=response_scores.device,
            dtype=torch.float32,
        )
        lengths = mask.sum(dim=1).to(dtype=torch.long).clamp(min=1)
        return response_scores.gather(1, (lengths - 1).view(-1, 1)).squeeze(1)

    def _response_logps_from_logits(self, logits, input_ids, prompt_lens, max_new_tokens: int):
        import torch

        labels = input_ids.to(device=logits.device, dtype=torch.long)[:, 1:]
        logps = torch.nn.functional.log_softmax(logits[:, :-1, :].float(), dim=-1)
        token_logps = logps.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        values, _ = self._select_response_values(token_logps, prompt_lens, max_new_tokens)
        return values

    def _response_entropies_from_logits(self, logits, prompt_lens, max_new_tokens: int):
        import torch

        log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :].float(), dim=-1)
        token_entropies = -(log_probs.exp() * log_probs).sum(dim=-1)
        values, _ = self._select_response_values(token_entropies, prompt_lens, max_new_tokens)
        return values

    def _response_values_from_logits(self, logits, input_ids, prompt_lens, max_new_tokens: int):
        del logits, input_ids, prompt_lens, max_new_tokens
        raise RuntimeConfigError(
            "value heads consume decoder hidden states, not logits; "
            "pipeline-parallel value training needs a hidden-state pipeline schedule"
        )

    def _logits_from_hidden_states(self, hidden_states):
        output_weight = None
        if getattr(self.model, "share_embeddings_and_output_weights", False):
            output_weight = self.model.shared_embedding_or_output_weight()
        if output_weight is None:
            output_layer = self.model.output_layer
            output_weight = getattr(output_layer, "weight", None)
        output_dtype = (
            output_weight.dtype
            if output_weight is not None
            else next(self.model.output_layer.parameters()).dtype
        )
        logits, _ = self.model.output_layer(
            hidden_states.to(dtype=output_dtype).transpose(0, 1).contiguous(),
            weight=output_weight if getattr(self.model, "share_embeddings_and_output_weights", False) else None,
            runtime_gather_output=True,
        )
        logits = self.model._scale_logits(logits)
        return logits.transpose(0, 1).contiguous()

    def _response_values_from_hidden_states(self, hidden_states, prompt_lens, max_new_tokens: int):
        if self.value_head is None:
            raise RuntimeConfigError("critic loss requires a value auxiliary head")
        head_dtype = next(self.value_head.parameters()).dtype
        token_values = self.value_head(hidden_states[:, :-1, :].to(dtype=head_dtype)).squeeze(-1)
        values, _ = self._select_response_values(token_values, prompt_lens, max_new_tokens)
        return values

    def _select_response_values(self, token_logps, prompt_lens, max_new_tokens: int):
        import torch

        selected = []
        masks = []
        width = token_logps.shape[1]
        for row, prompt_len in enumerate(prompt_lens):
            start = max(0, int(prompt_len) - 1)
            end = min(width, start + max_new_tokens)
            vals = token_logps[row, start:end]
            if vals.numel() < max_new_tokens:
                vals = torch.nn.functional.pad(vals, (0, max_new_tokens - vals.numel()))
            mask = torch.zeros(max_new_tokens, device=token_logps.device, dtype=torch.float32)
            mask[: max(0, end - start)] = 1.0
            selected.append(vals[:max_new_tokens])
            masks.append(mask)
        return torch.stack(selected, dim=0), torch.stack(masks, dim=0)

    def _select_response_hidden_states(self, token_values, prompt_lens, max_new_tokens: int):
        import torch

        selected = []
        masks = []
        width = token_values.shape[1]
        for row, prompt_len in enumerate(prompt_lens):
            start = max(0, int(prompt_len))
            end = min(width, start + max_new_tokens)
            values = token_values[row, start:end]
            if values.numel() < max_new_tokens:
                values = torch.nn.functional.pad(values, (0, max_new_tokens - values.numel()))
            mask = torch.zeros(max_new_tokens, device=token_values.device, dtype=torch.float32)
            mask[: max(0, end - start)] = 1.0
            selected.append(values[:max_new_tokens])
            masks.append(mask)
        return torch.stack(selected, dim=0), torch.stack(masks, dim=0)

    def generate_with_logps(
        self,
        input_ids,
        *,
        max_new_tokens: int,
        prompt_lens=None,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: int | None = None,
        generation_backend: str = "full_sequence",
    ):
        import torch

        self._require_model()
        if temperature <= 0.0:
            raise RuntimeConfigError("generation temperature must be > 0")
        if not 0.0 < top_p <= 1.0:
            raise RuntimeConfigError("generation top_p must be in (0, 1]")
        if top_k < 0:
            raise RuntimeConfigError("generation top_k must be >= 0")
        input_ids = input_ids.to(device=next(self.model.parameters()).device, dtype=torch.long)
        if prompt_lens is None:
            prompt_lens = [int(input_ids.shape[1])] * int(input_ids.shape[0])
        else:
            prompt_lens = [int(x) for x in prompt_lens]
        generator = None
        if seed is not None:
            generator = torch.Generator(device=input_ids.device)
            generator.manual_seed(int(seed))
        backend = _normalize_generation_backend(generation_backend)
        if backend == "static_kv":
            generated, counters, generated_old_logps = self._generate_tokens_static_kv(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                generator=generator,
            )
        else:
            generated, counters, generated_old_logps = self._generate_tokens_full_sequence(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                generator=generator,
            )
        if generated_old_logps is None:
            old_logps, response_mask = self.response_token_logps(
                generated,
                prompt_lens,
                max_new_tokens,
                grad=False,
            )
            counters["old_logps_from_generation"] = False
            counters["old_logp_forward_steps"] = 1
        else:
            old_logps = generated_old_logps
            response_mask = self._generated_response_mask(generated, max_new_tokens)
            counters["old_logps_from_generation"] = True
            counters["old_logp_forward_steps"] = 0
        self._last_generation_counters = counters
        return generated.detach(), old_logps.detach(), response_mask.detach(), prompt_lens

    def generate_quantile_phase(
        self,
        state_key: str,
        *,
        input_ids=None,
        prompt_lens=None,
        target_tokens: int,
        max_new_tokens: int,
        final_phase: bool,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: int | None = None,
        generation_backend: str = "static_kv",
        eos_token_ids=(),
        expected_completed_sequences: int | None = None,
        original_index_offset: int = 0,
    ):
        """
        Continue one quantile phase and emit only sequences completed in it.
        """
        import torch

        self._require_model()
        target_tokens = int(target_tokens)
        max_new_tokens = int(max_new_tokens)
        if not 0 < target_tokens <= max_new_tokens:
            raise RuntimeConfigError(
                f"generation phase target {target_tokens} must be in [1, max_new_tokens={max_new_tokens}]"
            )
        state = self._phased_generation_states.get(str(state_key))
        if state is None:
            if input_ids is None:
                raise RuntimeConfigError(f"first generation phase {state_key!r} requires input_ids")
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device=device, dtype=torch.long)
            prompt_lens = [int(x) for x in (prompt_lens or [input_ids.shape[1]] * input_ids.shape[0])]
            generator = None
            if seed is not None:
                generator = torch.Generator(device=input_ids.device)
                generator.manual_seed(int(seed))
            backend = _normalize_generation_backend(generation_backend)
            context = wrapper = None
            cache_hit = False
            cache_capacity = None
            if backend == "static_kv":
                context, wrapper, cache_hit, cache_capacity = self._static_kv_serving_context(
                    max_batch_size=int(input_ids.shape[0]),
                    max_sequence_length=int(input_ids.shape[1]) + max_new_tokens,
                )
                wrapper.prep_model_for_inference()
            if isinstance(eos_token_ids, int):
                eos_token_ids = (eos_token_ids,)
            state = _PhasedGenerationState(
                generated=input_ids,
                old_logps=torch.empty((input_ids.shape[0], 0), device=device, dtype=torch.float32),
                prompt_lens=prompt_lens,
                original_indices=(
                    torch.arange(input_ids.shape[0], device=device, dtype=torch.long)
                    + int(original_index_offset)
                ),
                max_new_tokens=max_new_tokens,
                tokens_generated=0,
                backend=backend,
                do_sample=bool(do_sample),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                eos_token_ids=tuple(int(x) for x in eos_token_ids if x is not None),
                generator=generator,
                static_context=context,
                static_wrapper=wrapper,
                static_cache_hit=cache_hit,
                static_cache_capacity=cache_capacity,
                pending_completions=[],
            )
            self._phased_generation_states[str(state_key)] = state
        else:
            if input_ids is not None:
                raise RuntimeConfigError(f"continuation generation phase {state_key!r} must not replace input_ids")
            if max_new_tokens != state.max_new_tokens:
                raise RuntimeConfigError("generation continuation changed max_new_tokens")
            if target_tokens < state.tokens_generated:
                raise RuntimeConfigError("generation continuation target moved backwards")

        active_at_start = int(state.generated.shape[0])
        emitted: list[tuple[int, Any, Any, int]] = []
        compacted = 0
        prefill_steps = 0
        decode_steps = 0
        was_training = bool(getattr(self.model, "training", False))
        try:
            while state.tokens_generated < target_tokens and state.generated.shape[0] > 0:
                if state.backend == "static_kv":
                    if state.tokens_generated == 0:
                        inference_input = state.static_wrapper.prep_inference_input(state.generated)
                        model_input = state.static_wrapper.get_batch_for_context_window(
                            inference_input,
                            0,
                            int(state.generated.shape[1]),
                        )
                        prefill_steps += 1
                    else:
                        previous_token = state.generated[:, -1:]
                        position_ids = torch.full_like(previous_token, int(state.static_context.sequence_len_offset))
                        model_input = {
                            "tokens": previous_token,
                            "position_ids": position_ids,
                            "attention_mask": None,
                        }
                        decode_steps += 1
                    with self._autocast_context():
                        logits = self._static_kv_forward_logits(
                            state.static_wrapper,
                            model_input,
                            expected_input_ids=model_input["tokens"],
                        )
                    if state.tokens_generated == 0:
                        state.static_context.enable_decode_mode()
                else:
                    logits = self.logits(state.generated, grad=False)[:, -1:, :]
                    decode_steps += 1
                step_logits = logits[:, -1, :].float()
                next_token = self._select_next_generation_token(
                    step_logits,
                    do_sample=state.do_sample,
                    temperature=state.temperature,
                    top_p=state.top_p,
                    top_k=state.top_k,
                    generator=state.generator,
                )
                selected_logps = self._selected_token_logps(step_logits, next_token).view(-1, 1)
                state.generated = torch.cat((state.generated, next_token), dim=1)
                state.old_logps = torch.cat((state.old_logps, selected_logps), dim=1)
                state.tokens_generated += 1

                finished = torch.zeros(next_token.shape[0], device=next_token.device, dtype=torch.bool)
                for eos_token_id in state.eos_token_ids:
                    finished |= next_token[:, 0] == eos_token_id
                if torch.any(finished):
                    finished_positions = torch.nonzero(finished, as_tuple=False).flatten()
                    for position in finished_positions.tolist():
                        emitted.append((
                            int(state.original_indices[position]),
                            state.generated[position].detach().clone(),
                            state.old_logps[position].detach().clone(),
                            state.prompt_lens[position],
                        ))
                    keep = torch.nonzero(~finished, as_tuple=False).flatten()
                    if state.backend == "static_kv" and keep.numel() > 0:
                        self._compact_static_kv_batch(state.static_context, keep)
                    compacted += int(finished.sum().item())
                    state.generated = state.generated.index_select(0, keep)
                    state.old_logps = state.old_logps.index_select(0, keep)
                    state.original_indices = state.original_indices.index_select(0, keep)
                    state.prompt_lens = [state.prompt_lens[int(i)] for i in keep.tolist()]
        finally:
            self.model.train(mode=was_training)

        if state.pending_completions is None:
            state.pending_completions = []
        state.pending_completions.extend(emitted)
        forced_completions = 0
        if final_phase:
            emitted = list(state.pending_completions)
            state.pending_completions.clear()
            for position in range(int(state.generated.shape[0])):
                emitted.append((
                    int(state.original_indices[position]),
                    state.generated[position].detach().clone(),
                    state.old_logps[position].detach().clone(),
                    state.prompt_lens[position],
                ))
            state.generated = state.generated[:0]
            state.old_logps = state.old_logps[:0]
            del self._phased_generation_states[str(state_key)]
        else:
            # Quantile breakpoints describe scheduling capacity, never sampling semantics. Emit
            # every naturally completed row at this boundary; active rows continue to the next phase
            # without truncation.
            emitted = list(state.pending_completions)
            state.pending_completions.clear()
        emitted.sort(key=lambda item: item[0])
        completion_ids, old_logps, response_mask, completed_prompt_lens, original_indices = (
            self._pack_completed_generation_phase(emitted)
        )
        counters = {
            "generation_backend": state.backend,
            "kv_cache_enabled": state.backend == "static_kv",
            "prefill_forward_steps": prefill_steps,
            "decode_forward_steps": decode_steps,
            "generation_forward_steps": prefill_steps + decode_steps,
            "kv_cache_layers": len(getattr(state.static_context, "key_value_memory_dict", {}) or {}),
            "kv_cache_reused": state.static_cache_hit,
            "kv_cache_pool_entries": len(self._static_kv_serving_cache),
            "kv_cache_pool_hits": self._static_kv_serving_hits,
            "kv_cache_pool_misses": self._static_kv_serving_misses,
            "kv_cache_pool_evictions": self._static_kv_serving_evictions,
            "kv_cache_max_batch_size": 0 if state.static_cache_capacity is None else state.static_cache_capacity[0],
            "kv_cache_max_sequence_length": 0 if state.static_cache_capacity is None else state.static_cache_capacity[1],
            "old_logps_from_generation": True,
            "old_logp_forward_steps": 0,
            "phase_active_at_start": active_at_start,
            "phase_active_at_end": int(state.generated.shape[0]),
            "phase_completed_sequences": len(emitted),
            "phase_compacted_sequences": compacted,
            "phase_forced_completions": forced_completions,
            "phase_deferred_completions": len(state.pending_completions),
            "phase_target_tokens": target_tokens,
            "phase_continuation": state.tokens_generated > (prefill_steps + decode_steps),
        }
        self._last_generation_counters = counters
        return completion_ids, old_logps, response_mask, completed_prompt_lens, original_indices, counters

    def _compact_static_kv_batch(self, context, keep_indices) -> None:
        """
        Move surviving sequence KV rows to the front of a static cache in place.
        """
        import torch

        # Megatron allocates static KV storage under inference mode, so PyTorch requires later
        # in-place cache compaction to enter the same mode.
        with torch.inference_mode():
            for layer_number, (key_memory, value_memory) in context.key_value_memory_dict.items():
                count = int(keep_indices.numel())
                key_memory[:, :count].copy_(key_memory.index_select(1, keep_indices).clone())
                value_memory[:, :count].copy_(value_memory.index_select(1, keep_indices).clone())
                context.key_value_memory_dict[layer_number] = (key_memory, value_memory)

    def _pack_completed_generation_phase(self, emitted):
        import torch

        if not emitted:
            device = next(self.model.parameters()).device
            empty_long = torch.empty((0, 0), device=device, dtype=torch.long)
            empty_float = torch.empty((0, 0), device=device, dtype=torch.float32)
            return empty_long, empty_float, empty_float, [], []
        max_sequence = max(int(item[1].numel()) for item in emitted)
        max_response = max(int(item[2].numel()) for item in emitted)
        completion_ids = torch.stack([
            torch.nn.functional.pad(item[1], (0, max_sequence - int(item[1].numel())))
            for item in emitted
        ])
        old_logps = torch.stack([
            torch.nn.functional.pad(item[2], (0, max_response - int(item[2].numel())))
            for item in emitted
        ])
        response_mask = torch.stack([
            torch.nn.functional.pad(torch.ones_like(item[2], dtype=torch.float32), (0, max_response - int(item[2].numel())))
            for item in emitted
        ])
        return (
            completion_ids,
            old_logps,
            response_mask,
            [int(item[3]) for item in emitted],
            [int(item[0]) for item in emitted],
        )

    def _generate_tokens_full_sequence(
        self,
        input_ids,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        generator,
    ):
        import torch

        generated = input_ids
        old_logps = []
        for _ in range(max_new_tokens):
            logits = self.logits(generated, grad=False)
            step_logits = logits[:, -1, :].float()
            next_token = self._select_next_generation_token(
                step_logits,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                generator=generator,
            )
            old_logps.append(self._selected_token_logps(step_logits, next_token))
            generated = torch.cat([generated, next_token], dim=1)
        return generated, {
            "generation_backend": "full_sequence",
            "kv_cache_enabled": False,
            "prefill_forward_steps": 0,
            "decode_forward_steps": int(max_new_tokens),
            "generation_forward_steps": int(max_new_tokens),
            "kv_cache_layers": 0,
        }, self._stack_generated_old_logps(input_ids, old_logps, max_new_tokens)

    def _generate_tokens_static_kv(
        self,
        input_ids,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        generator,
    ):
        import torch

        if max_new_tokens <= 0:
            return input_ids, {
                "generation_backend": "static_kv",
                "kv_cache_enabled": True,
                "prefill_forward_steps": 0,
                "decode_forward_steps": 0,
                "generation_forward_steps": 0,
                "kv_cache_layers": 0,
                "kv_cache_reused": False,
                "kv_cache_pool_entries": len(self._static_kv_serving_cache),
                "kv_cache_pool_hits": self._static_kv_serving_hits,
                "kv_cache_pool_misses": self._static_kv_serving_misses,
                "kv_cache_pool_evictions": self._static_kv_serving_evictions,
                "kv_cache_max_batch_size": 0,
                "kv_cache_max_sequence_length": 0,
            }, self._stack_generated_old_logps(input_ids, [], max_new_tokens)
        context, wrapper, cache_hit, cache_capacity = self._static_kv_serving_context(
            max_batch_size=int(input_ids.shape[0]),
            max_sequence_length=int(input_ids.shape[1]) + int(max_new_tokens),
        )
        was_training = bool(getattr(self.model, "training", False))
        old_logps = []
        try:
            wrapper.prep_model_for_inference()
            inference_input = wrapper.prep_inference_input(input_ids)
            prefill_input = wrapper.get_batch_for_context_window(
                inference_input,
                0,
                int(input_ids.shape[1]),
            )
            with self._autocast_context():
                logits = self._static_kv_forward_logits(
                    wrapper,
                    prefill_input,
                    expected_input_ids=prefill_input["tokens"],
                )
            step_logits = logits[:, -1, :].float()
            next_token = self._select_next_generation_token(
                step_logits,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                generator=generator,
            )
            old_logps.append(self._selected_token_logps(step_logits, next_token))
            generated = torch.cat(
                [
                    input_ids,
                    next_token,
                ],
                dim=1,
            )
            context.enable_decode_mode()
            for _ in range(1, max_new_tokens):
                previous_token = generated[:, -1:]
                position_ids = torch.full_like(previous_token, int(context.sequence_len_offset))
                with self._autocast_context():
                    logits = self._static_kv_forward_logits(
                        wrapper,
                        {
                            "tokens": previous_token,
                            "position_ids": position_ids,
                            "attention_mask": None,
                        },
                        expected_input_ids=previous_token,
                    )
                step_logits = logits[:, -1, :].float()
                next_token = self._select_next_generation_token(
                    step_logits,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    generator=generator,
                )
                old_logps.append(self._selected_token_logps(step_logits, next_token))
                generated = torch.cat(
                    [
                        generated,
                        next_token,
                    ],
                    dim=1,
                )
        finally:
            self.model.train(mode=was_training)
        return generated, {
            "generation_backend": "static_kv",
            "kv_cache_enabled": True,
            "prefill_forward_steps": 1,
            "decode_forward_steps": max(0, int(max_new_tokens) - 1),
            "generation_forward_steps": int(max_new_tokens),
            "kv_cache_layers": len(getattr(context, "key_value_memory_dict", {}) or {}),
            "kv_cache_reused": cache_hit,
            "kv_cache_pool_entries": len(self._static_kv_serving_cache),
            "kv_cache_pool_hits": self._static_kv_serving_hits,
            "kv_cache_pool_misses": self._static_kv_serving_misses,
            "kv_cache_pool_evictions": self._static_kv_serving_evictions,
            "kv_cache_max_batch_size": int(cache_capacity[0]),
            "kv_cache_max_sequence_length": int(cache_capacity[1]),
        }, self._stack_generated_old_logps(input_ids, old_logps, max_new_tokens)

    def _static_kv_forward_logits(self, wrapper, inference_input: dict[str, Any], *, expected_input_ids):
        import torch

        # The wrapper enters torch.inference_mode and Megatron may retain RoPE products beyond this
        # call. Record the transition so the next shared model training step invalidates those
        # tensor caches exactly once.
        self._inference_tensor_caches_dirty = True
        logits = wrapper.run_one_forward_step(inference_input)
        if not self._uses_pipeline_schedule():
            return self._normalize_generation_step_logits(logits, expected_input_ids)

        from megatron.core import parallel_state

        batch, seq = expected_input_ids.shape
        dtype = next(self.model.parameters()).dtype
        if parallel_state.is_pipeline_last_stage():
            if logits is None:
                raise RuntimeConfigError("pipeline static-KV last stage did not return logits")
            normalized = self._normalize_generation_step_logits(logits, expected_input_ids)
        else:
            assert self.architecture is not None
            normalized = torch.empty(
                (int(batch), 1, self.architecture.vocab_size),
                device=expected_input_ids.device,
                dtype=dtype,
            )
        self._pipeline_result_to_all_stages(normalized)
        return normalized

    @staticmethod
    def _pipeline_result_to_all_stages(tensor) -> None:
        """
        Distribute a generation result without an all-stage collective.

        Megatron's pipeline forward uses ordered point-to-point NCCL traffic. Following it with a
        broadcast on the same communicator can deadlock at PP>2. Generation needs the logits on
        every stage, so each rank advances identical token/cache state; use one grouped P2P round.
        Individual isend/irecv calls make ProcessGroupNCCL lazily create a distinct two-rank
        communicator per peer, which exhausts the CXI endpoint limit at deep pipeline parallelism.
        batch_isend_irecv keeps the traffic on the already-initialized pipeline communicator.
        """
        import torch
        from megatron.core import parallel_state

        group = parallel_state.get_pipeline_model_parallel_group()
        last_rank = parallel_state.get_pipeline_model_parallel_last_rank()
        try:
            ranks = tuple(int(rank) for rank in torch.distributed.get_process_group_ranks(group))
        except (AttributeError, KeyError, RuntimeError):
            first_rank = parallel_state.get_pipeline_model_parallel_first_rank()
            world_size = parallel_state.get_pipeline_model_parallel_world_size()
            ranks = tuple(first_rank + index for index in range(world_size))
        if len(ranks) <= 1:
            return
        if parallel_state.is_pipeline_last_stage():
            operations = [
                torch.distributed.P2POp(
                    torch.distributed.isend,
                    tensor,
                    rank,
                    group,
                )
                for rank in ranks
                if rank != last_rank
            ]
        else:
            operations = [
                torch.distributed.P2POp(
                    torch.distributed.irecv,
                    tensor,
                    last_rank,
                    group,
                )
            ]
        for request in torch.distributed.batch_isend_irecv(operations):
            request.wait()
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)

    @staticmethod
    def _post_grouped_pipeline_send(tensors, *, peer: int, group):
        """
        Post pipeline sends without introducing a blocking device edge.

        A fused bidirectional instruction stream may require the colocated model to advance before
        the matching pipeline receive is posted. A blocking send therefore adds a dependency that is
        absent from the validated fused DAG and can close a cycle. Keep detached aliases live until
        the caller waits so the allocator cannot recycle send storage.
        """
        import torch

        values = tensors if isinstance(tensors, list) else [tensors]
        retained = tuple(value.detach() for value in values if value is not None)
        if not retained:
            return (), ()
        operations = [
            torch.distributed.P2POp(
                torch.distributed.isend,
                value,
                peer,
                group,
            )
            for value in retained
        ]
        requests = tuple(torch.distributed.batch_isend_irecv(operations))
        return requests, retained

    @staticmethod
    def _post_grouped_pipeline_recv(
        tensor_shapes,
        *,
        peer: int,
        group,
        dtype,
        skip: bool,
    ):
        """
        Post pipeline receives asynchronously after a device-lane wait.

        Megatron's public recv_forward/recv_backward helpers wait for completion internally.
        Materialize the same receive buffers and return live requests, so the fused caller can
        separate receive launch from completion. The caller first advances its local CUDA-IPC
        predecessor; this avoids letting a persistent NCCL receive overtake a colocated model's TP
        collective on the shared GPU.
        """
        import torch

        single_shape = isinstance(tensor_shapes, torch.Size) or (
            isinstance(tensor_shapes, (list, tuple))
            and bool(tensor_shapes)
            and all(isinstance(dimension, int) for dimension in tensor_shapes)
        )
        shapes = [tensor_shapes] if single_shape else list(tensor_shapes)
        if skip:
            values = [None for _shape in shapes]
            received = values[0] if single_shape else values
            return received, ((), ())
        retained = tuple(
            torch.empty(
                shape,
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=dtype,
            )
            for shape in shapes
        )
        operations = [
            torch.distributed.P2POp(
                torch.distributed.irecv,
                value,
                peer,
                group,
            )
            for value in retained
        ]
        requests = tuple(torch.distributed.batch_isend_irecv(operations))
        received = retained[0] if single_shape else list(retained)
        return received, (requests, retained)

    @staticmethod
    def _wait_grouped_pipeline_sends(pending) -> int:
        requests, retained = pending
        for request in requests:
            request.wait()
        # retained deliberately stays referenced through every wait.
        del retained
        return len(requests)

    @staticmethod
    def _wait_grouped_pipeline_recvs(pending) -> int:
        requests, retained = pending
        for request in requests:
            request.wait()
        del retained
        return len(requests)

    @staticmethod
    def _synchronize_fused_cuda_ipc_event(event) -> None:
        """
        Make an offline fused-device dependency a host launch barrier.
        """
        event.synchronize()

    @staticmethod
    def _append_fused_progress(path: Path, payload: dict[str, Any]) -> None:
        """
        Append an opt-in fused launch marker for scale-only diagnostics.
        """
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _normalize_generation_step_logits(self, output, input_ids):
        if not hasattr(output, "shape"):
            raise RuntimeConfigError(f"Megatron generation forward returned unsupported type {type(output)!r}")
        if output.ndim != 3:
            raise RuntimeConfigError(
                f"Megatron generation forward returned shape {tuple(output.shape)}, expected rank 3"
            )
        if output.shape[0] == input_ids.shape[1] and output.shape[1] == input_ids.shape[0]:
            output = output.transpose(0, 1).contiguous()
        if output.shape[0] != input_ids.shape[0]:
            raise RuntimeConfigError(
                f"Megatron generation logits shape {tuple(output.shape)} is incompatible with input "
                f"{tuple(input_ids.shape)}"
            )
        if output.shape[1] == input_ids.shape[1]:
            return output[:, -1:, :].contiguous()
        if output.shape[1] == 1:
            return output.contiguous()
        raise RuntimeConfigError(
            f"Megatron generation logits shape {tuple(output.shape)} is incompatible with input "
            f"{tuple(input_ids.shape)}"
        )

    def _selected_token_logps(self, step_logits, next_token):
        import torch

        logps = torch.nn.functional.log_softmax(step_logits.float(), dim=-1)
        return logps.gather(-1, next_token.to(device=logps.device, dtype=torch.long)).squeeze(-1)

    def _stack_generated_old_logps(self, input_ids, old_logps: list, max_new_tokens: int):
        import torch

        if max_new_tokens <= 0:
            return torch.empty((int(input_ids.shape[0]), 0), device=input_ids.device, dtype=torch.float32)
        if len(old_logps) != int(max_new_tokens):
            return None
        return torch.stack(old_logps, dim=1).to(dtype=torch.float32)

    def _generated_response_mask(self, generated, max_new_tokens: int):
        import torch

        return torch.ones((int(generated.shape[0]), int(max_new_tokens)), device=generated.device, dtype=torch.float32)

    def _select_next_generation_token(
        self,
        step_logits,
        *,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        generator,
    ):
        import torch

        if do_sample:
            return _sample_next_token(
                step_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                generator=generator,
            )
        return torch.argmax(step_logits, dim=-1, keepdim=True)

    def _make_static_inference_context(self, *, max_batch_size: int, max_sequence_length: int):
        from megatron.core.inference.contexts.static_context import StaticInferenceContext

        return StaticInferenceContext(
            max_batch_size=max_batch_size,
            max_sequence_length=max_sequence_length,
        )

    def _make_gpt_inference_wrapper(self, inference_context):
        from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
            GPTInferenceWrapper,
        )

        return GPTInferenceWrapper(self.model, inference_context)

    def _static_kv_serving_context(self, *, max_batch_size: int, max_sequence_length: int):
        requested = self._static_kv_cache_capacity(
            max_batch_size=int(max_batch_size),
            max_sequence_length=int(max_sequence_length),
        )
        reusable_key = self._select_static_kv_serving_capacity(
            max_batch_size=requested[0],
            max_sequence_length=requested[1],
        )
        if reusable_key is not None:
            self._static_kv_serving_hits += 1
            context, wrapper = self._static_kv_serving_cache.pop(reusable_key)
            self._static_kv_serving_cache[reusable_key] = (context, wrapper)
            return context, wrapper, True, reusable_key
        context = self._make_static_inference_context(
            max_batch_size=requested[0],
            max_sequence_length=requested[1],
        )
        wrapper = self._make_gpt_inference_wrapper(context)
        self._static_kv_serving_misses += 1
        max_entries = int(getattr(self.config, "static_kv_cache_max_entries", 4) or 0)
        if max_entries <= 0:
            return context, wrapper, False, requested
        while len(self._static_kv_serving_cache) >= max_entries:
            self._static_kv_serving_cache.pop(next(iter(self._static_kv_serving_cache)))
            self._static_kv_serving_evictions += 1
        self._static_kv_serving_cache[requested] = (context, wrapper)
        return context, wrapper, False, requested

    def _static_kv_cache_capacity(self, *, max_batch_size: int, max_sequence_length: int) -> tuple[int, int]:
        batch = max(1, int(max_batch_size))
        sequence = max(1, int(max_sequence_length))
        batch_bucket = int(getattr(self.config, "static_kv_cache_batch_bucket", 0) or 0)
        sequence_bucket = int(getattr(self.config, "static_kv_cache_sequence_bucket", 0) or 0)
        if batch_bucket > 0:
            batch = _round_up_to_multiple(batch, batch_bucket)
        if sequence_bucket > 0:
            sequence = _round_up_to_multiple(sequence, sequence_bucket)
        return batch, sequence

    def _select_static_kv_serving_capacity(self, *, max_batch_size: int, max_sequence_length: int) -> tuple[int, int] | None:
        candidates = [
            key
            for key in self._static_kv_serving_cache
            if key[0] >= int(max_batch_size) and key[1] >= int(max_sequence_length)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda key: (key[0] * key[1], key[0], key[1]))

    def _clear_static_kv_serving_cache(self) -> None:
        self._static_kv_serving_cache.clear()
        self._static_kv_serving_hits = 0
        self._static_kv_serving_misses = 0
        self._static_kv_serving_evictions = 0

    def greedy_generate_with_logps(self, input_ids, *, max_new_tokens: int, prompt_lens=None):
        return self.generate_with_logps(
            input_ids,
            max_new_tokens=max_new_tokens,
            prompt_lens=prompt_lens,
            do_sample=False,
        )

    def _ensure_optimizer(self, learning_rate: float | None = None):
        if self._optimizer is None:
            from runtime.optimizer_sharding import FP32MasterAdamW

            mode = self._distributed_optimizer_mode()
            self._optimizer = FP32MasterAdamW(
                self.trainable_parameters(),
                lr=learning_rate or self.config.learning_rate,
                weight_decay=float(self.config.weight_decay),
                data_parallel_size=int(self.config.parallelism[0]),
                data_parallel_rank=self._data_parallel_rank(),
                shard_state=mode in {"zero1", "zero2"},
                flat_shard=mode == "zero2",
            )
            self._optimizer_parallelism = tuple(int(x) for x in self.config.parallelism)
        elif learning_rate is not None:
            for group in self._optimizer.param_groups:
                group["lr"] = float(learning_rate)
        return self._optimizer

    def release_model_for_layout_switch(self) -> MegatronBuildConfig:
        """
        Release the active BF16 model while retaining canonical ZeRO-2 state.
        """
        import torch
        from megatron.core import parallel_state

        self._require_model()
        assert self.config is not None
        source_config = self.config
        if self._optimizer is not None and self._optimizer.model_parameters_bound:
            if self._distributed_optimizer_mode() != "zero2":
                raise RuntimeConfigError(
                    "in-place RPEG layout switching with initialized optimizer state requires ZeRO-2"
                )
            self._optimizer.unbind_model_parameters()
        self.model = None
        self.value_head = None
        self.reward_head = None
        self._clear_static_kv_serving_cache()
        if self._model_parallel_initialized and parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        self._destroy_model_parallel_process_groups()
        self._model_parallel_initialized = False
        self._released_operational_parameter_specs = ()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return source_config

    def _destroy_model_parallel_process_groups(self) -> None:
        """
        Release pooled Megatron subgroups before constructing a new layout.

        Clearing Megatron's Python globals does not deterministically release NCCL communicators.
        Repeated RPEG layout switches can therefore retain CXI endpoints until garbage collection
        and exhaust the per-process fabric endpoint budget. Every worker rank creates groups in the
        same order, so destroying the unique local handles in reverse order is safe.
        """
        if not self._model_parallel_process_groups:
            return
        import torch

        non_member = getattr(torch.distributed.GroupMember, "NON_GROUP_MEMBER", None)
        for group in reversed(self._model_parallel_process_groups):
            if group is None or group is non_member:
                continue
            try:
                torch.distributed.destroy_process_group(group)
            except (RuntimeError, ValueError):
                # A backend may have already finalized a shared alias. The group list is
                # de-duplicated at creation, so this is defensive.
                pass
        self._model_parallel_process_groups.clear()

    def build_switched_layout(
        self,
        source_config: MegatronBuildConfig,
        *,
        parallelism: tuple[int, int, int],
        microbatch: int,
    ) -> None:
        self.build_model(
            replace(
                source_config,
                parallelism=tuple(int(x) for x in parallelism),
                microbatch=int(microbatch),
            )
        )

    def restore_canonical_optimizer_binding(self) -> bool:
        """
        Rebind a rebuilt training layout and materialize BF16 weights from FP32 masters.
        """
        if self._optimizer is None or self._optimizer_parallelism is None:
            return False
        assert self.config is not None
        if tuple(int(x) for x in self.config.parallelism) != self._optimizer_parallelism:
            return False
        mode = self._distributed_optimizer_mode()
        if not self._optimizer.model_parameters_bound:
            if mode != "zero2" or not self._optimizer.is_flat_sharded:
                raise RuntimeConfigError(
                    "only a flat-sharded ZeRO-2 optimizer can rebind a rebuilt model"
                )
            self._optimizer.bind_model_parameters(self.trainable_parameters())
        if mode == "zero2":
            self._sync_zero2_parameters()
        else:
            self._optimizer.copy_master_parameters_to_model()
            if mode == "zero1":
                self._sync_zero1_parameters()
        return True

    def _clip_gradients(self) -> None:
        import torch

        assert self.config is not None
        grad_clip = float(self.config.grad_clip)
        if grad_clip <= 0.0:
            return
        dp = int(self.config.parallelism[0])
        dense_world = int(self.config.parallelism[0]) * int(self.config.parallelism[1]) * int(self.config.parallelism[2])
        if dense_world > 1 and torch.distributed.is_initialized():
            from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
            from megatron.core.transformer.module import param_is_not_shared

            mode = self._distributed_optimizer_mode()
            count_this_dp_replica = mode == "zero2" or self._data_parallel_rank() == 0
            device = next(self.model.parameters()).device
            squared_norm = torch.zeros((), dtype=torch.float32, device=device)
            if mode == "zero2":
                gradient = self._zero2_local_grad_shard
                if gradient is None:
                    raise RuntimeConfigError("ZeRO-2 gradient clipping requires a reduce-scattered gradient shard")
                layout = self._zero2_flat_layout()
                for segment in layout.local_segments():
                    if not (
                        param_is_not_shared(segment.parameter)
                        and param_is_not_tensor_parallel_duplicate(segment.parameter)
                    ):
                        continue
                    local_offset = int(segment.global_offset) - int(layout.local_start)
                    squared_norm.add_(
                        gradient.detach()
                        .narrow(0, local_offset, int(segment.numel))
                        .float()
                        .pow(2)
                        .sum()
                    )
            elif count_this_dp_replica:
                for param in self.trainable_parameters():
                    if (
                        param.grad is not None
                        and param_is_not_shared(param)
                        and param_is_not_tensor_parallel_duplicate(param)
                    ):
                        squared_norm.add_(param.grad.detach().float().pow(2).sum())
            self._all_reduce_dense_world(squared_norm)
            total_norm = squared_norm.sqrt()
            coefficient = torch.clamp(grad_clip / (total_norm + 1.0e-6), max=1.0)
            if mode == "zero2":
                self._zero2_local_grad_shard.mul_(
                    coefficient.to(
                        device=self._zero2_local_grad_shard.device,
                        dtype=self._zero2_local_grad_shard.dtype,
                    )
                )
            else:
                for param in self.trainable_parameters():
                    if param.grad is not None:
                        param.grad.mul_(coefficient.to(device=param.grad.device, dtype=param.grad.dtype))
            return
        torch.nn.utils.clip_grad_norm_(self.trainable_parameters(), grad_clip)

    def _optimizer_step(self, optimizer) -> None:
        if self._distributed_optimizer_mode() == "zero2":
            if self._zero2_local_grad_shard is None:
                raise RuntimeConfigError("ZeRO-2 optimizer step requires a reduce-scattered gradient shard")
            if int(self.config.parallelism[0]) > 1:
                optimizer.set_sharded_gradient(self._zero2_local_grad_shard)
        hidden = self._zero1_hidden_gradients()
        try:
            optimizer.step()
        finally:
            self._restore_hidden_gradients(hidden)
        self._sync_zero1_parameters()
        self._sync_tensor_parallel_auxiliary_heads()

    @staticmethod
    def _all_reduce_dense_world(value) -> None:
        """
        Reduce over the dense DP x PP x TP world without a world NCCL communicator.

        Slingshot/CXI stack has a finite endpoint budget per process. Megatron already owns NCCL
        communicators along each independent dense parallel dimension, so constructing or lazily
        activating an additional full-world communicator for scalar loss/gradient statistics wastes
        one network endpoint. A reduction over TP, then PP, then DP is exactly a reduction over
        their Cartesian product and reuses the hot communicators used by the training step.
        """
        import torch
        from megatron.core import parallel_state

        if not torch.distributed.is_initialized():
            return
        groups = (
            parallel_state.get_tensor_model_parallel_group(),
            parallel_state.get_pipeline_model_parallel_group(),
            parallel_state.get_data_parallel_group(),
        )
        seen: set[int] = set()
        for group in groups:
            if group is None or id(group) in seen:
                continue
            seen.add(id(group))
            if torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.all_reduce(
                    value,
                    op=torch.distributed.ReduceOp.SUM,
                    group=group,
                )

    def _finish_weighted_step(self, loss, *, learning_rate: float | None = None) -> float:
        import torch

        optimizer = self._ensure_optimizer(learning_rate)
        optimizer.zero_grad(set_to_none=True)
        self._run_backward_with_zero2_overlap(loss.backward)
        self._average_data_parallel_gradients()
        self._clip_gradients()
        self._optimizer_step(optimizer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        value = loss.detach().float()
        contributors = torch.ones_like(value)
        if torch.distributed.is_initialized():
            self._all_reduce_dense_world(value)
            self._all_reduce_dense_world(contributors)
            value.div_(torch.clamp(contributors, min=1.0))
        return float(value.cpu())

    @staticmethod
    def _normalize_loss_sum(loss_sum, token_count):
        import torch

        denom = torch.clamp(token_count.to(device=loss_sum.device, dtype=loss_sum.dtype), min=1.0)
        return loss_sum / denom

    def _pipeline_loss_tuple(self, loss_sum, token_count, metrics, *, num_microbatches: int, global_token_count=None):
        import torch

        if global_token_count is None:
            global_token_count = token_count
        global_token_count = global_token_count.to(device=loss_sum.device)
        report = dict(metrics)
        report["rpeg_loss_sum"] = loss_sum.detach().float()
        report["rpeg_loss_tokens"] = token_count.detach().to(device=loss_sum.device, dtype=torch.float32)
        return loss_sum * int(num_microbatches), global_token_count.to(dtype=torch.int), report

    def _finish_fused_pipeline_loss_step(
        self,
        input_ids,
        loss_func,
        *,
        fused_pipeline_schedule: dict[str, Any],
        fused_model: str,
        fusion_sync_root: str,
        fusion_sync_token: str,
        fusion_sync_backend: str,
        fusion_timeout_s: float,
        hidden_output: bool,
        learning_rate: float | None = None,
    ) -> float:
        """
        Execute one side of a coordinated actor/critic microtask schedule.

        The actor and critic live in separate Megatron process groups, but map opposite logical PP
        ranks onto the same physical GPUs. Each process waits for its per-device predecessor before
        posting the next pipeline receive, executes exactly one scheduled compute task, then records
        a CUDA IPC event on its current stream. The producer records that event before posting its
        asynchronous send, so this launch order cannot create a send/receive cycle. It also prevents
        a persistent NCCL receive kernel from occupying a shared middle-stage GPU while the
        colocated process still needs to launch the predecessor's TP collective (observed at
        PP=4/TP=4 on 70B).
        """
        import torch
        from megatron.core import parallel_state
        from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
        from megatron.core.pipeline_parallel.schedules import (
            backward_step,
            deallocate_output_tensor,
            forward_step,
            get_tensor_shapes,
        )
        from megatron.core.utils import get_model_config
        from runtime.fused_pipeline import FusedPipelineSchedule

        schedule = FusedPipelineSchedule.from_dict(fused_pipeline_schedule)
        model_meta = next((model for model in schedule.models if model.name == fused_model), None)
        if model_meta is None:
            raise RuntimeConfigError(f"fused schedule has no model {fused_model!r}")
        model, build_config = self._require_model()
        dp, pp, tp = (int(value) for value in build_config.parallelism)
        if model_meta.pipeline_stages != pp:
            raise RuntimeConfigError(
                f"fused schedule PP={model_meta.pipeline_stages} does not match worker PP={pp}"
            )
        input_ids, position_ids, attention_mask = self._logit_inputs(input_ids)
        batch, seq = input_ids.shape
        num_microbatches, micro_batch_size = self._pipeline_microbatching(batch)
        if model_meta.microbatches != num_microbatches:
            raise RuntimeConfigError(
                f"fused schedule microbatches={model_meta.microbatches} does not match worker "
                f"microbatches={num_microbatches}"
            )
        dp_rank, pp_rank, tp_rank = self._rank_to_dp_pp_tp(self.rank)
        physical_stage = pp - pp_rank - 1 if model_meta.reverse else pp_rank
        device_order = schedule.device_order[physical_stage]
        task_by_id = {task.task_id: task for task in schedule.tasks}
        own_tasks = [
            task_by_id[task_id]
            for task_id in device_order
            if task_by_id[task_id].model == fused_model
        ]
        if len(own_tasks) != 2 * num_microbatches:
            raise RuntimeConfigError(
                f"fused schedule gives {len(own_tasks)} local tasks to {fused_model} stage "
                f"{pp_rank}, expected {2 * num_microbatches}"
            )
        if any(task.stage != pp_rank for task in own_tasks):
            raise RuntimeConfigError(
                f"fused schedule maps a non-local {fused_model} task onto PP rank {pp_rank}"
            )

        sync_root = Path(fusion_sync_root)
        if not sync_root.is_absolute() or ".." in sync_root.parts:
            raise RuntimeConfigError(f"invalid fused pipeline sync root {fusion_sync_root!r}")
        lane_root = (
            sync_root
            / fusion_sync_token
            / f"dp{dp_rank}-tp{tp_rank}-stage{physical_stage}"
        )
        lane_root.mkdir(parents=True, exist_ok=True)
        progress_path = None
        if (
            os.environ.get("RPEG_FUSION_PROGRESS_MARKERS", "0") == "1"
            and dp_rank == 0
            and tp_rank == 0
        ):
            progress_path = lane_root / f"{fused_model}.rank{self.rank}.progress.jsonl"

        def record_progress(task_id: str, phase: str) -> None:
            if progress_path is None:
                return
            self._append_fused_progress(progress_path, {
                "fused_model": fused_model,
                "phase": phase,
                "physical_stage": physical_stage,
                "pp_rank": pp_rank,
                "rank": self.rank,
                "task_id": task_id,
                "timestamp": time.time(),
            })

        sync_backend = str(fusion_sync_backend).strip().lower().replace("-", "_")
        if sync_backend not in {"cuda_ipc", "filesystem"}:
            raise RuntimeConfigError(f"unsupported fused pipeline sync backend {sync_backend!r}")
        if sync_backend == "cuda_ipc" and not torch.cuda.is_available():
            raise RuntimeConfigError("fused pipeline cuda_ipc synchronization requires CUDA")
        sync_counters = {
            "fusion_sync_backend": sync_backend,
            "fusion_cuda_ipc_event_exports": 0,
            "fusion_cuda_ipc_event_imports": 0,
            "fusion_cuda_ipc_event_waits": 0,
            "fusion_cuda_ipc_event_records": 0,
            "fusion_cpu_handle_polls": 0,
            "fusion_cpu_completion_polls": 0,
            "fusion_pipeline_async_sends": 0,
            "fusion_pipeline_async_send_waits": 0,
            "fusion_pipeline_async_recvs": 0,
            "fusion_pipeline_async_recv_waits": 0,
        }

        def marker(task_id: str) -> Path:
            safe = task_id.replace(":", "_").replace("/", "_")
            return lane_root / f"{safe}.done"

        def wait_for(path: Path, *, handle: bool = False) -> None:
            deadline = time.monotonic() + max(1.0, float(fusion_timeout_s))
            while not path.exists():
                if time.monotonic() >= deadline:
                    observed = sorted(candidate.name for candidate in lane_root.glob("*"))
                    raise RuntimeConfigError(
                        f"timed out waiting for fused pipeline marker {path}; "
                        f"observed_lane_markers={observed}"
                    )
                counter = "fusion_cpu_handle_polls" if handle else "fusion_cpu_completion_polls"
                sync_counters[counter] += 1
                time.sleep(0.0005)

        def event_handle_path(event_id: str) -> Path:
            safe = event_id.replace(":", "_").replace("/", "_")
            return lane_root / f"{safe}.cuda_event"

        local_events: dict[str, Any] = {}
        local_event_handles: dict[str, bytes] = {}
        imported_events: dict[str, Any] = {}
        if sync_backend == "cuda_ipc":
            local_event_ids = [task.task_id for task in own_tasks]
            if fused_model == "actor":
                local_event_ids.append("actor_optimizer")
            for event_id in local_event_ids:
                event = torch.cuda.Event(interprocess=True)
                handle = event.ipc_handle()
                local_events[event_id] = event
                local_event_handles[event_id] = handle

        def cuda_ipc_event(event_id: str):
            event = local_events.get(event_id)
            if event is None:
                event = imported_events.get(event_id)
            if event is not None:
                return event
            path = event_handle_path(event_id)
            # The producer publishes the handle only after cudaEventRecord has been issued.
            # cudaStreamWaitEvent is a no-op for a never-recorded event, so eager handle publication
            # would introduce a race.
            wait_for(path, handle=True)
            event = torch.cuda.Event.from_ipc_handle(
                torch.cuda.current_device(),
                path.read_bytes(),
            )
            imported_events[event_id] = event
            sync_counters["fusion_cuda_ipc_event_imports"] += 1
            return event

        def wait_for_completion(event_id: str) -> None:
            if sync_backend == "cuda_ipc":
                # A stream wait alone does not constrain the producer/consumer host launch order:
                # Python can enqueue the entire filtered model instruction stream while the paired
                # process does the same. NCCL P2P calls then overtake the offline device order and
                # can leave a later receive resident while an earlier TP collective still needs the
                # GPU (observed at PP=4 on 70B). The receive for this task has already been posted
                # above, so synchronizing the predecessor event cannot recreate the original
                # receive-before-marker deadlock. It makes each offline device task an actual
                # cross-process launch barrier, which is the execution model validated by
                # FusedPipelineSchedule.
                self._synchronize_fused_cuda_ipc_event(cuda_ipc_event(event_id))
                sync_counters["fusion_cuda_ipc_event_waits"] += 1
            else:
                wait_for(marker(event_id))

        def record_completion(event_id: str) -> None:
            if sync_backend == "cuda_ipc":
                event = local_events[event_id]
                event.record(torch.cuda.current_stream())
                path = event_handle_path(event_id)
                temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
                temporary.write_bytes(local_event_handles[event_id])
                temporary.replace(path)
                sync_counters["fusion_cuda_ipc_event_exports"] += 1
                sync_counters["fusion_cuda_ipc_event_records"] += 1
            else:
                marker(event_id).touch()

        predecessor = {
            task_id: device_order[index - 1] if index else None
            for index, task_id in enumerate(device_order)
        }
        config = get_model_config(model)
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        # Materialize the full PP communicator before issuing individual P2P calls. With lazy NCCL
        # initialization, an unbatched send/recv creates an extra two-rank communicator per
        # neighbor; two counter-flowing fused pipelines can otherwise consume CXI endpoints and
        # stall while the peer waits on a CUDA-IPC schedule marker.
        if pp > 1 and torch.distributed.is_initialized():
            torch.distributed.barrier(
                group=pp_group,
                device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None,
            )
        communicator = P2PCommunicator(
            pp_group=pp_group,
            config=config,
        )
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        tensor_shapes = get_tensor_shapes(
            seq_length=seq,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=None,
            config=config,
            tp_group=tp_group,
            cp_group=cp_group,
        )
        if bool(getattr(config, "variable_seq_lengths", False)) or bool(
            getattr(config, "mtp_standalone", False)
        ):
            raise RuntimeConfigError(
                "fused pipeline asynchronous receives require fixed tensor shapes"
            )
        optimizer = self._ensure_optimizer(learning_rate)
        optimizer.zero_grad(set_to_none=True)
        shards = list(self._iter_indexed_logits_microbatches(
            input_ids,
            position_ids,
            attention_mask,
            num_microbatches=num_microbatches,
        ))
        forward_data_store = []
        saved_inputs: dict[int, Any] = {}
        saved_outputs: dict[int, Any] = {}
        pending_forward_sends: dict[int, Any] = {}
        pending_backward_sends: list[Any] = []

        def forward_step_func(data_iterator, active_model, checkpoint_activations_microbatch=None):
            del checkpoint_activations_microbatch
            microbatch_index, ids, pos, mask = next(data_iterator)
            with self._autocast_context():
                if hidden_output:
                    preproc_output = active_model._preprocess(
                        input_ids=ids,
                        position_ids=pos,
                        decoder_input=None,
                        inference_context=None,
                        packed_seq_params=None,
                        padding_mask=None,
                    )
                    (
                        decoder_input,
                        rotary_pos_emb,
                        rotary_pos_cos,
                        rotary_pos_sin,
                        sequence_len_offset,
                        padding_mask,
                    ) = preproc_output[:6]
                    rotary_pos_cos_sin = preproc_output[6] if len(preproc_output) == 7 else None
                    output = active_model.decoder(
                        hidden_states=decoder_input,
                        attention_mask=mask,
                        inference_context=None,
                        rotary_pos_emb=rotary_pos_emb,
                        rotary_pos_cos=rotary_pos_cos,
                        rotary_pos_sin=rotary_pos_sin,
                        rotary_pos_cos_sin=rotary_pos_cos_sin,
                        packed_seq_params=None,
                        sequence_len_offset=sequence_len_offset,
                        padding_mask=padding_mask,
                    )
                else:
                    output = active_model(
                        input_ids=ids,
                        position_ids=pos,
                        attention_mask=mask,
                        labels=None,
                        runtime_gather_output=True,
                    )
            return output, partial(
                loss_func,
                microbatch_index=microbatch_index,
                input_ids_shard=ids,
                num_microbatches=num_microbatches,
            )

        def execute_microtasks() -> None:
            for task in own_tasks:
                # Enforce the offline physical-device instruction stream on the host before
                # launching another persistent NCCL receive. The completion event is recorded before
                # the send. The schedule's blocking-P2P validator proves that even if NCCL
                # host-blocks this send until its peer posts a receive, every process still advances
                # to completion.
                prior = predecessor[task.task_id]
                record_progress(task.task_id, "before_predecessor")
                if prior is not None:
                    wait_for_completion(prior)
                record_progress(task.task_id, "after_predecessor")
                if task.phase == "F":
                    received, pending_receive = self._post_grouped_pipeline_recv(
                        tensor_shapes,
                        peer=communicator.prev_rank,
                        group=pp_group,
                        dtype=config.pipeline_dtype,
                        skip=communicator.is_pp_first_stage,
                    )
                    output_grad = None
                else:
                    received = None
                    output_grad, pending_receive = self._post_grouped_pipeline_recv(
                        tensor_shapes,
                        peer=communicator.next_rank,
                        group=pp_group,
                        dtype=config.pipeline_dtype,
                        skip=communicator.is_pp_last_stage,
                    )
                sync_counters["fusion_pipeline_async_recvs"] += len(
                    pending_receive[0]
                )
                record_progress(task.task_id, "receive_posted")
                sync_counters["fusion_pipeline_async_recv_waits"] += (
                    self._wait_grouped_pipeline_recvs(pending_receive)
                )
                record_progress(task.task_id, "receive_completed")
                if task.phase == "F":
                    output, _num_tokens = forward_step(
                        forward_step_func,
                        iter((shards[task.microbatch],)),
                        model,
                        num_microbatches,
                        received,
                        forward_data_store,
                        config,
                        cp_group_size=cp_group.size(),
                        collect_non_loss_data=False,
                        current_microbatch=task.microbatch,
                        is_last_stage=communicator.is_pp_last_stage,
                    )
                    # The fused device-order dependency covers GPU compute, not completion of the
                    # following pipeline send. Both counter-flowing roots can otherwise block in
                    # send while their colocated peers wait for completion markers before advancing
                    # the matching receive path. Record on the compute stream first; CUDA IPC makes
                    # the peer wait for the actual kernels without serializing communication.
                    record_completion(task.task_id)
                    record_progress(task.task_id, "compute_recorded")
                    if not communicator.is_pp_last_stage:
                        pending = self._post_grouped_pipeline_send(
                            output,
                            peer=communicator.next_rank,
                            group=pp_group,
                        )
                        pending_forward_sends[task.microbatch] = pending
                        sync_counters["fusion_pipeline_async_sends"] += len(pending[0])
                    record_progress(task.task_id, "send_posted")
                    saved_inputs[task.microbatch] = received
                    saved_outputs[task.microbatch] = output
                    deallocate_output_tensor(output, config.deallocate_pipeline_outputs)
                else:
                    if task.microbatch not in saved_outputs:
                        raise RuntimeConfigError(
                            f"fused backward {task.task_id} has no retained forward activation"
                        )
                    pending_forward = pending_forward_sends.pop(task.microbatch, None)
                    if pending_forward is not None:
                        sync_counters["fusion_pipeline_async_send_waits"] += (
                            self._wait_grouped_pipeline_sends(pending_forward)
                        )
                    input_grad = backward_step(
                        saved_inputs.pop(task.microbatch),
                        saved_outputs.pop(task.microbatch),
                        output_grad,
                        config,
                    )
                    record_completion(task.task_id)
                    record_progress(task.task_id, "compute_recorded")
                    if not communicator.is_pp_first_stage:
                        pending = self._post_grouped_pipeline_send(
                            input_grad,
                            peer=communicator.prev_rank,
                            group=pp_group,
                        )
                        pending_backward_sends.append(pending)
                        sync_counters["fusion_pipeline_async_sends"] += len(pending[0])
                    record_progress(task.task_id, "send_posted")

        self._run_backward_with_zero2_overlap(execute_microtasks)
        for pending in pending_backward_sends:
            sync_counters["fusion_pipeline_async_send_waits"] += (
                self._wait_grouped_pipeline_sends(pending)
            )
        if saved_inputs or saved_outputs or pending_forward_sends:
            raise RuntimeConfigError("fused pipeline left retained microbatch activations")
        final_task = device_order[-1]
        wait_for_completion(final_task)
        if fused_model == "critic":
            wait_for_completion("actor_optimizer")
        self._average_data_parallel_gradients()
        self._clip_gradients()
        self._optimizer_step(optimizer)
        if fused_model == "actor":
            record_completion("actor_optimizer")
        loss = torch.zeros([], dtype=torch.float32, device=input_ids.device)
        contributors = torch.zeros([], dtype=torch.float32, device=input_ids.device)
        found = self._loss_from_forward_store(forward_data_store)
        if found is not None:
            loss.copy_(found.detach().float())
            contributors.fill_(1.0)
        if torch.distributed.is_initialized():
            self._all_reduce_dense_world(loss)
            self._all_reduce_dense_world(contributors)
            loss.div_(torch.clamp(contributors, min=1.0))
        self._last_fusion_counters = sync_counters
        return float(loss.detach().cpu())

    def _finish_pipeline_loss_step(self, input_ids, loss_func, *, learning_rate: float | None = None) -> float:
        import torch
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

        optimizer = self._ensure_optimizer(learning_rate)
        optimizer.zero_grad(set_to_none=True)
        input_ids, position_ids, attention_mask = self._logit_inputs(input_ids)
        batch, seq = input_ids.shape
        num_microbatches, micro_batch_size = self._pipeline_microbatching(batch)

        def forward_step_func(data_iterator, model, checkpoint_activations_microbatch=None):
            del checkpoint_activations_microbatch
            microbatch_index, ids, pos, mask = next(data_iterator)
            with self._autocast_context():
                output = model(
                    input_ids=ids,
                    position_ids=pos,
                    attention_mask=mask,
                    labels=None,
                    runtime_gather_output=True,
                )
            return output, partial(
                loss_func,
                microbatch_index=microbatch_index,
                input_ids_shard=ids,
                num_microbatches=num_microbatches,
            )

        forward_data_store = self._run_backward_with_zero2_overlap(
            lambda: get_forward_backward_func()(
                forward_step_func=forward_step_func,
                data_iterator=self._iter_indexed_logits_microbatches(
                    input_ids,
                    position_ids,
                    attention_mask,
                    num_microbatches=num_microbatches,
                ),
                model=self.model,
                num_microbatches=num_microbatches,
                seq_length=seq,
                micro_batch_size=micro_batch_size,
                forward_only=False,
                collect_non_loss_data=False,
            )
        )
        self._average_data_parallel_gradients()
        self._clip_gradients()
        self._optimizer_step(optimizer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        loss = torch.zeros([], dtype=torch.float32, device=input_ids.device)
        contributors = torch.zeros([], dtype=torch.float32, device=input_ids.device)
        found = self._loss_from_forward_store(forward_data_store)
        if found is not None:
            loss.copy_(found.detach().float())
            contributors.fill_(1.0)
        if torch.distributed.is_initialized():
            self._all_reduce_dense_world(loss)
            self._all_reduce_dense_world(contributors)
            loss.div_(torch.clamp(contributors, min=1.0))
        return float(loss.detach().cpu())

    def _finish_pipeline_hidden_loss_step(self, input_ids, loss_func, *, learning_rate: float | None = None) -> float:
        import torch
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

        optimizer = self._ensure_optimizer(learning_rate)
        optimizer.zero_grad(set_to_none=True)
        input_ids, position_ids, attention_mask = self._logit_inputs(input_ids)
        batch, seq = input_ids.shape
        num_microbatches, micro_batch_size = self._pipeline_microbatching(batch)

        def forward_step_func(data_iterator, model, checkpoint_activations_microbatch=None):
            del checkpoint_activations_microbatch
            microbatch_index, ids, pos, mask = next(data_iterator)
            with self._autocast_context():
                preproc_output = model._preprocess(
                    input_ids=ids,
                    position_ids=pos,
                    decoder_input=None,
                    inference_context=None,
                    packed_seq_params=None,
                    padding_mask=None,
                )
                (
                    decoder_input,
                    rotary_pos_emb,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    sequence_len_offset,
                    padding_mask,
                ) = preproc_output[:6]
                rotary_pos_cos_sin = preproc_output[6] if len(preproc_output) == 7 else None
                output = model.decoder(
                    hidden_states=decoder_input,
                    attention_mask=mask,
                    inference_context=None,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    rotary_pos_cos_sin=rotary_pos_cos_sin,
                    packed_seq_params=None,
                    sequence_len_offset=sequence_len_offset,
                    padding_mask=padding_mask,
                )
            return output, partial(
                loss_func,
                microbatch_index=microbatch_index,
                input_ids_shard=ids,
                num_microbatches=num_microbatches,
            )

        forward_data_store = self._run_backward_with_zero2_overlap(
            lambda: get_forward_backward_func()(
                forward_step_func=forward_step_func,
                data_iterator=self._iter_indexed_logits_microbatches(
                    input_ids,
                    position_ids,
                    attention_mask,
                    num_microbatches=num_microbatches,
                ),
                model=self.model,
                num_microbatches=num_microbatches,
                seq_length=seq,
                micro_batch_size=micro_batch_size,
                forward_only=False,
                collect_non_loss_data=False,
            )
        )
        self._average_data_parallel_gradients()
        self._clip_gradients()
        self._optimizer_step(optimizer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        loss = torch.zeros([], dtype=torch.float32, device=input_ids.device)
        contributors = torch.zeros([], dtype=torch.float32, device=input_ids.device)
        found = self._loss_from_forward_store(forward_data_store)
        if found is not None:
            loss.copy_(found.detach().float())
            contributors.fill_(1.0)
        if torch.distributed.is_initialized():
            self._all_reduce_dense_world(loss)
            self._all_reduce_dense_world(contributors)
            loss.div_(torch.clamp(contributors, min=1.0))
        return float(loss.detach().cpu())

    def _loss_from_forward_store(self, value):
        aggregated = self._aggregate_reported_loss(value)
        if aggregated is not None:
            return aggregated
        if isinstance(value, dict):
            for key in ("ppo ac loss", "ppo actor loss", "ppo critic loss", "grpo actor loss", "dpo loss", "lm loss"):
                if key in value:
                    return value[key]
            for item in value.values():
                found = self._loss_from_forward_store(item)
                if found is not None:
                    return found
        if isinstance(value, (list, tuple)):
            for item in reversed(value):
                found = self._loss_from_forward_store(item)
                if found is not None:
                    return found
        return None

    def _aggregate_reported_loss(self, value):
        reports = []
        self._collect_reported_losses(value, reports)
        if not reports:
            return None
        loss_sum = None
        token_count = None
        for report in reports:
            current_sum = report["rpeg_loss_sum"]
            current_tokens = report["rpeg_loss_tokens"]
            loss_sum = current_sum if loss_sum is None else loss_sum + current_sum.to(device=loss_sum.device)
            token_count = (
                current_tokens
                if token_count is None
                else token_count + current_tokens.to(device=token_count.device)
            )
        assert loss_sum is not None and token_count is not None
        return self._normalize_loss_sum(loss_sum, token_count)

    def _collect_reported_losses(self, value, out) -> None:
        if isinstance(value, dict):
            if "rpeg_loss_sum" in value and "rpeg_loss_tokens" in value:
                out.append(value)
                return
            for item in value.values():
                self._collect_reported_losses(item, out)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self._collect_reported_losses(item, out)

    def _pipeline_microbatching(self, batch: int) -> tuple[int, int]:
        assert self.config is not None
        num_microbatches = int(self.config.microbatch or 1)
        if num_microbatches < 1:
            raise RuntimeConfigError(f"invalid Megatron microbatch count {num_microbatches}")
        if self.config.parallelism[1] == 1:
            num_microbatches = 1
        if batch % num_microbatches != 0:
            raise RuntimeConfigError(
                f"batch size {batch} must be divisible by planned microbatch count {num_microbatches}"
            )
        return num_microbatches, batch // num_microbatches

    def _iter_loss_microbatches(
        self,
        input_ids,
        position_ids,
        attention_mask,
        labels,
        loss_mask,
        *,
        num_microbatches: int,
    ):
        for idx in range(num_microbatches):
            yield (
                input_ids.chunk(num_microbatches, dim=0)[idx].contiguous(),
                position_ids.chunk(num_microbatches, dim=0)[idx].contiguous(),
                attention_mask,
                labels.chunk(num_microbatches, dim=0)[idx].contiguous(),
                loss_mask.chunk(num_microbatches, dim=0)[idx].contiguous(),
            )

    def _iter_logits_microbatches(
        self,
        input_ids,
        position_ids,
        attention_mask,
        *,
        num_microbatches: int,
    ):
        for idx in range(num_microbatches):
            yield (
                input_ids.chunk(num_microbatches, dim=0)[idx].contiguous(),
                position_ids.chunk(num_microbatches, dim=0)[idx].contiguous(),
                attention_mask,
            )

    def _iter_indexed_logits_microbatches(
        self,
        input_ids,
        position_ids,
        attention_mask,
        *,
        num_microbatches: int,
    ):
        for idx in range(num_microbatches):
            yield (
                idx,
                input_ids.chunk(num_microbatches, dim=0)[idx].contiguous(),
                position_ids.chunk(num_microbatches, dim=0)[idx].contiguous(),
                attention_mask,
            )

    def _slice_pipeline_microbatch(self, value, microbatch_index: int, num_microbatches: int):
        if num_microbatches <= 1:
            return value
        if hasattr(value, "chunk"):
            return value.chunk(num_microbatches, dim=0)[int(microbatch_index)].contiguous()
        if isinstance(value, tuple):
            chunks = _split_sequence_evenly(value, num_microbatches)
            return tuple(chunks[int(microbatch_index)])
        if isinstance(value, list):
            chunks = _split_sequence_evenly(value, num_microbatches)
            return list(chunks[int(microbatch_index)])
        return value

    def ppo_actor_step(
        self,
        input_ids,
        prompt_lens,
        old_logps,
        advantages,
        response_mask,
        *,
        max_new_tokens: int,
        clip_range: float,
        entropy_coef: float = 0.0,
        learning_rate: float | None = None,
        fused_pipeline_schedule: dict[str, Any] | None = None,
        fused_model: str = "actor",
        fusion_sync_root: str = "/dev/shm/rpeg-fusion",
        fusion_sync_token: str = "default",
        fusion_sync_backend: str = "cuda_ipc",
        fusion_timeout_s: float = 600.0,
    ) -> float:
        import torch

        self._set_training_mode()
        device = next(self.model.parameters()).device
        input_ids = _ordinary_training_tensor(input_ids, device=device, dtype=torch.long)
        old_logps = _ordinary_training_tensor(old_logps, device=device, dtype=torch.float32)
        advantages = _ordinary_training_tensor(advantages, device=device, dtype=torch.float32)
        response_mask = _ordinary_training_tensor(response_mask, device=device, dtype=torch.float32)
        if self._uses_pipeline_schedule():
            prompt_lens_tuple = tuple(int(x) for x in prompt_lens)
            global_token_count = self._loss_token_count(response_mask)

            def loss_func(
                output_tensor,
                non_loss_data: bool = False,
                *,
                microbatch_index: int = 0,
                input_ids_shard=None,
                num_microbatches: int = 1,
            ):
                del non_loss_data
                ids = input_ids if input_ids_shard is None else input_ids_shard
                mb_prompt_lens = self._slice_pipeline_microbatch(prompt_lens_tuple, microbatch_index, num_microbatches)
                mb_old_logps = self._slice_pipeline_microbatch(old_logps, microbatch_index, num_microbatches)
                mb_advantages = self._slice_pipeline_microbatch(advantages, microbatch_index, num_microbatches)
                mb_response_mask = self._slice_pipeline_microbatch(response_mask, microbatch_index, num_microbatches)
                logits = self._normalize_logits(output_tensor, ids)
                current_logps = self._response_logps_from_logits(
                    logits,
                    ids,
                    mb_prompt_lens,
                    max_new_tokens,
                ).float()
                entropies = None
                if entropy_coef:
                    entropies = self._response_entropies_from_logits(
                        logits,
                        mb_prompt_lens,
                        max_new_tokens,
                    )
                loss_sum, token_count, metrics = self._ppo_actor_loss_tuple(
                    current_logps,
                    mb_old_logps,
                    mb_advantages,
                    mb_response_mask,
                    clip_range=clip_range,
                    entropy_coef=entropy_coef,
                    entropies=entropies,
                )
                return self._pipeline_loss_tuple(
                    loss_sum,
                    token_count,
                    metrics,
                    num_microbatches=num_microbatches,
                    global_token_count=global_token_count,
                )

            if fused_pipeline_schedule is not None:
                return self._finish_fused_pipeline_loss_step(
                    input_ids,
                    loss_func,
                    fused_pipeline_schedule=fused_pipeline_schedule,
                    fused_model=fused_model,
                    fusion_sync_root=fusion_sync_root,
                    fusion_sync_token=fusion_sync_token,
                    fusion_sync_backend=fusion_sync_backend,
                    fusion_timeout_s=fusion_timeout_s,
                    hidden_output=False,
                    learning_rate=learning_rate,
                )
            return self._finish_pipeline_loss_step(input_ids, loss_func, learning_rate=learning_rate)
        current_logps, _ = self.response_token_logps(input_ids, prompt_lens, max_new_tokens, grad=True)
        current_logps = current_logps.float()
        entropies = None
        if entropy_coef:
            logits = self.logits(input_ids, grad=True)
            entropies = self._response_entropies_from_logits(logits, prompt_lens, max_new_tokens)
        loss_sum, token_count, _metrics = self._ppo_actor_loss_tuple(
            current_logps,
            old_logps,
            advantages,
            response_mask,
            clip_range=clip_range,
            entropy_coef=entropy_coef,
            entropies=entropies,
        )
        loss = self._normalize_loss_sum(loss_sum, token_count)
        return self._finish_weighted_step(loss, learning_rate=learning_rate)

    def sequence_response_logps(self, input_ids, prompt_lens, max_new_tokens: int, *, grad: bool = False):
        logps, response_mask = self.response_token_logps(input_ids, prompt_lens, max_new_tokens, grad=grad)
        return (logps.float() * response_mask).sum(dim=1)

    def dpo_actor_step(
        self,
        input_ids,
        prompt_lens,
        ref_sequence_logps,
        *,
        max_new_tokens: int,
        beta: float,
        learning_rate: float | None = None,
    ) -> float:
        import torch

        self._set_training_mode()
        device = next(self.model.parameters()).device
        input_ids = _ordinary_training_tensor(input_ids, device=device, dtype=torch.long)
        ref_sequence_logps = _ordinary_training_tensor(
            ref_sequence_logps,
            device=device,
            dtype=torch.float32,
        )
        if ref_sequence_logps.ndim != 2 or ref_sequence_logps.shape[1] != 2:
            raise RuntimeConfigError("DPO reference logprobs must have shape [batch, 2]")
        if self._uses_pipeline_schedule():
            prompt_lens_tuple = tuple(int(x) for x in prompt_lens)

            def loss_func(
                output_tensor,
                non_loss_data: bool = False,
                *,
                microbatch_index: int = 0,
                input_ids_shard=None,
                num_microbatches: int = 1,
            ):
                del non_loss_data
                ids = input_ids if input_ids_shard is None else input_ids_shard
                mb_prompt_lens = self._slice_pipeline_microbatch(prompt_lens_tuple, microbatch_index, num_microbatches)
                mb_ref_sequence_logps = self._slice_pipeline_microbatch(
                    ref_sequence_logps,
                    microbatch_index,
                    num_microbatches,
                )
                logits = self._normalize_logits(output_tensor, ids)
                sequence_logps = self._response_logps_from_logits(
                    logits,
                    ids,
                    mb_prompt_lens,
                    max_new_tokens,
                ).float().sum(dim=1)
                if sequence_logps.numel() % 2 != 0:
                    raise RuntimeConfigError("DPO pipeline microbatch must contain complete chosen/rejected pairs")
                loss_sum, token_count, metrics = self._dpo_loss_tuple(
                    sequence_logps.view(-1, 2),
                    mb_ref_sequence_logps,
                    beta=beta,
                )
                return self._pipeline_loss_tuple(
                    loss_sum,
                    token_count,
                    metrics,
                    num_microbatches=num_microbatches,
                    global_token_count=torch.tensor(
                        ref_sequence_logps.shape[0],
                        device=loss_sum.device,
                        dtype=torch.int,
                    ),
                )

            return self._finish_pipeline_loss_step(input_ids, loss_func, learning_rate=learning_rate)
        sequence_logps = self.sequence_response_logps(input_ids, prompt_lens, max_new_tokens, grad=True).float()
        loss_sum, token_count, _metrics = self._dpo_loss_tuple(sequence_logps.view(-1, 2), ref_sequence_logps, beta=beta)
        loss = self._normalize_loss_sum(loss_sum, token_count)
        return self._finish_weighted_step(loss, learning_rate=learning_rate)

    def _dpo_loss_tuple(self, actor_sequence_logps, ref_sequence_logps, *, beta: float):
        import torch

        actor_margin = actor_sequence_logps[:, 0] - actor_sequence_logps[:, 1]
        ref_margin = ref_sequence_logps[:, 0] - ref_sequence_logps[:, 1]
        per_pair = -torch.nn.functional.logsigmoid(beta * (actor_margin - ref_margin))
        loss_sum = per_pair.sum()
        token_count = torch.tensor(per_pair.numel(), device=loss_sum.device, dtype=torch.int)
        reduced = self._normalize_loss_sum(loss_sum.detach(), token_count)
        return loss_sum, token_count, {"dpo loss": reduced}

    def grpo_actor_step(
        self,
        input_ids,
        prompt_lens,
        old_logps,
        ref_logps,
        rewards,
        response_mask,
        *,
        max_new_tokens: int,
        samples_per_prompt: int,
        clip_range: float,
        kl_coef: float,
        learning_rate: float | None = None,
    ) -> float:
        import torch

        self._set_training_mode()
        device = next(self.model.parameters()).device
        input_ids = _ordinary_training_tensor(input_ids, device=device, dtype=torch.long)
        old_logps = _ordinary_training_tensor(old_logps, device=device, dtype=torch.float32)
        ref_logps = _ordinary_training_tensor(ref_logps, device=device, dtype=torch.float32)
        rewards = _ordinary_training_tensor(rewards, device=device, dtype=torch.float32)
        response_mask = _ordinary_training_tensor(response_mask, device=device, dtype=torch.float32)
        if self._uses_pipeline_schedule():
            prompt_lens_tuple = tuple(int(x) for x in prompt_lens)
            advantages = self._grpo_advantages(rewards, samples_per_prompt=samples_per_prompt)
            global_token_count = self._loss_token_count(response_mask)

            def loss_func(
                output_tensor,
                non_loss_data: bool = False,
                *,
                microbatch_index: int = 0,
                input_ids_shard=None,
                num_microbatches: int = 1,
            ):
                del non_loss_data
                ids = input_ids if input_ids_shard is None else input_ids_shard
                mb_prompt_lens = self._slice_pipeline_microbatch(prompt_lens_tuple, microbatch_index, num_microbatches)
                mb_old_logps = self._slice_pipeline_microbatch(old_logps, microbatch_index, num_microbatches)
                mb_ref_logps = self._slice_pipeline_microbatch(ref_logps, microbatch_index, num_microbatches)
                mb_rewards = self._slice_pipeline_microbatch(rewards, microbatch_index, num_microbatches)
                mb_advantages = self._slice_pipeline_microbatch(advantages, microbatch_index, num_microbatches)
                mb_response_mask = self._slice_pipeline_microbatch(response_mask, microbatch_index, num_microbatches)
                logits = self._normalize_logits(output_tensor, ids)
                current_logps = self._response_logps_from_logits(
                    logits,
                    ids,
                    mb_prompt_lens,
                    max_new_tokens,
                ).float()
                loss_sum, token_count, metrics = self._grpo_actor_loss_tuple(
                    current_logps,
                    mb_old_logps,
                    mb_ref_logps,
                    mb_rewards,
                    mb_response_mask,
                    samples_per_prompt=samples_per_prompt,
                    clip_range=clip_range,
                    kl_coef=kl_coef,
                    advantages=mb_advantages,
                )
                return self._pipeline_loss_tuple(
                    loss_sum,
                    token_count,
                    metrics,
                    num_microbatches=num_microbatches,
                    global_token_count=global_token_count,
                )

            return self._finish_pipeline_loss_step(input_ids, loss_func, learning_rate=learning_rate)
        current_logps, _ = self.response_token_logps(input_ids, prompt_lens, max_new_tokens, grad=True)
        loss_sum, token_count, _metrics = self._grpo_actor_loss_tuple(
            current_logps.float(),
            old_logps,
            ref_logps,
            rewards,
            response_mask,
            samples_per_prompt=samples_per_prompt,
            clip_range=clip_range,
            kl_coef=kl_coef,
        )
        loss = self._normalize_loss_sum(loss_sum, token_count)
        return self._finish_weighted_step(loss, learning_rate=learning_rate)

    def _grpo_actor_loss_tuple(
        self,
        current_logps,
        old_logps,
        ref_logps,
        rewards,
        response_mask,
        *,
        samples_per_prompt: int,
        clip_range: float,
        kl_coef: float,
        advantages=None,
    ):
        import torch

        if advantages is None:
            advantages = self._grpo_advantages(rewards, samples_per_prompt=samples_per_prompt)
        advantages = advantages.reshape(-1, 1).expand_as(current_logps)
        token_count = self._loss_token_count(response_mask)
        ratio = torch.exp(current_logps - old_logps)
        clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        pg = -torch.minimum(ratio * advantages, clipped * advantages)
        kl = (current_logps - ref_logps) * response_mask
        loss_sum = (pg * response_mask).sum() + kl_coef * kl.sum()
        reduced = self._normalize_loss_sum(loss_sum.detach(), token_count)
        return loss_sum, token_count, {"grpo actor loss": reduced}

    def _ppo_actor_loss_tuple(
        self,
        current_logps,
        old_logps,
        advantages,
        response_mask,
        *,
        clip_range: float,
        entropy_coef: float = 0.0,
        entropies=None,
    ):
        import torch

        token_count = self._loss_token_count(response_mask)
        ratio = torch.exp(current_logps - old_logps)
        clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        pg = -torch.minimum(ratio * advantages, clipped * advantages)
        loss_sum = (pg * response_mask).sum()
        if entropy_coef:
            if entropies is None or tuple(entropies.shape) != tuple(current_logps.shape):
                raise RuntimeConfigError("PPO entropy bonus requires token entropies matching response logprobs")
            loss_sum = loss_sum - float(entropy_coef) * (entropies * response_mask).sum()
        reduced = self._normalize_loss_sum(loss_sum.detach(), token_count)
        return loss_sum, token_count, {"ppo actor loss": reduced}

    def _loss_token_count(self, response_mask):
        import torch

        return torch.clamp(response_mask.detach().float().sum().round().to(dtype=torch.int), min=1)

    def _grpo_advantages(self, rewards, *, samples_per_prompt: int):
        if samples_per_prompt <= 0 or rewards.numel() % samples_per_prompt != 0:
            raise RuntimeConfigError(
                f"GRPO rewards size {rewards.numel()} is incompatible with samples_per_prompt={samples_per_prompt}"
            )
        grouped = rewards.view(-1, samples_per_prompt)
        mean = grouped.mean(dim=1, keepdim=True)
        std = grouped.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
        return ((grouped - mean) / std).reshape(-1).detach()

    def ppo_ac_step(
        self,
        input_ids,
        prompt_lens,
        old_logps,
        advantages,
        returns,
        response_mask,
        *,
        max_new_tokens: int,
        clip_range: float,
        entropy_coef: float = 0.0,
        value_coef: float = 1.0,
        learning_rate: float | None = None,
    ) -> float:
        import torch

        self._set_training_mode()
        device = next(self.model.parameters()).device
        input_ids = _ordinary_training_tensor(input_ids, device=device, dtype=torch.long)
        old_logps = _ordinary_training_tensor(old_logps, device=device, dtype=torch.float32)
        advantages = _ordinary_training_tensor(advantages, device=device, dtype=torch.float32)
        returns = _ordinary_training_tensor(returns, device=device, dtype=torch.float32)
        response_mask = _ordinary_training_tensor(response_mask, device=device, dtype=torch.float32)
        if self._uses_pipeline_schedule():
            prompt_lens_tuple = tuple(int(x) for x in prompt_lens)
            global_token_count = self._loss_token_count(response_mask)

            def loss_func(
                output_tensor,
                non_loss_data: bool = False,
                *,
                microbatch_index: int = 0,
                input_ids_shard=None,
                num_microbatches: int = 1,
            ):
                del non_loss_data
                ids = input_ids if input_ids_shard is None else input_ids_shard
                mb_prompt_lens = self._slice_pipeline_microbatch(prompt_lens_tuple, microbatch_index, num_microbatches)
                mb_old_logps = self._slice_pipeline_microbatch(old_logps, microbatch_index, num_microbatches)
                mb_advantages = self._slice_pipeline_microbatch(advantages, microbatch_index, num_microbatches)
                mb_returns = self._slice_pipeline_microbatch(returns, microbatch_index, num_microbatches)
                mb_response_mask = self._slice_pipeline_microbatch(response_mask, microbatch_index, num_microbatches)
                hidden_states = self._normalize_hidden_states(output_tensor, ids)
                logits = self._logits_from_hidden_states(hidden_states)
                current_logps = self._response_logps_from_logits(
                    logits,
                    ids,
                    mb_prompt_lens,
                    max_new_tokens,
                ).float()
                current_values = self._response_values_from_hidden_states(
                    hidden_states,
                    mb_prompt_lens,
                    max_new_tokens,
                ).float()
                entropies = None
                if entropy_coef:
                    entropies = self._response_entropies_from_logits(
                        logits,
                        mb_prompt_lens,
                        max_new_tokens,
                    )
                loss_sum, token_count, metrics = self._ppo_ac_loss_tuple(
                    current_logps,
                    current_values,
                    mb_old_logps,
                    mb_advantages,
                    mb_returns,
                    mb_response_mask,
                    clip_range=clip_range,
                    entropy_coef=entropy_coef,
                    value_coef=value_coef,
                    entropies=entropies,
                )
                return self._pipeline_loss_tuple(
                    loss_sum,
                    token_count,
                    metrics,
                    num_microbatches=num_microbatches,
                    global_token_count=global_token_count,
                )

            return self._finish_pipeline_hidden_loss_step(input_ids, loss_func, learning_rate=learning_rate)
        if self.value_head is None:
            raise RuntimeConfigError("ppo_ac_step requires a value auxiliary head")
        logits = self.logits(input_ids, grad=True)
        current_logps = self._response_logps_from_logits(logits, input_ids, prompt_lens, max_new_tokens).float()
        hidden_states = self.hidden_states(input_ids, grad=True)
        current_values = self._response_values_from_hidden_states(hidden_states, prompt_lens, max_new_tokens).float()
        loss_sum, token_count, _metrics = self._ppo_ac_loss_tuple(
            current_logps,
            current_values,
            old_logps,
            advantages,
            returns,
            response_mask,
            clip_range=clip_range,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            entropies=self._response_entropies_from_logits(logits, prompt_lens, max_new_tokens)
            if entropy_coef
            else None,
        )
        loss = self._normalize_loss_sum(loss_sum, token_count)
        return self._finish_weighted_step(loss, learning_rate=learning_rate)

    def _ppo_ac_loss_tuple(
        self,
        current_logps,
        current_values,
        old_logps,
        advantages,
        returns,
        response_mask,
        *,
        clip_range: float,
        entropy_coef: float,
        value_coef: float,
        entropies=None,
    ):
        import torch

        actor_loss_sum, actor_tokens, actor_metrics = self._ppo_actor_loss_tuple(
            current_logps,
            old_logps,
            advantages,
            response_mask,
            clip_range=clip_range,
            entropy_coef=entropy_coef,
            entropies=entropies,
        )
        critic_loss_sum, critic_tokens, critic_metrics = self._ppo_critic_loss_tuple(
            current_values,
            returns,
            response_mask,
        )
        token_count = torch.maximum(actor_tokens, critic_tokens)
        loss_sum = actor_loss_sum + float(value_coef) * critic_loss_sum
        loss = self._normalize_loss_sum(loss_sum.detach(), token_count)
        metrics = {
            "ppo ac loss": loss,
            **actor_metrics,
            **critic_metrics,
        }
        return loss_sum, token_count, metrics

    def ppo_critic_step(
        self,
        input_ids,
        prompt_lens,
        returns,
        response_mask,
        *,
        max_new_tokens: int,
        learning_rate: float | None = None,
        fused_pipeline_schedule: dict[str, Any] | None = None,
        fused_model: str = "critic",
        fusion_sync_root: str = "/dev/shm/rpeg-fusion",
        fusion_sync_token: str = "default",
        fusion_sync_backend: str = "cuda_ipc",
        fusion_timeout_s: float = 600.0,
    ) -> float:
        import torch

        self._set_training_mode()
        device = next(self.model.parameters()).device
        input_ids = _ordinary_training_tensor(input_ids, device=device, dtype=torch.long)
        returns = _ordinary_training_tensor(returns, device=device, dtype=torch.float32)
        response_mask = _ordinary_training_tensor(response_mask, device=device, dtype=torch.float32)
        if self._uses_pipeline_schedule():
            prompt_lens_tuple = tuple(int(x) for x in prompt_lens)
            global_token_count = self._loss_token_count(response_mask)

            def loss_func(
                output_tensor,
                non_loss_data: bool = False,
                *,
                microbatch_index: int = 0,
                input_ids_shard=None,
                num_microbatches: int = 1,
            ):
                del non_loss_data
                ids = input_ids if input_ids_shard is None else input_ids_shard
                mb_prompt_lens = self._slice_pipeline_microbatch(prompt_lens_tuple, microbatch_index, num_microbatches)
                mb_returns = self._slice_pipeline_microbatch(returns, microbatch_index, num_microbatches)
                mb_response_mask = self._slice_pipeline_microbatch(response_mask, microbatch_index, num_microbatches)
                hidden_states = self._normalize_hidden_states(output_tensor, ids)
                values = self._response_values_from_hidden_states(
                    hidden_states,
                    mb_prompt_lens,
                    max_new_tokens,
                ).float()
                loss_sum, token_count, metrics = self._ppo_critic_loss_tuple(values, mb_returns, mb_response_mask)
                return self._pipeline_loss_tuple(
                    loss_sum,
                    token_count,
                    metrics,
                    num_microbatches=num_microbatches,
                    global_token_count=global_token_count,
                )

            if fused_pipeline_schedule is not None:
                return self._finish_fused_pipeline_loss_step(
                    input_ids,
                    loss_func,
                    fused_pipeline_schedule=fused_pipeline_schedule,
                    fused_model=fused_model,
                    fusion_sync_root=fusion_sync_root,
                    fusion_sync_token=fusion_sync_token,
                    fusion_sync_backend=fusion_sync_backend,
                    fusion_timeout_s=fusion_timeout_s,
                    hidden_output=True,
                    learning_rate=learning_rate,
                )
            return self._finish_pipeline_hidden_loss_step(input_ids, loss_func, learning_rate=learning_rate)
        values, _ = self.response_token_values(input_ids, prompt_lens, max_new_tokens, grad=True)
        values = values.float()
        loss_sum, token_count, _metrics = self._ppo_critic_loss_tuple(values, returns, response_mask)
        loss = self._normalize_loss_sum(loss_sum, token_count)
        return self._finish_weighted_step(loss, learning_rate=learning_rate)

    def _ppo_critic_loss_tuple(self, values, returns, response_mask):
        token_count = self._loss_token_count(response_mask)
        loss_sum = (((values - returns) ** 2) * response_mask).sum()
        reduced = self._normalize_loss_sum(loss_sum.detach(), token_count)
        return loss_sum, token_count, {"ppo critic loss": reduced}

    def training_step(self, input_ids, *, labels=None, loss_mask=None, learning_rate: float | None = None) -> float:
        self._require_model()
        import torch

        self.model.train()
        optimizer = self._ensure_optimizer(learning_rate)
        optimizer.zero_grad(set_to_none=True)
        forward_data_store = self._run_backward_with_zero2_overlap(
            lambda: self._forward_backward(input_ids, labels=labels, loss_mask=loss_mask, forward_only=False)
        )
        self._average_data_parallel_gradients()
        self._clip_gradients()
        self._optimizer_step(optimizer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        loss = torch.zeros([], dtype=torch.float32, device=next(self.model.parameters()).device)
        contributors = torch.zeros([], dtype=torch.float32, device=loss.device)
        if forward_data_store:
            maybe_loss = forward_data_store[-1]
            if isinstance(maybe_loss, dict) and "lm loss" in maybe_loss:
                loss.copy_(maybe_loss["lm loss"].detach().float())
                contributors.fill_(1.0)
        if torch.distributed.is_initialized():
            self._all_reduce_dense_world(loss)
            self._all_reduce_dense_world(contributors)
            loss.div_(torch.clamp(contributors, min=1.0))
        return float(loss.detach().cpu())

    def save_distributed_checkpoint(
        self,
        path: str | Path,
        *,
        step: int = 0,
        include_optimizer_state: bool = True,
    ) -> Path:
        self._require_model()
        from megatron.core import dist_checkpointing

        root = Path(path)
        checkpoint_root = root / f"iter_{step:07d}"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model.sharded_state_dict(),
            "rpeg_auxiliary_heads": {
                key: value
                for key, value in self.rpeg_state_dict().items()
                if key.startswith("rpeg_value_head.") or key.startswith("rpeg_reward_head.")
            },
            "rpeg_metadata": {
                "schema_version": "rpeg.megatron_dist_checkpoint.v1",
                "world_size": self.world_size,
                "parallelism": self.config.parallelism,
                "family": self.architecture.family if self.architecture else None,
                "rope_scaling": self.architecture.rope_scaling if self.architecture else False,
                "rope_scaling_factor": self.architecture.rope_scaling_factor if self.architecture else 1.0,
                "step": step,
            },
        }
        dist_checkpointing.save(
            state,
            str(checkpoint_root),
            content_metadata=state["rpeg_metadata"],
            validate_access_integrity=True,
        )
        if include_optimizer_state:
            self._save_optimizer_checkpoint_sidecar(checkpoint_root, step=step)
        return checkpoint_root

    def load_distributed_checkpoint(self, path: str | Path, *, strict: bool = True) -> None:
        self._require_model()
        from megatron.core import dist_checkpointing

        aux_template = {
            key: value
            for key, value in self.rpeg_state_dict().items()
            if key.startswith("rpeg_value_head.") or key.startswith("rpeg_reward_head.")
        }
        template = {
            "model": self.model.sharded_state_dict(),
            "rpeg_auxiliary_heads": aux_template,
        }
        loaded = dist_checkpointing.load(
            template,
            str(path),
            validate_access_integrity=True,
            strict="return_all" if strict else False,
        )
        if isinstance(loaded, tuple):
            loaded, missing, unexpected = loaded
            if strict and (missing or unexpected):
                raise RuntimeConfigError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        state = dict(loaded["model"])
        state.update(dict(loaded.get("rpeg_auxiliary_heads") or {}))
        self.load_rpeg_state_dict(state, strict=strict)
        self._load_optimizer_checkpoint_sidecar(Path(path), strict=strict)

    def _optimizer_checkpoint_sidecar_path(self, checkpoint_root: str | Path) -> Path:
        return Path(checkpoint_root) / f"rpeg_optimizer_rank{int(self.rank):05d}.pt"

    def _save_optimizer_checkpoint_sidecar(self, checkpoint_root: str | Path, *, step: int) -> None:
        optimizer = self._ensure_optimizer()
        import torch

        assert self.config is not None
        payload = {
            "schema_version": "rpeg.megatron_optimizer_checkpoint.v1",
            "rank": int(self.rank),
            "world_size": int(self.world_size),
            "parallelism": tuple(int(x) for x in self.config.parallelism),
            "step": int(step),
            "distributed_optimizer": self._distributed_optimizer_mode(),
            "state_dict": optimizer.state_dict(),
        }
        path = self._optimizer_checkpoint_sidecar_path(checkpoint_root)
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        try:
            torch.save(payload, tmp)
            tmp.replace(path)
        finally:
            if tmp.exists():
                tmp.unlink()

    def _load_optimizer_checkpoint_sidecar(self, checkpoint_root: str | Path, *, strict: bool) -> None:
        path = self._optimizer_checkpoint_sidecar_path(checkpoint_root)
        if not path.exists():
            if strict:
                raise RuntimeConfigError(f"missing optimizer checkpoint sidecar in {checkpoint_root}")
            return
        import torch

        assert self.config is not None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != "rpeg.megatron_optimizer_checkpoint.v1":
            raise RuntimeConfigError(f"unsupported optimizer checkpoint schema in {path}")
        expected = {
            "rank": int(self.rank),
            "world_size": int(self.world_size),
            "parallelism": tuple(int(x) for x in self.config.parallelism),
        }
        actual_parallelism = tuple(int(x) for x in payload.get("parallelism", ()))
        actual = {
            "rank": int(payload.get("rank", -1)),
            "world_size": int(payload.get("world_size", -1)),
            "parallelism": actual_parallelism,
        }
        mismatches = {key: (actual[key], value) for key, value in expected.items() if actual[key] != value}
        if mismatches and strict:
            raise RuntimeConfigError(f"optimizer checkpoint metadata mismatch in {path}: {mismatches}")
        optimizer = self._ensure_optimizer()
        optimizer.load_state_dict(payload.get("state_dict") or {})

    def load_hf_checkpoint(self, path: str | Path, *, strict: bool = False) -> dict[str, Any]:
        self._require_model()
        import torch
        from megatron.core import parallel_state

        try:
            from safetensors.torch import load_file as load_safetensors  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeDependencyError("safetensors is required for Hugging Face checkpoint conversion") from exc

        root = Path(path)
        hf_state = {}
        safetensor_files = sorted(root.glob("*.safetensors"))
        if safetensor_files:
            for shard in safetensor_files:
                hf_state.update(load_safetensors(shard))
        else:
            for shard in sorted(root.glob("pytorch_model*.bin")):
                hf_state.update(torch.load(shard, map_location="cpu"))
        if not hf_state:
            raise RuntimeConfigError(f"no Hugging Face checkpoint shards found in {root}")

        state = self.model.state_dict()
        mapped_keys: list[str] = []
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()

        def shard(tensor, dim: int, key: str, *, stride: int = 1):
            if tp_size == 1:
                return tensor.contiguous()
            chunks = tp_size * int(stride)
            if tensor.shape[dim] % chunks != 0:
                raise RuntimeConfigError(
                    f"HF tensor {key} dimension {dim} with shape {tuple(tensor.shape)} "
                    f"is not divisible by TP={tp_size} with partition stride={stride}"
                )
            pieces = torch.chunk(tensor, chunks, dim=dim)
            return torch.cat(pieces[tp_rank::tp_size], dim=dim).contiguous()

        def install(key: str, value) -> bool:
            """
            Copy one CPU HF tensor directly into its model destination.

            Accumulating converted CUDA tensors in a mapped state dictionary doubles the local model
            shard during initialization. A 70B PP=1/TP=2 reward worker reaches 94.6 GiB before the
            final 448 MiB tensor under that scheme. copy_ performs the dtype/device transfer into
            already-allocated parameter storage with only the current CPU shard live.
            """
            destination = state.get(key)
            if destination is None:
                return False
            if tuple(destination.shape) != tuple(value.shape):
                raise RuntimeConfigError(
                    f"HF tensor {key} has shape {tuple(value.shape)}, "
                    f"expected {tuple(destination.shape)}"
                )
            with torch.no_grad():
                destination.copy_(value, non_blocking=False)
            mapped_keys.append(key)
            return True

        def local_layer_to_global() -> dict[int, int]:
            decoder = getattr(self.model, "decoder", None)
            layers = getattr(decoder, "layers", None)
            result = {}
            if layers is None:
                return result
            for local_idx, layer in enumerate(layers):
                layer_number = getattr(layer, "layer_number", None)
                result[local_idx] = int(layer_number) - 1 if layer_number is not None else local_idx
            return result

        embedding_shard = None
        if "model.embed_tokens.weight" in hf_state:
            embedding_shard = shard(
                hf_state["model.embed_tokens.weight"], 0, "model.embed_tokens.weight"
            )
            install("embedding.word_embeddings.weight", embedding_shard)
        if "model.norm.weight" in hf_state:
            install("decoder.final_layernorm.weight", hf_state["model.norm.weight"])
        if "lm_head.weight" in hf_state:
            install(
                "output_layer.weight",
                shard(hf_state["lm_head.weight"], 0, "lm_head.weight"),
            )
        elif embedding_shard is not None:
            install("output_layer.weight", embedding_shard)
        assert self.architecture is not None
        layer_map = local_layer_to_global()
        for local_layer, global_layer in layer_map.items():
            prefix = f"model.layers.{global_layer}"
            mp = f"decoder.layers.{local_layer}"
            for hf_name, mg_name in [
                ("input_layernorm.weight", "input_layernorm.weight"),
                ("post_attention_layernorm.weight", "pre_mlp_layernorm.weight"),
            ]:
                key = f"{prefix}.{hf_name}"
                if key in hf_state:
                    install(f"{mp}.{mg_name}", hf_state[key])
            for hf_name, mg_name, dim in [
                ("self_attn.o_proj.weight", "self_attention.linear_proj.weight", 1),
                ("mlp.down_proj.weight", "mlp.linear_fc2.weight", 1),
            ]:
                key = f"{prefix}.{hf_name}"
                if key in hf_state:
                    install(f"{mp}.{mg_name}", shard(hf_state[key], dim, key))
            q_key = f"{prefix}.self_attn.q_proj.weight"
            k_key = f"{prefix}.self_attn.k_proj.weight"
            v_key = f"{prefix}.self_attn.v_proj.weight"
            if all(key in hf_state for key in (q_key, k_key, v_key)):
                q, k, v = hf_state[q_key], hf_state[k_key], hf_state[v_key]
                head_dim = self.architecture.kv_channels
                num_heads = self.architecture.num_attention_heads
                num_query_groups = self.architecture.num_query_groups
                heads_per_group = num_heads // num_query_groups
                q = q.view(num_query_groups, heads_per_group, head_dim, q.shape[1])
                k = k.view(num_query_groups, 1, head_dim, k.shape[1])
                v = v.view(num_query_groups, 1, head_dim, v.shape[1])
                qkv = torch.cat([q, k, v], dim=1).reshape(-1, q.shape[-1])
                install(
                    f"{mp}.self_attention.linear_qkv.weight",
                    shard(qkv, 0, f"{prefix}.self_attn.qkv.weight"),
                )
            q_bias_key = f"{prefix}.self_attn.q_proj.bias"
            k_bias_key = f"{prefix}.self_attn.k_proj.bias"
            v_bias_key = f"{prefix}.self_attn.v_proj.bias"
            if all(key in hf_state for key in (q_bias_key, k_bias_key, v_bias_key)):
                q, k, v = hf_state[q_bias_key], hf_state[k_bias_key], hf_state[v_bias_key]
                head_dim = self.architecture.kv_channels
                num_heads = self.architecture.num_attention_heads
                num_query_groups = self.architecture.num_query_groups
                heads_per_group = num_heads // num_query_groups
                q = q.view(num_query_groups, heads_per_group, head_dim)
                k = k.view(num_query_groups, 1, head_dim)
                v = v.view(num_query_groups, 1, head_dim)
                qkv = torch.cat([q, k, v], dim=1).reshape(-1)
                install(
                    f"{mp}.self_attention.linear_qkv.bias",
                    shard(qkv, 0, f"{prefix}.self_attn.qkv.bias"),
                )
            fc1_keys = [f"{prefix}.mlp.{name}_proj.weight" for name in ("gate", "up")]
            if all(key in hf_state for key in fc1_keys):
                fc1 = torch.cat([hf_state[key] for key in fc1_keys], dim=0)
                install(
                    f"{mp}.mlp.linear_fc1.weight",
                    shard(
                        fc1,
                        0,
                        f"{prefix}.mlp.gate_up_proj.weight",
                        stride=2,
                    ),
                )
        del embedding_shard
        loaded_auxiliary_heads = self._load_hf_auxiliary_heads(hf_state)
        return {
            "mapped_tensors": len(mapped_keys),
            "loaded_auxiliary_heads": list(loaded_auxiliary_heads),
        }

    def shutdown(self) -> None:
        try:
            import torch
            from megatron.core import parallel_state
        except (ModuleNotFoundError, RuntimeError):
            self._clear_static_kv_serving_cache()
            return
        if self._model_parallel_initialized and parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        self._destroy_model_parallel_process_groups()
        if self._owns_process_group and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        self._model_parallel_initialized = False
        self._owns_process_group = False
        self._clear_static_kv_serving_cache()
