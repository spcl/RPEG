# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Authors: Leonard Schmidt
#          Pierre Pang

import math

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from core.types import Allocation, Role
from model.hardware import Hardware
from model.models import ModelSuite, Model
from model.workload import Workload
from passes.lower import ExecNode, ExecutableRPEG, ExecNodeType


def training_activation_bytes_per_microbatch(
    node: ExecNode,
    suite: ModelSuite,
    workload: Workload,
    *,
    activation_checkpointing: bool,
) -> float:
    """
    Retained activation footprint of one local training microbatch.

    :param node: Vertex in the executable RPEG.
    :type node: ExecNode
    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param workload: Workload settings.
    :type workload: Workload
    :param activation_checkpointing: Flag, indicating whether activation checkpointing is enabled.
    :type activation_checkpointing: bool
    :return: Number of Bytes for the activations of a single local training microbatch.
    :rtype: float
    :raises ValueError: If the model cannot be properly accessed.
    """
    model = node.model
    if model is None or node.pi is None or node.mb is None:
        raise ValueError(f"{node.id}: training activation accounting requires model, pi, and mb")
    L, d, d_ff, sb = (
        suite.specs[model].L,
        suite.specs[model].d,
        suite.specs[model].d_ff,
        suite.specs[model].dtype_bytes,
    )
    B, K, S, T = workload.B, workload.K, workload.S, workload.T
    dp, pp, tp = node.pi
    b_local = B * K / (dp * node.mb)
    L_pp = math.ceil(L / pp)
    layer_peak = max(2 * d + 6 * d / tp, 2 * d + 2 * d_ff / tp)
    if activation_checkpointing:
        c = max(1, math.isqrt(L_pp))
        n_ckpt = math.ceil(L_pp / c)
        per_token = n_ckpt * d + c * layer_peak
    else:
        per_token = L_pp * layer_peak
    return sb * b_local * (S + T) * per_token


def _fused_training_activation_peak(
    members: List[ExecNode],
    mem_model: "NodeMemoryModel",
) -> float:
    """
    Determine the maximum of the combined live activation Bytes during a runtime fused schedule.

    :param members: List of vertices in the schedule.
    :type members: List[ExecNode]
    :param mem_model: Memory model for the graph nodes.
    :type mem_model: "NodeMemoryModel"
    :return: Number of Bytes for the activations of a single local training microbatch.
    :rtype: float
    """
    from runtime.fused_pipeline import FusedPipelineModel, build_fused_pipeline_schedule

    by_vertex = {
        member.op.template_vertex.id: member
        for member in members
        if member.op is not None
    }
    if set(by_vertex) != {"t_pi", "t_V"}:
        return sum(mem_model._mem_activations(member) for member in members)
    actor = by_vertex["t_pi"]
    critic = by_vertex["t_V"]
    if actor.pi is None or actor.mb is None or critic.pi != actor.pi or critic.mb != actor.mb:
        return sum(mem_model._mem_activations(member) for member in members)
    _dp, pp, _tp = actor.pi
    work_scale = max(
        float(mem_model.suite.n_params(actor.model)),
        float(mem_model.suite.n_params(critic.model)),
        1.0,
    )

    def model(name: str, node: ExecNode, *, reverse: bool) -> FusedPipelineModel:
        """
        Create a model that can be used inside a fused pipeline.

        :param name: Role name.
        :type name: str
        :param node: Vertex in the executable RPEG.
        :type node: ExecNode
        :param reverse: Flag, indicating the pipeline direction.
        :type reverse: bool
        :return: Model for fused pipeline.
        :rtype: FusedPipelineModel
        """
        work = float(mem_model.suite.n_params(node.model)) / work_scale
        return FusedPipelineModel.uniform(
            name,
            pipeline_stages=int(pp),
            microbatches=int(node.mb),
            forward_s=work,
            backward_s=2.0 * work,
            activation_bytes=int(training_activation_bytes_per_microbatch(
                node,
                mem_model.suite,
                mem_model.workload,
                activation_checkpointing=mem_model.activation_checkpointing,
            )),
            reverse=reverse,
        )

    schedule = build_fused_pipeline_schedule((
        model("actor", actor, reverse=False),
        model("critic", critic, reverse=True),
    ))
    return float(max(schedule.peak_activation_bytes, default=0))


@dataclass(frozen=True)
class MemoryPolicy:
    """
    Data class for handling practical overheads during the runtime.
    """
    optimizer_stage: int = 2
    activation_checkpointing: bool = True
    runtime_overhead_fraction: float = 0.0
    worker_overhead_bytes: float = 0.0
    semantic_activation_cache_bytes: float = 0.0
    max_worker_groups_per_gpu: int = 0

    def __post_init__(self):
        """
        Validation after class initialization.

        :raises ValueError: If one of the parameter is not valid.
        """
        if self.optimizer_stage not in {0, 1, 2}:
            raise ValueError("optimizer_stage must be 0, 1, or 2")
        if not 0.0 <= self.runtime_overhead_fraction < 1.0:
            raise ValueError("runtime_overhead_fraction must be in [0, 1)")
        if self.worker_overhead_bytes < 0.0:
            raise ValueError("worker_overhead_bytes must be >= 0")
        if self.semantic_activation_cache_bytes < 0.0:
            raise ValueError("semantic_activation_cache_bytes must be >= 0")
        if self.max_worker_groups_per_gpu < 0:
            raise ValueError("max_worker_groups_per_gpu must be >= 0")


def optimizer_stage(name: str) -> int:
    """
    Determine numeric optimizer stage based on the supplied name.

    :param name: Name of the stage.
    :type name: str
    :return: Numeric equivalent for the optimizer stage.
    :rtype: int
    :raises ValueError: If the stage name is not supported.
    """
    normalized = str(name or "none").strip().lower().replace("-", "_")
    if normalized in {"none", "adamw"}:
        return 0
    if normalized in {"zero1", "zero_1", "zero_stage_1", "stage1"}:
        return 1
    if normalized in {"zero2", "zero_2", "zero_stage_2", "stage2"}:
        return 2
    raise ValueError(f"unsupported distributed optimizer mode {name!r}")


@dataclass(frozen=True)
class NodeMemoryModel:
    """
    Data class for the memory model of the nodes in the executable RPEG.
    """
    hw: Hardware
    suite: ModelSuite
    workload: Workload
    eps_rt: float = 0.0 # runtime overhead fraction (0 = use full GPU memory)
    trainable: FrozenSet[Model] = frozenset() # trainable model set; empty = old behavior (all persistent)
    shadow_models: FrozenSet[Model] = frozenset() # subset of trainable with tau > 0 (persistent shadow copies)
    optimizer_stage: int = 2 # 0 = replicated Adam, 1 = optimizer sharded, 2 = optimizer+gradients sharded
    worker_overhead_bytes: float = 0.0
    semantic_activation_cache_bytes: float = 0.0
    max_worker_groups_per_gpu: int = 0
    activation_checkpointing: bool = True

    @classmethod
    def from_policy(
        cls,
        hw: Hardware,
        suite: ModelSuite,
        workload: Workload,
        policy: MemoryPolicy,
        *,
        trainable: FrozenSet[Model] = frozenset(),
        shadow_models: FrozenSet[Model] = frozenset(),
    ) -> "NodeMemoryModel":
        """
        Initialization from memory policy.

        :param hw: Hardware settings.
        :type hw: Hardware
        :param suite: Model hyperparameters.
        :type suite: ModelSuite
        :param workload: Workload settings.
        :type workload: Workload
        :param policy: Memory policy settings.
        :type policy: MemoryPolicy
        :param trainable: Trainable model set. Defaults to an empty frozen set.
        :type trainable: FrozenSet[Model]
        :param shadow_models: Subset of trainable with tau > 0 (persistent shadow copies). Defaults
                              to an empty frozen set.
        :type shadow_models: FrozenSet[Model]
        """
        return cls(
            hw,
            suite,
            workload,
            eps_rt=policy.runtime_overhead_fraction,
            trainable=trainable,
            shadow_models=shadow_models,
            optimizer_stage=policy.optimizer_stage,
            worker_overhead_bytes=policy.worker_overhead_bytes,
            semantic_activation_cache_bytes=policy.semantic_activation_cache_bytes,
            max_worker_groups_per_gpu=policy.max_worker_groups_per_gpu,
            activation_checkpointing=policy.activation_checkpointing,
        )

    def __post_init__(self) -> None:
        """
        Validation after class initialization and memorization cache initialization.

        :raises ValueError: If optimizer stage parameter is not within a valid range.
        """
        if self.optimizer_stage not in {0, 1, 2}:
            raise ValueError("optimizer_stage must be 0, 1, or 2")
        # Memorization cache for _w_stage_max(model, pp). This is a pure function of (model, pp) and
        # model-specific constants, so the result is immutable for the lifetime of this object.
        # Using object.__setattr__ because the dataclass is frozen.
        object.__setattr__(self, '_w_stage_max_cache', {})

    # Public functions
    def is_locally_feasible(self, node: ExecNode) -> bool: 
        """
        Check whether the execution of the vertex is feasible on a single GPU.

        :param node: Vertex in the executable RPEG.
        :type node: ExecNode
        :return: Flag, whether the execution of the vertex is feasible.
        :rtype: bool
        """
        cap = self._eff_capacity()
        return self.peak_bytes(node) <= cap  
    
    def peak_bytes(self, node: ExecNode) -> float:
        """
        Return the peak amount of memory used for the execution of the vertex.

        :param node: Vertex in the executable RPEG.
        :type node: ExecNode
        :return: Number of Bytes which are used at most during the execution of the vertex.
        :rtype: float
        """
        role = node.op.template_vertex.kind.role
        weights = self._mem_weights(node)
        if role == Role.TRAINING:
            return weights + self._mem_optimizer(node) + self._mem_gradients(node) + self._mem_activations(node)
        if role == Role.GENERATION:
            return weights + self._mem_kv_cache(node) + self._mem_activations_inference(node)
        return weights + self._mem_activations_inference(node) # ROLE.INFERENCE

    # Private helpers: parameter accounting
    def _eff_capacity(self) -> float:
        """
        Return the effective memory capacity of a GPU (reduced by an user-chosen epsilon).

        :return: Effective GPU memory capacity.
        :rtype: float
        """
        return (1 - self.eps_rt) * self.hw.memory_per_gpu

    def _w_layer(self, model: Model) -> int:
        """
        Return the number of layers for the model.

        :param model: Input model.
        :type model: Model
        :return: Number of model layers.
        :rtype: int
        """
        d, d_ff = self.suite.specs[model].d, self.suite.specs[model].d_ff
        return 4*d**2 + 2*d*d_ff 

    def _w_stage(self, model: Model, p: int, pp: int) -> Tuple[int, int]:
        """
        Return (tp_split_params, non_split_params) for pipeline stage p.

        Transformer layer weights are split across TP ranks (ColumnParallel/RowParallel). Embedding
        and output heads are not TP-split (full copy on each TP rank).

        :param model: Input model.
        :type model: Model
        :param p: Pipeline stage index.
        :type p: int
        :param pp: Pipeline parallelism degree.
        :type pp: int
        :return: Tuple of the number of tp_split and non_split parameters for the pipeline stage.
        :rtype: Tuple[int, int]
        :raises ValueError: If model type can not be processed.
        """
        L, V, d = self.suite.specs[model].L, self.suite.specs[model].V, self.suite.specs[model].d
        Embd = V * d
        Head = d
        tp_split = math.ceil(L / pp) * self._w_layer(model)

        if model == Model.ACTOR or model == Model.REFERENCE:
            non_split = (Embd if p == 1 else 0) + (Embd if p == pp else 0)
        elif model == Model.CRITIC or model == Model.REWARD:
            non_split = (Embd if p == 1 else 0) + (Head if p == pp else 0)
        elif model == Model.AC_SHARE:
            non_split = (Embd if p == 1 else 0) + (Embd if p == pp else 0) + (Head if p == pp else 0)
        else:
            raise ValueError

        return tp_split, non_split

    def _w_stage_max(self, model: Model, pp: int) -> Tuple[int, int]:
        """
        Return (max_tp_split, max_non_split) over all pipeline stages. Uses a cache for repeated
        calls.

        :param model: Input model.
        :type model: Model
        :param pp: Pipeline parallelism degree.
        :type pp: int
        :return: Tuple of the maximum number of tp_split and non_split parameters.
        :rtype: Tuple[int, int]
        """
        key = (model, pp)
        cache = self._w_stage_max_cache
        if key in cache:
            return cache[key]
        stages = [self._w_stage(model, p, pp) for p in range(1, pp + 1)]
        # Take the stage with the largest total footprint
        best = max(stages, key=lambda s: s[0] + s[1])
        cache[key] = best
        return best

    def _w_shard(self, node: ExecNode) -> int:
        """
        Return total number of weight parameters per GPU (one shard). Uses a cache for repeated
        calls.

        Transformer layers are split by TP; embedding/heads are not split.

        :param node: Vertex in the executable RPEG.
        :type node: ExecNode
        :return: Number of weight parameters per GPU.
        :rtype: int
        """
        model, pp, tp = node.model, node.pi[1], node.pi[2]
        key = (model, pp, tp)
        cache = self._w_stage_max_cache
        if key in cache:
            return cache[key]
        tp_split, non_split = self._w_stage_max(model=model, pp=pp)
        result = math.ceil(tp_split / tp) + non_split
        cache[key] = result
        return result

    # private helpers: memory components 
    def _mem_weights(self, node: ExecNode) -> float:
        """
        Return the number of Bytes per shard (= GPU) for the model weights.

        :param node: Vertex in the executable RPEG.
        :type node: ExecNode
        :return: Number of Bytes for the model weights per shard.
        :rtype: float
        """
        return self._w_shard(node) * self.suite.specs[node.model].dtype_bytes

    def _mem_optimizer(self, node: ExecNode) -> float:
        """
        Return the optimizer memory amount per shard (= GPU). ZeRO-1/2: optimizer state is sharded
        across DP ranks within each (p, t) communicator. Each rank holds 1/dp of the FP32 master
        weights + first moment + second moment. Stage 0 replicates it.

        :param node: Vertex in the executable RPEG.
        :type node: ExecNode
        :return: Number of Bytes for the optimizer memory per shard.
        :rtype: float
        """
        dp = node.pi[0] if self.optimizer_stage >= 1 else 1
        return 12 * self._w_shard(node) / dp

    def _mem_gradients(self, node: ExecNode) -> float:
        """
        Return the memory amount for the gradients per shard (= GPU).

        PyTorch accumulates gradients in the same dtype as the parameters (bf16). ZeRO-2: after
        ReduceScatter, each DP rank holds only 1/dp of the averaged gradients. Peak gradient memory
        = sb * w_shard / dp. Stages 0 and 1 retain a full gradient shard per rank.

        :param node: Vertex in the executable RPEG.
        :type node: ExecNode
        :return: Number of Bytes for the memory of the gradients per shard.
        :rtype: float
        """
        dp = node.pi[0] if self.optimizer_stage >= 2 else 1
        sb = self.suite.specs[node.model].dtype_bytes
        return sb * self._w_shard(node) / dp

    def _mem_activations(self, node: ExecNode) -> float:
        """
        Return the memory amount for the activations per shard (= GPU) during training.

        Activation checkpointing (selective / segment-based): Instead of retaining all L_pp =
        ceil(L/pp) layer activations, only checkpoint every c layers. During the backward pass,
        non-checkpointed layers are recomputed from the nearest checkpoint.

        Memory retained = ceil(L_pp / c) checkpoints + c recomputation buffer
        Optimal C = floor(sqrt(L_pp)) (minimises total retained layers).

        The recomputation adds ~33% to the forward pass cost (each segment recomputed once during
        the backward pass). This is already subsumed by the standard 3× training multiplier in the
        cost model (fwd + bwd + recompute ≈ 1 + 2 + 0.33 = 3.33, approximated as 3×).

        :param node: Vertex in the executable RPEG.
        :type node: ExecNode
        :return: Number of Bytes for the memory of the activations per shard.
        :rtype: float
        """
        assert node.mb is not None and node.pi is not None
        return min(node.mb, node.pi[1]) * training_activation_bytes_per_microbatch(
            node,
            self.suite,
            self.workload,
            activation_checkpointing=self.activation_checkpointing,
        )

    def _mem_activations_inference(self, node: ExecNode) -> float:
        """
        Return the memory amount for the inference activations per shard (= GPU).

        Unlike training, inference does not retain activations for the backward pass. Layers are
        processed sequentially — only one layer's tensors are alive at any time. However, within a
        single layer's forward pass, multiple intermediate tensors coexist in GPU memory.

        Key insight: the residual stream x is full d (not split by TP). TP splitting only happens
        inside ColumnParallel/RowParallel layers. So each layer has both full-d and split tensors
        alive simultaneously:

            Attention phase (peak):
             x (residual)   : (b, seq, d)        →  d           [full, not split]
             RMSNorm(x)     : (b, seq, d)        →  d           [full, not split]
             QKV projection : (b, seq, 3·d/tp)   →  3·d/tp      [TP-split]
             RoPE q,k       : (b, seq, 2·d/tp)   →  2·d/tp      [TP-split]
             attn output    : (b, seq, d/tp)      →  d/tp       [TP-split]
             ─────────────────────────────────────────────────────────────
             peak = 2·d + 6·d/tp per token

            FFN phase (peak):
             x (residual)   : (b, seq, d)        →  d           [full, not split]
             RMSNorm(x)     : (b, seq, d)        →  d           [full, not split]
             up projection  : (b, seq, d_ff/tp)  →  d_ff/tp     [TP-split]
             silu(up)       : (b, seq, d_ff/tp)  →  d_ff/tp     [TP-split]
             ─────────────────────────────────────────────────────────────
             peak = 2·d + 2·d_ff/tp per token

           LM head phase (last PP stage only):
             During lm_head(x), both the input (b, seq, d) and the output (b, seq, V) are alive
             simultaneously for the duration of the matrix multiply. The LM head is not TP-split —
             it is a full (d → V) linear on each rank (see _w_stage: embedding/head are non_split).
             For large vocabularies (V >> d), this dominates the activation peak.
             ─────────────────────────────────────────────────────────────
             peak = d + V  per token

        The overall peak is max(attn, ffn, lm_head) elements per token.

        Pipeline in-flight count:
           Inference/generation process microbatches sequentially — each stage handles one
           microbatch at a time (send, then receive next). So only 1 microbatch's activations are
           alive per stage at any point. This differs from training's 1F1B schedule where the warmup
           phase fills min(mb, pp) microbatches before backward passes begin.

        :param node: Vertex in the executable RPEG.
        :type node: ExecNode
        :return: Number of Bytes for the memory of inference activations per shard.
        :rtype: float
        """
        model = node.model
        spec = self.suite.specs[model]
        d, d_ff, V, sb = spec.d, spec.d_ff, spec.V, spec.dtype_bytes
        B, K, S, T = self.workload.B, self.workload.K, self.workload.S, self.workload.T
        dp, _, tp = node.pi
        mb = node.mb if node.mb is not None else 1
        b_local = B * K / (dp * mb)
        seq_len = S + T if node.op.template_vertex.kind.role != Role.GENERATION else S
        attn_peak = 2 * d + 6 * d / tp # full-d residual + split QKV/RoPE/attn
        ffn_peak  = 2 * d + 2 * d_ff / tp # full-d residual + split up + silu
        lm_head_peak = d + V # input (b,seq,d) + output (b,seq,V) during matmul
        peak_per_token = max(attn_peak, ffn_peak, lm_head_peak)
        return sb * b_local * seq_len * peak_per_token

    def _mem_kv_cache(self, node: ExecNode) -> float:
        """
        Return the memory amount for the KV cache elements per shard (= GPU).

        :param node: Vertex in the executable RPEG.
        :type node: ExecNode
        :return: Number of Bytes for the memory of the KV cache elements per shard.
        :rtype: float
        """
        model = node.model
        L, d, sb = self.suite.specs[model].L, self.suite.specs[model].d, self.suite.specs[model].dtype_bytes
        B, K, S, T = self.workload.B, self.workload.K, self.workload.S, self.workload.T
        dp, pp, tp = node.pi
        return 2 * sb * (B / dp) * K * (S + T) * (L / pp) * (d / tp)

    def _transient_bytes(self, node: ExecNode) -> float:
        """
        Return the amount of transient memory, which is only active while the operator is executing.

        :param node: Vertex in the executable RPEG.
        :type node: ExecNode
        :return: Number of Bytes for the transient memory.
        :rtype: float
        """
        role = node.op.template_vertex.kind.role
        if role == Role.TRAINING:
            return self._mem_gradients(node) + self._mem_activations(node)
        if role == Role.GENERATION:
            return self._mem_kv_cache(node) + self._mem_activations_inference(node)
        return self._mem_activations_inference(node) # ROLE.INFERENCE

    def _static_bytes(self, model: Model, w_shard: int, is_trained: bool, dp: int = 1) -> float:
        """
        Return the static per-model memory footprint given a pre-computed shard size.

        :param model: Model hyperparameters.
        :type model: Model
        :param w_shard: Number of weight parameters.
        :type w_shard: int
        :param is_trained: Flag indicating whether the model is trained.
        :type is_trained: bool
        :param dp: Data parallelism degree. Defaults to 1.
        :type dp: int
        :return: Static memory footprint of the the model in Bytes.
        :rtype: float
        """
        sb = self.suite.specs[model].dtype_bytes
        if is_trained:
            opt_dp = dp if self.optimizer_stage >= 1 else 1
            grad_dp = dp if self.optimizer_stage >= 2 else 1
            return sb * w_shard + 12 * w_shard / opt_dp + sb * w_shard / grad_dp
        return sb * w_shard # weights only


# Device enumeration helper
# Cached by (origin, extent): same rationale as scheduler._alloc_cache.
_alloc_cache: Dict[Tuple, frozenset] = {}

def _alloc_devices(alloc: Allocation) -> frozenset:
    """
    Return an enumeration of all devices in the allocation. Retrieved from cache if available,
    otherwise the enumeration is computed and added to the cache.

    :param alloc: Input allocation.
    :type alloc: Allocation
    :return: Set of all devices in the allocation.
    :rtype: frozenset
    """
    key = (alloc.origin, alloc.extent)
    result = _alloc_cache.get(key)
    if result is not None:
        return result
    o, e = alloc.origin, alloc.extent
    devices = set()
    for r in range(o[0], o[0] + e[0]):
        for n in range(o[1], o[1] + e[1]):
            for g in range(o[2], o[2] + e[2]):
                devices.add((r, n, g))
    result = frozenset(devices)
    _alloc_cache[key] = result
    return result


# ── Per-device peak computation ────────────────────────────────────────────
def _device_peaks(
    executable: ExecutableRPEG,
    mem_model: NodeMemoryModel,
) -> Dict[Tuple[int, int, int], float]:
    """
    Compute peak(d) for every device that has at least one OP node.

    Worker-liveness model (when mem_model.trainable is non-empty):
    - Trainable models: ZeRO-2 optimizer with Φ(12*ws/dp) is always resident. The training worker's
      BF16 weights contribute to its operator peak.
    - Separate generation/inference workers keep their active BF16 version resident from publication
      through consumption. This applies to both migration (tau=0) and shadow (tau>0): tau changes
      version freshness, not the receiving process's physical parameter storage.
    - Frozen models (non-trainable): BF16 weights are persistent, co-resident (summed across all
      frozen models on the device).

    peak(d) = Σ_train Φ(m)/dp + Σ_serving-worker W(m) + Σ_frozen W(m)
              + max_{v in V(d)} [W_trainable(v) + A(v)]

    Fallback (trainable is empty): co-resident sum model.

    :param executable: Graph to execute.
    :type executable: ExecutableRPEG
    :param mem_model: Memory model for the graph nodes.
    :type mem_model: NodeMemoryModel
    :return: Dictionary of devices and the memory amount in Bytes.
    :rtype: Dict[Tuple[int, int, int], float]
    """
    # 1. Group OP nodes by device
    device_nodes: Dict[Tuple[int,int,int], List[ExecNode]] = defaultdict(list)
    op_nodes: List[ExecNode] = []
    for node in executable.nodes:
        if node.kind != ExecNodeType.OP:
            continue
        op_nodes.append(node)
        for alloc in node.allocs:
            for d in _alloc_devices(alloc):
                device_nodes[d].append(node)

    use_worker_liveness_model = len(mem_model.trainable) > 0

    # Precompute per-node memory costs once
    # All mem_model helpers (_w_shard, _mem_weights, _mem_gradients, …) are pure functions of
    # (node.model, node.pi, node.mb) and mem_model constants. They do not depend on which device d
    # we're computing peaks for. Precomputing avoids recomputing identical values across ~N_devices
    # iterations (e.g. 128 devices × 24 nodes = 3072 calls → 24 calls).
    _n_w_shard: Dict[int, int] = {} # id(node) → w_shard
    _n_mem_w: Dict[int, float] = {} # id(node) → _mem_weights
    _n_transient: Dict[int, float] = {} # id(node) → _transient_bytes
    # For the worker-liveness pass 2: precompute the role-specific extras (everything except
    # _mem_weights, which may be zeroed per-device).
    _n_role_extra: Dict[int, float] = {} # id(node) → role extras (no weights)
    for node in op_nodes:
        nid = id(node)
        _n_w_shard[nid] = mem_model._w_shard(node)
        _n_mem_w[nid] = mem_model._mem_weights(node)
        _n_transient[nid] = mem_model._transient_bytes(node)
        role = node.op.template_vertex.kind.role
        if role == Role.TRAINING:
            _n_role_extra[nid] = mem_model._mem_gradients(node) + mem_model._mem_activations(node)
        elif role == Role.GENERATION:
            _n_role_extra[nid] = mem_model._mem_kv_cache(node) + mem_model._mem_activations_inference(node)
        else: # Role.INFERENCE
            _n_role_extra[nid] = mem_model._mem_activations_inference(node)

    fusion_members: Dict[str, List[ExecNode]] = defaultdict(list)
    for node in op_nodes:
        if node.fusion_group is not None:
            fusion_members[node.fusion_group].append(node)
    fused_activation_peak = {
        group: _fused_training_activation_peak(members, mem_model)
        for group, members in fusion_members.items()
    }

    # 2. Per-device peak
    peaks: Dict[Tuple[int,int,int], float] = {}
    for d, nodes in device_nodes.items():
        # Runtime workers are grouped by (model, allocation). Each process owns a CUDA context and
        # backend workspaces not represented by model tensors, so count the same unique groups for
        # every device.
        worker_keys = {
            (node.model, node.allocs[0].origin, node.allocs[0].extent)
            for node in nodes
        }
        # Every PP/TP process in a worker receives one full semantic DP shard. Charge the
        # configured, runtime-enforced CUDA cache limit once per co-resident worker group,
        # independently of generic CUDA-context and library workspace overhead.
        worker_overhead = len(worker_keys) * (
            mem_model.worker_overhead_bytes
            + mem_model.semantic_activation_cache_bytes
        )
        if (
            mem_model.max_worker_groups_per_gpu > 0
            and len(worker_keys) > mem_model.max_worker_groups_per_gpu
        ):
            peaks[d] = math.inf
            continue

        if not use_worker_liveness_model:
            # Co-resident sum model
            model_info: Dict[Model, List] = {}
            for node in nodes:
                m = node.model
                w_shard = _n_w_shard[id(node)]
                is_trained = (node.op.template_vertex.kind.role == Role.TRAINING)
                dp = node.pi[0]
                if m not in model_info:
                    model_info[m] = [w_shard, is_trained, dp]
                else:
                    model_info[m][0] = max(model_info[m][0], w_shard)
                    if is_trained:
                        model_info[m][1] = True
                        model_info[m][2] = dp # use dp from training node
                    # else: keep existing dp (from training node if present)

            static = sum(
                mem_model._static_bytes(m, info[0], info[1], info[2])
                for m, info in model_info.items()
            )
            fused_transients: Dict[str, float] = defaultdict(float)
            transient = 0.0
            for node in nodes:
                node_transient = _n_transient[id(node)]
                if node.fusion_group is not None:
                    fused_transients[node.fusion_group] += (
                        node_transient - mem_model._mem_activations(node)
                    )
                else:
                    transient = max(transient, node_transient)
            if fused_transients:
                for group in fused_transients:
                    fused_transients[group] += fused_activation_peak[group]
                transient = max(transient, max(fused_transients.values()))
            peaks[d] = worker_overhead + static + transient
            continue

        # Worker-liveness model
        # Persistent baseline: optimizer state on training devices, active BF16 weights in separate
        # serving workers, and frozen model BF16 weights (loaded once, never evicted).
        #
        # A training worker's BF16 module is accounted in the maximum active operator. A distinct
        # long-lived generation/inference worker cannot be treated that way: direct publication
        # installs its next version before the consumer is dispatched, so it can coexist with
        # another colocated worker's optimizer and layout-switch peak. Count one active shard for
        # every distinct serving worker, independent of tau.
        #
        # Frozen exception: frozen model weights are co-resident (summed, not maxed). They are
        # read-only, needed every K_eval steps, and cheap to keep resident.
        #
        # Device roles:
        #   Training device: Φ persistent, BF16 W transient per-op
        #   Serving worker : BF16 W persistent from publication to consumer
        #   Frozen model   : BF16 W persistent (co-resident, summed)

        # Pass 1: identify persistent state per model on this device
        optimizer_shards: Dict[Model, Tuple[int, int]] = {} # model -> (max_w_shard, dp)
        training_worker_keys: Dict[Model, set[Tuple]] = defaultdict(set)
        for node in nodes:
            if (
                node.model in mem_model.trainable
                and node.op.template_vertex.kind.role == Role.TRAINING
            ):
                training_worker_keys[node.model].add(
                    (node.model, node.allocs[0].origin, node.allocs[0].extent)
                )
        serving_shards: Dict[Tuple, int] = {} # worker key -> max active BF16 shard
        frozen_shards: Dict[Model, float] = {} # model -> max BF16 weight bytes (frozen, co-resident)

        for node in nodes:
            m = node.model
            w_shard = _n_w_shard[id(node)]
            is_training_op = (node.op.template_vertex.kind.role == Role.TRAINING)

            if m in mem_model.trainable:
                if is_training_op:
                    dp = node.pi[0]
                    if m not in optimizer_shards or w_shard > optimizer_shards[m][0]:
                        optimizer_shards[m] = (w_shard, dp)
                else:
                    worker_key = (m, node.allocs[0].origin, node.allocs[0].extent)
                    if worker_key not in training_worker_keys[m]:
                        serving_shards[worker_key] = max(
                            serving_shards.get(worker_key, 0),
                            w_shard,
                        )
            else:
                # Frozen model: BF16 weights are persistent (co-resident). Track max shard size per
                # model (different ops may have different pi → different w_shard; take the largest).
                frozen_shards[m] = max(frozen_shards.get(m, 0.0), _n_mem_w[id(node)])

        # Persistent baseline: optimizer state on training devices, BF16 weights in distinct serving
        # workers, frozen BF16 weights (summed).
        persistent_static = 0.0
        for m, (ws, dp) in optimizer_shards.items():
            opt_dp = dp if mem_model.optimizer_stage >= 1 else 1
            persistent_static += 12 * ws / opt_dp
        for worker_key, ws in serving_shards.items():
            sb = mem_model.suite.specs[worker_key[0]].dtype_bytes
            persistent_static += sb * ws # serving weights
        for m, w_bytes in frozen_shards.items():
            persistent_static += w_bytes # frozen weights (co-resident)

        # Pass 2: per-operator transient (BF16 W + role extras)
        # Separate serving-worker and frozen weights are already persistent → skip.
        max_operator = 0.0
        fused_operator_costs: Dict[str, float] = defaultdict(float)
        for node in nodes:
            nid = id(node)
            m = node.model

            worker_key = (m, node.allocs[0].origin, node.allocs[0].extent)
            if worker_key in serving_shards or m in frozen_shards:
                op_cost = _n_role_extra[nid] # weights already in persistent baseline
            else:
                op_cost = _n_mem_w[nid] + _n_role_extra[nid]

            if node.fusion_group is not None:
                # Members of an intra-stage actor/critic group execute concurrently on the same
                # pipeline. Their model weights, gradients and exact schedule-live activations
                # coexist.
                fused_operator_costs[node.fusion_group] += (
                    op_cost - mem_model._mem_activations(node)
                )
            else:
                max_operator = max(max_operator, op_cost)

        if fused_operator_costs:
            for group in fused_operator_costs:
                fused_operator_costs[group] += fused_activation_peak[group]
            max_operator = max(max_operator, max(fused_operator_costs.values()))

        peaks[d] = worker_overhead + persistent_static + max_operator

    return peaks


# ── Public feasibility predicates ──────────────────────────────────────────
def mem_feasible(executable: ExecutableRPEG, mem_model: NodeMemoryModel) -> bool:
    """
    Returns whether the executable RPEG is feasible memory-wise, i.e., each operator stays within
    the effective memory limit.

    :param executable: Graph to execute.
    :type executable: ExecutableRPEG
    :param mem_model: Memory model for the graph nodes.
    :type mem_model: NodeMemoryModel
    :return: Flag, whether the executable RPEG is memory feasible.
    :rtype: bool
    """
    cap = mem_model._eff_capacity()
    for d, peak in _device_peaks(executable, mem_model).items():
        if peak > cap:
            return False
    return True


def mem_worst_violation(
    executable: ExecutableRPEG,
    mem_model: NodeMemoryModel,
) -> Optional[Tuple[Tuple[int,int,int], float, float]]:
    """
    Returns (device, peak_bytes, capacity_bytes) for the device with the highest peak/capacity
    ratio, or None if all devices are feasible.

    :param executable: Graph to execute.
    :type executable: ExecutableRPEG
    :param mem_model: Memory model for the graph nodes.
    :type mem_model: NodeMemoryModel
    :return: Tuple of device, highest peak, maximum effective capacity for highest peak device or
             None if all devices are feasible.
    :rtype: Optional[Tuple[Tuple[int,int,int], float, float]]
    """
    cap = mem_model._eff_capacity()
    worst_dev, worst_peak = None, 0.0
    for d, peak in _device_peaks(executable, mem_model).items():
        if peak > cap and peak > worst_peak:
            worst_dev, worst_peak = d, peak
    if worst_dev is None:
        return None
    return worst_dev, worst_peak, cap
