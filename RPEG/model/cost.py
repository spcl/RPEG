# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Authors: Leonard Schmidt
#          Pierre Pang

import math

from dataclasses import dataclass
from typing import NamedTuple

from core.types import Allocation, Role, Collective
from model.hardware import Hardware
from model.models import ModelSuite, Model
from model.workload import Workload
from passes.lower import ExecNode, ExecNodeType


class OpTimeBreakdown(NamedTuple):
    """
    Time breakdown for an operator node in seconds. Its sum is equal to time(node).
    """
    gemm: float # GEMM compute (no bubble)
    bubble: float # pipeline bubble overhead
    intra_comm: float # TP AllReduce + PP P2P + DP collectives
    non_gemm: float # RMSNorm, RoPE, FlashAttn, SiLU, residuals
    optimizer_step: float # Adam parameter update (training only)


@dataclass(frozen=True)
class NodeCostModel:
    """
    Data class for a graph node-specific cost model.
    """
    hw: Hardware
    suite: ModelSuite
    workload: Workload
    optimizer_stage: int = 2 # 0 = replicated Adam, 1 = optimizer sharded, 2 = optimizer+gradients sharded
    activation_checkpointing: bool = True

    def __post_init__(self):
        """
        Validation after class initialization.

        :raises ValueError: If one of the parameter is not valid.
        """
        if self.optimizer_stage not in {0, 1, 2}:
            raise ValueError("optimizer_stage must be 0, 1, or 2")
        for name in (
            "op_generation_scale",
            "op_inference_scale",
            "op_training_scale",
            "weight_comm_scale",
            "reshard_scale",
        ):
            value = float(getattr(self.hw, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"hardware.{name} must be finite and > 0")
        overhead = float(self.hw.activation_comm_overhead_s)
        if not math.isfinite(overhead) or overhead < 0.0:
            raise ValueError("hardware.activation_comm_overhead_s must be finite and >= 0")

    def time(self, node: ExecNode) -> float:
        """
        Wrapper function that selects the function to call for the node in question based on its
        type.

        Implements a time map function that maps each node to an execution time in seconds.

        :param node: Node to compute the execution time for.
        :type node: ExecNode
        :return: Execution time of the node (in seconds).
        :rtype: float
        :raises ValueError: If the node type is not supported.
        """
        if node.kind == ExecNodeType.OP:
            return self._op_time(node)

        if node.kind == ExecNodeType.COMM:
            return self._comm_time(node)

        if node.kind == ExecNodeType.RESHARD:
            return self._reshard_time(node)

        raise ValueError(f"NodeCostModel.time: unknown ExecNodeType {node.kind}")

    def time_breakdown(self, node: ExecNode) -> OpTimeBreakdown:
        """
        Decompose an operator node's time into components. For analysis only. Returns an
        OpTimeBreakdown whose fields sum to self.time(node). Only valid for OP nodes.

        :param node: Node to compute the execution time breakdown for.
        :type node: ExecNode
        :return: Execution time breakdown of the node (in seconds).
        :rtype: OpTimeBreakdown
        :raises ValueError: If it is not an operator node.
        """
        if node.kind != ExecNodeType.OP:
            raise ValueError("time_breakdown is only valid for OP nodes")

        role = node.op.template_vertex.kind.role
        pd, pp, pt = node.pi
        mb_eff = self._effective_mb(node)
        F_eff = self._effective_flops(node)

        gemm = self._work_cost(node) / (F_eff * pd * pp * pt)
        bubble = gemm * (pp - 1) / mb_eff
        intra_comm = self._intra_comm_cost(node)
        non_gemm = self._non_gemm_overhead(node)

        opt_step = 0.0
        if role == Role.TRAINING:
            model = node.model
            L, d, d_ff, V = (self.suite.specs[model].L, self.suite.specs[model].d,
                             self.suite.specs[model].d_ff, self.suite.specs[model].V)
            params = L * (4*d**2 + 2*d*d_ff) + V*d
            optimizer_dp = pd if self.optimizer_stage >= 1 else 1
            params_per_rank = params / (optimizer_dp * pp * pt)
            opt_step = (params_per_rank * 18) / self.hw.memory_bandwidth

        scale = self._op_time_scale(role)
        return OpTimeBreakdown(*(
            component * scale
            for component in (gemm, bubble, intra_comm, non_gemm, opt_step)
        ))

    # Auxiliary functions
    def _effective_batch(self, node: ExecNode) -> float:
        """
        Return the effective batch size depending on the workload type: n_active for generation
        sub-nodes, n_seqs for inference sub-nodes, B*K otherwise.

        :param node: Node to compute the batch size for.
        :type node: ExecNode
        :return: Batch size.
        :rtype: float
        """
        if node.n_active is not None:
            return node.n_active
        if node.n_seqs is not None:
            return node.n_seqs
        return self.workload.B * self.workload.K

    def _effective_mb(self, node: ExecNode) -> int:
        """
        Return the effective microbatch count, clamped for sub-nodes with small batches.

        :param node: Node to compute the microbatch count for.
        :type node: ExecNode
        :return: Microbatch count.
        :rtype: int
        """
        mb = node.mb
        pd = node.pi[0]
        n = self._effective_batch(node)
        return max(1, min(mb, int(n) // max(1, pd)))

    def _op_time(self, node: ExecNode) -> float:
        """
        Estimates the time in seconds to execute an operator node.

        :param node: Node for which the execution time is estimated.
        :type node: ExecNode
        :return: Execution time estimate in seconds.
        :rtype: float
        """
        role = node.op.template_vertex.kind.role
        pd, pp, pt = node.pi
        mb_eff = self._effective_mb(node)
        bubble_factor = (mb_eff + pp - 1) / mb_eff
        F_eff = self._effective_flops(node)
        exec_time = (self._work_cost(node) / (F_eff * pd*pp*pt)) * bubble_factor + self._intra_comm_cost(node)

        # Non-GEMM memory-bound overhead: RMSNorm, RoPE, FlashAttn, SiLU, residuals. These ops are
        # not counted in _work_cost (which only counts GEMM FLOP). F_peak absorbs them at tp=1, but
        # at tp>1 they don't fully scale, creating an error that grows with tp. We model them as
        # BW-bound ops.
        exec_time += self._non_gemm_overhead(node)

        # Optimizer step cost (training only): Adam reads/writes params + optimizer states
        if role == Role.TRAINING:
            model = node.model
            L, d, d_ff, V = (self.suite.specs[model].L, self.suite.specs[model].d,
                             self.suite.specs[model].d_ff, self.suite.specs[model].V)
            params = L * (4*d**2 + 2*d*d_ff) + V*d
            optimizer_dp = pd if self.optimizer_stage >= 1 else 1
            params_per_rank = params / (optimizer_dp * pp * pt)
            exec_time += (params_per_rank * 18) / self.hw.memory_bandwidth

        return exec_time * self._op_time_scale(role)

    def _op_time_scale(self, role: Role) -> float:
        """
        Helper function to return the correct scale factor based on the model role.

        :param role: Model role.
        :type role: Role
        :return: Scale factor.
        :rtype: float
        """
        if role == Role.GENERATION:
            return self.hw.op_generation_scale
        if role == Role.INFERENCE:
            return self.hw.op_inference_scale
        if role == Role.TRAINING:
            return self.hw.op_training_scale
        return 1.0

    def _non_gemm_overhead(self, node: ExecNode) -> float:
        """
        Additional overhead (time in seconds) from non-GEMM ops (RMSNorm, RoPE, FlashAttn, SiLU,
        residuals).

        These ops are not counted in _work_cost, which only counts GEMM FLOP. F_peak is calibrated
        from the full model at tp=1, so non-GEMM cost is already absorbed at tp=1. At tp>1, GEMMs
        shrink by 1/tp, but some non-GEMM ops don't scale (RMSNorm, residuals), creating an error.

        Uses calibrated per-layer component times from hw_*.yaml:
          non_gemm_constant: RMSNorm×2 + residual×2 (doesn't scale with tp)
          non_gemm_scaling: RoPE + FlashAttn + SiLU (scales with 1/tp)

        The calibrated values are measured at (B_cal, seq_cal) = (B, S+T) from calibrate.py. Since
        these ops process b_local*seq elements per layer, their time scales linearly with batch:
        ng(b) = ng(B_cal) * b_local / B_cal.

        The ADDITIONAL overhead beyond what F_peak absorbs:
          additional(tp) = L/pp * constant * (1 - 1/tp) * (b_local / B_cal)
        This is zero at tp=1 and grows with tp. It also scales down at small b_local (high dp),
        which correctly captures that non-GEMM ops do less work per GPU when the batch is split
        across more devices.

        :param node: Node for which the non-GEMM overhead is estimated.
        :type node: ExecNode
        :return: Time estimate of the non-GEMM overhead in seconds.
        :rtype: float
        """
        if self.hw.non_gemm_constant == 0.0 and self.hw.non_gemm_scaling == 0.0:
            return 0.0 # not calibrated

        role = node.op.template_vertex.kind.role
        if role == Role.GENERATION:
            return 0.0 # generation decode dominated by weight loading

        pd, pp, pt = node.pi
        L = self.suite.specs[node.model].L
        Lpp = L / pp

        # Batch scaling: non_gemm was calibrated at hw.non_gemm_batch_cal samples on one GPU. Each
        # non-GEMM op processes b_local elements per layer, so time scales linearly: ng(b_local) =
        # ng(B_cal) × b_local / B_cal.
        mb_eff = self._effective_mb(node)
        b_local = self._effective_batch(node) / (pd * mb_eff)
        B_cal = self.hw.non_gemm_batch_cal
        batch_scale = b_local / B_cal if B_cal > 0 else 1.0

        # Non-GEMM time at tp vs what F_peak absorbs (tp=1 scaled by 1/tp)
        ng_at_tp = Lpp * (self.hw.non_gemm_constant + self.hw.non_gemm_scaling / pt)
        ng_at_tp1_scaled = Lpp * (self.hw.non_gemm_constant + self.hw.non_gemm_scaling) / pt

        additional = (ng_at_tp - ng_at_tp1_scaled) * batch_scale

        # Training: forward + backward + one recompute forward when checkpointing
        if role == Role.TRAINING:
            additional *= 4 if self.activation_checkpointing else 3

        return max(0.0, additional)

    def _effective_flops(self, node: ExecNode) -> float:
        """
        Roofline-aware effective FLOP/s per GPU with GEMM SM-utilization correction.

        :param node: Node indicating the role and workload to be processed.
        :type node: ExecNode
        :returns: Effective FLOP/s per GPU.
        :rtype: float
        """
        role = node.op.template_vertex.kind.role
        if role == Role.GENERATION:
            return self._effective_flops_generation(node)
        I = self._arithmetic_intensity(node)
        eta = self._gemm_efficiency(node)
        I_star = self.hw.flops_per_gpu / self.hw.memory_bandwidth
        if I >= I_star:
            return self.hw.flops_per_gpu * eta
        else:
            return I * self.hw.memory_bandwidth

    def _gemm_efficiency(self, node: ExecNode) -> float:
        """
        GEMM SM utilization as function of batch dimension and TP-split dimension.

        eta = eta_batch × eta_tp where:
          eta_batch = M / (M + M_half)         — batch-dimension efficiency (calibrated)
          eta_tp    = K_min / (K_min + K_half) — TP-narrowing efficiency (calibrated)
          K_min     = min(d/tp, d_ff/tp)       — narrowest GEMM dimension after TP split

        :param node: Graph node for which the GEMM efficiency is estimated.
        :type node: ExecNode
        :return: GEMM SM utilization as a fraction.
        :rtype: float
        """
        M_half = self.hw.gemm_m_half
        K_half = getattr(self.hw, 'gemm_k_half', 0.0)

        pd, pp, pt = node.pi
        mb_eff = self._effective_mb(node)
        S, T = self.workload.S, self.workload.T
        b_local = self._effective_batch(node) / (pd * mb_eff)
        seq_len = S + T
        M_eff = b_local * seq_len

        # Batch dimension efficiency
        eta_batch = M_eff / (M_eff + M_half) if M_half > 0 else 1.0

        # TP-narrowing efficiency: smallest GEMM dimension after TP split
        if pt > 1 and K_half > 0:
            model = node.model
            d = self.suite.specs[model].d
            d_ff = self.suite.specs[model].d_ff
            K_min = min(d // pt, d_ff // pt)
            eta_tp = K_min / (K_min + K_half)
        else:
            eta_tp = 1.0

        return eta_batch * eta_tp

    def _effective_flops_generation(self, node: ExecNode) -> float:
        """
        Weighted effective FLOP/s for generation (prefill + decode have different I).

        :param node: Node indicating the workload to be processed.
        :type node: ExecNode
        :returns: Effective FLOP/s per GPU for generation.
        :rtype: float
        """
        model = node.model
        pd, pp, pt = node.pi
        P = pd * pp * pt

        S, T = self.workload.S, self.workload.T
        L, d, d_ff, V = (self.suite.specs[model].L, self.suite.specs[model].d,
                         self.suite.specs[model].d_ff, self.suite.specs[model].V)
        sb = self.suite.specs[model].dtype_bytes
        Lpp = L / pp
        mb_eff = self._effective_mb(node)
        b_local = self._effective_batch(node) / (pd * mb_eff)

        # Phase-specific decode parameters (sub-node or full)
        if node.phase_steps is not None:
            decode_steps = node.phase_steps
            avg_past = S + node.phase_t_start + decode_steps / 2
            has_prefill = node.is_prefill_phase
        else:
            decode_steps = T - 1
            avg_past = S + T / 2
            has_prefill = True

        # Weight Bytes per device (loaded once per microbatch phase)
        omega_W = (Lpp / pt) * (4*d**2 + 2*d*d_ff) * sb

        d_head = self.suite.specs[model].d_head
        M_half = self.hw.gemm_m_half
        I_star = self.hw.flops_per_gpu / self.hw.memory_bandwidth

        EFF_WEIGHTS = self.hw.decode_bw_eff
        EFF_KV = min(1.0, self.hw.decode_bw_eff * 3.5)

        W_total = self._work_cost(node)
        total_time = 0.0

        # Prefill component (first phase only)
        if has_prefill:
            eta_prefill = (b_local * S) / (b_local * S + M_half) if M_half > 0 else 1.0
            flops_prefill_dev = b_local * S * (2 * Lpp / pt) * (4*d**2 + 2*d*d_ff + S*d)
            omega_A_prefill = 2 * b_local * S * d * Lpp * sb
            I_prefill = flops_prefill_dev / (omega_W + omega_A_prefill) if (omega_W + omega_A_prefill) > 0 else float('inf')
            F_prefill = (self.hw.flops_per_gpu * eta_prefill if I_prefill >= I_star
                         else I_prefill * self.hw.memory_bandwidth)
            n_pf = self._effective_batch(node)
            W_prefill = n_pf * ((S*2*L) * (4*d**2 + 2*d*d_ff + S*d) + 2*V*d)
            total_time += W_prefill / (P * F_prefill) if F_prefill > 0 else 0

        # Decode component
        if decode_steps > 0:
            eta_decode = b_local / (b_local + M_half) if M_half > 0 else 1.0
            flops_decode_dev = b_local * (2 * Lpp / pt) * (4*d**2 + 2*d*d_ff + avg_past*d)
            omega_A_decode = 2 * b_local * d * Lpp * sb
            omega_KV = 2 * avg_past * (d / pt) * sb * b_local * Lpp
            omega_decode = omega_W + omega_A_decode + omega_KV
            eff_decode = (omega_W * EFF_WEIGHTS + omega_KV * EFF_KV + omega_A_decode * 1.0) / omega_decode if omega_decode > 0 else EFF_WEIGHTS
            bw_decode = self.hw.memory_bandwidth * eff_decode
            I_decode = flops_decode_dev / omega_decode if omega_decode > 0 else float('inf')
            F_decode = (self.hw.flops_per_gpu * eta_decode if I_decode >= I_star
                        else I_decode * bw_decode)
            W_decode = W_total - W_prefill if has_prefill else W_total
            total_time += W_decode / (P * F_decode) if F_decode > 0 else 0

        # Return effective F such that W_total / (P * F_eff) = total_time
        if total_time > 0:
            return W_total / (P * total_time)
        return self.hw.flops_per_gpu

    def _arithmetic_intensity(self, node: ExecNode) -> float:
        """
        Arithmetic intensity (FLOP / byte of HBM traffic) per device.

        For training, both FLOP and memory traffic scale by 4x (with activation checkpointing), so
        the intensity is the same as for inference.

        :param node: Node indicating the workload to be processed.
        :type node: ExecNode
        :returns: Arithmetic intensity per device.
        :rtype: float
        """
        model = node.model
        pd, pp, pt = node.pi

        S, T = self.workload.S, self.workload.T
        L, d, d_ff = (self.suite.specs[model].L, self.suite.specs[model].d,
                      self.suite.specs[model].d_ff)
        sb = self.suite.specs[model].dtype_bytes
        Lpp = L / pp
        seq_len = S + T

        mb_eff = self._effective_mb(node)
        b_local = self._effective_batch(node) / (pd * mb_eff)

        # Per-device FLOP for one microbatch (forward pass equivalent)
        flops_dev = b_local * seq_len * (2 * Lpp / pt) * (4*d**2 + 2*d*d_ff + seq_len*d)

        # Weight Bytes per device (loaded from HBM once per microbatch)
        omega_W = (Lpp / pt) * (4*d**2 + 2*d*d_ff) * sb

        # Activation Bytes per device (read + write per layer)
        omega_A = 2 * b_local * seq_len * d * Lpp * sb

        # FlashAttention: score matrix is never materialized in HBM → omega_S = 0.
        total_bytes = omega_W + omega_A
        if total_bytes <= 0:
            return float('inf')
        return flops_dev / total_bytes

    def _comm_time(self, node: ExecNode) -> float:
        """
        Returns the time estimate in seconds to execute a communication node.

        :param node: Communication node for which the execution time is estimated for.
        :type node: ExecNode
        :return: Execution time estimate in seconds.
        :rtype: float
        """
        alpha, beta = self.hw.tier(node.allocs)
        p_src = node.allocs[0].size()
        p_dst = node.allocs[1].size()
        p_min = min(p_src, p_dst)
        modeled = alpha + (node.payload_in_bytes / p_min) * beta
        if str(node.label or "").startswith("W_"):
            return modeled * self.hw.weight_comm_scale
        return modeled + self.hw.activation_comm_overhead_s

    def _reshard_time(self, node: ExecNode) -> float:
        """
        Returns the time estimate in seconds to execute a reshard node.

        :param node: Reshard node for which the execution time is estimated for.
        :type node: ExecNode
        :return: Execution time estimate in seconds.
        :rtype: float
        """
        P = node.allocs[0].size() # devices in the mesh (both allocs are identical)
        return self.hw.reshard_scale * self._collective_cost(
            coll = Collective.ALLTOALL,
            M = node.payload_in_bytes / P,
            allocs = node.allocs,
            p = P
        )

    def _work_cost(self, node: ExecNode) -> float:
        """
        Estimates the number of FLOP (no communication) that need to be performed by this operator
        node to complete its task.

        :param node: Operator node for which the number of FLOP is estimated for.
        :type node: ExecNode
        :return: Number of FLOP estimate for operator node.
        :rtype: float
        :raises ValueError: If model role is not supported.
        """
        role = node.op.template_vertex.kind.role
        model = node.model

        B, K, S, T = self.workload.B, self.workload.K, self.workload.S, self.workload.T
        L, d, d_ff, V = self.suite.specs[model].L, self.suite.specs[model].d, self.suite.specs[model].d_ff, self.suite.specs[model].V

        if role == Role.GENERATION:
            if node.n_active is not None:
                # Sub-node: phase-specific work
                n = node.n_active
                steps = node.phase_steps
                avg_past = S + node.phase_t_start + steps / 2
                work_decode = n * (2 * steps * L * (4*d**2 + 2*d*d_ff + avg_past*d) + 2*steps*V*d)
                if node.is_prefill_phase:
                    work_prefill = n * ((S*2*L) * (4*d**2 + 2*d*d_ff + S*d) + 2*V*d)
                    return work_prefill + work_decode
                return work_decode
            # Work is not split
            work_prefill = B * K * ((S*2*L) * (4*d**2 + 2*d*d_ff + S*d) + 2*V*d)
            work_decode = B * K * (2*(T-1)*L*(4*d**2 + 2*d*d_ff + (S+(T/2))*d) + 2*(T-1)*V*d)
            return work_prefill + work_decode

        if role == Role.INFERENCE:
            n = node.n_seqs if node.n_seqs is not None else B * K
            if model == Model.ACTOR or model == Model.REFERENCE:
                return n*(S+T)*(2*L*(4*d**2 + 2*d*d_ff + (S+T)*d) + 2*V*d)
            if model == Model.CRITIC or model == Model.AC_SHARE:
                return n*(S+T)*(2*L*(4*d**2 + 2*d*d_ff + (S+T)*d) + 2*d)
            if model == Model.REWARD:
                return n*(S+T)*(2*L*(4*d**2 + 2*d*d_ff + (S+T)*d)) + n*(2*d)

        if role == Role.TRAINING:
            multiplier = 4 if self.activation_checkpointing else 3
            if model == Model.ACTOR:
                work_fw_bw = multiplier * B*K*(S+T)*(2*L*(4*d**2 + 2*d*d_ff + (S+T)*d) + 2*V*d)
                return work_fw_bw
            if model == Model.CRITIC:
                work_fw_bw = multiplier * B*K*(S+T)*(2*L*(4*d**2 + 2*d*d_ff + (S+T)*d) + 2*d)
                return work_fw_bw
            if model == Model.AC_SHARE:
                work_fw_bw = multiplier * B*K*(S+T)*(2*L*(4*d**2 + 2*d*d_ff + (S+T)*d) + 2*V*d + 2*d)
                return work_fw_bw

        raise ValueError(f"_work_cost: unhandled (role={role}, model={model})")


    def _intra_comm_cost(self, node: ExecNode) -> float:
        """
        Returns an estimate for the time (in seconds) spend on communication induced by the
        intra-model 3D parallel configuration (dp, pp, tp).

        :param node: Node for which the intra-model communication time is estimated for.
        :type node: ExecNode
        :return: Execution time estimate for the intra-model communication of the 3D parallel
                 configuration (in seconds).
        :rtype: float
        :raises ValueError: If model role is not supported.
        """
        role  = node.op.template_vertex.kind.role
        model = node.model
        par = node.pi
        allocs = node.allocs

        B, K, S, T = self.workload.B, self.workload.K, self.workload.S, self.workload.T
        L, d, d_ff, V = self.suite.specs[model].L, self.suite.specs[model].d, self.suite.specs[model].d_ff, self.suite.specs[model].V
        pd, pp, pt = par
        m = self._effective_mb(node)

        sb = self.suite.specs[model].dtype_bytes

        # TP AllReduce: uses collective parameters (switch-aggregated BW).
        alpha_tp, beta_tp = self.hw.alpha_coll_g, self.hw.beta_coll_g
        def tp_allreduce(M: float, p: int) -> float:
            return 2 * (p - 1) * alpha_tp + 2 * (p - 1) / p * M * beta_tp

        # PP P2P: uses point-to-point parameters (single-link BW).
        alloc = allocs[0]
        r_ext, n_ext, g_per_node = alloc.extent
        s = g_per_node // pt # stages packed per node
        nodes_per_rep = math.ceil(pp / s) if s > 0 else pp
        total_pp_edges = pp - 1
        inter_pp_edges = nodes_per_rep - 1
        intra_pp_edges = total_pp_edges - inter_pp_edges

        # Inter-node PP tier: same-rack or cross-rack, based on the replica node layout.
        if r_ext > 1:
            alpha_pp_inter, beta_pp_inter = self.hw.alpha_p2p_r, self.hw.beta_p2p_r
        elif n_ext > 1:
            alpha_pp_inter, beta_pp_inter = self.hw.alpha_p2p_n, self.hw.beta_p2p_n
        else:
            alpha_pp_inter, beta_pp_inter = self.hw.alpha_p2p_g, self.hw.beta_p2p_g

        alpha_pp_intra, beta_pp_intra = self.hw.alpha_p2p_g, self.hw.beta_p2p_g

        def pp_p2p(M: float, n_steps: int) -> float:
            """
            Weighted PP P2P time for the n_steps communication steps in the pipeline.

            :param M: Data amount in Bytes to communicate.
            :type M: float
            :param n_steps: Number of pipeline communication steps.
            :type n_steps: int
            :return: Execution time in seconds.
            :rtype: float
            """
            if total_pp_edges == 0 or n_steps == 0:
                return 0.0
            cost_intra = (intra_pp_edges / total_pp_edges) * n_steps * (alpha_pp_intra + M * beta_pp_intra)
            cost_inter = (inter_pp_edges / total_pp_edges) * n_steps * (alpha_pp_inter + M * beta_pp_inter)
            return cost_intra + cost_inter

        Lpp = L / pp   # layers per pipeline stage

        if role == Role.GENERATION:
            if node.n_active is not None:
                # Sub-node: phase-specific comm
                n = node.n_active
                steps = node.phase_steps
                cost = 0.0
                if node.is_prefill_phase:
                    # Prefill PP + TP
                    cost += pp_p2p((n*S*d*sb)/(pd*m), m + pp - 2)
                    cost += 2 * Lpp * m * tp_allreduce((n*S*d*sb)/(pd*m), pt)
                # Decode PP + TP
                cost += pp_p2p((n*d*sb)/(pd*m), steps * (m + pp - 2))
                cost += 2 * steps * Lpp * m * tp_allreduce((n*d*sb)/(pd*m), pt)
                return cost
            # No splitting
            return (pp_p2p((B*K*S*d*sb)/(pd*m), m + pp - 2)
                    + pp_p2p((B*K*d*sb)/(pd*m), (T-1) * (m + pp - 2))
                    + 2 * Lpp * m * tp_allreduce((B*K*S*d*sb)/(pd*m), pt)
                    + 2 * (T-1) * Lpp * m * tp_allreduce((B*K*d*sb)/(pd*m), pt))

        if role == Role.INFERENCE:
            n = node.n_seqs if node.n_seqs is not None else B * K
            return (pp_p2p((n*(S+T)*d*sb)/(pd*m), m + pp - 2)
                    + 2 * Lpp * m * tp_allreduce((n*(S+T)*d*sb)/(pd*m), pt))

        if role == Role.TRAINING:
            params = L * (4*d**2 + 2*d*d_ff) + V*d + (d if model in (Model.CRITIC, Model.AC_SHARE) else 0)
            grad_msg = (params * sb) / (pp * pt) # BF16 gradient shard per DP group
            weight_msg = (params * sb) / (pp * pt) # BF16 weight shard per DP group
            if self.optimizer_stage == 0:
                dp_comm = self._collective_cost(Collective.ALLREDUCE, grad_msg, allocs, pd)
            elif self.optimizer_stage == 1:
                dp_comm = (
                    self._collective_cost(Collective.ALLREDUCE, grad_msg, allocs, pd)
                    + self._collective_cost(Collective.ALLGATHER, weight_msg, allocs, pd)
                )
            else:
                dp_comm = (
                    self._collective_cost(Collective.REDUCESCATTER, grad_msg, allocs, pd)
                    + self._collective_cost(Collective.ALLGATHER, weight_msg, allocs, pd)
                )
            tp_collectives = 6 if self.activation_checkpointing else 4

            return (dp_comm
                    + pp_p2p((B*K*(S+T)*d*sb)/(pd*m),  m + 2*pp - 3)
                    + tp_collectives * Lpp * m * tp_allreduce((B*K*(S+T)*d*sb)/(pd*m), pt))

        raise ValueError(f"_intra_comm_cost: unhandled (role={role}, model={model})")


    def _collective_cost(self, coll: Collective, M: int, allocs: Allocation, p: int) -> float:
        """
        Generic function that returns the time induced by a collective communication operation.

        :param coll: Type of collective communication operation.
        :type coll: Collective
        :param M: Data amount in Bytes per participant.
        :type M: int
        :param allocs: Device allocations.
        :type allocs: Allocation
        :param p: Number of participants.
        :type p: int
        :return: Estimated execution time of the collective communication operation in seconds.
        """
        # Collectives (AllReduce, ReduceScatter, AllGather, AllToAll) use collective parameters.
        # P2P uses p2p parameters.
        if coll in (Collective.ALLGATHER, Collective.REDUCESCATTER, Collective.ALLREDUCE, Collective.ALLTOALL):
            alpha, beta = self.hw.tier_coll(allocs)
        else:
            alpha, beta = self.hw.tier_p2p(allocs)

        exec_time = 0.0
        if coll == Collective.ALLGATHER or coll == Collective.REDUCESCATTER:
            exec_time = (p - 1) * alpha + (p - 1) / p * M * beta
        elif coll == Collective.ALLREDUCE:
            exec_time = 2 * (p - 1) * alpha + 2 * (p - 1) / p * M * beta
        elif coll == Collective.ALLTOALL:
            exec_time = (p - 1) * alpha + M * beta
        else:
            exec_time = alpha + M * beta

        return exec_time
