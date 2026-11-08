# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt
#
# Contributions: Pierre Pang

from dataclasses import dataclass
from typing import Tuple

from core.types import Allocation


@dataclass(frozen=True)
class Hardware:
    """
    Data class representing a target cluster.
    """
    flops_per_gpu: float        # workload-effective FLOP/s (from compute benchmark)
    memory_per_gpu: float       # GPU memory in bytes
    memory_bandwidth: float     # HBM bandwidth in bytes/s (from STREAM benchmark)

    # Collective communication: AllReduce, ReduceScatter, AllGather
    # (NCCL collectives use switch fabric → higher aggregate BW than single link)
    alpha_coll_g: float         # intra-node latency (s)
    beta_coll_g: float          # intra-node inverse bandwidth (s/byte)
    alpha_coll_n: float         # inter-node latency
    beta_coll_n: float          # inter-node inverse bandwidth
    alpha_coll_r: float         # inter-rack latency
    beta_coll_r: float          # inter-rack inverse bandwidth

    # Point-to-point communication: PP P2P send/recv, inter-model resharding
    # (single sender → single receiver → limited to single-link BW)
    alpha_p2p_g: float          # intra-node latency
    beta_p2p_g: float           # intra-node inverse bandwidth
    alpha_p2p_n: float          # inter-node latency
    beta_p2p_n: float           # inter-node inverse bandwidth
    alpha_p2p_r: float          # inter-rack latency
    beta_p2p_r: float           # inter-rack inverse bandwidth

    # GEMM SM utilization: eta = eta_batch × eta_tp
    #   eta_batch = M / (M + gemm_m_half)         — batch dim (M = b_local × seq)
    #   eta_tp    = K_min / (K_min + gemm_k_half)  — TP-narrowed dim
    gemm_m_half: float = 5.0
    gemm_k_half: float = 0.0   # 0 = disabled (no TP penalty)

    # Autoregressive decode HBM efficiency (from decode benchmark)
    decode_bw_eff: float = 0.15

    # Non-GEMM overhead per layer (from per-component benchmark)
    # non_GEMM(tp) = L * (non_gemm_constant + non_gemm_scaling / tp)
    non_gemm_constant: float = 0.0  # seconds/layer (RMSNorm×2 + residual×2)
    non_gemm_scaling: float = 0.0   # seconds/layer at tp=1 (RoPE + FlashAttn + SiLU)
    non_gemm_batch_cal: float = 64  # batch size used during non_gemm calibration

    # End-to-end runtime corrections learned from an actual graph trace. The analytical kernels
    # remain the extrapolation model across strategies; these factors account for backend/runtime
    # effects absent from isolated microbenchmarks (framework dispatch, model implementation, and
    # control latency). Defaults preserve the analytical model exactly.
    op_generation_scale: float = 1.0
    op_inference_scale: float = 1.0
    op_training_scale: float = 1.0
    activation_comm_overhead_s: float = 0.0
    weight_comm_scale: float = 1.0
    reshard_scale: float = 1.0

    def tier_coll(self, allocs: Tuple[Allocation, ...]) -> Tuple[float, float]:
        """
        Select collective (alpha, beta) by physical placement.

        :param allocs: Involved allocations.
        :type allocs: Tuple[Allocation, ...]
        :return: Tuple of alpha and beta for the collective performance model according to the
                 actual physical placement.
        :rtype: Tuple[float, float]
        """
        min_r = min(a.origin[0] for a in allocs)
        max_r = max(a.origin[0] + a.extent[0] for a in allocs)
        min_n = min(a.origin[1] for a in allocs)
        max_n = max(a.origin[1] + a.extent[1] for a in allocs)
        if max_r - min_r > 1:
            return self.alpha_coll_r, self.beta_coll_r
        elif max_n - min_n > 1:
            return self.alpha_coll_n, self.beta_coll_n
        else:
            return self.alpha_coll_g, self.beta_coll_g

    def tier_p2p(self, allocs: Tuple[Allocation, ...]) -> Tuple[float, float]:
        """
        Select P2P (alpha, beta) by physical placement.

        :param allocs: Involved allocations.
        :type allocs: Tuple[Allocation, ...]
        :return: Tuple of alpha and beta for the P2P performance model according to the actual
                 physical placement.
        :rtype: Tuple[float, float]
        """
        min_r = min(a.origin[0] for a in allocs)
        max_r = max(a.origin[0] + a.extent[0] for a in allocs)
        min_n = min(a.origin[1] for a in allocs)
        max_n = max(a.origin[1] + a.extent[1] for a in allocs)
        if max_r - min_r > 1:
            return self.alpha_p2p_r, self.beta_p2p_r
        elif max_n - min_n > 1:
            return self.alpha_p2p_n, self.beta_p2p_n
        else:
            return self.alpha_p2p_g, self.beta_p2p_g

    def tier(self, allocs: Tuple[Allocation, ...]) -> Tuple[float, float]:
        """
        Backward compatibility: tier() delegates to tier_p2p (used by _comm_time, _reshard_time).

        :param allocs: Involved allocations.
        :type allocs: Tuple[Allocation, ...]
        :return: Tuple of alpha and beta for the P2P performance model according to the actual
                 physical placement.
        :rtype: Tuple[float, float]
        """
        return self.tier_p2p(allocs)
