#!/usr/bin/env python3

# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt

"""
Hardware calibration micro-benchmark suite for the analytical cost model.

Measures 7 parameters in ~3 minutes:
  1. NCCL AllReduce BW       → alpha_coll, beta_coll  (collective communication)
  2. NCCL P2P Send/Recv BW   → alpha_p2p, beta_p2p    (point-to-point communication)
  3. Forward-pass throughput → F_peak                 (workload-effective FLOP/s)
  4. HBM bandwidth           → memory_bandwidth       (STREAM-like copy)
  5. GEMM efficiency sweep   → gemm_m_half            (SM utilization versus batch dimension)
  6. Decode-step latency     → decode_bw_eff          (autoregressive decode efficiency)
  7. Non-GEMM overhead                                (per-component non-GEMM overhead)

Output: YAML hardware config ready for the optimizer.

Usage (via SLURM):
    srun --ntasks=N --gpus-per-task=1 python calibrate.py [--output hw_config.yaml]
"""

import argparse
import numpy as np
import os
import sys
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F

from typing import Callable, Dict, List, Optional, Tuple


# Global compute dtype — set in main() via --dtype flag
COMPUTE_DTYPE = torch.bfloat16 # default; overridden to fp16 for V100

def init_distributed() -> Tuple[int, int, int]:
    """
    Distributed communicator setup.

    :return: Tuple of the (global) rank, the (global) communicator size and the local rank.
    :rtype: Tuple[int, int, int]
    :raises RuntimeError: If the script was not started within a SLURM or torchrun environment.
    """
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        raise RuntimeError("Must be launched via srun or torchrun")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        'nccl',
        rank = rank,
        world_size = world_size,
        device_id = torch.device('cuda', local_rank)
    )
    return rank, world_size, local_rank


def bench_allreduce(
    sizes_bytes: List[int],
    n_warmup: int = 3,
    n_iters: int = 10,
    group: Optional[dist.ProcessGroup] = None
) -> List[Tuple[int, float]]:
    """
    Benchmark 1: NCCL AllReduce (collective BW)

    :param sizes_bytes: List of data sizes to communicate and measure.
    :type sizes_bytes: List[int]
    :param n_warmup: Number of warm-up executions to perform before the measurements. Defaults to 3.
    :type n_warmup: int
    :param n_iters: Number of measurements per data size. Defaults to 10.
    :type n_iters: int
    :param group: Communicator to use. Defaults to None, in which case the default Torch process
                  group is used.
    :type group: Optional[dist.ProcessGroup]
    :return: List of data sizes and the median execution times for respective data sizes (in
             seconds).
    :rtype: List[Tuple[int, float]]
    """
    device = torch.cuda.current_device()
    results = []
    for size in sizes_bytes:
        buf = torch.randn(max(1, size // 2), dtype=torch.bfloat16, device=device)
        for _ in range(n_warmup):
            dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=group)
        torch.cuda.synchronize(); dist.barrier(group=group)
        times = []
        for _ in range(n_iters):
            torch.cuda.synchronize(); dist.barrier(group=group)
            t0 = time.perf_counter()
            dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=group)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        results.append((size, float(np.median(times))))
    return results


def bench_repeated_allreduce(
    size_bytes: int,
    n_repeats: int = 64,
    n_warmup: int = 3,
    n_iters: int = 10,
    group: Optional[dist.ProcessGroup] = None
) -> Tuple[int, float]:
    """
    Benchmark 1b: Repeated AllReduce (per-layer TP pattern)

    Measure the amortized cost of n_repeats back-to-back AllReduces.

    This captures the real per-call overhead when AllReduce is called many times in sequence (as in
    TP AllReduce per transformer layer). The microbenchmark for a single large AllReduce
    underestimates this because of NCCL kernel launch gaps and the accumulation of GPU
    synchronization overhead.

    :param sizes_bytes: Data size in Bytes to communicate and measure.
    :type sizes_bytes: int
    :param n_repeats: Number of times to repeat AllReduce within a single measurement. Defaults to
                      64.
    :type n_repeats: int
    :param n_warmup: Number of warm-up executions to perform before the measurements. Defaults to 3.
    :type n_warmup: int
    :param n_iters: Number of measurements. Defaults to 10.
    :type n_iters: int
    :param group: Communicator to use. Defaults to None, in which case the default Torch process
                  group is used.
    :type group: Optional[dist.ProcessGroup]
    :return: Data size in Bytes and the median of the amortized execution times (in seconds).
    :rtype: Tuple[int, float]
    """
    device = torch.cuda.current_device()
    buf = torch.randn(max(1, size_bytes // 2), dtype=torch.bfloat16, device=device)

    # Warm-up
    for _ in range(n_warmup):
        for _ in range(n_repeats):
            dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=group)
    torch.cuda.synchronize()
    if group is not None:
        dist.barrier(group=group)
    else:
        dist.barrier()

    # Timed
    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        if group is not None:
            dist.barrier(group=group)
        else:
            dist.barrier()
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=group)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) / n_repeats) # amortized per call

    return (size_bytes, float(np.median(times)))


def bench_p2p(
    sizes_bytes: List[int],
    src: int = 0,
    dst: int = 1,
    n_warmup: int = 3,
    n_iters: int = 10
) -> List[Tuple[int, float]]:
    """
    Benchmark 2: NCCL P2P Send/Recv (point-to-point BW)

    Uses the default process group.

    :param sizes_bytes: List of data sizes to communicate and measure.
    :type sizes_bytes: List[int]
    :param src: Process rank, who will send the data. Defaults to 0.
    :type src: int
    :param dst: Process rank, who will receive the data. Defaults to 1.
    :type dst: int
    :param n_warmup: Number of warm-up executions to perform before the measurements. Defaults to 3.
    :type n_warmup: int
    :param n_iters: Number of measurements per data size. Defaults to 10.
    :type n_iters: int
    :return: List of data sizes and the median communication times for those data size (in seconds).
             An empty list is returned for any process not participating in the p2p communication.
    :rtype: List[Tuple[int, float]]
    """
    rank = dist.get_rank()
    if dist.get_world_size() < 2:
        return []
    device = torch.cuda.current_device()
    results = []
    for size in sizes_bytes:
        buf = torch.randn(max(1, size // 2), dtype=torch.bfloat16, device=device)
        recv_buf = torch.empty_like(buf)
        for _ in range(n_warmup):
            if rank == src:
                dist.send(buf, dst=dst); dist.recv(recv_buf, src=dst)
            elif rank == dst:
                dist.recv(recv_buf, src=src); dist.send(buf, dst=src)
            dist.barrier()
        torch.cuda.synchronize(); dist.barrier()
        times = []
        for _ in range(n_iters):
            torch.cuda.synchronize(); dist.barrier()
            t0 = time.perf_counter()
            if rank == src:
                dist.send(buf, dst=dst); dist.recv(recv_buf, src=dst)
            elif rank == dst:
                dist.recv(recv_buf, src=src); dist.send(buf, dst=src)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) / 2)
        if rank in (src, dst):
            results.append((size, float(np.median(times))))
    return results



def bench_compute(
    L: int = 32,
    d: int = 4096,
    d_ff: int = 11008,
    n_heads: int = 32,
    V: int = 32000,
    B: int = 64,
    seq: int = 640,
    n_warmup: int = 3,
    n_iters: int = 5
) -> Tuple[int, float, float]:
    """
    Benchmark 3: Forward-pass throughput (F_peak)

    Uses the actual model architecture (RMSNorm, RoPE, FlashAttention, SiLU-gated FFN) to capture
    non-GEMM overhead that the cost model's F_peak must absorb.

    :param L: Number of model layers. Defaults to 32.
    :type L: int
    :param d: Size of the hidden dimension. Defaults to 4096.
    :type d: int
    :param d_ff: Size of the FFN dimension. Defaults to 11008.
    :type d_ff: int
    :param n_head: Number of heads. Defaults to 32.
    :type n_head: int
    :param V: Vocabulary size. Defaults to 32000.
    :type V: int
    :param B: Batch size. Defaults to 64.
    :type B: int
    :param seq: Number of tokens in the input sequence. Defaults to 640.
    :type seq: int
    :param n_warmup: Number of warm-up executions to perform before the measurements. Defaults to 3.
    :type n_warmup: int
    :param n_iters: Number of measurements. Defaults to 5.
    :type n_iters: int
    :return: Tuple of the number of FLOP for a single forward-pass, median execution time (in
             seconds) and the forward-pass throughput.
    :rtype: Tuple[int, float, float]
    """
    device = torch.cuda.current_device()
    dtype = COMPUTE_DTYPE
    head_dim = d // n_heads

    # Build a minimal Transformer (same number of operations as bench.py's TransformerStage)
    class RMSNorm(torch.nn.Module):
        """
        Apply the Root Mean Square Layer Normalization.
        """
        def __init__(self, dim: int) -> None:
            super().__init__()
            """
            Initialize the class.

            :param dim: Size of the dimension.
            :type dim: int
            """
            self.weight = torch.nn.Parameter(torch.ones(dim, dtype=dtype))
        def forward(self, x):
            """
            Apply the RMS norm.
            """
            return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(dtype) * self.weight

    class Layer(torch.nn.Module):
        """
        Model a layer of the Transformer.
        """
        def __init__(self):
            """
            Initialize the layer structure.
            """
            super().__init__()
            self.attn_norm = RMSNorm(d)
            self.W_qkv = torch.nn.Linear(d, 3*d, bias=False, dtype=dtype)
            self.W_o = torch.nn.Linear(d, d, bias=False, dtype=dtype)
            self.ffn_norm = RMSNorm(d)
            self.up = torch.nn.Linear(d, d_ff, bias=False, dtype=dtype)
            self.down = torch.nn.Linear(d_ff, d, bias=False, dtype=dtype)

        def forward(self, x, cos, sin):
            """
            Forward-pass through the layer.
            """
            h = self.attn_norm(x)
            qkv = self.W_qkv(h).view(B, seq, 3, n_heads, head_dim)
            q, k, v = [t.transpose(1, 2) for t in qkv.unbind(2)]
            # RoPE
            q1, q2 = q.chunk(2, dim=-1)
            q = torch.cat([q1*cos - q2*sin, q2*cos + q1*sin], dim=-1)
            k1, k2 = k.chunk(2, dim=-1)
            k = torch.cat([k1*cos - k2*sin, k2*cos + k1*sin], dim=-1)
            # FlashAttention via SDPA
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn = attn.transpose(1, 2).contiguous().view(B, seq, d)
            x = x + self.W_o(attn)
            h = self.ffn_norm(x)
            x = x + self.down(F.silu(self.up(h)))
            return x

    layers = torch.nn.ModuleList([Layer() for _ in range(L)]).to(device)
    head = torch.nn.Linear(d, V, bias=False, dtype=dtype, device=device)

    # RoPE precompute
    freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq, device=device).float()
    angles = torch.outer(t, freqs)
    cos = torch.cos(angles).to(dtype).unsqueeze(0).unsqueeze(0)  # (1,1,seq,head_dim/2)
    sin = torch.sin(angles).to(dtype).unsqueeze(0).unsqueeze(0)

    x = torch.randn(B, seq, d, dtype=dtype, device=device)

    @torch.no_grad()
    def fwd():
        """
        Forward-pass through the entire model.
        """
        h = x
        for layer in layers:
            h = layer(h, cos, sin)
        return head(h)

    # Warm-up
    for _ in range(n_warmup):
        fwd()
    torch.cuda.synchronize()

    # Timed
    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fwd()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    median_t = float(np.median(times))
    flops = B * seq * (2*L*(4*d**2 + 2*d*d_ff + seq*d) + 2*V*d)
    F_peak = flops / median_t

    # Cleanup
    del layers, head, x
    torch.cuda.empty_cache()

    return flops, median_t, F_peak


def bench_memory_bandwidth(
    L: int = 32,
    d: int = 4096,
    d_ff: int = 11008,
    n_heads: int = 32,
    n_warmup: int = 3,
    n_iters: int = 10
) -> float:
    """
    Benchmark 4: HBM bandwidth (STREAM-like)

    Measure workload-realistic HBM bandwidth by running a single forward-pass at B=1, seq=1 (heavily
    memory-bound: must load all weights, minimal compute).

    Uses eager execution (no CUDA graphs) — this intentionally includes per-layer Python dispatch
    overhead, which matches the cost model's F_peak calibration (also eager). The decode benchmark
    (benchmark 6) uses CUDA graphs, so decode_bw_eff = decode_BW / HBM_BW < 1.0 captures the KV
    cache access penalty relative to pure weight loading.

    :param L: Number of model layers. Defaults to 32.
    :type L: int
    :param d: Size of the hidden dimension. Defaults to 4096.
    :type d: int
    :param d_ff: Size of the FFN dimension. Defaults to 11008.
    :type d_ff: int
    :param n_head: Number of heads. Defaults to 32.
    :type n_head: int
    :param n_warmup: Number of warm-up executions to perform before the measurements. Defaults to 3.
    :type n_warmup: int
    :param n_iters: Number of measurements. Defaults to 10.
    :type n_iters: int
    :return: HBM bandwidth in Bytes/second.
    :rtype: float
    """
    device = torch.cuda.current_device()
    dtype = COMPUTE_DTYPE
    head_dim = d // n_heads
    sb = 2

    class RMSNorm(torch.nn.Module):
        """
        Apply the Root Mean Square Layer Normalization.
        """
        def __init__(self, dim: int) -> None:
            """
            Initialize the class.

            :param dim: Size of the dimension.
            :type dim: int
            """
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(dim, dtype=dtype))
        def forward(self, x):
            """
            Apply the RMS norm.
            """
            return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(dtype) * self.weight

    class Layer(torch.nn.Module):
        """
        Model a layer of the Transformer.
        """
        def __init__(self):
            """
            Initialize the layer structure.
            """
            super().__init__()
            self.attn_norm = RMSNorm(d)
            self.W_qkv = torch.nn.Linear(d, 3*d, bias=False, dtype=dtype)
            self.W_o = torch.nn.Linear(d, d, bias=False, dtype=dtype)
            self.ffn_norm = RMSNorm(d)
            self.up = torch.nn.Linear(d, d_ff, bias=False, dtype=dtype)
            self.down = torch.nn.Linear(d_ff, d, bias=False, dtype=dtype)

        def forward(self, x):
            """
            Forward-pass through the layer (simplified).
            """
            h = self.attn_norm(x)
            qkv = self.W_qkv(h)
            o = qkv[:, :d] # simplified attention (B=1 → no real attention needed)
            x = x + self.W_o(o)
            h = self.ffn_norm(x)
            x = x + self.down(F.silu(self.up(h)))
            return x

    layers = torch.nn.ModuleList([Layer() for _ in range(L)]).to(device)
    x = torch.randn(1, d, dtype=dtype, device=device)  # B=1, seq=1

    @torch.no_grad()
    def fwd():
        """
        Forward-pass through the entire model.
        """
        h = x
        for layer in layers:
            h = layer(h)
        return h

    # Warm-up
    for _ in range(n_warmup):
        fwd()
    torch.cuda.synchronize()

    # Timed
    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fwd()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    median_t = float(np.median(times))

    # At B=1 seq=1, compute is negligible → time ≈ weight_bytes / BW
    weight_bytes = L * (4*d**2 + 2*d*d_ff) * sb
    act_bytes = 2 * d * L * sb
    total_bytes = weight_bytes + act_bytes

    bw = total_bytes / median_t

    # Cleanup
    del layers, x
    torch.cuda.empty_cache()

    return bw


def bench_gemm_sweep(
    d: int = 4096,
    d_ff: int = 11008,
    n_warmup: int = 3,
    n_iters: int = 10
) -> List[Tuple[int, float]]:
    """
    Benchmark 5: GEMM sweep (gemm_m_half), which measures the SM utilization versus batch dimension.

    :param d: Size of the hidden dimension. Defaults to 4096.
    :type d: int
    :param d_ff: Size of the FFN dimension. Defaults to 11008.
    :type d_ff: int
    :param n_warmup: Number of warm-up executions to perform before the measurements. Defaults to 3.
    :type n_warmup: int
    :param n_iters: Number of measurements. Defaults to 10.
    :type n_iters: int
    :return: List of tuples of matrix size and the corresponding performance in TFLOP/s.
    :rtype: List[Tuple[int, float]]
    """
    device = torch.cuda.current_device()
    dtype = COMPUTE_DTYPE
    M_values = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536]
    W = torch.randn(d_ff, d, dtype=dtype, device=device)
    results = []
    for M in M_values:
        x = torch.randn(M, d, dtype=dtype, device=device)
        flops = 2 * M * d * d_ff
        for _ in range(n_warmup):
            _ = x @ W.T
        torch.cuda.synchronize()
        times = []
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = x @ W.T
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        results.append((M, flops / float(np.median(times)) / 1e12))
    del W; torch.cuda.empty_cache()
    return results


def fit_gemm_m_half(gemm_results: List[Tuple[int, float]]) -> Tuple[float, float]:
    """
    Fit the results to determine the peak and sustainable GEMM performance.

    :param gemm_results: Results from the GEMM sweep.
    :type gemm_results: List[Tuple[int, float]]
    :return: Tuple of the peak and sustainable GEMM performance.
    :rtype: Tuple[float, float]
    """
    M_vals = np.array([r[0] for r in gemm_results], dtype=np.float64)
    tflops = np.array([r[1] for r in gemm_results], dtype=np.float64)
    peak = tflops[-3:].mean()
    estimates = []
    for m, t in zip(M_vals, tflops):
        ratio = t / peak
        if 0.3 < ratio < 0.9:
            estimates.append(m * (peak / t - 1))
    return peak, float(np.median(estimates)) if estimates else 0.0


def bench_decode_step(
    L: int = 32,
    d: int = 4096,
    d_ff: int = 11008,
    n_heads: int = 32,
    V: int = 32000,
    B: int = 1,
    past_len: int = 256,
    n_warmup: int = 5,
    n_iters: int = 30
) -> Tuple[float, int, int, int]:
    """
    Benchmark 6: Decode step with KV cache (DECODE_BW_EFF)

    Measure decode step throughput using CUDA graphs with a pre-allocated KV cache.

    Mimics real generation decode: fixed-size KV buffers, scatter-based writes, full-buffer
    attention with causal mask. CUDA graphs eliminate Python dispatch overhead, giving the true
    GPU-limited decode time.

    Measures at B=1 (weight-dominated) to calibrate decode_bw_eff.

    :param L: Number of model layers. Defaults to 32.
    :type L: int
    :param d: Size of the hidden dimension. Defaults to 4096.
    :type d: int
    :param d_ff: Size of the FFN dimension. Defaults to 11008.
    :type d_ff: int
    :param n_head: Number of heads. Defaults to 32.
    :type n_head: int
    :param V: Vocabulary size. Defaults to 32000.
    :type V: int
    :param B: Batch size. Defaults to 1.
    :type B: int
    :param past_len: Part of the KV cache, that has already been prefilled. Defaults to 256.
    :type past_len: int
    :param n_warmup: Number of warm-up executions to perform before the measurements. Defaults to 5.
    :type n_warmup: int
    :param n_iters: Number of decode steps within a single measurement. Defaults to 30, however at
                    least 64 steps are measured.
    :type n_iters: int
    :return: Tuple of the median time (in seconds) of a decode step, weight size (in Bytes), KV
             cache size (in Bytes) and the total memory amount (in Bytes).
    :rtype: Tuple[float, int, int, int]
    """
    device = torch.cuda.current_device()
    dtype = COMPUTE_DTYPE
    head_dim = d // n_heads
    sb = 2
    n_steps = max(n_iters, 64)
    max_kv_len = past_len + n_warmup + 1 + n_steps + 10 # enough for warmup + capture + all replays

    class RMSNorm(torch.nn.Module):
        """
        Apply the Root Mean Square Layer Normalization.
        """
        def __init__(self, dim: int) - None:
            """
            Initialize the class.

            :param dim: Size of the dimension.
            :type dim: int
            """
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(dim, dtype=dtype))
        def forward(self, x):
            """
            Apply the RMS norm.
            """
            return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(dtype) * self.weight

    class DecodeLayer(torch.nn.Module):
        """
        Model a decoder layer of the Transformer.
        """
        def __init__(self):
            super().__init__()
            self.attn_norm = RMSNorm(d)
            self.W_qkv = torch.nn.Linear(d, 3 * d, bias=False, dtype=dtype)
            self.W_o = torch.nn.Linear(d, d, bias=False, dtype=dtype)
            self.ffn_norm = RMSNorm(d)
            self.up = torch.nn.Linear(d, d_ff, bias=False, dtype=dtype)
            self.down = torch.nn.Linear(d_ff, d, bias=False, dtype=dtype)

    layers = torch.nn.ModuleList([DecodeLayer() for _ in range(L)]).to(device)

    # Pre-allocate a fixed-size KV cache (CUDA graph compatible)
    k_bufs = [torch.randn(B, n_heads, max_kv_len, head_dim, dtype=dtype, device=device) for _ in range(L)]
    v_bufs = [torch.randn(B, n_heads, max_kv_len, head_dim, dtype=dtype, device=device) for _ in range(L)]

    # Static input/output buffers
    x = torch.randn(B, 1, d, dtype=dtype, device=device)

    # GPU tensor for cache position (updated in-place for CUDA graph replay)
    cache_pos = torch.tensor([past_len], dtype=torch.long, device=device)

    # Pre-compute full RoPE table
    freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t_pos = torch.arange(max_kv_len, device=device, dtype=torch.float32)
    rope_freqs = torch.outer(t_pos, freqs)
    cos_table = rope_freqs.cos().to(dtype) # (max_kv_len, head_dim//2)
    sin_table = rope_freqs.sin().to(dtype)

    @torch.no_grad()
    def decode_step_graph():
        """
        One decode step using scatter writes + full-buffer attention.
        """
        h = x
        cos = cos_table.index_select(0, cache_pos).view(1, 1, 1, head_dim // 2)
        sin = sin_table.index_select(0, cache_pos).view(1, 1, 1, head_dim // 2)
        for i, layer in enumerate(layers):
            # Attention
            normed = layer.attn_norm(h)
            qkv = layer.W_qkv(normed).view(B, 1, 3, n_heads, head_dim)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q = q.transpose(1, 2)  # (B, n_heads, 1, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            # RoPE
            q1, q2 = q.chunk(2, dim=-1)
            q = torch.cat([q1*cos - q2*sin, q2*cos + q1*sin], dim=-1)
            k1, k2 = k.chunk(2, dim=-1)
            k = torch.cat([k1*cos - k2*sin, k2*cos + k1*sin], dim=-1)
            # Scatter write to KV cache (fixed addresses for CUDA graph)
            idx = cache_pos.view(1, 1, 1, 1).expand(B, n_heads, 1, head_dim)
            k_bufs[i].scatter_(2, idx, k)
            v_bufs[i].scatter_(2, idx, v)
            # Full-buffer attention (is_causal=True masks future positions)
            attn = F.scaled_dot_product_attention(q, k_bufs[i], v_bufs[i], is_causal=True)
            attn = attn.transpose(1, 2).contiguous().view(B, 1, -1)
            h = h + layer.W_o(attn)
            # FFN
            h2 = layer.ffn_norm(h)
            h = h + layer.down(F.silu(layer.up(h2)))
        x.copy_(h)

    # Eager warmup (populate state)
    for i in range(n_warmup):
        cache_pos.fill_(past_len + i)
        decode_step_graph()
    torch.cuda.synchronize()

    # Capture CUDA graph
    cache_pos.fill_(past_len + n_warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode_step_graph()
    torch.cuda.synchronize()

    # Timed replay
    # Run many steps and measure total time for accurate per-step average
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n_steps):
        cache_pos.fill_(past_len + n_warmup + 1 + i)
        graph.replay()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    median_t = (t1 - t0) / n_steps

    # Memory traffic: weights + KV cache + activations
    omega_W = L * (4*d**2 + 2*d*d_ff) * sb
    omega_KV = 2 * past_len * d * sb * B * L
    omega_A = 2 * B * d * L * sb
    total_bytes = omega_W + omega_KV + omega_A

    del layers, k_bufs, v_bufs, x, graph
    torch.cuda.empty_cache()

    return median_t, omega_W, omega_KV, total_bytes


def bench_non_gemm_components(
    L: int = 32,
    d: int = 4096,
    d_ff: int = 11008,
    n_heads: int = 32,
    B: int = 64,
    seq: int = 640,
    n_warmup: int = 5,
    n_iters: int = 20
) -> Tuple[float, float, Dict[str, float]]:
    """
    Benchmark 7: Per-component non-GEMM overhead

    Measure per-layer time of each non-GEMM component.

    Returns (t_constant, t_scaling) in seconds per layer:
      t_constant: ops that don't scale with tp (RMSNorm×2 + residual×2)
      t_scaling: ops that do scale with 1/tp (RoPE + FlashAttn + SiLU)

    The cost model uses: non_GEMM_overhead(tp) = L * (t_constant + t_scaling/tp).

    :param L: Number of model layers. Defaults to 32.
    :type L: int
    :param d: Size of the hidden dimension. Defaults to 4096.
    :type d: int
    :param d_ff: Size of the FFN dimension. Defaults to 11008.
    :type d_ff: int
    :param n_head: Number of heads. Defaults to 32.
    :type n_head: int
    :param B: Batch size. Defaults to 64.
    :type B: int
    :param seq: Number of tokens in the input sequence. Defaults to 640.
    :type seq: int
    :param n_warmup: Number of warm-up executions to perform before the measurements. Defaults to 5.
    :type n_warmup: int
    :param n_iters: Number of measurements. Defaults to 20.
    :type n_iters: int
    :return: Tuple of the constant and scaling components and a dictionary with the median execution
             times of the individual components (everything in seconds).
    :rtype: Tuple[float, float, Dict[str, float]]
    """
    device = torch.cuda.current_device()
    dtype = COMPUTE_DTYPE
    head_dim = d // n_heads
    sb = 2

    # Build weights for one layer
    W_qkv = torch.randn(3*d, d, dtype=dtype, device=device) * 0.01
    W_o = torch.randn(d, d, dtype=dtype, device=device) * 0.01
    W_up = torch.randn(d_ff, d, dtype=dtype, device=device) * 0.01
    W_down = torch.randn(d, d_ff, dtype=dtype, device=device) * 0.01
    rms_w = torch.ones(d, dtype=dtype, device=device)
    x = torch.randn(B, seq, d, dtype=dtype, device=device)

    # RoPE precompute
    freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t_arr = torch.arange(seq, device=device).float()
    angles = torch.outer(t_arr, freqs)
    cos = torch.cos(angles).to(dtype).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(angles).to(dtype).unsqueeze(0).unsqueeze(0)

    def _bench(fn: Callable, iters: int = n_iters) -> float:
        """
        Benchmark a function after a warm-up period and return the median execution time.

        :param fn: Function to benchmark.
        :type: fn: Callable
        :param iters: Number of measurements. Defaults to the global n_iters.
        :type iters: int
        :return: Median execution time of the function in seconds.
        :rtype: float
        """
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        return float(np.median(times))

    # Constant ops (don't scale with tp)
    # RMSNorm: 2 per layer
    def rmsnorm():
        """
        Apply the Root Mean Square Layer Normalization.
        """
        h = x.float()
        return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)).to(dtype) * rms_w
    t_rmsnorm = _bench(rmsnorm)

    # Residual add: 2 per layer
    y = torch.randn_like(x)
    def residual():
        """
        Add a residual connection.
        """
        return x + y
    t_residual = _bench(residual)

    t_constant = 2 * t_rmsnorm + 2 * t_residual # per layer

    # Scaling ops (scale with 1/tp)
    # RoPE
    qkv_out = x @ W_qkv.T
    q = qkv_out[:, :, :d].view(B, seq, n_heads, head_dim).transpose(1, 2)
    k = qkv_out[:, :, d:2*d].view(B, seq, n_heads, head_dim).transpose(1, 2)
    def rope():
        """
        Apply the rotary positional embedding.
        """
        q1, q2 = q.chunk(2, dim=-1)
        torch.cat([q1*cos - q2*sin, q2*cos + q1*sin], dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        torch.cat([k1*cos - k2*sin, k2*cos + k1*sin], dim=-1)
    t_rope = _bench(rope)

    # FlashAttention
    v = qkv_out[:, :, 2*d:].view(B, seq, n_heads, head_dim).transpose(1, 2)
    qr = q  # use un-rotated for simplicity (same compute cost)
    def flash_attn():
        """
        Apply FlashAttention.
        """
        return F.scaled_dot_product_attention(qr, k, v, is_causal=True)
    t_flash = _bench(flash_attn)

    # SiLU
    up_out = x @ W_up.T
    def silu():
        """
        Apply the Sigmoid Linear Unit activation function.
        """
        return F.silu(up_out)
    t_silu = _bench(silu)

    t_scaling = t_rope + t_flash + t_silu # per layer at tp=1

    # Cleanup
    del W_qkv, W_o, W_up, W_down, x, y, qkv_out, q, k, v, up_out
    torch.cuda.empty_cache()

    return t_constant, t_scaling, {
        'rmsnorm': t_rmsnorm, 'residual': t_residual,
        'rope': t_rope, 'flash_attn': t_flash, 'silu': t_silu
    }


# Fitting
def fit_alpha_beta(results: List[Tuple[int, float]]) -> Tuple[float, float]:
    """
    Fit t = alpha + M * beta from (size_bytes, time_seconds) pairs.

    Returns (alpha, beta_e2e) where beta_e2e is the end-to-end inverse BW:
    t_allreduce(M) = alpha + M * beta_e2e.

    For collectives (AllReduce/ReduceScatter/AllGather), NCCL internally uses a ring algorithm that
    moves 2*(p-1)/p * M bytes per rank. beta_e2e includes this ring factor.

    To get the link-level beta for the cost model's ring formula (which applies 2*(p-1)/p itself),
    use to_link_params().

    :param results: Benchmarking results.
    :type results: List[Tuple[int, float]]
    :return: Tuple of alpha and beta_e2e, where beta_e2e is the end-to-end inverse BW.
    :rtype: Tuple[float, float]
    """
    if len(results) < 2:
        return 0.0, 0.0
    sizes = np.array([r[0] for r in results], dtype=np.float64)
    times = np.array([r[1] for r in results], dtype=np.float64)
    large = sizes > 1e6
    if large.sum() >= 2:
        sizes_fit, times_fit = sizes[large], times[large]
    else:
        sizes_fit, times_fit = sizes, times
    A = np.column_stack([np.ones_like(sizes_fit), sizes_fit])
    alpha, beta = np.linalg.lstsq(A, times_fit, rcond=None)[0]
    return max(0, alpha), max(0, beta)


def to_link_params(alpha_e2e: float, beta_e2e: float, p: int) -> Tuple[float, float]:
    """
    Convert end-to-end AllReduce alpha/beta to link-level values.

    The cost model's ring formula: t = 2*(p-1)*alpha_link + 2*(p-1)/p * M * beta_link
    The calibration measures: t = alpha_e2e + M * beta_e2e

    Both alpha_e2e and beta_e2e already contain the ring factors:
      alpha_e2e = 2*(p-1) * alpha_link  →  alpha_link = alpha_e2e / (2*(p-1))
      beta_e2e  = 2*(p-1)/p * beta_link →  beta_link  = beta_e2e * p / (2*(p-1))

    After conversion, the cost model reproduces the calibration exactly:
      2*(p-1)*alpha_link + 2*(p-1)/p*M*beta_link = alpha_e2e + M*beta_e2e

    For p=2: alpha_link = alpha_e2e/2, beta_link = beta_e2e (no beta correction)
    For p=4: alpha_link = alpha_e2e/6, beta_link = beta_e2e * 2/3

    :param alpha_e2e: End-to-end AllReduce alpha.
    :type alpha_e2e: float
    :param beta_e2e: End-to-end AllReduce beta.
    :type beta_e2e: float
    :param p: Number of processes in the ring.
    :type p: int
    :return: Link-level values of alpha and beta.
    :rtype: Tuple[float, float]
    """
    if p <= 1:
        return alpha_e2e, beta_e2e
    factor = 2.0 * (p - 1)
    alpha_link = alpha_e2e / factor
    beta_link = beta_e2e * p / factor
    return alpha_link, beta_link


def main():
    """
    Main function executing all benchmarks and fitting functions with the calibration results being
    written to an output file.
    """
    p = argparse.ArgumentParser(description="Hardware calibration (~3 min)")
    p.add_argument("--output", default="hw_config.yaml")
    # Model architecture for compute benchmarks (fixed reference workload)
    p.add_argument("--L", type=int, default=32, help=argparse.SUPPRESS)
    p.add_argument("--d", type=int, default=4096, help=argparse.SUPPRESS)
    p.add_argument("--d_ff", type=int, default=11008, help=argparse.SUPPRESS)
    p.add_argument("--V", type=int, default=32000, help=argparse.SUPPRESS)
    p.add_argument("--n_heads", type=int, default=32, help=argparse.SUPPRESS)
    p.add_argument("--B", type=int, default=64, help=argparse.SUPPRESS)
    p.add_argument("--seq", type=int, default=640, help=argparse.SUPPRESS)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"],
                   help="Compute dtype: bf16 (H100/A100) or fp16 (V100)")
    p.add_argument("--skip-cross-p2p", action="store_true",
                   help="Skip cross-node P2P benchmark (use collective BW as estimate)")
    args = p.parse_args()

    rank, world_size, local_rank = init_distributed()

    # Set global compute dtype (V100 needs FP16 for TensorCore acceleration)
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # Message sizes from 256KB to 512MB, covering both decode (0.5MB) and inference (336MB) regimes
    sizes = [256*1024, 512*1024] + [int(s) for s in np.logspace(np.log10(1e6), np.log10(512e6), 8)]

    if rank == 0:
        print("=" * 70)
        print("Hardware Calibration Micro-Benchmark Suite")
        print("=" * 70)
        print(f"GPUs: {world_size} | Device: {torch.cuda.get_device_name()}")
        print()

    # Create intra-node and cross-node subgroups
    gpus_per_node = 4
    node_id = rank // gpus_per_node
    local_rank_in_node = rank % gpus_per_node
    n_nodes = world_size // gpus_per_node

    # Intra-node group: all GPUs on the same node
    intra_groups = []
    for n in range(n_nodes):
        ranks_on_node = list(range(n * gpus_per_node, (n + 1) * gpus_per_node))
        g = dist.new_group(ranks_on_node)
        intra_groups.append((ranks_on_node, g))
    my_intra_group = intra_groups[node_id][1]

    # Cross-node group: one GPU per node (local_rank=0 from each node)
    cross_ranks = [n * gpus_per_node for n in range(n_nodes)]
    cross_group = dist.new_group(cross_ranks)

    # 1a: Intra-node AllReduce (TP uses this)
    if rank == 0:
        print("── 1/6: NCCL AllReduce ──")
        print(f"  (a) Intra-node ({gpus_per_node} GPUs, 1 node) — used for TP AllReduce")
    ar_intra_results = bench_allreduce(sizes, group=my_intra_group)
    alpha_coll_g = beta_coll_g = bw_coll_g = 0.0
    if rank == 0:
        alpha_e2e, beta_e2e = fit_alpha_beta(ar_intra_results)
        # Convert both alpha and beta to link-level for the cost model's ring formula
        p_intra = gpus_per_node
        alpha_coll_g, beta_coll_g = to_link_params(alpha_e2e, beta_e2e, p_intra)
        bw_coll_g = 1.0 / beta_coll_g if beta_coll_g > 0 else float('inf')
        bw_e2e = 1.0 / beta_e2e if beta_e2e > 0 else float('inf')
        print(f"      e2e:  alpha = {alpha_e2e*1e6:.0f} us, BW = {bw_e2e/1e9:.1f} GB/s")
        print(f"      link: alpha = {alpha_coll_g*1e6:.0f} us, BW = {bw_coll_g/1e9:.1f} GB/s (p={p_intra}, factor={p_intra/(2*(p_intra-1)):.2f})")
        for size, t in ar_intra_results:
            pred = alpha_e2e + size * beta_e2e
            print(f"        M={size/1e6:>6.1f}MB  meas={t*1000:>7.2f}ms  pred={pred*1000:>7.2f}ms")

    # 1a2: Repeated intra-node AllReduce (real TP pattern: 64 calls back-to-back)
    # Only measured for validation: the values from step 1a are used henceforth
    if rank == 0:
        print("  (a2) Repeated intra-node AllReduce (64 calls, TP pattern)")
    # Test at typical TP AllReduce message sizes: B*seq*d*sb/(dp*mb)
    # B=64, seq=640, d=4096, sb=2 → 336 MB for dp=1, 168 MB for dp=2, 84 MB for dp=4
    tp_msg_sizes = [84 * 1024**2, 168 * 1024**2, 336 * 1024**2]
    alpha_coll_g_rep = beta_coll_g_rep = bw_coll_g_rep = 0.0
    rep_results = []
    for msg_size in tp_msg_sizes:
        result = bench_repeated_allreduce(msg_size, n_repeats=64, n_warmup=2, n_iters=5,
                                           group=my_intra_group)
        rep_results.append(result)
    if rank == 0:
        alpha_rep_e2e, beta_rep_e2e = fit_alpha_beta(rep_results)
        # Convert both alpha and beta to link-level
        alpha_coll_g_rep, beta_coll_g_rep = to_link_params(alpha_rep_e2e, beta_rep_e2e, p_intra)
        bw_coll_g_rep = 1.0 / beta_coll_g_rep if beta_coll_g_rep > 0 else float('inf')
        bw_rep_e2e = 1.0 / beta_rep_e2e if beta_rep_e2e > 0 else float('inf')
        print(f"      e2e:  alpha = {alpha_rep_e2e*1e6:.0f} us, BW = {bw_rep_e2e/1e9:.1f} GB/s")
        print(f"      link: alpha = {alpha_coll_g_rep*1e6:.0f} us, BW = {bw_coll_g_rep/1e9:.1f} GB/s")
        print(f"      (vs single-call link: alpha={alpha_coll_g*1e6:.0f}us, BW={bw_coll_g/1e9:.1f}GB/s)")
        for size, t in rep_results:
            pred = alpha_rep_e2e + size * beta_rep_e2e
            print(f"        M={size/1e6:>6.1f}MB  meas={t*1000:>7.2f}ms  pred={pred*1000:>7.2f}ms")
        # Use single-call measurements (repeated AR has NCCL pipelining artifact)
        print(f"      → Using SINGLE-CALL values for alpha_coll_g/beta_coll_g (link-level)")
        # Don't override — keep the single-call values which are already link-level

    # 1b: Cross-node AllReduce (DP ZeRO-2 uses this)
    if rank == 0:
        print(f"  (b) Cross-node (1 GPU per node, {n_nodes} nodes) — used for DP ZeRO-2")
    ar_cross_results = []
    if local_rank_in_node == 0 and n_nodes >= 2:
        ar_cross_results = bench_allreduce(sizes, group=cross_group)
    alpha_coll_n = beta_coll_n = bw_coll_n = 0.0
    if rank == 0:
        if ar_cross_results:
            alpha_cross_e2e, beta_cross_e2e = fit_alpha_beta(ar_cross_results)
            # Convert both alpha and beta to link-level (cross-node group has p = n_nodes)
            p_cross = n_nodes
            alpha_coll_n, beta_coll_n = to_link_params(alpha_cross_e2e, beta_cross_e2e, p_cross)
            bw_coll_n = 1.0 / beta_coll_n if beta_coll_n > 0 else float('inf')
            bw_cross_e2e = 1.0 / beta_cross_e2e if beta_cross_e2e > 0 else float('inf')
            print(f"      e2e:  alpha = {alpha_cross_e2e*1e6:.0f} us, BW = {bw_cross_e2e/1e9:.1f} GB/s")
            print(f"      link: alpha = {alpha_coll_n*1e6:.0f} us, BW = {bw_coll_n/1e9:.1f} GB/s (p={p_cross}, factor={p_cross/(2*(p_cross-1)):.2f})")
            for size, t in ar_cross_results:
                pred = alpha_cross_e2e + size * beta_cross_e2e
                print(f"        M={size/1e6:>6.1f}MB  meas={t*1000:>7.2f}ms  pred={pred*1000:>7.2f}ms")
        else:
            alpha_coll_n, beta_coll_n = alpha_coll_g, beta_coll_g
            bw_coll_n = bw_coll_g
            print(f"      (single node — using intra-node values)")
        print()

    # 2a: Intra-node P2P (rank 0 ↔ rank 1, same node)
    if rank == 0:
        print("── 2/6: NCCL P2P Send/Recv ──")
        print("  (a) Intra-node (rank 0 ↔ rank 1)")
    p2p_intra_results = bench_p2p(sizes, src=0, dst=1)
    alpha_p2p_g = beta_p2p_g = bw_p2p_g = 0.0
    if rank == 0:
        alpha_p2p_g, beta_p2p_g = fit_alpha_beta(p2p_intra_results)
        bw_p2p_g = 1.0 / beta_p2p_g if beta_p2p_g > 0 else float('inf')
        print(f"      alpha = {alpha_p2p_g*1e6:.0f} us, BW = {bw_p2p_g/1e9:.1f} GB/s")

    # 2b: Cross-node P2P (rank 0 ↔ rank gpus_per_node, different nodes)
    if rank == 0:
        print("  (b) Cross-node (rank 0 ↔ rank 4)")
    p2p_cross_results = []
    if n_nodes >= 2:
        p2p_cross_results = bench_p2p(sizes, src=0, dst=gpus_per_node)
    alpha_p2p_n = beta_p2p_n = bw_p2p_n = 0.0
    if rank == 0:
        if p2p_cross_results:
            alpha_p2p_n, beta_p2p_n = fit_alpha_beta(p2p_cross_results)
            bw_p2p_n = 1.0 / beta_p2p_n if beta_p2p_n > 0 else float('inf')
            print(f"      alpha = {alpha_p2p_n*1e6:.0f} us, BW = {bw_p2p_n/1e9:.1f} GB/s")
        else:
            alpha_p2p_n, beta_p2p_n = alpha_p2p_g, beta_p2p_g
            bw_p2p_n = bw_p2p_g
            print(f"      (single node — using intra-node values)")
        print()

    # 3: Compute (real model with FlashAttn)
    F_peak = 0.0
    if rank == 0:
        print("── 3/6: Forward Pass (real model, FlashAttention) ──")
        flops, median_t, F_peak = bench_compute(
            L = args.L,
            d = args.d,
            d_ff = args.d_ff,
            n_heads = args.n_heads,
            V = args.V,
            B = args.B,
            seq = args.seq
        )
        print(f"  FLOP: {flops:.2e} | Time: {median_t*1000:.0f}ms | F_peak: {F_peak/1e12:.0f} TFLOP/s")
        print()

    # 4: HBM BW (workload-realistic)
    mem_bw = 2.0e12
    if rank == 0:
        print("── 4/6: HBM Bandwidth (workload-realistic, B=1 forward pass) ──")
        mem_bw = bench_memory_bandwidth(L=args.L, d=args.d, d_ff=args.d_ff, n_heads=args.n_heads)
        print(f"  {mem_bw/1e12:.2f} TB/s  (weight loading through real model at B=1)")
        print()

    # 5: GEMM sweep
    gemm_m_half = 5.0
    if rank == 0:
        print("── 5/6: GEMM Sweep (gemm_m_half) ──")
        gemm_results = bench_gemm_sweep(d=args.d, d_ff=args.d_ff)
        gemm_peak, gemm_m_half = fit_gemm_m_half(gemm_results)
        print(f"  Peak GEMM: {gemm_peak:.0f} TFLOP/s | gemm_m_half: {gemm_m_half:.1f}")
        for M, tflops in gemm_results:
            eta = M / (M + gemm_m_half) if gemm_m_half > 0 else 1.0
            print(f"    M={M:>6d}  {tflops:>6.0f} TFLOP/s  eta={eta:.3f}")
        print()

    # 6: Decode step
    decode_bw_eff = 0.15
    if rank == 0:
        print("── 6/6: Decode Step (DECODE_BW_EFF, B=1 weight-dominated) ──")
        decode_t, omega_W, omega_KV, total_bytes = bench_decode_step(
            L=args.L, d=args.d, d_ff=args.d_ff, n_heads=args.n_heads, B=1
        )
        # decode_bw_eff = omega_W / (mem_bw * decode_t) but simpler:
        # effective BW for weight loading = omega_W / decode_t (B=1 → KV negligible)
        eff_bw = omega_W / decode_t # weight-loading BW
        decode_bw_eff = eff_bw / mem_bw
        print(f"  Step time: {decode_t*1000:.2f}ms | omega_W: {omega_W/1e9:.1f}GB | "
              f"eff_BW: {eff_bw/1e9:.0f} GB/s | decode_bw_eff: {decode_bw_eff:.3f}")
        print()

    # 7: Non-GEMM per-component overhead
    non_gemm_constant = 0.0
    non_gemm_scaling = 0.0
    if rank == 0:
        print("── 7/7: Non-GEMM Component Overhead (RMSNorm, RoPE, FlashAttn, SiLU) ──")
        ng_const, ng_scale, ng_details = bench_non_gemm_components(
            L=1, d=args.d, d_ff=args.d_ff, n_heads=args.n_heads,
            B=args.B, seq=args.seq
        )
        non_gemm_constant = ng_const # per layer (seconds)
        non_gemm_scaling = ng_scale # per layer at tp=1 (seconds)
        print(f"  Per-layer breakdown:")
        print(f"    RMSNorm×2:     {ng_details['rmsnorm']*2*1000:.3f}ms")
        print(f"    Residual×2:    {ng_details['residual']*2*1000:.3f}ms")
        print(f"    RoPE:          {ng_details['rope']*1000:.3f}ms")
        print(f"    FlashAttn:     {ng_details['flash_attn']*1000:.3f}ms")
        print(f"    SiLU:          {ng_details['silu']*1000:.3f}ms")
        print(f"  Constant (no TP scaling): {non_gemm_constant*1000:.3f}ms/layer")
        print(f"  Scaling (1/tp):           {non_gemm_scaling*1000:.3f}ms/layer at tp=1")
        print(f"  Total at tp=1:            {(non_gemm_constant+non_gemm_scaling)*1000:.3f}ms/layer")
        print(f"  Total at tp=4:            {(non_gemm_constant+non_gemm_scaling/4)*1000:.3f}ms/layer")
        print()

    # Output YAML
    if rank == 0:
        yaml = f"""# Hardware configuration — auto-generated by calibrate.py
# Device: {torch.cuda.get_device_name()}
# GPUs: {world_size} ({n_nodes} nodes x {gpus_per_node} GPUs)
# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

# Compute
flops_per_gpu: {F_peak:.6e}    # {F_peak/1e12:.0f} TFLOP/s
memory_per_gpu: {torch.cuda.get_device_properties(0).total_memory:.6e}
memory_bandwidth: {mem_bw:.6e}  # {mem_bw/1e12:.2f} TB/s (workload-realistic)

# Collective communication (AllReduce, ReduceScatter, AllGather)
# beta values are LINK-LEVEL: ring overhead removed from calibration.
# Cost model applies ring factor internally: t = 2(p-1)*a + 2(p-1)/p * M * beta
# Intra-node: calibrated with p={gpus_per_node} GPUs on 1 node (used for TP AllReduce)
alpha_coll_g: {alpha_coll_g:.6e}    # {alpha_coll_g*1e6:.0f} us
beta_coll_g: {beta_coll_g:.6e}     # {bw_coll_g/1e9:.1f} GB/s (link-level)
# Cross-node: calibrated with p={n_nodes} nodes, 1 GPU each (used for DP ZeRO-2)
alpha_coll_n: {alpha_coll_n:.6e}    # {alpha_coll_n*1e6:.0f} us
beta_coll_n: {beta_coll_n:.6e}     # {bw_coll_n/1e9:.1f} GB/s (link-level)
alpha_coll_r: {alpha_coll_n*2:.6e}
beta_coll_r: {beta_coll_n*2:.6e}

# Point-to-point communication (PP P2P)
# Intra-node: rank 0 <-> rank 1 (same node)
alpha_p2p_g: {alpha_p2p_g:.6e}     # {alpha_p2p_g*1e6:.0f} us
beta_p2p_g: {beta_p2p_g:.6e}      # {bw_p2p_g/1e9:.1f} GB/s
# Cross-node: rank 0 <-> rank {gpus_per_node} (different nodes)
alpha_p2p_n: {alpha_p2p_n:.6e}     # {alpha_p2p_n*1e6:.0f} us
beta_p2p_n: {beta_p2p_n:.6e}      # {bw_p2p_n/1e9:.1f} GB/s
alpha_p2p_r: {alpha_p2p_n*2:.6e}
beta_p2p_r: {beta_p2p_n*2:.6e}

# GEMM & decode
gemm_m_half: {gemm_m_half:.1f}
decode_bw_eff: {decode_bw_eff:.3f}

# Non-GEMM overhead per layer (RMSNorm, RoPE, FlashAttn, SiLU, residuals)
# non_GEMM(tp) = L * (non_gemm_constant + non_gemm_scaling / tp)
# F_peak absorbs this at tp=1; cost model adds the ADDITIONAL overhead at tp>1.
non_gemm_constant: {non_gemm_constant:.6e}   # {non_gemm_constant*1000:.3f} ms/layer (RMSNorm×2 + residual×2)
non_gemm_scaling: {non_gemm_scaling:.6e}   # {non_gemm_scaling*1000:.3f} ms/layer at tp=1 (RoPE + FlashAttn + SiLU)
non_gemm_batch_cal: {args.B}               # batch size used for non_gemm calibration
"""
        with open(args.output, 'w') as f:
            f.write(yaml)

        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"  F_peak:          {F_peak/1e12:.0f} TFLOP/s")
        print(f"  HBM BW:          {mem_bw/1e12:.2f} TB/s (workload-realistic)")
        print(f"  Coll BW intra:   {bw_coll_g/1e9:.1f} GB/s  (alpha={alpha_coll_g*1e6:.0f}us)")
        print(f"  Coll BW cross:   {bw_coll_n/1e9:.1f} GB/s  (alpha={alpha_coll_n*1e6:.0f}us)")
        print(f"  P2P BW intra:    {bw_p2p_g/1e9:.1f} GB/s  (alpha={alpha_p2p_g*1e6:.0f}us)")
        print(f"  P2P BW cross:    {bw_p2p_n/1e9:.1f} GB/s  (alpha={alpha_p2p_n*1e6:.0f}us)")
        print(f"  gemm_m_half:     {gemm_m_half:.1f}")
        print(f"  decode_bw_eff:   {decode_bw_eff:.3f}")
        print(f"\n  Config: {args.output}")
        print("=" * 70)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
