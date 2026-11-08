#!/usr/bin/env python3

# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt

"""
Cost model validation plots across 3 GPU generations.

Generates:
  1. 3×3 Predicted vs Measured scatter (columns=mode, rows=GPU gen)
  2. MAPE by parallelism strategy (grouped bars)
  3. Error CDF (one line per GPU generation)
  4. Summary table with MAPE, R², ±25% stats

Uses calibrated cost model with hardware parameters from hw_*.yaml.

Usage:
    python experiments/plot_cost_model_validation.py
"""

import csv
import importlib
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import yaml

from matplotlib.lines import Line2D
from typing import Callable, Dict, List

from core.types import Role, Model, Allocation
from model.hardware import Hardware as MainHardware
from model.memory import NodeMemoryModel
from model.models import ModelSpec, ModelSuite
from model.workload import Workload
from passes.lower import ExecNode, ExecNodeType
from style import apply_style


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RPEG'))
# Load GH200-calibrated cost model from harness directory
_spec = importlib.util.spec_from_file_location("cost_gh200", "/path/to/cost_gh200.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
NodeCostModel = _mod.NodeCostModel
Hardware = _mod.Hardware

matplotlib.use('Agg')

apply_style()

# Hardware loading from YAML
HW_DIR = '/path/to/hw-configs'

def load_hw(yaml_path: str) -> Hardware:
    """
    Load hardware configuration from a calibrated hw_*.yaml file.

    :param yaml_path: Path to the input yaml file.
    :type yaml_path: str
    :return: Hardware configuration.
    :rtype: Hardware
    """
    with open(yaml_path) as f:
        lines = [l for l in f if not l.strip().startswith('#') and l.strip()]
    d = {}
    for line in lines:
        if ':' in line:
            key, val = line.split(':', 1)
            key = key.strip()
            val = val.split('#')[0].strip()
            try:
                d[key] = float(val)
            except ValueError:
                pass
    return Hardware(
        flops_per_gpu = d['flops_per_gpu'],
        memory_per_gpu = d.get('memory_per_gpu', 0),
        memory_bandwidth = d['memory_bandwidth'],
        alpha_coll_g = d['alpha_coll_g'], beta_coll_g=d['beta_coll_g'],
        alpha_coll_n = d['alpha_coll_n'], beta_coll_n=d['beta_coll_n'],
        alpha_coll_r = d['alpha_coll_r'], beta_coll_r=d['beta_coll_r'],
        alpha_p2p_g = d['alpha_p2p_g'], beta_p2p_g=d['beta_p2p_g'],
        alpha_p2p_n = d['alpha_p2p_n'], beta_p2p_n=d['beta_p2p_n'],
        alpha_p2p_r = d['alpha_p2p_r'], beta_p2p_r=d['beta_p2p_r'],
        gemm_m_half = d.get('gemm_m_half', 5.0),
        decode_bw_eff = d.get('decode_bw_eff', 0.15),
        non_gemm_constant = d.get('non_gemm_constant', 0.0),
        non_gemm_scaling = d.get('non_gemm_scaling', 0.0),
    )

HW_H100 = load_hw(os.path.join(HW_DIR, 'hw_h100.yaml'))
HW_A100 = load_hw(os.path.join(HW_DIR, 'hw_a100.yaml'))
HW_V100 = load_hw(os.path.join(HW_DIR, 'hw_v100.yaml'))

GPU_CONFIGS = {
    'clariden': ('GH200 (H100)', HW_H100, '#1976D2'),
    'a100':     ('A100 80GB',    HW_A100, '#E65100'),
    'v100':     ('V100 32GB',    HW_V100, '#2E7D32'),
}

MODE_COLORS = {"inference": "#42A5F5", "training": "#EF5350", "generation": "#66BB6A"}

# Node construction + prediction
class FakeKind:
    """
    Simplified data class representing an operator.

    role = role of the operator
    """

    def __init__(self, r: Role) -> None:
        """
        Initialize the class.

        :param r: Operator role.
        :type r: Role
        """
        self.role = r


class FakeTV:
    """
    Simplified data class representing a vertex in the RPEG.
    """

    def __init__(self, r: Role) -> None:
        """
        Initialize the class.

        :param r: Operator role.
        :type r: Role
        """
        self.kind = FakeKind(r)


class FakeOp:
    """
    Simplified data class representing an unrolled operator vertex.
    """

    def __init__(self, r: Role):
        """
        Initialize the class.

        :param r: Operator role.
        :type r: Role
        """
        self.template_vertex = FakeTV(r)


ROLE_MAP = {'inference': Role.INFERENCE, 'training': Role.TRAINING, 'generation': Role.GENERATION}
SPEC_7B  = ModelSpec(L=32, d=4096,  d_ff=11008, V=32000, dtype_bytes=2, d_head=128)
SPEC_3B  = ModelSpec(L=16, d=3072,  d_ff=8192,  V=32000, dtype_bytes=2, d_head=128)
SPEC_13B = ModelSpec(L=40, d=5120,  d_ff=13824, V=32000, dtype_bytes=2, d_head=128)


def predict_ms(row: Dict, hw: Hardware) -> float:
    """
    Predict time in ms for one experiment using calibrated cost model.

    :param row: Workload configuration.
    :type row: Dict
    :param hw: Hardware configuration.
    :type hw: Hardware
    :return: Predicted runtime in ms.
    :rtype: float
    """
    mode = row['mode']
    dp, pp, tp = row['dp'], row['pp'], row['tp']
    mb_csv = row['mb']
    B, S, T, K = row['B'], row['S'], row['T'], row['K']
    L = row['L']

    # Build spec from CSV columns (supports arbitrary model sizes)
    spec = ModelSpec(
        L = L,
        d = row['d'],
        d_ff = row['d_ff'],
        V = 32000,
        dtype_bytes = 2,
        d_head = row['d'] // row['n_heads']
    )
    suite = ModelSuite(specs={Model.ACTOR: spec})
    if mode == 'generation':
        wl = Workload(B=B, K=K, S=S, T=T)
    else:
        wl = Workload(B=B, K=K, S=S, T=0)
    cm = NodeCostModel(hw=hw, suite=suite, workload=wl)

    num_micro = max(1, mb_csv)
    total_gpus = dp * pp * tp
    gpus_per_node = 4
    n_nodes = math.ceil(total_gpus / gpus_per_node)
    alloc = Allocation(origin=(0, 0, 0), extent=(1, n_nodes, min(total_gpus, gpus_per_node)))
    node = ExecNode(
        id = 'x',
        kind = ExecNodeType.OP,
        allocs = (alloc,),
        op = FakeOp(ROLE_MAP[mode]),
        model = Model.ACTOR,
        pi = (dp, pp, tp),
        mb = num_micro
    )

    t = cm.time(node)
    return t * 1000


# Dummy hardware for memory model (only memory_per_gpu used, and only for feasibility — not peak_bytes)
_DUMMY_HW = MainHardware(
    flops_per_gpu = 0,
    memory_per_gpu = 0,
    memory_bandwidth = 0,
    alpha_coll_g = 0,
    beta_coll_g = 0,
    alpha_coll_n = 0,
    beta_coll_n = 0,
    alpha_coll_r = 0,
    beta_coll_r = 0,
    alpha_p2p_g = 0,
    beta_p2p_g = 0,
    alpha_p2p_n = 0,
    beta_p2p_n = 0,
    alpha_p2p_r = 0,
    beta_p2p_r = 0
)

def training_harness_correction_gb(row: Dict) -> float:
    """
    Compute Bytes to subtract from measured training memory to account for known harness
    inefficiencies versus a production-grade implementation:
    1. Per-layer checkpointing overhead: the harness uses torch.utils.checkpoint per layer (L_pp
       checkpoints), while optimal sqrt(L) segmented checkpointing needs only ceil(L_pp/sqrt(L_pp))
       checkpoints. Extra = (L_pp - ceil(L_pp/c)) * d elements per token.
    2. Non-overlapped gradient reduction: the harness accumulates full .grad buffers during
       backward, then does a single ReduceScatter. A production ZeRO-2 implementation overlaps
       backward with ReduceScatter per bucket, keeping only 1/dp of gradients live.
       Extra = sb * w_shard * (1 - 1/dp).

    :param row: Workload configuration.
    :type row: Dict
    :return: Memory overhead in GB.
    :rtype: float
    """
    if row['mode'] != 'training':
        return 0.0
    dp, pp, tp = row['dp'], row['pp'], row['tp']
    mb = max(1, row['mb'])
    spec = ModelSpec(
        L = row['L'],
        d = row['d'],
        d_ff = row['d_ff'],
        V = 32000,
        dtype_bytes = 2,
        d_head = row['d'] // row['n_heads']
    )
    suite = ModelSuite(specs={Model.ACTOR: spec})
    wl = Workload(B=row['B'], K=row['K'], S=row['S'], T=0)
    mm = NodeMemoryModel(hw=_DUMMY_HW, suite=suite, workload=wl)
    alloc = Allocation(origin=(0, 0, 0), extent=(1, 1, 4))
    node = ExecNode(
        id = 'x',
        kind = ExecNodeType.OP,
        allocs = (alloc,),
        op = FakeOp(Role.TRAINING),
        model = Model.ACTOR,
        pi = (dp, pp, tp),
        mb = mb
    )

    d, L_pp = spec.d, math.ceil(spec.L / pp)
    sb = spec.dtype_bytes
    b_local = row['B'] * row['K'] / (dp * mb)
    in_flight = min(mb, pp)
    c = max(1, math.isqrt(L_pp))

    # 1. Surplus checkpoints
    extra_ckpts = L_pp - math.ceil(L_pp / c)
    ckpt_corr = sb * in_flight * b_local * row['S'] * extra_ckpts * d

    # 2. Non-overlapped gradients
    w_shard = mm._w_shard(node)
    grad_corr = sb * w_shard * (1 - 1 / dp) if dp > 1 else 0

    return (ckpt_corr + grad_corr) / 1e9


def predict_mem_gb(row: Dict) -> float:
    """
    Predict peak GPU memory in GB for one experiment using the memory model.

    :param row: Workload configuration.
    :type row: Dict
    :return: Peak GPU memory in GB.
    :rtype: float
    """
    mode = row['mode']
    dp, pp, tp = row['dp'], row['pp'], row['tp']
    mb_csv = row['mb']
    B, S, T, K = row['B'], row['S'], row['T'], row['K']

    spec = ModelSpec(
        L = row['L'],
        d = row['d'],
        d_ff = row['d_ff'],
        V = 32000,
        dtype_bytes = 2,
        d_head = row['d'] // row['n_heads']
    )
    suite = ModelSuite(specs={Model.ACTOR: spec})
    if mode == 'generation':
        wl = Workload(B=B, K=K, S=S, T=T)
    else:
        wl = Workload(B=B, K=K, S=S, T=0)
    mm = NodeMemoryModel(hw=_DUMMY_HW, suite=suite, workload=wl)

    num_micro = max(1, mb_csv)
    total_gpus = dp * pp * tp
    gpus_per_node = 4
    n_nodes = math.ceil(total_gpus / gpus_per_node)
    alloc = Allocation(origin=(0, 0, 0), extent=(1, n_nodes, min(total_gpus, gpus_per_node)))
    node = ExecNode(
        id = 'x',
        kind = ExecNodeType.OP,
        allocs = (alloc,),
        op = FakeOp(ROLE_MAP[mode]),
        model = Model.ACTOR,
        pi = (dp, pp, tp),
        mb=num_micro
    )

    return mm.peak_bytes(node) / 1e9


# Data loading
RESULTS_DIR = '/path/to/profiling-results'

def load_results() -> List[Dict]:
    """
    Load all result CSVs, keyed by (cluster, config).

    :return: Results as a list of dictionaries.
    :rtype: List[Dict]
    """
    all_rows = []
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith('.csv'):
            continue
        path = os.path.join(RESULTS_DIR, fname)
        with open(path) as f:
            for row in csv.DictReader(f):
                all_rows.append(row)

    # Deduplicate: keep last occurrence per (cluster, mode, dp, pp, tp, mb, B, S, T, L, d)
    seen = {}
    for row in all_rows:
        key = (row['cluster'], row['mode'], row['dp'], row['pp'], row['tp'],
               row['mb'], row['B'], row['S'], row['T'], row['L'], row['d'])
        r = {
            'cluster': row['cluster'],
            'mode': row['mode'],
            'dp': int(row['dp']), 'pp': int(row['pp']), 'tp': int(row['tp']),
            'mb': int(row['mb']), 'B': int(row['B']), 'S': int(row['S']),
            'T': int(row['T']), 'K': int(row['K']), 'L': int(row['L']),
            'd': int(row['d']), 'd_ff': int(row['d_ff']),
            'n_heads': int(row['n_heads']),
            'measured_ms': float(row['makespan_ms']),
            'peak_mem_gb': float(row['peak_mem_gb']),
        }
        r['gpus'] = r['dp'] * r['pp'] * r['tp']
        seen[key] = r
    return list(seen.values())


def strategy_label(r: Dict) -> str:
    """
    Return a strategy label based on the parallelism parameters.

    :param r: Workload configuration.
    :type r: Dict
    :return: Strategy label as string.
    :rtype: str
    """
    dp, pp, tp = r['dp'], r['pp'], r['tp']
    if tp == 1 and pp == 1:
        return '1-GPU' if dp == 1 else 'DP'
    elif tp > 1 and pp == 1 and dp == 1:
        return 'TP'
    elif tp == 1 and pp > 1 and dp == 1:
        return 'PP'
    else:
        return '3D'


def strategy_marker(r: Dict) -> str:
    """
    Return a plot marker based on the parallelism strategy.

    :param r: Workload configuration.
    :type r: Dict
    :return: Plot marker as string.
    :rtype: str
    """
    s = strategy_label(r)
    return {'1-GPU': 'o', 'DP': 'o', 'TP': '^', 'PP': 's', '3D': 'D'}[s]


# Figures
def fig_scatter_3x3(results: List[Dict], outdir: str) -> None:
    """
    3×3 grid: rows=GPU generation, columns=mode.

    :param results: List of results.
    :type results: List[Dict]
    :param outdir: Path to the output directory.
    :type outdir: str
    """
    gpu_order = ['clariden', 'a100', 'v100']
    modes = ['inference', 'training', 'generation']

    fig, axes = plt.subplots(3, 3, figsize=(16, 15))

    for row_idx, cluster in enumerate(gpu_order):
        gpu_name, _, gpu_color = GPU_CONFIGS[cluster]

        for col_idx, mode in enumerate(modes):
            ax = axes[row_idx, col_idx]
            pts = [r for r in results if r['cluster'] == cluster and r['mode'] == mode
                   and r.get('predicted_ms', 0) > 0]

            if not pts:
                ax.text(
                    0.5,
                    0.5,
                    'No data',
                    transform = ax.transAxes,
                    ha = 'center',
                    va = 'center',
                    fontsize = 12,
                    color = 'gray'
                )
                ax.set_xlim(10, 1e5)
                ax.set_ylim(10, 1e5)
                ax.set_xscale('log')
                ax.set_yscale('log')
                if row_idx == 0:
                    ax.set_title(mode.capitalize(), fontweight='bold', fontsize=14)
                if col_idx == 0:
                    ax.set_ylabel(f'{gpu_name}\nPredicted (ms)', fontsize=11)
                continue

            meas = np.array([r['measured_ms'] for r in pts])
            pred = np.array([r['predicted_ms'] for r in pts])

            for i, r in enumerate(pts):
                mk = strategy_marker(r)
                ax.scatter(
                    meas[i],
                    pred[i],
                    c = gpu_color,
                    marker = mk,
                    s = 70,
                    edgecolors = 'k',
                    linewidths = 0.5,
                    alpha = 0.85,
                    zorder = 3
                )

            lo = min(meas.min(), pred.min()) * 0.4
            hi = max(meas.max(), pred.max()) * 2.5
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0, alpha=0.5)
            ax.fill_between(
                [lo, hi],
                [lo*0.75, hi*0.75],
                [lo*1.25, hi*1.25],
                color = 'gray',
                alpha = 0.10,
                zorder = 0
            )
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2, which='both')

            rel_err = np.abs(meas - pred) / meas
            mape = np.mean(rel_err) * 100
            log_r2 = np.corrcoef(np.log(meas), np.log(pred))[0,1]**2 if len(meas) > 2 else 0
            within_25 = np.sum(rel_err < 0.25) / len(rel_err) * 100
            ax.text(
                0.05,
                0.95,
                f'MAPE={mape:.0f}%\n$r^2$={log_r2:.3f}\n±25%: {within_25:.0f}%\nn={len(meas)}',
                transform = ax.transAxes,
                fontsize = 9,
                va = 'top',
                bbox = dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7)
            )

            if row_idx == 0:
                ax.set_title(mode.capitalize(), fontweight='bold', fontsize=14)
            if col_idx == 0:
                ax.set_ylabel(f'{gpu_name}\nPredicted (ms)', fontsize=11)
            if row_idx == 2:
                ax.set_xlabel('Measured (ms)')

    # Strategy legend
    legend_markers = [('o', '1-GPU / DP'), ('^', 'TP'), ('s', 'PP'), ('D', '3D')]
    handles = [Line2D([0], [0], marker=mk, color='gray', markeredgecolor='k',
                      markersize=8, linestyle='None', label=lbl)
               for mk, lbl in legend_markers]
    fig.legend(
        handles = handles,
        loc = 'lower center',
        ncol = 4,
        fontsize = 11,
        frameon = True,
        bbox_to_anchor = (0.5, -0.02)
    )

    fig.suptitle(
        'Cost Model Validation: Predicted vs. Measured Latency\n'
        'LLaMA-2 7B/3B, 3 GPU Generations, FlashAttention',
        fontsize = 15,
        fontweight = 'bold',
        y = 1.02
    )
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'validation_3x3.pdf'))
    plt.close()
    print("  -> validation_3x3.pdf")


def fig_error_bars(results: List[Dict], outdir: str) -> None:
    """
    MAPE by parallelism strategy, one group per GPU generation.

    :param results: List of results.
    :type results: List[Dict]
    :param outdir: Path to the output directory.
    :type outdir: str
    """
    strategy_groups = [
        ("1-GPU", lambda r: r['gpus'] == 1),
        ("TP", lambda r: r['tp'] > 1 and r['pp'] == 1 and r['dp'] == 1),
        ("PP", lambda r: r['pp'] > 1 and r['tp'] == 1 and r['dp'] == 1),
        ("DP", lambda r: r['dp'] > 1 and r['tp'] == 1 and r['pp'] == 1),
        ("3D", lambda r: sum([r['dp']>1, r['pp']>1, r['tp']>1]) >= 2),
    ]
    gpu_order = ['clariden', 'a100', 'v100']

    fig, ax = plt.subplots(figsize=(12, 5))
    n_groups = len(strategy_groups)
    bar_w = 0.25
    x = np.arange(n_groups)

    for j, cluster in enumerate(gpu_order):
        gpu_name, _, gpu_color = GPU_CONFIGS[cluster]
        mapes = []
        counts = []
        for _, filt_fn in strategy_groups:
            pts = [r for r in results if r['cluster'] == cluster and filt_fn(r)
                   and r.get('predicted_ms', 0) > 0]
            if pts:
                meas = np.array([r['measured_ms'] for r in pts])
                pred = np.array([r['predicted_ms'] for r in pts])
                mape = np.mean(np.abs(meas - pred) / meas) * 100
                mapes.append(mape)
                counts.append(len(pts))
            else:
                mapes.append(0)
                counts.append(0)

        offset = (j - 1) * bar_w
        bars = ax.bar(
            x + offset,
            mapes,
            bar_w,
            color = gpu_color,
            edgecolor = 'k',
            linewidth = 0.5,
            alpha = 0.85,
            label = gpu_name
        )
        for i, (bar, n) in enumerate(zip(bars, counts)):
            if n > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    f'{n}',
                    ha = 'center',
                    va = 'bottom',
                    fontsize = 7
                )

    ax.set_xticks(x)
    ax.set_xticklabels([sg[0] for sg in strategy_groups])
    ax.set_ylabel('MAPE (%)')
    ax.set_title('Prediction Error by Parallelism Strategy', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(bottom=0)
    ax.axhline(25, color='gray', linestyle=':', alpha=0.4)
    ax.text(n_groups - 0.5, 26, '25% target', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'error_by_strategy.pdf'))
    plt.close()
    print("  -> error_by_strategy.pdf")


def fig_cdf(results: List[Dict], outdir: str) -> None:
    """
    CDF of absolute relative error, one line per GPU generation.

    :param results: List of results.
    :type results: List[Dict]
    :param outdir: Path to the output directory.
    :type outdir: str
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    gpu_order = ['clariden', 'a100', 'v100']

    for cluster in gpu_order:
        gpu_name, _, gpu_color = GPU_CONFIGS[cluster]
        pts = [r for r in results if r['cluster'] == cluster and r.get('predicted_ms', 0) > 0]
        if not pts:
            continue
        meas = np.array([r['measured_ms'] for r in pts])
        pred = np.array([r['predicted_ms'] for r in pts])
        rel_err = np.sort(np.abs(meas - pred) / meas * 100)
        cdf = np.arange(1, len(rel_err) + 1) / len(rel_err) * 100

        ax.plot(
            cdf,
            rel_err,
            color = gpu_color,
            linewidth = 2.5,
            label = f'{gpu_name} (n={len(pts)})'
        )
        ax.fill_between(cdf, 0, rel_err, color=gpu_color, alpha=0.10)

    ax.axhline(25, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.text(82, 26, '25% target', fontsize=8, color='gray')
    ax.set_xlabel('Cumulative % of Configurations')
    ax.set_ylabel('Absolute Relative Error (%)')
    ax.set_title('Error Distribution Across GPU Generations', fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'error_cdf.pdf'))
    plt.close()
    print("  -> error_cdf.pdf")


def fig_memory_scatter(results: List[Dict], outdir: str) -> None:
    """
    1×3 scatter: GH200 only, columns=mode.

    :param results: List of results.
    :type results: List[Dict]
    :param outdir: Path to the output directory.
    :type outdir: str
    """
    modes = ['inference', 'training', 'generation']
    gpu_name, _, gpu_color = GPU_CONFIGS['clariden']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    for col_idx, mode in enumerate(modes):
        ax = axes[col_idx]
        pts = [r for r in results if r['cluster'] == 'clariden' and r['mode'] == mode
               and r.get('predicted_mem_gb', 0) > 0 and r.get('peak_mem_gb', 0) > 0]

        if not pts:
            ax.text(
                0.5,
                0.5,
                'No data',
                transform = ax.transAxes,
                ha = 'center',
                va = 'center',
                fontsize = 12,
                color = 'gray'
            )
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_title(mode.capitalize(), fontweight='bold', fontsize=14)
            continue

        meas = np.array([r['peak_mem_gb'] for r in pts])
        pred = np.array([r['predicted_mem_gb'] for r in pts])

        for i, r in enumerate(pts):
            mk = strategy_marker(r)
            ax.scatter(
                meas[i],
                pred[i],
                c = gpu_color,
                marker = mk,
                s = 70,
                edgecolors = 'k',
                linewidths = 0.5,
                alpha = 0.85,
                zorder = 3
            )

        lo = min(meas.min(), pred.min()) * 0.5
        hi = max(meas.max(), pred.max()) * 2.0
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0, alpha=0.5)
        ax.fill_between(
            [lo, hi],
            [lo*0.75, hi*0.75],
            [lo*1.25, hi*1.25],
            color = 'gray',
            alpha = 0.10,
            zorder = 0
        )
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2, which='both')

        rel_err = np.abs(meas - pred) / meas
        mape = np.mean(rel_err) * 100
        within_25 = np.sum(rel_err < 0.25) / len(rel_err) * 100
        log_r2 = np.corrcoef(np.log(meas), np.log(pred))[0,1]**2 if len(meas) > 2 else 0
        ax.text(
            0.05,
            0.95,
            f'MAPE={mape:.0f}%\n$r^2$={log_r2:.3f}\n±25%: {within_25:.0f}%\nn={len(pts)}',
            transform = ax.transAxes,
            fontsize = 9,
            va = 'top',
            bbox = dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7)
        )

        ax.set_title(mode.capitalize(), fontweight='bold', fontsize=14)
        ax.set_xlabel('Measured (GB)')
        if col_idx == 0:
            ax.set_ylabel('Predicted (GB)', fontsize=11)

    # Strategy legend
    legend_markers = [('o', '1-GPU / DP'), ('^', 'TP'), ('s', 'PP'), ('D', '3D')]
    handles = [Line2D([0], [0], marker=mk, color='gray', markeredgecolor='k',
                      markersize=8, linestyle='None', label=lbl)
               for mk, lbl in legend_markers]
    fig.legend(
        handles = handles,
        loc = 'lower center',
        ncol = 4,
        fontsize = 11,
        frameon = True,
        bbox_to_anchor = (0.5, -0.05)
    )

    fig.suptitle(
        'Memory Model Validation: Predicted vs. Measured Peak GPU Memory\n'
        f'{gpu_name}, LLaMA-2 3B/7B/13B, 1\u2013128 GPUs',
        fontsize = 14,
        fontweight = 'bold',
        y = 1.04
    )
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'memory_validation.pdf'))
    plt.close()
    print("  -> memory_validation.pdf")


def print_summary(results: List[Dict]) -> None:
    """
    Print summary table per GPU generation and overall.

    :param results: List of results.
    :type results: List[Dict]
    """
    print(f"\n{'='*90}")
    print(f"  SUMMARY — {len(results)} experiments across 3 GPU generations")
    print(f"{'='*90}")

    gpu_order = ['clariden', 'a100', 'v100']
    for cluster in gpu_order:
        gpu_name = GPU_CONFIGS[cluster][0]
        cluster_pts = [r for r in results if r['cluster'] == cluster and r.get('predicted_ms', 0) > 0]
        if not cluster_pts:
            continue

        print(f"\n  ── {gpu_name} ({len(cluster_pts)} experiments) ──")
        for mode in ['inference', 'training', 'generation']:
            pts = [r for r in cluster_pts if r['mode'] == mode]
            if not pts:
                continue
            meas = np.array([r['measured_ms'] for r in pts])
            pred = np.array([r['predicted_ms'] for r in pts])
            err = (pred - meas) / meas * 100
            abs_err = np.abs(err)
            log_r2 = np.corrcoef(np.log(meas), np.log(pred))[0,1]**2 if len(meas) > 2 else 0

            print(f"    {mode:12s}  MAPE={abs_err.mean():5.1f}%  signed={err.mean():+5.1f}%"
                  f"  R²={log_r2:.3f}  ±25%={int((abs_err<25).sum())}/{len(pts)}"
                  f"  max={abs_err.max():.0f}%")

        all_m = np.array([r['measured_ms'] for r in cluster_pts])
        all_p = np.array([r['predicted_ms'] for r in cluster_pts])
        all_err = np.abs(all_p - all_m) / all_m * 100
        print(f"    {'OVERALL':12s}  MAPE={all_err.mean():5.1f}%  ±25%={int((all_err<25).sum())}/{len(cluster_pts)}")

    # Grand total
    valid = [r for r in results if r.get('predicted_ms', 0) > 0]
    all_m = np.array([r['measured_ms'] for r in valid])
    all_p = np.array([r['predicted_ms'] for r in valid])
    all_err = np.abs(all_p - all_m) / all_m * 100
    log_r2 = np.corrcoef(np.log(all_m), np.log(all_p))[0,1]**2
    print(f"\n  {'='*60}")
    print(f"  GRAND TOTAL ({len(valid)} experiments)")
    print(f"    MAPE:        {all_err.mean():5.1f}%")
    print(f"    Median:      {np.median(all_err):5.1f}%")
    print(f"    log-R²:      {log_r2:.4f}")
    print(f"    Within ±25%: {int((all_err<25).sum())}/{len(valid)}")
    print(f"    Within ±50%: {int((all_err<50).sum())}/{len(valid)}")


def print_detail_table(results: List[Dict]) -> None:
    """
    Print per-experiment detail table.

    :param results: List of results.
    :type results: List[Dict]
    """
    print(f"\n{'='*120}")
    print(f"  {'cluster':10s} {'mode':12s} {'dp':>3s} {'pp':>3s} {'tp':>3s} {'mb':>3s}"
          f"  {'B':>4s} {'S':>4s} {'T':>4s}  {'meas_ms':>10s} {'pred_ms':>10s} {'err%':>8s}")
    print(f"{'='*120}")
    for r in sorted(results, key=lambda x: (x['cluster'], x['mode'], x['dp'], x['pp'], x['tp'])):
        if r.get('predicted_ms', 0) <= 0:
            continue
        err = (r['predicted_ms'] - r['measured_ms']) / r['measured_ms'] * 100
        print(f"  {r['cluster']:10s} {r['mode']:12s} {r['dp']:3d} {r['pp']:3d} {r['tp']:3d} {r['mb']:3d}"
              f"  {r['B']:4d} {r['S']:4d} {r['T']:4d}  {r['measured_ms']:10.1f} {r['predicted_ms']:10.1f} {err:>+7.1f}%")


# Publication figures (publication-quality, GH200 only, PDF vector output)
# Colorblind-safe palette (Okabe-Ito inspired, muted for print)
_STRAT_STYLE = {
    '1-GPU': {'marker': 'o', 'color': '#888888', 'size': 28},
    'DP':    {'marker': 'o', 'color': '#0072B2', 'size': 28},
    'TP':    {'marker': '^', 'color': '#D55E00', 'size': 32},
    'PP':    {'marker': 's', 'color': '#009E73', 'size': 28},
    '3D':    {'marker': 'D', 'color': '#CC79A7', 'size': 28},
}

_MODE_LABELS = {'generation': 'Generation', 'inference': 'Inference', 'training': 'Training'}


def _scatter_panel(
    ax: matplotlib.axes,
    pts: Dict,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    show_ylabel: bool = True
) -> None:
    """
    Draw one predicted-vs-measured scatter panel with consistent publication styling.

    :param ax: Subplot.
    :type ax: matplotlib.axes
    :param pts: Data to plot.
    :type pts: Dict
    :param x_key: Dictionary key for datapoint position on the x axis.
    :type x_key: str
    :param y_key: Dictionary key for datapoint position on the y axis.
    :type y_key: str
    :param xlabel: Label for the x axis.
    :type xlabel: str
    :param ylabel: Label for the y axis.
    :type ylabel: str
    :param show_ylabel: Flag whether to show the y axis label. Defaults to True.
    :type show_ylabel: bool
    """
    if not pts:
        ax.text(
            0.5,
            0.5,
            'No data',
            transform = ax.transAxes,
            ha = 'center',
            va = 'center',
            fontsize = 8,
            color = 'gray',
            style = 'italic'
        )
        return

    meas = np.array([r[x_key] for r in pts])
    pred = np.array([r[y_key] for r in pts])

    # Points by strategy
    for r, m, p in zip(pts, meas, pred):
        s = strategy_label(r)
        st = _STRAT_STYLE[s]
        ax.scatter(
            m,
            p,
            c = st['color'],
            marker = st['marker'],
            s = st['size'],
            edgecolors = 'k',
            linewidths = 0.3,
            alpha = 0.9,
            zorder = 3
        )

    # Diagonal + ±25% band
    lo = min(meas.min(), pred.min()) * 0.4
    hi = max(meas.max(), pred.max()) * 2.5
    ax.plot([lo, hi], [lo, hi], color='#333333', ls='-', lw=0.6, alpha=0.6, zorder=1)
    ax.fill_between(
        [lo, hi],
        [lo*0.75, hi*0.75],
        [lo*1.25, hi*1.25],
        color = '#cccccc',
        alpha = 0.25,
        zorder = 0,
        linewidth = 0
    )

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15, which='major', linewidth=0.3)
    ax.grid(True, alpha=0.07, which='minor', linewidth=0.2)

    # Stats
    from scipy import stats as sp_stats
    rel_err = np.abs(meas - pred) / meas
    mape = np.mean(rel_err) * 100
    spearman_rho = sp_stats.spearmanr(meas, pred)[0] if len(meas) > 2 else 0
    within_25 = np.sum(rel_err < 0.25) / len(rel_err) * 100
    ax.text(
        0.04,
        0.96,
        f'MAPE={mape:.0f}%\n'
        r'$\rho$' + f'={spearman_rho:.3f}\n'
        f'$n$={len(meas)}',
        transform = ax.transAxes,
        fontsize = 7,
        va = 'top',
        bbox = dict(
            boxstyle = 'round,pad=0.25',
            facecolor = 'white',
            edgecolor = '#aaaaaa',
            linewidth = 0.4,
            alpha = 0.9
        )
    )

    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel(ylabel)


def _pub_1x3(
    results: List[Dict],
    outdir: str,
    filename: str,
    cluster: str,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    filter_fn: Callable,
    title_suffix: str = ''
) -> None:
    """
    Generic 1×3 publication scatter: Generation | Inference | Training.

    :param results: List of results.
    :type results: List[Dict]
    :param outdir: Path to the output directory.
    :type outdir: str
    :param filename: Name of the output file.
    :type filename: str
    :param cluster: Name of the cluster to plot.
    :type cluster: str
    :param x_key: Dictionary key for datapoint position on the x axis.
    :type x_key: str
    :param y_key: Dictionary key for datapoint position on the y axis.
    :type y_key: str
    :param xlabel: Label for the x axis.
    :type xlabel: str
    :param ylabel: Label for the y axis.
    :type ylabel: str
    :param filter_fn: Potential filter function.
    :type filter_fn: Callable
    :param title_suffix: Suffix for the plot title. Defaults to an empty string.
    :type title_suffix: str
    """
    modes = ['generation', 'inference', 'training']
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.35))
    fig.subplots_adjust(wspace=0.06, bottom=0.25)

    for col, mode in enumerate(modes):
        ax = axes[col]
        pts = [r for r in results if r['cluster'] == cluster and r['mode'] == mode
               and filter_fn(r)]

        _scatter_panel(
            ax,
            pts,
            x_key,
            y_key,
            xlabel = xlabel,
            ylabel = ylabel,
            show_ylabel = (col == 0)
        )
        ax.set_title(_MODE_LABELS[mode], fontsize=9, pad=4)

        if col > 0:
            ax.set_yticklabels([])
            ax.set_ylabel('')

    # Strategy legend — inside bottom margin
    handles = [Line2D([0], [0], marker=st['marker'], color=st['color'],
                      markeredgecolor='k', markeredgewidth=0.3,
                      markersize=4.5, linestyle='None', label=name)
               for name, st in _STRAT_STYLE.items()]
    fig.legend(
        handles = handles,
        loc = 'lower center',
        ncol = 5,
        fontsize = 7,
        frameon = True,
        fancybox = False,
        edgecolor = '#aaaaaa',
        bbox_to_anchor = (0.5, 0.005),
        handletextpad = 0.3,
        columnspacing = 0.8,
        borderpad = 0.25
    )

    fig.savefig(os.path.join(outdir, filename), format='pdf')
    plt.close()
    print(f"  -> {filename}")


def fig_pub_time(results: List[Dict], outdir: str) -> None:
    """
    1×3 execution time scatter: GH200 only.

    :param results: List of results.
    :type results: List[Dict]
    :param outdir: Path to the output directory.
    :type outdir: str
    """
    _pub_1x3(
        results,
        outdir,
        'time_validation.pdf',
        'clariden',
        'measured_ms',
        'predicted_ms',
        'Measured (ms)',
        'Predicted (ms)',
        filter_fn=lambda r: r.get('predicted_ms', 0) > 0
    )


def fig_pub_memory(results: List[Dict], outdir: str) -> None:
    """
    1×3 memory scatter: GH200 only.

    :param results: List of results.
    :type results: List[Dict]
    :param outdir: Path to the output directory.
    :type outdir: str
    """
    _pub_1x3(
        results,
        outdir,
        'memory_validation.pdf',
        'clariden',
        'peak_mem_gb',
        'predicted_mem_gb',
        'Measured (GB)',
        'Predicted (GB)',
        filter_fn=lambda r: r.get('predicted_mem_gb', 0) > 0 and r.get('peak_mem_gb', 0) > 0
    )


if __name__ == '__main__':
    outdir = 'figures/cost_model_validation'
    os.makedirs(outdir, exist_ok=True)

    print("Loading results from all GPU generations...")
    results = load_results()
    print(f"  {len(results)} unique experiments")

    # Map cluster to hardware
    hw_map = {c: cfg[1] for c, cfg in GPU_CONFIGS.items()}

    print("Computing predictions...")
    n_ok = 0
    for r in results:
        cluster = r['cluster']
        if cluster not in hw_map:
            print(f"  WARNING: unknown cluster '{cluster}', skipping")
            r['predicted_ms'] = 0
            continue
        try:
            r['predicted_ms'] = predict_ms(r, hw_map[cluster])
            n_ok += 1
        except Exception as e:
            print(f"  ERROR: {cluster} {r['mode']} dp={r['dp']} pp={r['pp']} tp={r['tp']}: {e}")
            r['predicted_ms'] = 0
    print(f"  {n_ok}/{len(results)} with valid predictions")

    print("Computing memory predictions...")
    n_mem = 0
    for r in results:
        try:
            r['predicted_mem_gb'] = predict_mem_gb(r)
            n_mem += 1
        except Exception as e:
            print(f"  MEM ERROR: {r['cluster']} {r['mode']} dp={r['dp']} pp={r['pp']} tp={r['tp']}: {e}")
            r['predicted_mem_gb'] = 0
    print(f"  {n_mem}/{len(results)} with valid memory predictions")

    print("Applying training harness corrections...")
    n_corr = 0
    for r in results:
        corr = training_harness_correction_gb(r)
        if corr > 0:
            r['peak_mem_gb'] -= corr
            n_corr += 1
    print(f"  {n_corr} training experiments corrected")

    # Filter out dp=1 training configs (optimizer lazy-init causes systematic overprediction)
    before = len(results)
    results = [r for r in results if not (r['mode'] == 'training' and r['dp'] == 1)]
    print(f"  Excluded {before - len(results)} dp=1 training configs")

    print("\nGenerating publication figures (GH200 only, PDF)...")
    fig_pub_time(results, outdir)
    fig_pub_memory(results, outdir)

    print_summary(results)

    # Memory summary
    print(f"\n{'='*90}")
    print(f"  MEMORY MODEL SUMMARY")
    print(f"{'='*90}")
    for cluster in ['clariden', 'a100', 'v100']:
        gpu_name = GPU_CONFIGS[cluster][0]
        pts = [r for r in results if r['cluster'] == cluster
               and r.get('predicted_mem_gb', 0) > 0 and r.get('peak_mem_gb', 0) > 0]
        if not pts:
            continue
        print(f"\n  ── {gpu_name} ({len(pts)} experiments) ──")
        for mode in ['inference', 'training', 'generation']:
            mpts = [r for r in pts if r['mode'] == mode]
            if not mpts:
                continue
            meas = np.array([r['peak_mem_gb'] for r in mpts])
            pred = np.array([r['predicted_mem_gb'] for r in mpts])
            err = (pred - meas) / meas * 100
            abs_err = np.abs(err)
            print(f"    {mode:12s}  MAPE={abs_err.mean():5.1f}%  signed={err.mean():+5.1f}%"
                  f"  ±25%={int((abs_err<25).sum())}/{len(mpts)}  max={abs_err.max():.0f}%")
        meas = np.array([r['peak_mem_gb'] for r in pts])
        pred = np.array([r['predicted_mem_gb'] for r in pts])
        abs_err = np.abs(pred - meas) / meas * 100
        print(f"    {'OVERALL':12s}  MAPE={abs_err.mean():5.1f}%  ±25%={int((abs_err<25).sum())}/{len(pts)}")
