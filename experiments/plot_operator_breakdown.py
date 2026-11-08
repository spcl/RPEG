#!/usr/bin/env python3

# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt

"""
Figure 9 style: per-operator wall-time breakdown, greedy vs optimized.
"""
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from style import apply_style


matplotlib.use("Agg")

apply_style()

with open("evaluation/results/operator_breakdown.json") as f:
    data = json.load(f)

ops = ["Generation (Actor)", "Inference (Critic)", "Inference (Ref)",
       "Inference (Reward)", "Training (Actor)", "Training (Critic)"]
op_short = ["Actor\nGen.", "Critic\nInf.", "Ref.\nInf.", "Reward\nInf.", "Actor\nTrain", "Critic\nTrain"]

C_GREEDY = "#9E9E9E"
C_CONC   = "#5C6BC0"
C_OURS   = "#c44e52"

gpu_rows = [16, 32, 64]

fig, axes = plt.subplots(len(gpu_rows), 2, figsize=(6.5, len(gpu_rows) * 1.6),
                          gridspec_kw={"width_ratios": [1, 3.5], "wspace": 0.25})
fig.subplots_adjust(hspace=0.50)

methods = [
    ("baseline",   "Greedy",     C_GREEDY),
    ("concurrent", "Concurrent", C_CONC),
    ("optimized",  "RPEG",       C_OURS),
]

for row, gpus in enumerate(gpu_rows):
    d = data[str(gpus)]
    ax_total = axes[row, 0]
    ax_ops = axes[row, 1]

    # Left panel: Total makespan (3 bars)
    w_t = 0.25
    makespans = [d[m[0]]["makespan"] for m in methods]
    for mi, (mkey, mlabel, mcolor) in enumerate(methods):
        ax_total.bar(mi, makespans[mi], w_t * 2, color=mcolor, edgecolor="white", linewidth=0.3)

    ax_total.set_xticks(range(len(methods)))
    ax_total.set_xticklabels([m[1] for m in methods], fontsize=5.5, rotation=30, ha="right")
    ax_total.set_ylabel(f"{gpus} GPUs\nTime (s)", fontsize=8, fontweight="bold")
    ax_total.set_title("Makespan" if row == 0 else "", fontsize=8)
    ax_total.spines['top'].set_visible(False)
    ax_total.spines['right'].set_visible(False)
    ax_total.set_ylim(0)

    # Speedup annotation on Ours bar
    base_ms = d["baseline"]["makespan"]
    opt_ms = d["optimized"]["makespan"]
    speedup = (1 - opt_ms / base_ms) * 100
    ax_total.annotate(f"-{abs(speedup):.0f}%", (2, opt_ms),
                     textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=6, fontweight="bold", color="#333")

    # Right panel: Per-operator (3 grouped bars per operator)
    x = np.arange(len(ops))
    w_op = 0.25

    for mi, (mkey, mlabel, mcolor) in enumerate(methods):
        m_ops = [d[mkey]["ops"].get(op, 0) for op in ops]
        offset = (mi - 1) * w_op
        label = mlabel if row == 0 else None
        ax_ops.bar(x + offset, m_ops, w_op, color=mcolor,
                   edgecolor="white", linewidth=0.3, label=label)

    ax_ops.set_xticks(x)
    ax_ops.set_xticklabels(op_short, fontsize=6.5)
    if row == 0:
        ax_ops.set_title("Per-Operator Wall Time", fontsize=8)
    ax_ops.spines['top'].set_visible(False)
    ax_ops.spines['right'].set_visible(False)
    ax_ops.set_ylim(0)
    ax_ops.set_ylabel("Time (s)", fontsize=7)

    # Degree of parallelism annotation below each total bar
    for mi, (mkey, mlabel, mcolor) in enumerate(methods):
        m_sum = sum(d[mkey]["ops"].get(op, 0) for op in ops)
        m_ms = d[mkey]["makespan"]
        ratio = m_sum / m_ms
        ax_total.annotate(f"{ratio:.1f}$\\times$", (mi, m_ms * 0.5),
                         ha="center", va="center", fontsize=4, fontweight="bold",
                         color="white", alpha=0.9)

# Legend
handles = [
    mpatches.Patch(facecolor=C_GREEDY, edgecolor="white", label="Greedy heuristic"),
    mpatches.Patch(facecolor=C_CONC, edgecolor="white", label="Concurrent heuristic"),
    mpatches.Patch(facecolor=C_OURS, edgecolor="white", label="RPEG"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5,
           bbox_to_anchor=(0.5, -0.04), frameon=True, framealpha=1.0,
           edgecolor="#ccc", fancybox=False, borderpad=0.3)

out = "figures/scaling/operator_breakdown.pdf"
fig.savefig(out, dpi=600, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
