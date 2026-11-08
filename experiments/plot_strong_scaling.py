#!/usr/bin/env python3

# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt

"""
Strong scaling: LLaMA-2 13B, B=1024, 8→128 GPUs. Publication quality.
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import re

from pathlib import Path
from style import apply_style


matplotlib.use("Agg")

apply_style()

# Muted, print-safe palette
COLORS = {
    "greedy":   "#8c8c8c",
    "conc":     "#4a72b0",
    "opt_tau1": "#2e8b57",
    "opt_eff":  "#c44e52",
}
LABELS = {
    "greedy":   "Greedy heuristic",
    "conc":     "Concurrent heuristic",
    "opt_tau1": "RPEG (throughput)",
    "opt_eff":  "RPEG (util)",
}
MARKERS = {"greedy": "s", "conc": "D", "opt_tau1": "o", "opt_eff": "^"}
LWIDTH  = {"greedy": 1.2, "conc": 1.2, "opt_tau1": 1.8, "opt_eff": 1.8}
MSIZE   = {"greedy": 4, "conc": 4, "opt_tau1": 5, "opt_eff": 5}
ZORDER  = {"greedy": 2, "conc": 2, "opt_tau1": 4, "opt_eff": 3}

log_dir = Path("logs/scaling/strong")
gpu_counts = [8, 16, 32, 64, 128]
oom_thresh = 100

# Parse
data = {}
for gpus in gpu_counts:
    data[gpus] = {}
    for suffix, key in [("tau1", "opt_tau1"), ("tau1_eff", "opt_eff")]:
        log = log_dir / f"{gpus}gpu_all13b_B1024_{suffix}.log"
        if not log.exists():
            continue
        with open(log) as f:
            text = f.read()
        conc_line = re.search(r"Concurrent heuristic:\s+([\d.e+\-]+)\s+tokens/s", text)
        tp = re.findall(r"Throughput \(tokens/s\)\s+([\d.e+\-inf]+)\s+([\d.e+\-inf]+)\s+([\d.e+\-inf]+)", text)
        if tp:
            vals = [float(v) if v != "-inf" else 0 for v in tp[-1]]
            data[gpus]["greedy"] = vals[0]
            data[gpus]["conc"] = float(conc_line.group(1)) if conc_line else vals[1]
            data[gpus][key] = vals[2]

# Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.0))
fig.subplots_adjust(wspace=0.42)

x = np.arange(len(gpu_counts))

# Panel (a): Throughput
for key in ["greedy", "conc", "opt_tau1", "opt_eff"]:
    vals = [data[g].get(key, 0) for g in gpu_counts]
    px = [x[i] for i, v in enumerate(vals) if v > oom_thresh]
    py = [v for v in vals if v > oom_thresh]
    if py:
        ax1.plot(px, py, color=COLORS[key], marker=MARKERS[key],
                label=LABELS[key], linewidth=LWIDTH[key], markersize=MSIZE[key],
                zorder=ZORDER[key], markeredgecolor='white', markeredgewidth=0.3)

# OOM markers on panel (a): red "OOM" text at bottom
# Group OOM keys per GPU count to avoid stacking labels
for i, g in enumerate(gpu_counts):
    oom_keys = [k for k in ["greedy", "conc", "opt_tau1", "opt_eff"]
                if data[g].get(k, 0) <= oom_thresh]
    if oom_keys:
        # Find the max valid value at this GPU count for y positioning
        max_valid = max((data[g].get(k, 0) for k in ["greedy", "conc", "opt_tau1", "opt_eff"]), default=0)
        oom_y = max_valid * 0.08 if max_valid > oom_thresh else 2000
        oom_names = ", ".join({"greedy": "G", "conc": "C", "opt_tau1": "O", "opt_eff": "E"}[k] for k in oom_keys)
        ax1.annotate(f"OOM", (x[i], oom_y), ha="center", va="bottom",
                    fontsize=6, fontweight="bold", color="#C62828", zorder=10)

# Shaded gap
gv = [data[g].get("greedy", 0) for g in gpu_counts]
ov = [data[g].get("opt_tau1", 0) for g in gpu_counts]
mask = [(a > oom_thresh and b > oom_thresh) for a, b in zip(gv, ov)]
sx = [x[i] for i, m in enumerate(mask) if m]
slo = [gv[i] for i, m in enumerate(mask) if m]
shi = [ov[i] for i, m in enumerate(mask) if m]
if sx:
    ax1.fill_between(sx, slo, shi, alpha=0.08, color=COLORS["opt_tau1"], linewidth=0)

ax1.set_xticks(x)
ax1.set_xticklabels([str(g) for g in gpu_counts])
ax1.set_xlabel("Number of GPUs")
ax1.set_ylabel("Throughput (tokens/s)")
ax1.set_title("(a) Throughput", fontsize=9, pad=3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim(bottom=0)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}k"))

# Panel (b): Speedup over greedy
# Collect all speedup data first so we can offset overlapping annotations
all_speedups = {}
for key in ["opt_tau1", "opt_eff"]:
    speedups, vx = [], []
    for i, g in enumerate(gpu_counts):
        gr = data[g].get("greedy", 0)
        op = data[g].get(key, 0)
        if gr > oom_thresh and op > oom_thresh:
            speedups.append((op / gr - 1) * 100)
            vx.append(x[i])
    all_speedups[key] = (vx, speedups)

for key in ["opt_tau1", "opt_eff"]:
    vx, speedups = all_speedups[key]
    if speedups:
        ax2.plot(vx, speedups, color=COLORS[key], marker=MARKERS[key],
                label=LABELS[key], linewidth=LWIDTH[key], markersize=MSIZE[key],
                zorder=ZORDER[key], markeredgecolor='white', markeredgewidth=0.3)

# Annotate all points: green (opt_tau1) below its line, red (opt_eff) above its line
vx_tau1, sp_tau1 = all_speedups["opt_tau1"]
vx_eff, sp_eff = all_speedups["opt_eff"]

for xi, sp in zip(vx_tau1, sp_tau1):
    ax2.annotate(f"+{sp:.0f}%", (xi, sp), textcoords="offset points",
                xytext=(0, -12), ha="center", fontsize=6.5,
                color=COLORS["opt_tau1"], fontweight="bold")

for xi, sp in zip(vx_eff, sp_eff):
    ax2.annotate(f"+{sp:.0f}%", (xi, sp), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=6.5,
                color=COLORS["opt_eff"], fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels([str(g) for g in gpu_counts])
ax2.set_xlabel("Number of GPUs")
ax2.set_ylabel("Improvement over greedy (%)")
ax2.set_title("(b) Optimizer advantage", fontsize=9, pad=3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylim(bottom=10)  # start at 10% to zoom into the differences

# Legend
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.5,
           bbox_to_anchor=(0.5, -0.22), frameon=True, framealpha=1.0,
           edgecolor="#ccc", fancybox=False,
           handlelength=1.8, handletextpad=0.4, columnspacing=1.0,
           borderpad=0.4)

out = Path("figures/scaling/strong_scaling.pdf")
fig.savefig(str(out), dpi=600, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
