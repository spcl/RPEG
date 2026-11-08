#!/usr/bin/env python3

# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt

"""
Plot 3x3 scaling figure — thesis/paper quality.

Rows    : actor scaling, critic scaling, both scaling
Columns : T=128 (B=512), T=384 (B=256), T=896 (B=128)

Each cell: grouped bar chart with 4 bars per cluster size:
  1. Fully colocated (sequential baseline)
  2. Naive concurrent (human-tuned partitioning)
  3. Optimized tau=1 (max throughput)
  4. Optimized tau=1 (max throughput × utilization)

Usage:
    python experiments/plot_scaling.py [--output figures/scaling/scaling_3x3.pdf]
    python experiments/plot_scaling.py --metric throughput
"""

import argparse
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import re
import sys

from collections import defaultdict
from pathlib import Path
from typing import Dict

from style import apply_style


matplotlib.use("Agg")

apply_style()

# Experiment layout
EXPERIMENTS = [
    ("actor",  "Scale Actor",  [8, 16, 32, 64]),
    ("critic", "Scale Critic", [8, 16, 32, 64]),
    ("both",   "Scale Both",   [16, 32, 64, 128]),
]

T_VALUES = [128, 384, 896]
B_VALUES = [512, 256, 128]

ACTOR_LABELS  = {8: "7B/7B", 16: "13B/7B", 32: "34B/7B", 64: "70B/7B"}
CRITIC_LABELS = {8: "7B/7B", 16: "7B/13B", 32: "7B/34B", 64: "7B/70B"}
BOTH_LABELS   = {16: "7B/7B", 32: "13B/13B", 64: "34B/34B", 128: "70B/70B"}

MODEL_LABELS = {
    "actor":  ACTOR_LABELS,
    "critic": CRITIC_LABELS,
    "both":   BOTH_LABELS,
}

# MLSys-style palette (NeurIPS/ICML inspired, colorblind-friendly)
BAR_COLORS = {
    "greedy":    "#9E9E9E",   # neutral grey — greedy per-vertex baseline
    "conc":      "#5C6BC0",   # indigo       — concurrent heuristic
    "opt_tau1":  "#26A69A",   # teal         — optimized throughput
    "opt_eff":   "#EF5350",   # coral red    — optimized throughput × utilization
}
BAR_HATCH = {
    "greedy":    "",
    "conc":      "",
    "opt_tau1":  "",
    "opt_eff":   "",
}
BAR_LABELS = {
    "greedy":    "Greedy heuristic",
    "conc":      "Concurrent heuristic",
    "opt_tau1":  "RPEG (throughput)",
    "opt_eff":   "RPEG (util)",
}

VARIANTS = ["tau1", "tau1_eff"]


def parse_log(path: str) -> Dict:
    """
    Parse a pipeline log file for throughput, makespan, PFLOP/s, and utilization.

    Extracts:
      - conc_*  : concurrent heuristic baseline (from "Concurrent heuristic:" line)
      - naive_* : naive concurrent (from results table column 2)
      - opt_*   : optimized (from results table column 3)

    :param path: Path to the input log file.
    :type path: str
    :return: Dictionary with results.
    :rtype: Dict
    """
    result = {}
    if not os.path.exists(path):
        return result
    with open(path) as f:
        text = f.read()

    # Concurrent heuristic baseline (always present, always feasible)
    conc_line = re.search(r"Concurrent heuristic:\s+([\d.e+\-]+)\s+tokens/s\s+\|\s+makespan:\s+([\d.e+\-]+)", text)
    if conc_line:
        result["conc_tp"] = float(conc_line.group(1))
        result["conc_ms"] = float(conc_line.group(2))

    # Results table (3-column: Greedy  Conc.Heuristic  Optimized)
    tp3 = re.search(r"Throughput \(tokens/s\)\s+([\d.e+\-inf]+)\s+([\d.e+\-inf]+)\s+([\d.e+\-inf]+)", text)
    ms3 = re.search(r"Makespan \(s\)\s+([\d.e+\-nan]+)\s+([\d.e+\-nan]+)\s+([\d.e+\-nan]+)", text)
    pf3 = re.search(r"Throughput \(PFLOP/s\)\s+([\d.e+\-]+)\s+([\d.e+\-]+)\s+([\d.e+\-]+)", text)
    ut3 = re.search(r"GPU utilization\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%", text)

    if tp3:
        result["greedy_tp"] = float(tp3.group(1)) if tp3.group(1) != "-inf" else 0
        result.setdefault("conc_tp", float(tp3.group(2)) if tp3.group(2) != "-inf" else 0)
        result["opt_tp"] = float(tp3.group(3)) if tp3.group(3) != "-inf" else 0
    else:
        # Fallback: old 2-column format (Concurrent, Optimized)
        tp2 = re.search(r"Throughput \(tokens/s\)\s+([\d.e+\-]+)\s+([\d.e+\-]+)", text)
        if tp2:
            result.setdefault("conc_tp", float(tp2.group(1)))
            result["greedy_tp"] = float(tp2.group(1))
            result["opt_tp"] = float(tp2.group(2))

    if ms3:
        result["greedy_ms"] = float(ms3.group(1)) if ms3.group(1) != "nan" else 0
        result["opt_ms"] = float(ms3.group(3)) if ms3.group(3) != "nan" else 0

    if pf3:
        result["greedy_pf"] = float(pf3.group(1))
        result["conc_pf"] = float(pf3.group(2))
        result["opt_pf"] = float(pf3.group(3))

    if ut3:
        result["greedy_util"] = float(ut3.group(1)) / 100
        result["conc_util"] = float(ut3.group(2)) / 100
        result["opt_util"] = float(ut3.group(3)) / 100

    # MFU = actual_PFLOP/s / (num_gpus × peak_TFLOP/s × 1e-3)
    gpu_match = re.search(r"→\s+(\d+)\s+GPUs total", text)
    hw_match = re.search(r"compute\s+([\d.]+)\s+TFLOP/s", text)
    if gpu_match:
        result["num_gpus"] = int(gpu_match.group(1))
    if gpu_match and hw_match and "opt_pf" in result:
        num_gpus = int(gpu_match.group(1))
        peak_tflops = float(hw_match.group(1))
        peak_pflops = num_gpus * peak_tflops * 1e-3
        if peak_pflops > 0:
            result["greedy_mfu"] = result.get("greedy_pf", 0) / peak_pflops
            result["conc_mfu"] = result.get("conc_pf", 0) / peak_pflops
            result["opt_mfu"] = result.get("opt_pf", 0) / peak_pflops

    # Baseline feasibility status
    feas_match = re.search(r"baseline_feasible:\s*(True|False)", text)
    result["baseline_feasible"] = feas_match.group(1) == "True" if feas_match else True

    return result


def _get_metric(d: Dict, metric: str, kind: str) -> float:
    """
    Extract value from parsed dictionary.
    kind can have the following values: 'greedy', 'conc', 'opt'.

    :param d: Dictionary parsed from the log file.
    :type d: Dict
    :param metric: Metric type.
    :type metric: str
    :param kind: Method.
    :type kind: str
    :return: Relevant value based on metric and the method.
    :rtype: float
    """
    key_map = {
        "throughput": f"{kind}_tp",
        "pflops":     f"{kind}_pf",
        "makespan":   f"{kind}_ms",
        "mfu":        f"{kind}_mfu",
        "utilization": f"{kind}_util",
    }
    if metric == "speedup":
        greedy = d.get("greedy_tp", 0)
        val    = d.get(f"{kind}_tp", 0)
        if kind == "greedy":
            return 1.0
        if greedy <= 0 or val / greedy > 20:
            return 0
        return val / greedy
    if metric == "efficiency":
        pf = d.get(f"{kind}_pf", 0)
        gpus = d.get("num_gpus", 1)
        return pf / gpus if gpus > 0 else 0
    return d.get(key_map.get(metric, ""), 0)


def _pct_label(opt: float, base: float) -> str:
    """
    Format percentage delta for annotation.

    :param opt: Value of the optimized method.
    :type opt: float
    :param base: Base value.
    :type base: float
    :return: Formatted percentage delta or empty string if value is too small or something went wrong.
    :rtype: str
    """
    if base <= 0 or opt <= 0:
        return ""
    delta = (opt / base - 1) * 100
    if abs(delta) < 0.5:
        return ""
    return f"+{delta:.0f}%"


def plot_3x3(results: Dict, metric: str, output_path: str) -> None:
    """
    Generate the 3x3 grid figure.

    :param results: Dictionary with the results.
    :type results: Dict
    :param metric: Metric type.
    :type metric: str
    :param output_path: Path to the output file, where the plot is stored.
    :type output_path: str
    """
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 6.5))
    fig.subplots_adjust(hspace=0.55, wspace=0.35, top=0.92, bottom=0.12,
                        left=0.09, right=0.91)

    bar_keys = ["greedy", "conc", "opt_tau1", "opt_eff"]
    w = 0.19

    for row, (exp, exp_title, gpu_counts) in enumerate(EXPERIMENTS):
        labels_map = MODEL_LABELS[exp]
        for col, (T, B) in enumerate(zip(T_VALUES, B_VALUES)):
            ax = axes[row, col]

            x_labels = []
            bar_data = {k: [] for k in bar_keys}

            baseline_oom = []
            for gpus in gpu_counts:
                x_labels.append(f"{gpus}G\n{labels_map.get(gpus, '?')}")
                tau1_d = results[exp][T][gpus].get("tau1", {})
                eff_d  = results[exp][T][gpus].get("tau1_eff", {})

                # Greedy and concurrent come from the tau1 run; opt_eff from eff run
                bar_data["greedy"].append(_get_metric(tau1_d, metric, "greedy"))
                bar_data["conc"].append(_get_metric(tau1_d, metric, "conc"))
                bar_data["opt_tau1"].append(_get_metric(tau1_d, metric, "opt"))
                bar_data["opt_eff"].append(_get_metric(eff_d, metric, "opt"))

                # OOM detection: always based on THROUGHPUT, not the plotted metric.
                # A plan with near-zero throughput is OOM regardless of util/makespan.
                greedy_tp = tau1_d.get("greedy_tp", 0)
                conc_tp   = tau1_d.get("conc_tp", 0)
                opt_tp    = max(tau1_d.get("opt_tp", 0), eff_d.get("opt_tp", 0), 1e-9)
                baseline_oom.append({
                    "greedy": greedy_tp < opt_tp * 0.05,
                    "conc":   conc_tp   < opt_tp * 0.05,
                })

            x = np.arange(len(gpu_counts))
            offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]

            # Draw bars; mark OOM baselines with hatched placeholder
            for key, off in zip(bar_keys, offsets):
                heights = bar_data[key]
                for i, h in enumerate(heights):
                    is_baseline = key in ("greedy", "conc")
                    is_oom = is_baseline and baseline_oom[i].get(key, False)
                    if is_oom:
                        # Use the max bar height as placeholder
                        oom_h = max(bar_data["opt_tau1"][i], bar_data["opt_eff"][i],
                                    bar_data["greedy"][i], bar_data["conc"][i], 1e-9)
                        ax.bar(x[i] + off, oom_h, w,
                               color="none", hatch="xxx",
                               edgecolor="#999", linewidth=0.4, zorder=3)
                        ax.annotate("OOM", (x[i] + off, oom_h),
                                    textcoords="offset points", xytext=(0, 2),
                                    ha="center", va="bottom",
                                    fontsize=5, fontweight="bold", color="#C62828")
                    else:
                        ax.bar(x[i] + off, max(h, 0), w,
                               color=BAR_COLORS[key], hatch=BAR_HATCH[key],
                               edgecolor="#444", linewidth=0.4, zorder=3)

            # Annotations depend on metric:
            # - throughput/pflops/speedup: percentage improvement of opt_tau1 over greedy
            # - utilization: pp gain of opt_eff over opt_tau1
            # - makespan: no annotation (lower is better, percentage would be confusing)
            for i in range(len(gpu_counts)):
                if metric == "utilization":
                    # Annotate opt_eff bar with pp gain over opt_tau1
                    tau1_val = bar_data["opt_tau1"][i]
                    eff_val  = bar_data["opt_eff"][i]
                    pp_gain = (eff_val - tau1_val) * 100 # already in [0,1] → pp
                    if pp_gain >= 0.5:
                        lbl = f"+{pp_gain:.0f}pp"
                        bar_x = x[i] + offsets[3] # opt_eff position
                        ax.annotate(lbl, (bar_x, eff_val),
                                    textcoords="offset points", xytext=(0, 3),
                                    ha="center", va="bottom",
                                    fontsize=6, fontweight="bold", color="#333")
                elif metric not in ("makespan",):
                    # Percentage improvement of opt_tau1 over greedy
                    if baseline_oom[i].get("greedy", False):
                        continue # greedy is OOM — skip percentage
                    base_val = bar_data["greedy"][i]
                    if base_val <= 0:
                        continue
                    best_val = bar_data["opt_tau1"][i]
                    lbl = _pct_label(best_val, base_val)
                    if lbl:
                        bar_x = x[i] + offsets[2] # opt_tau1 position
                        ax.annotate(lbl, (bar_x, best_val),
                                    textcoords="offset points", xytext=(0, 3),
                                    ha="center", va="bottom",
                                    fontsize=6, fontweight="bold", color="#333")

            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=6.5)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

            if metric == "throughput":
                ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

            # Column header (top row only)
            if row == 0:
                ax.set_title(f"$T = {T}$,  $B = {B}$", fontsize=11, fontweight="bold",
                             pad=10)

            # Row label (rightmost column)
            if col == 2:
                ax.annotate(
                    exp_title, xy=(1.08, 0.5), xycoords="axes fraction",
                    fontsize=14, fontweight="bold", rotation=270,
                    ha="center", va="center",
                )

            # Styling
            ax.grid(axis="y", alpha=0.25, linewidth=0.4, zorder=0)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylim(bottom=0)

    # Global y-axis label
    ylabel_map = {
        "throughput": "Throughput (tokens/s)",
        "makespan": "Makespan (s)",
        "pflops": "Throughput (PFLOP/s)",
        "mfu": "MFU (Model FLOPs Utilization)",
        "speedup": "Speedup over concurrent baseline",
        "efficiency": "Efficiency (PFLOP/s per GPU)",
        "utilization": "GPU Utilization",
    }
    fig.text(0.01, 0.5, ylabel_map.get(metric, metric),
             va="center", ha="center", rotation="vertical",
             fontsize=14, fontweight="bold")

    # Legend at bottom
    legend_handles = []
    for key in bar_keys:
        patch = mpatches.Patch(
            facecolor=BAR_COLORS[key], edgecolor="#444", linewidth=0.4,
            hatch=BAR_HATCH[key], label=BAR_LABELS[key],
        )
        legend_handles.append(patch)
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02), frameon=True, framealpha=0.95,
               edgecolor="#ccc", fancybox=False,
               handlelength=1.8, handletextpad=0.5, columnspacing=1.2,
               title="Model sizes shown as Actor / Critic", title_fontsize=9)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    """
    Main functionality including argument parsing.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="figures/scaling/scaling_3x3.pdf",
                        help="Output path for the 3x3 figure")
    parser.add_argument("--log-dir", default="logs/scaling",
                        help="Base directory for log files")
    parser.add_argument("--metric", choices=["throughput", "makespan", "pflops", "mfu", "speedup", "efficiency", "utilization"], default="throughput",
                        help="Which metric to plot (default: mfu)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    missing = []

    for exp, _, gpu_counts in EXPERIMENTS:
        for T in T_VALUES:
            for gpus in gpu_counts:
                for variant in VARIANTS:
                    log_path = log_dir / exp / f"{gpus}gpu_t{T}_{variant}.log"
                    parsed = parse_log(str(log_path))
                    if parsed:
                        results[exp][T][gpus][variant] = parsed
                    else:
                        missing.append(f"{exp}/{gpus}gpu_t{T}_{variant}")

    if missing:
        print(f"Warning: {len(missing)} log files missing or unparseable:")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        print()

    total_data = sum(
        1 for e in results for t in results[e]
        for g in results[e][t] for v in results[e][t][g]
    )
    if total_data == 0:
        print("No data found. Run the experiments first:")
        print("  bash experiments/run_all_scaling.sh")
        sys.exit(1)

    print(f"Found data for {total_data} experiment-variant combinations.\n")
    plot_3x3(results, args.metric, args.output)


if __name__ == "__main__":
    main()
