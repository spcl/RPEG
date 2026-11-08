# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Main authors: Leonard Schmidt
#               Pierre Pang
#
# Contributions: Robert Gerstenberger


import json
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "athena_results"

METHOD_LABELS = {
    "native": "ReaL",
    "pardistrlm": "RPEG",
}

MODEL_ORDER = [7, 13, 32, 70]
GPU_ORDER = [8, 16, 32, 64, 128]

METRIC_PATHS = {
    "runtime": ("benchmark_wall_time_s",),
    "throughput": ("tokens_per_s",),
    "utilization": ("hardware_utilization", "avg_utilization_pct"),
}


# written by Pierre Pang
def _nested_get(data, path):
    value = data
    for key in path:
        if value is None:
            return None
        value = value.get(key)
        if value is None:
            return None
    return value


# written by Pierre Pang
def load_metric(metric_name):
    metric_path = METRIC_PATHS[metric_name]
    values = {
        method: {model_size: {gpu_count: None for gpu_count in GPU_ORDER} for model_size in MODEL_ORDER}
        for method in METHOD_LABELS.values()
    }

    for report_path in sorted(RESULTS_DIR.glob("*/metrics_report.json")):
        gpu_str, model_str = report_path.parent.name.split("-", 1)
        gpu_count = int(gpu_str)
        model_size = int(model_str)

        if gpu_count not in GPU_ORDER or model_size not in MODEL_ORDER:
            continue

        with report_path.open() as f:
            report = json.load(f)

        for schedule_name, label in METHOD_LABELS.items():
            runtime = report.get("schedules", {}).get(schedule_name, {}).get("runtime", {})
            values[label][model_size][gpu_count] = _nested_get(runtime, metric_path)

    return values


# written by Pierre Pang with contributions by Robert Gerstenberger
def hardware_validation():
    runtime = load_metric("runtime")
    throughput = load_metric("throughput")
    utilization = load_metric("utilization")
    gpus = GPU_ORDER

    # Define visual elements
    colors = {
        "ReaL": "#C0843D",
        "RPEG": "#2F6B99"
    }

    MISSING_ALLOCATIONS = {
        ("ReaL", 32, 16): "No allocation found",
    }

    categories = [
        "LLaMA-2-7B/LLaMA-2-7B",
        "LLaMA-2-13B/LLaMA-2-13B",
        "Qwen2.5-32B/Qwen2.5-3B",
        "LLaMA-2-70B/\nLLaMA-2-7B"
    ]

    matplotlib.rc("pdf")
    fig, ax = plt.subplots(3, 1, figsize=(10, 8.4))
    fig.subplots_adjust(bottom=0.20)

    width = 0.35

    # Calculate positions for bars and grid/ticks
    x_tickpos = []
    x_ticklabels = []
    category_positions = []
    group_sizes = []
    current_position = 1
    for model_size, gpu_counts in runtime['ReaL'].items(): # assume that the dictionary also has the same keys and that non-available configurations have None as value
        category_start = current_position
        group_size = 0
        for gpu_count in gpu_counts.items():
            if gpu_count[1] != None or runtime['RPEG'][model_size][gpu_count[0]] != None:
                x_tickpos.append(current_position)
                x_ticklabels.append(gpu_count[0])
                current_position += 1
                group_size += 1
        current_position += 0.5 # Add space between categories
        category_positions.append((category_start + (current_position - 1.5 - category_start) / 2))
        group_sizes.append(group_size)

    model_sizes = MODEL_ORDER

    max_value = [0, 0]
    max_value[0] = max(
        value
        for method_data in runtime.values()
        for model_data in method_data.values()
        for value in model_data.values()
        if value is not None
    )
    max_value[1] = max(
        value
        for method_data in throughput.values()
        for model_data in method_data.values()
        for value in model_data.values()
        if value is not None
    )

    for i in list(range(3)):
        # x axis
        ax[i].set_xticks(x_tickpos)
        if i < 2:
            ax[i].set_xticklabels([])
            # Remove ticks
            for tic in ax[i].xaxis.get_major_ticks():
                tic.tick1line.set_visible(False)
        else:
            ax[2].set_xticklabels(x_ticklabels)

        # Grid
        ax[i].minorticks_off()
        ax[i].grid(True, which="major")
        ax[i].set_axisbelow(True) # draw grid behind the bars

        ax[i].set_xlim([0.5, x_tickpos[-1]+0.5])

    ax[0].set_ylim([0, max_value[0] * 1.1])
    ax[2].set_ylim([0, 100])

    ax[2].set_xlabel("Number of GPUs", y=0.04)

    # y axis
    ax[0].set_ylabel("Makespan in seconds\n(lower is better)")
    ax[1].set_ylabel("Throughput in token/s\n(higher is better)")
    ax[2].set_ylabel("GPU utilization in %\n(higher is better)")

    # Actual plotting
    # Makespan
    for i, method in enumerate(runtime):
        cur_idx = 0
        for j, model_size in enumerate(model_sizes):
            x_data = []
            y_data = []
            for x_idx, gpu_count in enumerate(gpus):
                value = runtime[method][model_size][gpu_count]
                if value is None:
                    if (method, model_size, gpu_count) in MISSING_ALLOCATIONS:
                        x_pos = x_tickpos[cur_idx] + (i - 0.5) * width
                        ax[0].plot(x_pos, max_value[0] * 0.04, marker="x", color=colors[method], markersize=6)
                        ax[0].text(
                            x_pos,
                            max_value[0] * 0.08,
                            MISSING_ALLOCATIONS[(method, model_size, gpu_count)],
                            rotation = 90,
                            ha = "center",
                            va = "bottom",
                            fontsize = 8,
                            color = "#919191",
                        )
                        cur_idx += 1
                    continue
                x_data.append(x_tickpos[cur_idx] + (i - 0.5) * width)
                y_data.append(value)
                cur_idx += 1
            ax[0].bar(x_data, y_data, color=colors[method], width=width, label=method)
    # Throughput
    for i, method in enumerate(throughput):
        cur_idx = 0
        for j, model_size in enumerate(model_sizes):
            x_data = []
            y_data = []
            for x_idx, gpu_count in enumerate(gpus):
                value = throughput[method][model_size][gpu_count]
                if value is None:
                    if (method, model_size, gpu_count) in MISSING_ALLOCATIONS:
                        x_pos = x_tickpos[cur_idx] + (i - 0.5) * width
                        ax[1].plot(x_pos, max_value[1] * 0.04, marker="x", color=colors[method], markersize=6)
                        ax[1].text(
                            x_pos,
                            max_value[1] * 0.08,
                            MISSING_ALLOCATIONS[(method, model_size, gpu_count)],
                            rotation = 90,
                            ha = "center",
                            va = "bottom",
                            fontsize = 8,
                            color = "#919191",
                        )
                        cur_idx += 1
                    continue
                x_data.append(x_tickpos[cur_idx] + (i - 0.5) * width)
                y_data.append(value)
                cur_idx += 1
            ax[1].bar(x_data, y_data, color=colors[method], width=width, label=method)
    # Utilization
    for i, method in enumerate(utilization):
        cur_idx = 0
        for j, model_size in enumerate(model_sizes):
            x_data = []
            y_data = []
            for x_idx, gpu_count in enumerate(gpus):
                value = utilization[method][model_size][gpu_count]
                if value is None:
                    if (method, model_size, gpu_count) in MISSING_ALLOCATIONS:
                        x_pos = x_tickpos[cur_idx] + (i - 0.5) * width
                        ax[2].plot(x_pos, 4, marker="x", color=colors[method], markersize=6)
                        ax[2].text(
                            x_pos,
                            8,
                            MISSING_ALLOCATIONS[(method, model_size, gpu_count)],
                            rotation = 90,
                            ha = "center",
                            va = "bottom",
                            fontsize = 8,
                            color = "#919191",
                        )
                        cur_idx += 1
                    continue
                x_data.append(x_tickpos[cur_idx] + (i - 0.5) * width)
                y_data.append(value)
                cur_idx += 1
            ax[2].bar(x_data, y_data, color=colors[method], width=width, label=method)

    # Add category labels
    for pos, category in zip(category_positions, categories):
        ax[0].text(
            pos,
            1.06,
            category,
            transform = ax[0].get_xaxis_transform(),
            horizontalalignment = 'center',
            verticalalignment = 'top',
            fontsize = 8,
            fontweight = 'bold',
            clip_on = False,
        )

    # Legend
    pa2 = Patch(facecolor=colors["ReaL"])
    pa3 = Patch(facecolor=colors["RPEG"])
    ax[0].legend(handles=[pa2, pa3], labels=["ReaL", "RPEG"], ncol=1)

    plt.savefig("hardware_validation.pdf", bbox_inches="tight")


# Call the plotting functions
hardware_validation()
