#!/usr/bin/env python3

# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt

"""
Generate ablation configurations for three wide search space configurations.

Small models on many GPUs = huge candidate directory, many feasible plans. Tests whether search
algorithm features (pair moves, polish, critical-path, soft OOM) matter when the landscape is rich
rather than constrained.
"""

import copy
import os
import yaml

from typing import Dict


CONFIGS = [
    ("configs/scaling/critic/64gpu_t384_tau1.yaml",  "critic_64gpu_t384"),
    ("configs/scaling/critic/32gpu_t384_tau1.yaml",  "critic_32gpu_t384"),
    ("configs/scaling/actor/8gpu_t384_tau1.yaml",    "actor_8gpu_t384"),
]

SEEDS = [1, 2, 3, 4, 5]

# Ablation modifiers (identical to gen_ablation_tight.py)
def no_change(cfg: Dict):
    """
    No changes to the configuration for the baseline necessary.

    :param cfg: Configuration to modify.
    :type cfg: Dict
    """
    pass

def a1_no_overlap(cfg: Dict):
    """
    No overlap of early generation with downstream inference and reference evaluation.

    :param cfg: Configuration to modify.
    :type cfg: Dict
    """
    if "gen_quantiles" in cfg.get("workload", {}):
        del cfg["workload"]["gen_quantiles"]

def a2_sync_only(cfg):
    """
    Synchronous execution only.

    :param cfg: Configuration to modify.
    :type cfg: Dict
    """
    cfg["optimizer"]["staleness_candidates"] = {"actor": [0], "critic": [0]}

def a3_fixed_mb(cfg):
    """
    Fixed microbatch count similar to ReaL (mb=pp for gen/inf, mb=2*pp for train).

    :param cfg: Configuration to modify.
    :type cfg: Dict
    """
    cfg["optimizer"]["fixed_mb"] = True

def a4_no_pair(cfg):
    """
    Only single-vertex moves are allowed.

    :param cfg: Configuration to modify.
    :type cfg: Dict
    """
    cfg["optimizer"]["alpha_pair"] = 0.0

def a5_no_restarts(cfg):
    """
    Disable warm restarts.

    :param cfg: Configuration to modify.
    :type cfg: Dict
    """
    cfg["optimizer"]["stag_window"] = 999999

def a6_uniform_vertex(cfg):
    """
    Replaces the critical-path vertex bias with uniform random selection.

    :param cfg: Configuration to modify.
    :type cfg: Dict
    """
    cfg["optimizer"]["uniform_vertex"] = True

def a7_no_polish(cfg):
    """
    Disable polish phase.

    :param cfg: Configuration to modify.
    :type cfg: Dict
    """
    cfg["optimizer"]["no_polish"] = True

def a8_hard_oom(cfg):
    """
    Hard rejection of memory-infeasible configurations during search.

    :param cfg: Configuration to modify.
    :type cfg: Dict
    """
    cfg["optimizer"]["hard_oom"] = True

def a9_cooling(cfg):
    """
    Use geometric cooling of the temperature.

    :param cfg: Configuration to modify.
    :type cfg: Dict
    """
    cfg["optimizer"]["t_init"] = 0.30
    cfg["optimizer"]["t_warm"] = 0.10
    cfg["optimizer"]["t_min"] = 0.01

ABLATIONS = [
    ("baseline",              no_change),
    ("A1_no_gen_inf_overlap", a1_no_overlap),
    ("A2_sync_only",          a2_sync_only),
    ("A3_fixed_mb",           a3_fixed_mb),
    ("A4_no_pair_moves",      a4_no_pair),
    ("A5_no_restarts",        a5_no_restarts),
    ("A6_uniform_vertex",     a6_uniform_vertex),
    ("A7_no_polish",          a7_no_polish),
    ("A8_hard_oom",           a8_hard_oom),
    ("A9_geometric_cooling",  a9_cooling),
]

total = 0

for src_path, short_name in CONFIGS:
    with open(src_path) as f:
        base = yaml.safe_load(f.read())
    for ablation_name, modifier in ABLATIONS:
        out_dir = f"configs/ablation_wide/{ablation_name}"
        log_dir = f"logs/ablation_wide/{ablation_name}"
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        for seed in SEEDS:
            cfg = copy.deepcopy(base)
            modifier(cfg)
            tag = f"{short_name}_s{seed}"
            cfg.setdefault("output", {})
            cfg["output"]["seed"] = seed
            cfg["output"]["plots"] = False
            cfg["output"]["log_file"] = f"{log_dir}/{tag}.log"
            path = os.path.join(out_dir, f"{tag}.yaml")
            with open(path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            total += 1

print(f"Generated {total} configs")
