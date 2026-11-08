#!/usr/bin/env python3

# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt

"""
Generate ablation A2 configurations: use existing synchronous configurations with 5 seeds.
"""

import copy
import os
import yaml


# Use the synchronous counterparts of our 3 ablation configurations
CONFIGS = [
    ("configs/scaling/both/16gpu_t896_sync.yaml",    "both_16gpu"),
    ("configs/scaling/actor/32gpu_t128_sync.yaml",   "actor_32gpu"),
    ("configs/scaling/critic/64gpu_t128_sync.yaml",  "critic_64gpu"),
]

SEEDS = [1, 2, 3, 4, 5]

ABLATION = "A2_sync_only"
out_dir = f"configs/ablation/{ABLATION}"
log_dir = f"logs/ablation/{ABLATION}"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

total = 0
for src_path, short_name in CONFIGS:
    with open(src_path) as f:
        base = yaml.safe_load(f.read())

    for seed in SEEDS:
        cfg = copy.deepcopy(base)
        tag = f"{short_name}_s{seed}"
        cfg.setdefault("output", {})
        cfg["output"]["seed"] = seed
        cfg["output"]["plots"] = False
        cfg["output"]["log_file"] = f"{log_dir}/{tag}.log"

        path = os.path.join(out_dir, f"{tag}.yaml")
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        total += 1

print(f"Generated {total} configs for {ABLATION}")
