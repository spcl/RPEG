#!/bin/bash

# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt

# Run ablation configs for a specific ablation ID.
# Usage: bash experiments/run_ablation.sh <ablation_id>
#   e.g.: bash experiments/run_ablation.sh baseline
#         bash experiments/run_ablation.sh A1_no_gen_inf_overlap
set -euo pipefail
cd /path/to/RPEG
TARGET_NODE=

ABLATION=${1:?Usage: bash experiments/run_ablation.sh <ablation_id>}
CFG_DIR="configs/ablation/$ABLATION"
LOG_DIR="logs/ablation/$ABLATION"

if [ ! -d "$CFG_DIR" ]; then
    echo "Error: $CFG_DIR does not exist"
    exit 1
fi

mkdir -p "$LOG_DIR"

CONFIGS=("$CFG_DIR"/*.yaml)
echo "Launching ${#CONFIGS[@]} configs for ablation: $ABLATION"

for cfg in "${CONFIGS[@]}"; do
    tag=$(basename "$cfg" .yaml)
    echo "  $tag"
    srun -w $(TARGET_NODE) --ntasks=1 --cpus-per-task=6 --job-name="abl_${tag}" \
        python pipeline.py --config "$cfg" > /dev/null 2>&1 &
done

echo ""
echo "All ${#CONFIGS[@]} jobs launched. Waiting for completion..."
