#!/bin/bash

# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt

# ─────────────────────────────────────────────────────────────────────────────
# Run all scaling-study configs via SLURM.
# Each config is submitted as a parallel srun job.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd /path/to/RPEG
TARGET_NODE=

# Kill any leftover scaling jobs from previous runs
echo "Cancelling any previous scaling jobs..."
scancel --name="sc_*" 2>/dev/null || true
pkill -f "pipeline.py.*scaling" 2>/dev/null || true
sleep 2

# Generate configs (idempotent)
python experiments/gen_scaling_configs.py

# Wipe old logs to prevent corruption from interleaved runs
echo ""
echo "Cleaning old logs..."
rm -rf logs/scaling/actor logs/scaling/critic logs/scaling/both
mkdir -p logs/scaling/{actor,critic,both}

CONFIGS=()
for exp in actor critic both; do
    for cfg in configs/scaling/"$exp"/*.yaml; do
        [ -f "$cfg" ] && [[ "$cfg" != *"256gpu"* ]] && [[ "$cfg" != *"512gpu"* ]] && [[ "$cfg" != *"1024gpu"* ]] && CONFIGS+=("$cfg")
    done
done

echo ""
echo "Found ${#CONFIGS[@]} configs to run."
echo ""

PIDS=()
TAGS=()
TIMING_LOG="logs/scaling/timing.csv"
echo "config,wall_seconds" > "$TIMING_LOG"

for cfg in "${CONFIGS[@]}"; do
    tag=$(basename "$cfg" .yaml)
    exp=$(basename "$(dirname "$cfg")")
    job_name="sc_${exp}_${tag}"

    echo "  Launching: $exp/$tag"
    srun -w $(TARGET_NODE) --ntasks=1 --cpus-per-task=6 --job-name="$job_name" \
        bash -c "
            START=\$(date +%s.%N)
            python pipeline.py --hw hw_calibrated.yaml --config $cfg > /dev/null 2>&1
            END=\$(date +%s.%N)
            ELAPSED=\$(echo \"\$END - \$START\" | bc)
            echo \"$exp/$tag,\$ELAPSED\" >> $TIMING_LOG
        " &
    PIDS+=($!)
    TAGS+=("$exp/$tag")
done

echo ""
echo "All ${#CONFIGS[@]} jobs launched. Waiting for completion..."
echo ""
