# Reproducibility

Most evaluation figures from the paper can be reproduced from scratch on a SLURM cluster with calibrated hardware.

| Paper Figure                         | Config Generator         | Run Script           | Plot Script                            |
|--------------------------------------|--------------------------|----------------------|----------------------------------------|
| throughput - weak scaling (Fig. 6)   | `gen_scaling_configs.py` | `run_all_scaling.sh` | `plot_scaling.py`                      |
| utilization - weak scaling (Fig. 7)  | `gen_scaling_configs.py` | `run_all_scaling.sh` | `plot_scaling.py --metric utilization` |
| throughput - strong scaling (Fig. 8) | `gen_scaling_configs.py` | `run_all_scaling.sh` | `plot_strong_scaling.py`               |
| operator breakdown (Fig. 9)          | `gen_scaling_configs.py` | `run_all_scaling.sh` | `plot_operator_breakdown.py`           |
| time validation (Fig. 4)             | —                        | `calibrate.py`       | `plot_cost_model_validation.py`        |
| memory validation (Fig. 5)           | —                        | `calibrate.py`       | `plot_cost_model_validation.py`        |
| ablation table (Table V)             | `gen_ablation_*.py`      | `run_ablation.sh`    | `optimizer_bench.py`                   |


## Running from Scratch

The commands assume that they are started from the main directory of the repository.

```bash
# 1. Calibrate hardware (one-time, ~3 min, requires PyTorch + SLURM)
srun --ntasks=8 --ntasks-per-node=4 --gres=gpu:4 \
    python calibrate.py --output hw_calibrated.yaml

# 2. Scaling study (36 weak-scaling + strong-scaling experiments)
python experiments/gen_scaling_configs.py
edit the repository path and target node in lines 16 and 17 of experiments/run_all_scaling.sh
bash experiments/run_all_scaling.sh

# 3. Ablation study (4 scenarios x 10 ablations x 5 seeds)
python experiments/gen_ablation_baseline.py
edit the repository path and target node in lines 16 and 17 of experiments/run_ablation.sh
bash experiments/run_ablation.sh

# 4. Plot results from logs
python experiments/plot_scaling.py --log-dir logs/scaling
python experiments/plot_strong_scaling.py --log-dir logs/scaling
```
