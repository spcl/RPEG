# Runtime Job Configuration

A runtime configuration binds a fixed execution plan to model checkpoints, input data, workers, and output locations.
The plan determines where the computation is run, while the run configuration supplies the environment and numerical settings needed to carry the plan out.


## Running RPEG

Running RPEG consists of three steps:

<div align="center">

| Input | Purpose | Entry point |
|---|---|---|
| Workload analysis | Explore graph structures and placements using cost estimates | `pipeline.py --config ...` |
| Runtime job creation | Search with runtime policies and emit a complete plan/run pair | `python -m runtime.job_plan --config ... --out-dir ...` |
| Job execution | Compile and execute an existing plan artifact | `python -m runtime.run --config ...` |

</div>

The [job planner](job_plan.py) handles the second step, while [schema.py](schema.py) defines the third, including validation and defaults.

We suggest to use absolute model, tokenizer, data, inventory, and output paths, especially when generated jobs will run from another directory.
In a strict run configuration, the compiler resolves relative `optimized_plan.path` and `hardware.calibrated_hw_path` against the run file.
Other paths are passed to their consumers without general rebasing and normally resolve against the process working directory.
[RuntimeRunConfig.load()](schema.py) reads the mapping, but does not normalize every path.


## Run Configuration Sections

We now give an overview over the different sections of a run conifguration:

<div align="center">

| Section | Important fields | Meaning |
|---|---|---|
| `optimized_plan` | `path`, `expected_hash`, `require_exact_match` | Select and validate the plan artifact |
| `algorithm` | `name`, loss coefficients, `epochs`, `minibatch_size`, sampling settings | Select PPO, GRPO, or DPO and its numerical behavior |
| `training` | `max_steps` | Set the compiled execution horizon |
| `workload` | `batch_size`, `samples_per_prompt`, `context_length`, `generation_length`, `gen_quantiles` | Define $B$, $K$, $S$, $T$, and optional generation phases |
| `models`, `tokenizer` | `source`, `checkpoint_path`, model `dtype` | Locate architecture, weights, and tokenization assets |
| `device_space` | `racks`, `nodes_per_rack`, `gpus_per_node` | Describe the logical device grid |
| `hardware` | Memory capacity, reserve fraction, calibration path, worker/cache overhead | Supply feasibility and cost-model inputs |
| `cluster` | Inventory, scheduler, launcher | Map logical allocations to actual hosts and devices |
| `runtime` | Backend, transports, stores, trace path, fusion, schedule requirements | Select execution mechanisms |
| `data` | `train_prompts` or `preferences`, ordering and shard settings | Locate and partition training records |
| `optimizer_runtime` | Learning rates, `distributed_optimizer`, `activation_checkpointing` | Configure parameter updates and training memory |
| `checkpoints` | Output directory, interval, retention, resume source | Control durable training state |
| `monitoring` | Backend-specific reporting and replay settings | Configure observation and recovery |

</div>

Fields that affect graph geometry or reservations must agree with the plan. Editing `workload.gen_quantiles`, fusion, device space, or transport overlap after planning can invalidate it. Regenerate the plan/run pair when changing those choices.


## Tensor Execution

The runtime supports a compilation-only mode with `--dry-run`, which prints a compilation summary, as well as a metadata validation of the computation job graph with `--execute-graph-metadata` with a local Torch backend.

```bash
python -m runtime.run \
    --config generated/grpo_metadata_run.yaml --dry-run

python -m runtime.run \
    --config generated/grpo_metadata_run.yaml \
    --execute-graph-metadata

python -m runtime.plot_trace \
    --trace traces/grpo_metadata_full.jsonl \
    --config generated/grpo_metadata_run.yaml \
    --output grpo.gantt.svg
```

The metadata validation contains architectural information only.
Tensor execution requires a complete supported model checkpoint and tokenizer, suitable input data, and the dependencies for the selected backend.
[Model adapters](models/README.md) describe supported architectures, which is currently Megatron.

For PPO and GRPO, supply `data.train_prompts` and a reward model configuration. For DPO, supply `data.preferences`; the example record has `prompt`, `chosen`, and `rejected` fields. The model source `actor` lets a frozen reference start from the actor checkpoint while retaining its own role.

Check these coupled settings before launching workers:

- **Memory:** `optimizer_runtime.distributed_optimizer` (`none`, `zero1`, or `zero2`) and activation checkpointing must match the intended memory policy. Reserve persistent worker and semantic-cache overhead in `hardware`.
- **Generation:** `algorithm.generation_backend` selects `full_sequence` or `static_kv`. Sampling settings and workload lengths should match those used to measure a completion-length profile.
- **Direct transports:** Slurm UCX/NCCL direct transfers require `cluster.launcher.shared_step: true`, so roles share the required network context. Source overlap also changes reservations and must be included during planning.
- **Recovery:** `monitoring.graph_replay_retries` enables bounded retries in the Megatron graph path. Use durable activation storage and retain usable interval checkpoints; see [recovery](recovery.md).


## Error Debugging

<div align="center">

| Error | What to inspect |
|---|---|
| Workload or device-space mismatch | Run and referenced plan were generated for different dimensions |
| Execution-schedule horizon mismatch | `training.max_steps` differs from the saved schedule's horizon |
| Local execution batch is not divisible | DP, training minibatch size, and pipeline microbatch **count** violate the [batch constraints](execution.md#compilation-pipeline) |
| Positive-lag publication overlaps | A Megatron serving allocation overlaps its train publisher despite nonzero staleness |
| Transfer resource policy mismatch | Transport or `direct_source_overlap` changed after planning |
| CUDA activation cache requires a reservation | Add the intended cache budget to hardware and recheck feasibility |
| Backend dependency or model-file failure | Metadata assets were used for tensor execution, or the selected runtime environment is incomplete |

</div>

Compilation errors are reported as `runtime error: ...` with exit code 2 by `runtime.run`. A successful metadata run should also have a successful trace-verification report; [validation and recovery](recovery.md) explains each execution mode.
