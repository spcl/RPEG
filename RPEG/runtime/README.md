# Distributed Runtime

The analytical pipeline produces a graph, a placement and parallelization plan, and a predicted schedule.
The runtime turns those choices into persistent model workers and explicit operations, while preserving the dependencies and resource ownership established during planning.
The same lowered graph therefore describes both the estimate and the execution to be verified.

A runtime job consists of three related artifacts:
[job_plan.py](job_plan.py) handles the *high-level job input*, namely model locations, workload, device topology, search settings, as well as the backend policy.
[serialize.py](../ir/serialize.py) and [schema.py](schema.py) create a *runtime-specific optimized plan*, which includes the chosen assignments, structural choices, workload metadata, hashes, and optional exact schedule.
Finally the *runtime job* is created by [schema.py](schema.py) and [compiler.py](compiler.py), which incorporates the plan and execution, data, transport, optimizer, and checkpoint settings.


## Planning a Runtime Job

[job_plan.py](job_plan.py) builds the graph and candidate plans using the same memory and transfer policies requested for execution.
It evaluates feasibility, searches compatible structural families, serializes the selected schedule, and compiles the emitted run as a consistency check.
A failed feasibility check prevents a run from being emitted as a valid job.

```bash
python -m runtime.job_plan \
    --config /path/to/optimizer_runtime.yaml \
    --out-dir /tmp/rpeg-job \
    --emit-sbatch
```

Run from the repository root with `PYTHONPATH` set to `RPEG`.

The planner writes baseline and optimized plan artifacts, input/output schedule SVGs, an analytical summary, and `run.yaml`.
With `--emit-sbatch`, it also writes a Slurm submission batch file `submit.sbatch` without submitting it.


### Step 1 Compilation and Exact Schedules

[compile_runtime_plan()](compiler.py) reads the strict schema, verifies hashes and model geometry, rebuilds the algorithm graph, unrolls it, and lowers explicit transfers.
It then assigns workers and checks bounds, parallelism, batch and microbatch divisibility, memory capacity, and worker density.

For exact schedule replay, the recorded graph, intervals, dependencies, and transfer reservations must agree with the compiled run.
This matters because equal operator placement does not imply equal execution resources: a transport that reserves both endpoints permits less source overlap than a destination-only policy.
Changing that policy after search can invalidate an otherwise plausible schedule.

Recorded intervals are predictions checked during compilation.
Execution advances on actual completions.
The stored order supplies a dispatch preference rather than wall-clock start deadlines.
The [execution guide](execution.md#dependency-driven-dispatch) specifies the runtime's priority rules and exactness checks.

Positive-lag publication requires disjoint publisher and serving allocations under the distributed runtime contract.
A stale version cannot be modeled as an in-place layout switch on the same worker.
Synchronous colocated examples use zero staleness, while asynchronous execution needs a plan with distinct serving and training ownership.

You can inspect the compilation without launching model workers:

```bash
python -m runtime.run --config /tmp/rpeg-job/run.yaml --dry-run
```


### Step 2 Graph Execution

[graph.py](graph.py) converts the lowered graph into executable node specifications with dependencies and device reservations.
[executor.py](executor.py) dispatches ready nodes by logical step, transfer priority, and preferred graph order, while enforcing resource ownership and completion dependencies.
Model-specific handlers supply the meaning of `OP`, `COMM`, and `RESHARD` nodes.
`OP` nodes generate responses, assess these responses, or apply a training update.
`COMM` nodes move activation records or publish a specific weight version between allocations.
`RESHARD` change parameter and optimizer ownership to the destination parallel layout.

The metadata handler executes these graph contracts without model tensor operations:

```bash
python -m runtime.run --config /tmp/rpeg-job/run.yaml --execute-graph-metadata
```

This mode produces a trace and verification report.
It does not measure training throughput or establish numerical loss correctness.

For real execution, algorithm handlers coordinate local Torch operations or distributed Megatron workers.
[semantic_controller.py](semantic_controller.py) manages worker lifecycle and requests; [semantic_worker.py](semantic_worker.py) owns semantic operations and transfer endpoints; the [Megatron worker](workers/README.md) performs tensor computation.
Workers can persist across several graph nodes, so parameter state and compatible serving caches survive between calls.


## Generation Distributions and Phases

The [lowering pass](../passes/README.md#response-lengthaware-sub-node-splitting) uses cumulative completion-length quantiles to divide generation into phases.
An empirical profile from [generation_profile.py](generation_profile.py) can supply these boundaries.
`runtime.job_plan` accepts a `generation_profile.path` when `workload.gen_quantiles` is absent and checks that its maximum generated length matches the workload.

For fractions $f_q$ and boundaries $t_q$, the analytical model predicts approximately $BK(f_q-f_{q-1})$ completions in phase $q$.
Actual generation does not force that many rows to finish.
At a boundary, rows that have reached EOS are emitted, while surviving rows retain their KV state and continue decoding.
A phase can be empty or have a different volume from its estimate.

Every record retains global sample indices.
These indices let downstream DP layouts route sparse completion sets and restore their original ordering, even after surviving rows have been compacted.
The activation transfer and verification paths must therefore distinguish sparse indexed records from dense contiguous ranges.
See the [algorithm handlers](algorithms/README.md) for how those records become rewards, values, log probabilities, and training inputs.


## Transfer and Version Ownership

Portable activation transfers use integrity-checked shards in the semantic store.
Depending on layout and locality, movement can reuse files or copy/repartition them.
Direct transports provide TCP, Gloo, NCCL, UCX, and applicable CUDA IPC paths.
Typed tensor-tree records carry ranges, byte counts, and integrity metadata.

The reservation policy is derived in [transfer_policy.py](transfer_policy.py) and applied through [transfer_resources.py](../passes/transfer_resources.py):

- Both endpoints are reserved by default for runtime jobs.
- With `direct_source_overlap` enabled, UCX weight transfers and UCX/TCP activation transfers can select destination-only reservations.
- Activation movement that changes DP degrees still reserves both endpoints.

Artifacts are released only after all graph consumers have completed.
Publication edges identify exact model versions: [weight_versions.py](weight_versions.py) governs their selection and retention.
[reshard.py](reshard.py) changes parameter and optimizer layouts while preserving the version being transferred.
Required old versions remain available, so that an incomplete consumer can still observe them.


## Fused Training and Memory

Intra-stage actor/critic fusion retains separate models, but executes compatible training operators as one composite on a shared allocation.
[fused_pipeline.py](fused_pipeline.py) builds the coordinated bidirectional microtask schedule.
Both the analytical scheduler and runtime use this construction, and the memory model accounts for its simultaneously live activations.

The [memory policy](../model/README.md#runtime-memory-and-cost-policies) also distinguishes persistent optimizer state, separate serving copies, frozen weights, backend overhead, and semantic caches.
ZeRO-1 and ZeRO-2 change optimizer and gradient ownership.
Checkpointing changes activation liveness and recomputation work.


## Checkpoints, Traces, and Verification

Interval checkpoints preserve training state and the weight-version artifacts required by the selected recovery path.
On an eligible failure, recovery restores a completed interval, republishes its required versions, and replays the uncovered graph suffix.
The verification path checks attempt boundaries, restored versions, skipped prefix nodes, and eventual graph completion.
The [recovery guide](recovery.md) details checkpoint selection, boundary publication replay, retention, and validation levels.

[trace.py](trace.py) writes events, from which [metrics.py](metrics.py) derives summaries, alerts, dashboards, and Prometheus output.
[verification.py](verification.py) checks execution semantics.

A SVG renderer works can render the runtime trace into a Gantt plot:

```bash
python -m runtime.plot_trace \
    --trace /path/to/run.jsonl \
    --config /path/to/run.yaml \
    --output /path/to/run.gantt.svg
```

[cost_calibration.py](cost_calibration.py) compares successful measured events with analytical node estimates.
It records provenance and incomplete coverage, so a later search can use calibration appropriate to its model, backend, scale, and transport.
The running job remains a fixed compiled plan: it does not migrate operators or re-optimize online.


## Additional Details

Additional information is available in guides dedicated to specific topics:
* [configuration and walkthrough](configuration.md): job preparation, compilation and execution
* [compilation and execution](execution.md): translation of graph nodes become worker operations, and a discussion of when they are executed
* [data and version lifetimes](dataflow.md): movement of samples, KV state, and stale weight versions through the graph
* [checkpoints and verification](recovery.md): which artifacts survive a failure
* [algorithm handlers](algorithms/README.md): translation of graph records into PPO, GRPO, and DPO updates
* [model adapters](models/README.md) and [workers](workers/README.md): architectual description and where tensor computation is performed
