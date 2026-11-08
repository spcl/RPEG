# Scheduling and Graph Score

Given the [executable graph](../passes/README.md#lowering) $\hat{G}$ and the [node-level time map](../model/README.md#summary-time-map) $\mathcal T$, it is possible to determine the execution time of the entire graph and derive a throughput-based score that serves as the optimization objective.

The key challenge is that the graph's makespan cannot be obtained by simply summing node execution times: nodes may execute concurrently when their precedence constraints are satisfied and their device allocations do not conflict, but must serialize when they compete for the same devices.
Computing an optimal schedule under these constraints is NP-hard; we therefore use a greedy list-scheduling heuristic.


## Scheduling

A schedule $\left(s(\cdot), f(\cdot)\right)$ for $\hat{G} = (\hat{V}, \hat{E})$ assigns each node $v \in \hat{V}$ a start time $s(v) \in \mathbb{R}\_{\ge 0}$ and finish time $f(v) = s(v) + \mathcal T(v)$ such that:
1. **Precedence.** $\forall (u \to v) \in \hat{E}: f(u) \le s(v).$ That is, a node may only start after all its predecessors have finished.
2. **Device reservation.** $\forall u \ne v \in \hat{V}: \sigma(u) \cap \sigma(v) \neq \emptyset \implies [s(u), f(u)) \cap [s(v), f(v)) = \emptyset.$  That is, nodes that reserve overlapping devices may not execute concurrently. Here $\sigma(v)$ denotes the reserved device allocation: the full allocation for *Op* nodes, and the allocations selected by the [transfer-resource policy](#transfer-resource-policy) for *Comm*/*Reshard* nodes.

Members of one fused composite operator intentionally share a coordinated interval and the reservation constraint applies to that composite as one scheduling unit.

The makespan of a schedule $\left(s(\cdot), f(\cdot)\right)$ is the latest finish time across all nodes:
```math
\mathrm{MS}(\hat{G}) = \max_{v \in \hat{V}} f(v).
```


### Scheduling Heuristic

The scheduler constructs a feasible schedule by processing nodes greedily, assigning each node the earliest start time that satisfies both constraints from the schedule definition above.
When multiple nodes are ready simultaneously, the scheduler must decide which to schedule first.
We prioritize nodes that lie on or near the critical path, since delaying them is most likely to increase the makespan.
This is captured by the [downstream rank](scheduler.py#L71):
```math
\mathrm{rank}(v) = \mathcal T(v) + \max_{(v \to w) \in \hat{E}} \mathrm{rank}(w),
```
with $\max \emptyset = 0$.
The rank approximates the remaining work on the longest path from $v$ to any sink.

For a ready node $v$ (all predecessors scheduled), the earliest feasible start time is
```math
\mathrm{EST}(v) = \max\Bigl\{
\max_{(u \to v) \in \hat{E}} f(u),\;
\max_{d \in \sigma(v)} \mathrm{avail}(d)
\Bigr\},
```
where $\mathrm{avail}(d)$ tracks when device $d$ becomes free.
The first term enforces precedence; the second ensures all blocking devices are available.

**Selection and Tie-Breaking.**
At each step, the scheduler selects the ready node with the smallest $\mathrm{EST}$.
Ties are broken by:
1. *Comm*/*Reshard* nodes before *Op* nodes (to avoid artificial serialization of transfers),
2. a lexicographically sorted template-ID key for operators to ensure consistent ordering across unrolled steps,
3. larger downstream rank $\mathrm{rank}(v)$.
After scheduling $v$ at $s(v) = \mathrm{EST}(v)$, the availability of all devices in $\sigma(v)$ is updated to $f(v)$.
Two additional tie-breakers (sub-node rank inheritance and RPEG-order consistency) are described after the algorithm.


### Algorithm

<p align="center">
  <img src="../../paper/pics/algo_scheduling.svg" width="60%">
</p>

The algorithm returns $\mathrm{MS}(\hat{G})$.
Since $|\hat{V}| = O(K_\mathrm{eval} \cdot |V^\circ|)$ and $|\hat{E}| = O(K_\mathrm{eval} \cdot |E^\circ|)$, the runtime is 
```math
O(K_\mathrm{eval} \cdot |V^\circ| \log(K_\mathrm{eval} \cdot |V^\circ|) + K_\mathrm{eval} \cdot |E^\circ|).
```
The [implementation](scheduler.py#L270) uses a min-heap with lazy deletion, which accounts for the $\log$ factor.

**Sub-Node Rank Inheritance.**
When [generation quantile splitting](../passes/README.md#response-lengthaware-sub-node-splitting) produces a chain of sub-nodes $x_{v,k}^{g_1} \to \cdots \to x_{v,k}^{g_Q}$ on the same devices, rank naturally decreases along the chain (later sub-nodes have shorter remaining paths).
Without correction, the scheduler may interleave unrelated operators between generation phases, which disrupts KV-cache residency (the cache built during earlier phases would be evicted before later phases use it).
To prevent this, all sub-nodes in a chain inherit the rank of the chain head (the first sub-node), ensuring that chain continuations are always preferred over competing operators on the same devices.

**RPEG-Order Tie-Breaking.**
Beyond rank, the scheduler uses the RPEG vertex index as an additional tie-breaker: operators from the same RPEG position across different unrolled steps are scheduled in the same relative order.
This produces steady-state regularity in the schedule, which is important for asynchronous execution where overlapping iterations should exhibit consistent device-usage patterns.


## Optimization Objective

The executable graph $\hat{G}$ spans $K_\mathrm{eval}$ unrolled iterations.
Under bounded staleness, work from different iterations may overlap in time, so a per-iteration latency metric is not applicable: it cannot capture how consecutive iterations overlap.
We instead use global throughput over the full evaluation horizon as the optimization objective.


### Throughput Objective

The [throughput score](trace.py#L40) of a configuration $C = (G^\circ, P)$ is
```math
\mathrm{Score}(C) = \frac{K_\mathrm{eval} \cdot W_\mathrm{step}}{\mathrm{MS}(\hat{G})},
```
where $W_\mathrm{step} = BK(S + T)$ is the number of tokens processed per iteration and $\mathrm{MS}(\hat{G})$ is the makespan from the algorithm above.

Since $W_\mathrm{step}$ is constant for a fixed workload, maximizing throughput is equivalent to minimizing the makespan.
The optimizer uses $\mathrm{Score}\_\mathrm{penalized}(C)$ as its search objective; when the utilization objective is enabled, the penalty is applied to $\mathrm{Score}\_\mathrm{eff}(C)$ instead.
The score captures the end-to-end effect of all optimization decisions: the RPEG structure $G^\circ$, the staleness budget $\tau$, the execution plan $P = (\sigma, \Pi, \mathrm{mb})$, and structural choices such as stage fusion and placement strategies.

**Memory Penalty.**
Not all configurations are memory-feasible.
Rather than rejecting infeasible configurations outright (which would create hard boundaries in the search landscape), the optimizer applies a soft exponential penalty to the raw throughput score:
```math
\mathrm{Score}_\mathrm{penalized}(C) = \mathrm{Score}(C) \cdot \exp\!\bigl(-\beta_\mathrm{oom} \cdot \max(0,\, r_\mathrm{peak} - 1)\bigr),
```
where $r_\mathrm{peak} = \max_d\, \mathrm{peak}(d) / M_\mathrm{eff}$ is the worst-case peak-to-capacity ratio across all devices and $\beta_\mathrm{oom} = 40$ controls the penalty steepness.
When $r_\mathrm{peak} \le 1$ (feasible), the penalty is $1$ and the score is unchanged.
For moderately infeasible configurations ($1$ – $3$% overflow), the penalty is small enough that the optimizer can temporarily accept high-throughput configurations that slightly exceed memory, then refine them toward feasibility through subsequent moves.
For large violations ($>10$%), the score is effectively zero.
This smooth boundary is critical for search quality: hard rejection would create a discontinuous landscape that prevents the [search](../search/README.md) from crossing memory-constrained regions to reach better feasible solutions.
The optimizer never returns an infeasible configuration as its final result; the penalty only guides intermediate search decisions.


### Cost-Efficiency (Throughput×Utilization) Objective

Plans that partition devices across operators may achieve high throughput but leave a fraction of devices idle between operators.
For example, two plans on the same 32-GPU cluster may achieve similar throughput, but one keeps all devices busy while the other leaves 8 devices idle whenever generation is not active.
To steer the optimizer toward plans that are both fast and resource-efficient, we define an optional efficiency-aware score.
Let $t_\mathrm{busy}(d)$ denote the total time device $d$ spends executing vertices of $\hat{G}$, and define the [mean device utilization](trace.py#L53) as
```math
\mathrm{Util}(C) = \frac{1}{|D|} \sum_{d \in D} \frac{t_\mathrm{busy}(d)}{\mathrm{MS}(\hat{G})}.
```

Given the mean device utilization of configuration $C$, we define the efficiency score (throughput×utilization) as
```math
\mathrm{Score}_\mathrm{eff}(C) = \mathrm{Score}(C) \cdot
\begin{cases}
\mathrm{Util}(C), & \text{if } \mathrm{Score}(C) \ge (1 - \delta) \cdot \mathrm{Score}^\star, \\[4pt]
\lambda_\mathrm{floor}, & \text{otherwise},
\end{cases}
```
where $\mathrm{Score}^\star$ is the best raw throughput found so far during the search, $\delta$ is a user-specified maximum throughput loss, and $\lambda_\mathrm{floor} = 10^{-12}$ penalizes plans that fall below the throughput floor.

The user controls this objective via a single parameter $\delta$ in the optimizer's settings: $\delta = 0$ (the default) recovers pure throughput maximization; $\delta = 0.10$ allows up to 10\% throughput loss in exchange for higher utilization, but ultimately lower cost.


## Transfer-Resource Policy

The analytical model uses by default a destination-only policy while transfering resources, while the runtime normally reserves both endpoints unless their transport mechanism explicitly supports source overlap.
Activation repartitioning across different DP degrees reserves both endpoints in either case.

Selected reservations are stored in [Trace.reserved_allocations](trace.py#L29) and serialized with exact runtime schedules.
Changing the underlying transport mechanism can therefore change feasibility and makespan even when the operator placement is unchanged.
During the [runtime compilation](../runtime/README.md#compilation-and-exact-schedules), a respective check is done before the operator is executed.


## Composite Operators

It is possible to fuse actor and critic training under certain conditions.
The fused actor/critic training operator is dispatched once both dependency sets of the original operators are ready.
During the scheduling of the analytical model their common interval is costed using the same [bidirectional pipeline schedule](../runtime/fused_pipeline.py) used by the runtime, rather than as the sum of two independently scheduled operators.
The trace itself preserves both member IDs, while utilization merges their shared device intervals.
