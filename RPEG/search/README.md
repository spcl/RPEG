# Search and Optimization

The deterministic evaluation pipeline compiles any [configuration](evaluate.py#L38) $C = (G^\circ, P)$ into a quantitative score.
We now define how the optimizer searches over the configuration space to maximize that score.

The [search](optimizer.py#L365) operates on two levels.
An outer loop enumerates structural variants of the RPEG (staleness budgets, actor-critic sharing), while an inner loop optimizes the execution plan $P = (\sigma, \Pi, \mathrm{mb})$ for each fixed RPEG using simulated annealing with multi-vertex moves and elite-seeded warm restarts.
The procedure is an anytime heuristic: the best feasible configuration found so far can be returned at any point, making it practical for users who simply set a time or evaluation budget.


## Optimization Problem

In the following, the workload $\mathcal W$, model specifications $\mathcal S$, hardware $\mathcal H$, and evaluation horizon $K_\mathrm{eval}$ are fixed.
The optimization variable is the configuration $C = (G^\circ, P)$.

Given a [scoring function](../schedule/README.md#optimization-objective), the optimizer seeks
```math
C^\star = \underset{C \in \Omega}{\mathrm{argmax}}\; \mathrm{Score}(C),
\qquad
\Omega = \{C : \textsc{Valid}(C)\}.
```

A configuration $C = (G^\circ, P)$ is [valid](evaluate.py#L143) if:  
V1 *Well-formedness.* $G^\circ$ is [well-formed](../ir/README.md#well-formedness).  
V2 *[Plan feasibility.](#plan-level-search-space)* For every $v \in V^\circ$: $(\sigma(v), \Pi(v), \mathrm{mb}(v)) \in X_\mathrm{cand}(v)$.  
V3 *[Staleness feasibility.](#inter-model-search-space)* $\tau \in Y_\mathrm{cand}(G^\circ)$  
V4 *Memory feasibility.* The peak memory on every device fits within GPU capacity. The final returned configuration must satisfy this condition. During search, however, the hard constraint is relaxed via the [soft penalty](../schedule/README.md#throughput-objective), which allows the optimizer to temporarily explore near the memory boundary and refine toward feasibility.

The optimizer configuration can be found in [optimizer.py](optimizer.py#L33).

### Search Space

The configuration space is the Cartesian product of two subspaces: the structural space over RPEGs $G^\circ$ and the plan space over execution plans $P = (\sigma, \Pi, \mathrm{mb})$.
Both are constrained to exclude structurally invalid or obviously wasteful configurations without eliminating any viable optimum.


#### Plan-Level Search Space

Using the nested 3D parallelization model, the plan for each vertex $v$ must satisfy several constraints to ensure that the logical parallel structure maps cleanly onto the physical device topology.
Let $B$ be the batch size, $L_m$ the layer count of the model invoked at $v$, and $B_\mathrm{local} = BK / P_d(v)$ the per-replica sequence count.

**Parallelization degrees.**  
P1 *Tensor parallelism.* $P_t(v) \mid G:$ TP is restricted to intra-node placement, so the group size must divide the GPU count per node.  
P2 *Pipeline parallelism.* $P_p(v) \mid L_m$ and $P_p(v)$ is $1$ or even: each stage receives an equal share of layers. Odd depths $>1$ cause asymmetric packing and are excluded.  
P3 *Data parallelism.* $P_d(v) \mid B$ and $P_d(v) \le R \cdot N \cdot G$ (the total GPU count). Every replica processes the same number of samples. When the pipeline fits on a single node, intra-node DP is permitted.

**Microbatch count.**  
P4 *No pipeline, no microbatching.* If $P_p = 1$, then $\mathrm{mb} = 1$.  
P5 *Microbatch alignment.* If $P_p > 1$:
```math
\mathrm{mb}(v) \mid B_\mathrm{local},
\qquad
\mathrm{mb}(v) \ge P_p(v),
\qquad
\frac{B_\mathrm{local}}{\mathrm{mb}(v)} \ge 8.
```
The first term ensures equal-sized microbatches, the second component provides enough microbatches to fill the pipeline, while the third part maintains a minimum per-device microbatch size for SM utilization.

**Device packing.**
Given $(P_d, P_p, P_t)$, each node can host $s = \min(P_p, \lfloor G / P_t \rfloor)$ pipeline stages (limited by either the pipeline depth or the available GPUs after TP).
A single data-parallel replica spans $n_\mathrm{rep} = P_p / s$ nodes (divisibility guaranteed by P8 below), so the operator's total allocation requires $n_\mathrm{need} = P_d \cdot n_\mathrm{rep}$ nodes.  
P6 *Feasible layout.* Either $n_\mathrm{need} \le N$ (all required nodes fit within a single rack) or $N \mid n_\mathrm{need}$ with $n_\mathrm{need} / N \le R$ (the allocation spans one or more complete racks).  
P7 *Aligned origin.* The allocation origin is a multiple of the extent in each dimension, so allocations tile the device space without overlap.  
P8 *Exact packing.* $s \mid P_p$, so each allocated GPU hosts exactly one logical shard of the model: $|\sigma(v)| = P_d \cdot P_p \cdot P_t$.

Combining all of the above, the [per-vertex plan candidate set](moves.py#L489) collects every admissible combination of placement, parallelization strategy, and microbatch count:
```math
X_\mathrm{cand}(v) = \{(\sigma, \Pi, \mathrm{mb}) \mid \text{P1--P8 hold}\}.
```
Together, these constraints guarantee a unique deterministic mapping of logical shards to physical GPUs.


#### Inter-Model Search Space

Plan-level modifications optimize intra-model parallelism and placement for a fixed RPEG.
Inter-model modifications, in contrast, change the RPEG $G^\circ$ itself. The optimizer re-runs the plan-level search for each structural variant.

The inter-model search space is intentionally small.
The base RPEG (PPO, GRPO, or DPO) is chosen by the user and fixed for a given optimization run.
Within that graph, the optimizer searches over [two structural dimensions](../ir/README.md#rlm-pipeline-execution-graphs): (i) whether actor and critic parameters are shared, and (ii) the staleness budgets $\tau$ that control asynchronous execution.
A third structural transformation, [response-length quantile splitting](../passes/README.md#lowering), is not represented as an explicit structural move but applied automatically during lowering whenever the optimizer places generation and inference operators on disjoint device allocations.

We represent inter-model choices by an inter-model configuration
```math
y = (\mathit{ac},\tau),
\qquad
\mathit{ac} \in \{0,1\},
\qquad
\tau:M_{\mathrm{train}}\to\mathbb N,
```
where $\mathit{ac} = 1$ indicates that actor–critic parameter sharing is enabled and $\tau$ is the [staleness budget](../ir/README.md#rlm-pipeline-execution-graphs).

**Structural and inter-model constraints.**
We enforce the following constraints on $y$:  
S1 *Applicability of AC-sharing.* The actor–critic shared variant ($\mathit{ac} = 1$) may only be selected if the base RPEG contains both an actor and a critic model. We write this condition as $\text{ACApplicable}(G^\circ)$. Not all base RPEGs satisfy this precondition: GRPO and DPO have no separate critic, so $\mathit{ac} = 1$ is only applicable to PPO.  
S2 *Staleness candidate set.* Since $\tau: M_\mathrm{train} \to \mathbb N$ is a function, the search enumerates its image componentwise: for each trainable model $m \in M_\mathrm{train}$,
```math
\tau(m) \in T_{\mathrm{cand}}(m),
```
where $T_{\mathrm{cand}}(m)$ is a finite set of allowable budgets (typically $\{0,1\}$).

The **inter-model candidate set** collects all admissible combinations of sharing flag and staleness budget:
```math
Y_{\mathrm{cand}}
=
\Bigl\{
(\mathit{ac},\tau)
\ \Bigm|\
\text{constraints (S1)–(S2) hold}
\Bigr\}.
```
Selecting $(\mathit{ac},\tau) \in Y_{\mathrm{cand}}$ determines the structural RPEG variant and the staleness budget, which controls how publication edges are instantiated during [unrolling](../passes/README.md#unrolling).


## Evaluation Pipeline

Having defined what the optimizer searches over, we now describe how it scores each candidate configuration.
Since local search frequently proposes configurations that have already been evaluated (e.g., a candidate that was explored in an earlier restart), results are memoized in a hash-based [cache](cache.py#L22).
The cache key is a [canonical hash](cache.py#L160) $\text{Hash}(C)$ derived from the RPEG structure (vertices, edges, containers, staleness budget $\tau$) and the execution plan $P=(\sigma, \Pi, \mathrm{mb})$.
The evaluation budget $E$ counts only cache misses ([full pipeline executions](evaluate.py#L155)); cache hits are free.

The algorithm first computes the raw score via the standard pipeline (unroll, lower, schedule), then applies the [soft memory penalty](../schedule/README.md#throughput-objective).

<p align="center">
  <img src="../../paper/pics/algo_evaluatecached.svg" width="60%">
</p>


**Runtime.**
On a cache miss, [EvaluateCached](evaluate.py#L329) executes the full pipeline. The dominant step is [scheduling](../schedule/README.md#scheduling-heuristic), so the total cost per evaluation is
```math
O\bigl(K_\mathrm{eval} \cdot |V^\circ| \log(K_\mathrm{eval} \cdot |V^\circ|) + K_\mathrm{eval} \cdot |E^\circ|\bigr).
```
Cache hits are $O(1)$ (hash lookup).


## Optimization Strategy

The inter-model candidate set $Y_\mathrm{cand}$ is small and can be enumerated exhaustively.
The combinatorial difficulty lies in the plan space: each vertex $v$ admits $|X_\mathrm{cand}(v)|$ valid choices, and the joint plan space grows as $\prod_{v \in V^\circ} |X_\mathrm{cand}(v)|$, which is exponential in the number of RPEG vertices.
Similar to [ReaL](https://github.com/openpsi-project/ReaLHF), each configuration is associated with a cost defined by its (penalized) score and formulate the search as sampling from an energy-based distribution over configurations.
This approach is augmented with multi-vertex proposals, best-of-K candidate selection, and elite-seeded warm restarts.
Therefore a two-level strategy is adopted:
1. **Outer loop (structural enumeration).** Enumerate all $y \in Y_\mathrm{cand}$. For each structural variant, run an independent plan-level search and retain the best feasible configuration found.
2. **Inner loop (plan search).** For a fixed RPEG $G^\circ$, optimize the execution plan $P$ via multi-start simulated annealing (SA). Each iteration selects one or two RPEG vertices, proposes alternative plan assignments, and accepts or rejects the best proposal according to the Metropolis criterion. Multiple SA chains (restarts) are run per structural variant, seeded from an elite set of previously discovered plans.

The overall procedure is anytime: it maintains the best feasible configuration across all structural variants and can return it whenever a wall-clock time limit or evaluation budget $E_\mathrm{max}$ is reached.


### Energy-Based Formulation

The search is formulated as sampling from an energy-based distribution over configurations.
The target distribution over configurations, parameterized by temperature $t > 0$, is
```math
p(C) \;\propto\; \exp\!\left(\frac{\mathrm{Score}(C)}{t}\right).
```

The temperature $t$ controls the tradeoff between exploration and exploitation: at high $t$, $p$ is nearly uniform and the sampler accepts most proposals; at low $t$, $p$ concentrates on the highest-scoring configurations and the sampler behaves nearly greedily.
The search constructs a Markov chain that approximately samples from $p$ using a modified Metropolis acceptance criterion.
Given the current configuration $C$ with score $S$ and a proposal $C'$ with score $S'$, the acceptance probability is
```math
p_\mathrm{acc}(C \to C') \;=\; \min\!\left\{1,\;\exp\!\left(\frac{S' - S}{|S| \cdot t}\right)\right\}.
```
Since $\mathrm{Score}(C)$ is a product of a positive throughput and a positive penalty factor, $S > 0$ always holds.
The normalization by $|S|$ makes the acceptance threshold scale-invariant: the same temperature $t$ produces comparable acceptance rates regardless of the absolute score magnitude.
This differs from the standard Metropolis ratio $\exp((S'-S)/t)$, which would require re-tuning $t$ for every workload.

Improving proposals are always accepted.
Crucially, worsening proposals are accepted with non-zero probability, which allows the chain to escape local optima by temporarily traversing lower-scoring regions of the landscape.
This is the central advantage over greedy search: a purely greedy optimizer can become permanently trapped in a suboptimal basin, whereas the stochastic acceptance mechanism provides a principled way to trade short-term score regression for long-term exploration.
$t$ is held [constant](#temperature-and-chain-termination), unless geometric cooling is enabled, throughout each chain rather than decreased as in classical SA.
The exploration–exploitation tradeoff is instead controlled through the [multi-start mechanism](#restarts-and-elite-seeding).


### Move Proposals

Evaluating a full configuration through the pipeline is expensive relative to enumerating valid strategy tuples.
To avoid wasting evaluations on invalid or obviously poor candidates, the optimizer pre-computes a candidate directory for each RPEG vertex before the SA search begins.
The directory contains all valid strategy tuples satisfying constraints P1–P8, filtered by per-vertex memory feasibility (strategies that exceed GPU capacity even in isolation are excluded; the [soft memory penalty](../schedule/README.md#throughput-objective) handles joint infeasibility across co-resident operators during search) and sorted by estimated execution time.
Candidates are sampled uniformly from the full directory.

Each SA iteration proposes up to $K_x=6$ neighbor configurations by modifying the plan of one or two RPEG vertices while keeping $G^\circ$ fixed.
The [proposal mechanism](moves.py#L645) uses two move kernels.

**Single-vertex kernel.**
A target vertex is selected with a bias toward scheduling bottlenecks: with probability $0.6$, the vertex is sampled proportionally to its execution time on the critical path (the longest path through the current schedule).
Otherwise it is chosen uniformly at random.
For the selected vertex, $K_x$ candidate strategies are sampled uniformly from the directory and each is paired with a randomly chosen aligned device origin. 

**Donor-receiver pair kernel.**
Single-vertex moves cannot redistribute devices between operators in a single step: shrinking one vertex's allocation without simultaneously expanding another leaves devices idle.
To address this, a [donor-receiver kernel](moves.py#L848) proposes coordinated two-vertex moves.
A donor is selected with probability proportional to its device count, and a receiver is chosen among vertices whose allocation overlaps with the donor's.
The donor is shrunk to a smaller contiguous region, and the receiver is re-sampled to potentially expand into the freed devices.
For example, if generation occupies two full racks while inference is constrained to a single node, a pair move can transfer one rack from generation to inference in one step.
No sequence of single-vertex moves can reach this configuration without passing through intermediate states where devices are left unassigned.
The pair kernel is selected with probability $p_\mathrm{pair} = 0.28$.

**Best-of-K selection.**
All $K_x$ proposals are evaluated via [EvaluateCached](#evaluation-pipeline), with cache misses dispatched to a thread pool for parallel evaluation.
Only the highest-scoring proposal $C_\mathrm{best}$ is subjected to the [Metropolis acceptance step](#energy-based-formulation).
This best-of-K strategy amortizes evaluation cost across multiple local directions before committing to a single chain update.


### Temperature and Chain Termination

Unlike classical simulated annealing, which decreases $t$ over the course of a single run, our optimizer holds $t$ constant within each chain.
The first chain runs at $t = t_\mathrm{init}$; warm-restarted chains run at $t = t_\mathrm{warm} > t_\mathrm{init}$, which allows broader exploration around the elite seed.
The higher restart temperature is safe because the chain starts from a known-good plan rather than an arbitrary initial point.
The exploration–exploitation tradeoff is therefore managed across chains (via the two temperature levels and the multi-start mechanism in the subsequent section) rather than within a single chain.
The implementation also supports geometric cooling ($t \leftarrow \max\{t_\mathrm{min}, \gamma \cdot t\}$) as an alternative; an ablation finds that it degrades throughput by $1$ – $5$% compared to constant temperature.

Chains are not run for a fixed number of iterations.
Instead, the optimizer monitors consecutive rejections: if $w_\mathrm{stag}=10$ proposals are rejected in a row, the chain is terminated and a new restart begins.
This ensures that productive chains continue while stuck chains are recycled quickly.


### Restarts and Elite Seeding

Random restarts are a standard technique to escape local optima in SA, but they waste budget rediscovering plans that earlier chains already found.
Instead warm restarts are used: the first chain ($r = 0$) is seeded from a concurrent baseline heuristic that groups operators into concurrent groups (sets of operators with no data dependencies that can run in parallel) and partitions the device space so that each operator within a group receives a disjoint allocation, selecting the partition that minimizes the estimated step time.
All subsequent chains begin from the best feasible plan discovered so far (the elite seed) and run at $t_\mathrm{warm} > t_\mathrm{init}$, allowing broader exploration around the known-good starting point.
The optimizer maintains an elite set of the top $K_\mathrm{elite} = 3$ feasible plans per vertex set, shared across staleness variants with the same vertices (since a good plan under one $\tau$ is a structurally sound seed for another).


### Outer Loop and Budget

The outer loop iterates over all inter-model candidates $y = (\mathit{ac}, \tau) \in Y_\mathrm{cand}$.
Termination is governed by either a wall-clock time limit or an evaluation budget $E_\mathrm{max}$ (cache misses), whichever is reached first.
In time-limited mode, the remaining wall-clock time is split evenly among the remaining structural variants; in budget-limited mode, each variant receives a fixed budget $E_\mathrm{struct}$.
The [candidate directory](#move-proposals) is rebuilt only when the vertex set changes (e.g., when transitioning from the base RPEG to the AC-shared variant); staleness variants that share the same vertices reuse the same directory.

The best feasible configuration $C^\star$ is maintained across all structural variants.
In practice, a 10-minute time budget suffices for workloads up to 1024 GPUs, producing $75$ to $220$ thousand evaluations depending on cluster and model size.
After the SA search completes, a deterministic [polish phase](#polish-phase) performs greedy coordinate descent to extract any remaining local improvements.
The following algorithm shows the complete procedure.

<p align="center">
  <img src="../../paper/pics/algo_optimize.svg" width="60%">
</p>

### Polish Phase

The SA search is stochastic: even when the chain has converged near a good basin, it may not have evaluated every neighbor of the final plan.
The [polish phase](optimizer.py#L949) addresses this by performing deterministic greedy coordinate descent on the best plan returned by *Optimize*.
It visits each RPEG vertex in a fixed order and, for each vertex, evaluates all feasible strategies in the candidate directory.
If any strategy strictly improves the score, the best one is applied before moving to the next vertex.
This process repeats in passes over all vertices until a full pass yields no improvement.

Two properties make the polish phase efficient.
First, scores are origin-independent: the cost model depends on the parallelization strategy and allocation extent, but not on the absolute position of the allocation within the device space.
The [communication tier](../model#intra-operator-communication) is determined by the extent (whether the allocation spans multiple racks or nodes), not by which specific racks or nodes are used.
This means each strategy needs to be evaluated only once regardless of how many valid origins it admits.
Second, many candidates will already reside in the evaluation cache from the preceding SA search, so the polish phase primarily performs hash lookups rather than full pipeline evaluations.
Cache misses are dispatched to the thread pool in parallel, as in the SA inner loop.

The polish phase is guaranteed to terminate (the score increases strictly at each accepted move, and the candidate set is finite) and is deterministic (no random choices).
It cannot discover plans that require coordinated multi-vertex changes, but it reliably extracts any single-vertex improvements that the stochastic search may have missed.
