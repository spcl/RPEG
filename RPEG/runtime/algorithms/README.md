# Algorithm Handlers

The runtime graph describes dependencies and resources, where as an algorithm handler gives its nodes their semantic meaning specific to the respective policy optimization algorithm.
The local handlers provide a compact execution path for numerical checks, while the Megatron handler coordinates persistent distributed workers.

<div align="center">

| Module | Responsibility |
|---|---|
| [dpo.py](dpo.py), [dpo_graph.py](dpo_graph.py) | Local preference loss and graph execution for DPO |
| [ppo_graph.py](ppo_graph.py) | Local PPO and shared actor-critic graph operations |
| [grpo_graph.py](grpo_graph.py) | GRPO graph entry point and backend dispatch |
| [megatron_ppo_graph.py](megatron_ppo_graph.py) | Distributed semantic graph handling, publication, and recovery |
| [ppo_math.py](../ppo_math.py) | Shared loss, return, and advantage calculations |

</div>


## From Containers to Updates

For PPO, generation produces completions and policy information.
Reference and reward inference as well as critic evaluation supply the inputs for KL-shaped rewards.
Training applies the configured clipped policy and value losses and updates the corresponding model versions.
Actor-critic sharing combines policy and value behavior around one trainable backbone.
GRPO produces several responses per prompt and derives group-relative advantages without a separate critic.
DPO consumes chosen/rejected preference pairs and compares actor and reference sequence log probabilities, but has no online generation.
The [IR templates](../../ir/README.md) define the graph interfaces for each algorithm.

Generation phases can deliver sparse or empty completion sets.
Handlers preserve global sample indices when assembling downstream batches, so a changed completion order does not change which response belongs to which prompt or rollout group.
Publication and checkpoint logic belong to the graph execution path rather than an independent trainer loop.


### PPO

The rollout is collected under a selected actor version.
Its response tokens, mask, and old policy log probabilities remain fixed while actor and critic updates consume it.
Reference inference supplies frozen-policy log probabilities, reward inference supplies a score per response, and critic inference supplies token values.

The local target construction in [build_ppo_targets()](../ppo_math.py) uses token rewards

```math
r_t=-\beta\bigl(\log\pi_{\mathrm{old}}(a_t\mid x_t)-\log\pi_{\mathrm{ref}}(a_t\mid x_t)\bigr)
+\mathbf{1}_{t=T_i}\,\mathrm{score}_i,
```

where $T_i$ is the final valid response token for sample $i$. Padding is masked. Generalized advantage estimation then computes

```math
\delta_t=r_t+\gamma V_{t+1}-V_t,
\qquad A_t=\delta_t+\gamma\lambda A_{t+1},
\qquad R_t=A_t+V_t,
```

with continuation terms masked at the response boundary. Returns are formed before optional advantage normalization, and rollout targets are detached from the training computation graph. Response masks must be binary, left-aligned, contiguous, and contain at least one response token per sample.

Actor training uses old policy information and advantages for the clipped policy objective.
Critic training uses returns and old values for its value objective.
`algorithm.epochs` and `algorithm.minibatch_size` govern repeated updates over the rollout, while the plan's microbatch count governs pipeline execution within a local batch.


### GRPO

GRPO replaces critic-derived advantages with relative reward information within each prompt's group of $K$ responses.
It therefore has no separate critic training or inference worker.

Completion phases may split one prompt's responses across several transfers.
Global sample identities must survive that split, so the training path can reconstruct the original prompt groups.
The [data-flow guide](../dataflow.md#activation-records) illustrates sparse and empty phases.


### DPO

DPO reads chosen/rejected response pairs. Actor and frozen-reference log probabilities supply the preference comparison

```math
\mathcal{L}_{\mathrm{DPO}}
=-\log\sigma\!\left(\beta\left[
\log\frac{\pi_\theta(y^+\mid x)}{\pi_{\mathrm{ref}}(y^+\mid x)}
-\log\frac{\pi_\theta(y^-\mid x)}{\pi_{\mathrm{ref}}(y^-\mid x)}
\right]\right).
```

The local implementation is in [dpo.py](dpo.py), with graph integration in [dpo_graph.py](dpo_graph.py). It consumes preference records rather than generating new responses or evaluating a separate reward model. Its graph still expresses data movement and actor updates explicitly.


## Sharing and Fusion

Actor-critic sharing uses one trainable backbone for policy and value behavior. Intra-stage fusion retains two models and coordinates their training microtasks on a shared allocation. The latter requires compatible layouts and uses [fused_pipeline.py](../fused_pipeline.py) for the bidirectional schedule. Selecting fusion does not turn the actor and critic into one parameter set.

Across these variants, the handler must preserve the same operation boundary: consume the declared records and model version, produce aligned outputs or an update, and report counters under the lowered node ID. Backend-specific transfers, retention, and checkpoint hooks support that boundary. [execution.md](../execution.md#interface) maps these responsibilities to the executor, controller, and workers.
