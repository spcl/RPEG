# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt

"""
DPO (Direct Preference Optimization) RPEG template.

DPO eliminates both the reward model AND the critic. It also has no generation step — training runs
on a static preference dataset of (prompt, y_chosen, y_rejected) triples that were collected offline
(e.g. from an SFT model + human labellers).

Key insight (Rafailov et al., 2023): the RLHF reward can be analytically rearranged so that the
policy is optimised directly from preference pairs, without ever fitting an explicit reward model:

L_DPO = -log σ(β · [(log π(y_w|x) - log π(y_l|x)) - (log π_ref(y_w|x) - log π_ref(y_l|x))])

Models
------
Actor / Policy  (trainable)   — the model being aligned
Reference       (frozen)      — snapshot of the pre-DPO actor, for KL anchor

Pipeline (per training step)
----------------------------
1. Load a batch of B preference pairs from the static dataset
2. Forward-pass all 2B sequences (chosen + rejected, concatenated along the batch dimension)
   through the actor → log_pi (one batched fwd)
3. Forward-pass the same 2B sequences through the reference → log_ref
4. Compute the DPO loss from (log_pi, log_ref), backprop, update actor

Steps 2 and 3 are independent (anti-chain in the DAG) and can execute concurrently on disjoint
device sets.

Why one vertex per model, not two (chosen / rejected)?
------------------------------------------------------
In every major DPO implementation (TRL, DeepSpeed-Chat, OpenRLHF) the chosen and rejected sequences
are concatenated into a single batch and processed in one forward pass — same weights, no data
dependency, better GPU utilisation. Two vertices would wrongly suggest they could be placed on
different devices with different parallelism, which never happens. The ×2 factor is therefore baked
into the container counts, analogous to how K (samples_per_prompt) scales containers in GRPO.

Container sizing
----------------
Every batch element is a preference pair, so all sequence-level containers carry a factor of 2:
X_pref :  B × 2 × (S + T)    input tokens (prompt + response, both variants)
log_pi :  B × 2 × T          actor log-probs on response tokens
log_ref:  B × 2 × T          reference log-probs on response tokens

Weight containers are the same as in any other template.
"""

from typing import Optional

from core.types import Container, EdgeKind, Model, OperatorKind, Role, TemplateVertex, TemplateEdge
from ir.rpeg import RPEG
from model.workload import Workload
from model.models import ModelSuite


def build_DPO_RPEG(suite: ModelSuite, workload: Workload, tau: Optional[int] = None, verbose: bool = False) -> RPEG:
    """
    Construct the DPO RPEG.

    Vertices:  i_pi  ──┐
                        ├──►  t_pi  ──(PUB W_pi)──►  i_pi  (next step)
               i_ref ──┘

    3 vertices, 2 data edges, 1 publication edge.
    Trainable: {ACTOR}. No generation, no reward, no critic.

    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param workload: Workload settings.
    :type workload: Workload
    :param tau: Staleness budget. Defaults to None.
    :type tau: Optional[int]
    :param verbose: Verbose flag for additional output. Defaults to False.
    :type verbose: bool
    :return: RLM Pipeline Execution Graph.
    :rtype: RPEG
    """
    B, S, T = workload.B, workload.S, workload.T

    trainable = {Model.ACTOR}

    # Transient containers
    # Factor of 2: every batch element is a preference pair (chosen + rejected). Both variants are
    # concatenated along the batch dimension and processed in a single forward pass per model, so
    # the container count reflects the true tensor size seen by each operator.
    X_pref  = Container(name="X_pref",  count=B * 2 * (S + T), elem_size_bytes=2)
    log_pi  = Container(name="log_pi",  count=B * 2 * T,       elem_size_bytes=2)
    log_ref = Container(name="log_ref", count=B * 2 * T,       elem_size_bytes=2)

    # Weight containers (cross-step via publication edges)
    W_pi  = Container(name="W_pi",  count=suite.n_params(Model.ACTOR),     elem_size_bytes=2)
    W_ref = Container(name="W_ref", count=suite.n_params(Model.REFERENCE), elem_size_bytes=2)

    containers     = (log_pi, log_ref, W_pi, W_ref)
    in_containers  = (X_pref, W_ref)   # static preference data + frozen reference weights
    out_containers = ()

    # Operator kinds
    # i_pi: forward-pass the actor on 2B sequences → log-probs
    o_i_pi = OperatorKind(
        role=Role.INFERENCE, model=Model.ACTOR,
        read=(W_pi, X_pref), write=(log_pi,),
    )
    # i_ref: forward-pass the reference on the same 2B sequences → log-probs
    o_i_ref = OperatorKind(
        role=Role.INFERENCE, model=Model.REFERENCE,
        read=(W_ref, X_pref), write=(log_ref,),
    )
    # t_pi: compute DPO loss from the two sets of log-probs, backprop, update
    o_t_pi = OperatorKind(
        role=Role.TRAINING, model=Model.ACTOR,
        read=(W_pi, X_pref, log_pi, log_ref), write=(W_pi,),
    )

    # Vertices
    i_pi  = TemplateVertex(id="i_pi",  kind=o_i_pi)
    i_ref = TemplateVertex(id="i_ref", kind=o_i_ref)
    t_pi  = TemplateVertex(id="t_pi",  kind=o_t_pi)

    vertices = (i_pi, i_ref, t_pi)

    # Edges
    # Data edges: both inference outputs feed into training.
    # i_pi and i_ref have NO mutual dependency → anti-chain (concurrent).
    e1 = TemplateEdge(src=i_pi,  dst=t_pi, container=log_pi,  kind=EdgeKind.DATA)
    e2 = TemplateEdge(src=i_ref, dst=t_pi, container=log_ref, kind=EdgeKind.DATA)

    # Publication edge: training updates actor weights for the next step's inference pass.
    # (Reference weights are frozen — no pub edge for i_ref.)
    e3 = TemplateEdge(src=t_pi, dst=i_pi, container=W_pi, kind=EdgeKind.PUB)

    data_edges = (e1, e2)
    pub_edges  = (e3,)

    # Staleness
    # Default: synchronous (τ=0). With τ>0 the actor inference could run on stale weights while the
    # previous training step finishes.
    if tau is None:
        tau = {Model.ACTOR: 0}

    # DPO RPEG
    dpo_rpeg = RPEG(
        vertices, data_edges, pub_edges,
        containers, in_containers, out_containers,
        tau, trainable,
    )

    dpo_rpeg.well_formed(verbose)

    if verbose:
        print("RPEG instantiated successfully for DPO.")
        print(f"    Template Nodes: {len(dpo_rpeg.vertices)}.")
        print(f"    Data edges: {len(dpo_rpeg.data_edges)}.")
        print(f"    Publication edges: {len(dpo_rpeg.pub_edges)}.")
        print(f"    Containers: {len(dpo_rpeg.containers)}.")
        print(f"    IN_Containers: {len(dpo_rpeg.in_containers)}.")
        print(f"    OUT_Containers: {len(dpo_rpeg.out_containers)}.")

    return dpo_rpeg
