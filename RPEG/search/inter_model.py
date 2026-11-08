# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt
#
# Contributions: Pierre Pang

import dataclasses
import itertools

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Set, Tuple

from ir.ac_share import build_AC_share_RPEG, is_AC_share_applicable
from ir.dpo import build_DPO_RPEG
from ir.grpo import build_GRPO_RPEG
from ir.ppo import build_PPO_RPEG
from ir.rpeg import RPEG
from model.models import Model, ModelSuite
from model.workload import Workload


DEFAULT_T_CAND: Dict[Model, Set[int]] = {
    Model.ACTOR: {0, 1},
    Model.CRITIC: {0, 1},
    Model.AC_SHARE: {0, 1},
}

@dataclass
class InterModelConfig:
    """
    Data class which records the structural setting producing this RPEG.
    """
    ac_share: bool
    tau: Dict[Model, int]  # mutable dict; frozen=True was incompatible with dict fields
    algorithm: str = "ppo"  # "ppo", "grpo", or "dpo"
    intra_stage_fusion: bool = False


def _iter_staleness(
    trainable: Set[Model],
    T_cand: Dict[Model, Set[int]]
) -> Iterator[Dict[Model, int]]:
    """
    Returns all staleness possibilities:
        {Model.ACTOR: 0, Model.CRITIC: 0}   
        {Model.ACTOR: 0, Model.CRITIC: 1}      
        {Model.ACTOR: 1, Model.CRITIC: 0}   
        {Model.ACTOR: 1, Model.CRITIC: 1}   
        ...

    :param trainable: Set of trainable models.
    :type trainable: Set[Model]
    :param T_cand: Candidates for the staleness of each model.
    :type T_cand: Dict[Model, Set[int]]
    :return: Iterator over all staleness possibilities for the given models.
    :rtype: Iterator[Dict[Model, int]]
    """
    models = sorted(list(trainable), key=lambda m: m.value) # models = [Model.ACTOR, Model.CRITIC]
    value_sets = [sorted(T_cand[m]) for m in models] # value_sets = [[0,1,2,3], [0,1,2,3]]
    for combo in itertools.product(*value_sets): # <=> itertools.product([0,1,2,3], [0,1,2,3]) = [(0,0),(0,1),(0,2),...,(3,3)]
        yield dict(zip(models, combo)) # combo = (0,0), models = [Model.ACTOR, Model.CRITIC], zipped -> [(Model.ACTOR, 0), (Model.CRITIC, 0)]


def Iter_inter_model_candidates(
    suite: ModelSuite,
    workload: Workload,
    T_cand: Optional[Dict[Model, Set[int]]] = None,
    algorithm: str = "ppo",
) -> Iterator[Tuple[RPEG, InterModelConfig]]:
    """
    Outer inter-model candidate iterator.

    For PPO:  yields PPO variants (+ AC-share if applicable).
    For GRPO: yields GRPO variants (no AC-share — no critic to fuse).
    For DPO:  yields DPO variants (no generation, no reward, no critic).

    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param workload: Workload settings.
    :type workload: Workload
    :param T_cand: Candidates for the staleness of each model. Defaults to None.
    :type T_cand: Optional[Dict[Model, Set[int]]]
    :param algorithm: Name of the policy optimization algorithm. Defaults to 'ppo'.
    :type algorithm: str
    :return: Iterator for the inter-model candidates.
    :rtype: Iterator[Tuple[RPEG, InterModelConfig]]
    """
    if T_cand is None:
        T_cand = {m: set(s) for m, s in DEFAULT_T_CAND.items()}  # copy to avoid mutating module default

    if algorithm == "dpo":
        dpo_rpeg = build_DPO_RPEG(suite, workload)
        for tau in _iter_staleness(dpo_rpeg.trainable, T_cand):
            yield dataclasses.replace(dpo_rpeg, tau=tau), InterModelConfig(ac_share=False, tau=tau, algorithm="dpo")
        return

    if algorithm == "grpo":
        grpo_rpeg = build_GRPO_RPEG(suite, workload)
        for tau in _iter_staleness(grpo_rpeg.trainable, T_cand):
            yield dataclasses.replace(grpo_rpeg, tau=tau), InterModelConfig(ac_share=False, tau=tau, algorithm="grpo")
        return

    # b=False: base PPO template
    ppo_rpeg = build_PPO_RPEG(suite, workload)
    for tau in _iter_staleness(ppo_rpeg.trainable, T_cand):
        for intra_stage_fusion in (False, True):
            yield dataclasses.replace(ppo_rpeg, tau=tau), InterModelConfig(
                ac_share = False,
                tau = tau,
                intra_stage_fusion = intra_stage_fusion
            )

    # b=True: AC-shared template (only if applicable)
    if is_AC_share_applicable(ppo_rpeg):
        ac_rpeg = build_AC_share_RPEG(suite, workload)
        for tau in _iter_staleness(ac_rpeg.trainable, T_cand):
            yield dataclasses.replace(ac_rpeg, tau=tau), InterModelConfig(ac_share=True, tau=tau)
