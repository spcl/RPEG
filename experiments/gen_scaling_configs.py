#!/usr/bin/env python3

# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt

"""
Generate scaling experiment configurations.

Configurations contain only topology + models + workload + optimizer.
Hardware parameters come from calibrated hw YAML via --hw flag at runtime.
"""

from pathlib import Path
from typing import List, Tuple


CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "scaling"

# LLaMA-2 model specs
LLAMA = {
    "7b":  {"layers": 32, "hidden_dim": 4096,  "ffn_dim": 11008, "vocab_size": 32000, "head_dim": 128},
    "13b": {"layers": 40, "hidden_dim": 5120,  "ffn_dim": 13824, "vocab_size": 32000, "head_dim": 128},
    "34b": {"layers": 48, "hidden_dim": 8192,  "ffn_dim": 22016, "vocab_size": 32000, "head_dim": 128},
    "70b": {"layers": 80, "hidden_dim": 8192,  "ffn_dim": 28672, "vocab_size": 32000, "head_dim": 128},
}

TB_PAIRS = [
    (128,  512),
    (384,  256),
    (896,  128),
]

def gen_quantiles_for_T(T: int) -> List[Tuple[Float, int]]:
    """
    Response-length distribution quantiles (cum_fraction, token_count).
    Breakpoints scale with T: 50% finish by T/4, 85% by T/2, 100% by T.

    :param T: Maximum number of tokens for the responses.
    :type T: int
    :return: List of response-length distribution quantiles.
    :rtype: List[Tuple[Float, int]]
    """
    return [
        (0.50, T // 4),
        (0.85, T // 2),
        (1.00, T),
    ]

# Budgets calibrated for ~15 min total per config (multi-threaded, 6 cores).
OPTIMIZER_BY_GPUS = {
    8:   {"e_max": 55000, "e_struct": 55000},
    16:  {"e_max": 45000, "e_struct": 45000},
    32:  {"e_max": 40000, "e_struct": 40000},
    64:  {"e_max": 30000, "e_struct": 30000},
    128: {"e_max": 55000, "e_struct": 55000},
}

STALENESS_VARIANTS = [
    ("sync", "[0]", "[0]"),
    ("tau1", "[1]", "[0]"),
    ("tau2", "[2]", "[1]"),
]


def model_block(name: str, size: str) -> str:
    """
    Return the configuration block of the model role as a string.

    :param name: Model role.
    :type name: str
    :param size: Model size including a b at the end.
    :type size: str
    :return: Model block configuration string.
    :rtype: str
    """
    s = LLAMA[size]
    comment = f"# LLaMA-2 {size.upper()}"
    if size == "34b":
        comment = "# Code Llama 34B"
    return f"""\
  {name}:                        {comment}
    layers: {s['layers']}
    hidden_dim: {s['hidden_dim']}
    ffn_dim: {s['ffn_dim']}
    vocab_size: {s['vocab_size']}
    head_dim: {s['head_dim']}"""


def gen_config(
    experiment: str,
    gpus: int,
    actor_size: str,
    critic_size: str,
    reward_size: str,
    T: int,
    B: int,
    variant: str,
    actor_staleness: List[int],
    critic_staleness: List[int]
) -> str:
    """
    Generate a single configuration based on the parameters. The policy optimization algorithm is
    always PPO.

    Cluster configuration is based on the virtual Alps cluster Clariden at CSCS.

    :param experiment: Experiment name.
    :type experiment: str
    :param gpus: Number of available GPUs.
    :type gpus: int
    :param actor_size: Size of the actor/reference models including a b at the end.
    :type actor_size: str
    :param critic_size: Size of the critic model including a b at the end.
    :type critic_size: str
    :param reward_size: Size of the reward model including a b at the end.
    :type reward_size: str
    :param T: Maximum number of response tokens.
    :type T: int
    :param B: Batch size.
    :type B: int
    :param variant: Name of the staleness variant.
    :type variant: str
    :param actor_staleness: Number of iterations that the actor model can be out of sync.
    :type actor_staleness: List[int]
    :param critic_staleness: Number of iterations that the critic model can be out of sync.
    :type critic_staleness: List[int]
    :return: Configuration as a string.
    :rtype: str
    """
    gpus_per_node = 4
    total_nodes = gpus // gpus_per_node
    racks = 2 if gpus >= 64 else 1
    nodes_per_rack = total_nodes // racks

    models = "\n".join([
        "models:",
        model_block("actor", actor_size),
        model_block("critic", critic_size),
        model_block("reward", reward_size),
    ])
    opt = OPTIMIZER_BY_GPUS[gpus]
    quantiles = gen_quantiles_for_T(T)
    gq_lines = "\n".join(f"    - [{q[0]}, {q[1]}]" for q in quantiles)

    return f"""\
# ──────────────────────────────────────────────────────────────────────────────
# Scaling experiment: {experiment} | {gpus} GPUs | T={T}, B={B} | {variant}
# Actor={actor_size.upper()}, Critic={critic_size.upper()}, Reward={reward_size.upper()}
# Hardware params provided at runtime via --hw flag (calibrated hw YAML)
# ──────────────────────────────────────────────────────────────────────────────

hardware:
  racks: {racks}
  nodes_per_rack: {nodes_per_rack}
  gpus_per_node: 4

{models}

workload:
  batch_size: {B}              # B: prompts per step
  samples_per_prompt: 1        # K: rollouts per prompt
  context_length: 128          # S: prompt tokens
  generation_length: {T}       # T: generated tokens
  gen_quantiles:               # response-length distribution (50%@T/4, 85%@T/2, 100%@T)
{gq_lines}

policy:
  algorithm: ppo
  allow_ac_share: false

optimizer:
  k_eval: 8

  # ── Search budget ────────────────────────────────────────────────────────
  e_max: {opt['e_max']}
  e_struct: {opt['e_struct']}

  # ── Move generation ──────────────────────────────────────────────────────
  k_x: 6
  t_max: 50

  # ── SA temperature (constant T, no cooling) ──────────────────────────────
  t_init: 0.05
  t_warm: 0.10
  t_min: 0.05

  # ── Stagnation: restart after this many consecutive rejections ───────────
  stag_window: 10

  # ── Scoring penalties ────────────────────────────────────────────────────
  lambda_ac: 0.0
  score_eps: 1.0e-4

  # ── Staleness ────────────────────────────────────────────────────────────
  staleness_candidates:
    actor:  {actor_staleness}
    critic: {critic_staleness}

output:
  plots: true
  plots_dir: figures/runs/scaling/{experiment}/{gpus}gpu_t{T}_{variant}/
  log_file: logs/scaling/{experiment}/{gpus}gpu_t{T}_{variant}.log
"""


# Experiment definitions
ACTOR_SCALING = [
    (8,   "7b",  "7b", "7b"),
    (16,  "13b", "7b", "7b"),
    (32,  "34b", "7b", "7b"),
    (64,  "70b", "7b", "7b"),
]

CRITIC_SCALING = [
    (8,   "7b", "7b",  "7b"),
    (16,  "7b", "13b", "13b"),
    (32,  "7b", "34b", "34b"),
    (64,  "7b", "70b", "70b"),
]

# Both-model scaling: 16 → 32 → 64 → 128 GPUs
BOTH_SCALING = [
    (16,  "13b", "13b", "13b"),
    (32,  "13b", "13b", "13b"),
    (64,  "34b", "34b", "13b"),
    (128, "70b", "70b", "13b"),
]

ALL_EXPERIMENTS = [
    ("actor",  ACTOR_SCALING),
    ("critic", CRITIC_SCALING),
    ("both",   BOTH_SCALING),
]


def main():
    """
    Main function to generate configurations for the weak and strong scaling experiments.
    """
    count = 0
    for experiment, configs in ALL_EXPERIMENTS:
        out_dir = CONFIGS_DIR / experiment
        out_dir.mkdir(parents=True, exist_ok=True)
        for gpus, actor, critic, reward in configs:
            for T, B in TB_PAIRS:
                for variant, actor_staleness, critic_staleness in STALENESS_VARIANTS:
                    fname = f"{gpus}gpu_t{T}_{variant}.yaml"
                    content = gen_config(experiment, gpus, actor, critic, reward, T, B,
                                         variant, actor_staleness, critic_staleness)
                    (out_dir / fname).write_text(content)
                    count += 1
                    print(f"  wrote configs/scaling/{experiment}/{fname}")
    print(f"\nGenerated {count} config files.")


if __name__ == "__main__":
    main()
