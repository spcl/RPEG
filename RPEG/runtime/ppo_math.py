# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

from runtime.errors import RuntimeConfigError


def build_ppo_targets(
    old_logps,
    ref_logps,
    scores,
    values,
    response_mask,
    *,
    beta: float,
    gamma: float,
    gae_lambda: float,
    normalize_advantages: bool = False,
):
    """
    Build the token rewards, GAE advantages, and returns from a frozen rollout.
    """
    import torch

    tensors = {
        "old_logps": old_logps,
        "ref_logps": ref_logps,
        "values": values,
        "response_mask": response_mask,
    }
    expected = tuple(values.shape)
    if values.ndim != 2:
        raise RuntimeConfigError(f"PPO values must have shape [batch, response_tokens], got {expected}")
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != expected:
            raise RuntimeConfigError(f"PPO {name} shape {tuple(tensor.shape)} does not match values {expected}")
    if scores.ndim != 1 or int(scores.shape[0]) != expected[0]:
        raise RuntimeConfigError(f"PPO rewards must have shape [{expected[0]}], got {tuple(scores.shape)}")

    mask = response_mask.detach().float()
    if torch.any((mask != 0) & (mask != 1)):
        raise RuntimeConfigError("PPO response_mask must be binary")
    lengths = mask.sum(dim=1).to(dtype=torch.long)
    if torch.any(lengths <= 0):
        raise RuntimeConfigError("PPO rollouts must contain at least one response token")
    positions = torch.arange(expected[1], device=mask.device).view(1, -1)
    expected_mask = positions < lengths.view(-1, 1)
    if not torch.equal(mask.bool(), expected_mask):
        raise RuntimeConfigError("PPO response_mask must be left-aligned and contiguous")

    old = old_logps.detach().float()
    ref = ref_logps.detach().float()
    old_values = values.detach().float()
    token_rewards = -float(beta) * (old - ref) * mask
    token_rewards = token_rewards.scatter_add(
        1,
        (lengths - 1).view(-1, 1),
        scores.detach().float().view(-1, 1),
    )

    advantages = torch.zeros_like(old_values)
    next_advantage = torch.zeros(expected[0], device=old_values.device, dtype=old_values.dtype)
    for token_index in range(expected[1] - 1, -1, -1):
        if token_index + 1 < expected[1]:
            next_mask = mask[:, token_index + 1]
            next_value = old_values[:, token_index + 1]
        else:
            next_mask = torch.zeros_like(next_advantage)
            next_value = torch.zeros_like(next_advantage)
        delta = (
            token_rewards[:, token_index]
            + float(gamma) * next_value * next_mask
            - old_values[:, token_index]
        )
        next_advantage = delta + float(gamma) * float(gae_lambda) * next_mask * next_advantage
        advantages[:, token_index] = next_advantage * mask[:, token_index]

    returns = (advantages + old_values) * mask
    if normalize_advantages:
        token_count = mask.sum().clamp(min=1.0)
        mean = (advantages * mask).sum() / token_count
        variance = (((advantages - mean) * mask) ** 2).sum() / token_count
        advantages = ((advantages - mean) * torch.rsqrt(variance + 1e-6)) * mask
    return token_rewards.detach(), advantages.detach(), returns.detach()


def minibatch_indices(batch_size: int, minibatch_size: int, *, epochs: int, seed: int):
    """
    Yield deterministic shuffled rollout indices for every PPO epoch.
    """
    import torch

    if batch_size <= 0:
        raise RuntimeConfigError("PPO local rollout batch must be positive")
    if minibatch_size <= 0 or batch_size % minibatch_size != 0:
        raise RuntimeConfigError(
            f"PPO local rollout batch {batch_size} must be divisible by local minibatch size {minibatch_size}"
        )
    if epochs <= 0:
        raise RuntimeConfigError("PPO epochs must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    for epoch in range(epochs):
        permutation = torch.randperm(batch_size, generator=generator)
        for start in range(0, batch_size, minibatch_size):
            yield epoch, permutation[start:start + minibatch_size]
