# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import random

from runtime.errors import RuntimeConfigError


def dataset_batch_indices(
    length: int,
    *,
    step: int,
    batch_size: int,
    offset: int = 0,
    shuffle: bool = False,
    seed: int = 0,
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[int]:
    if length <= 0:
        raise RuntimeConfigError("dataset length must be positive")
    if batch_size <= 0:
        raise RuntimeConfigError("dataset batch_size must be positive")
    if offset < 0:
        raise RuntimeConfigError("dataset offset must be >= 0")
    if num_shards <= 0:
        raise RuntimeConfigError("dataset num_shards must be > 0")
    if shard_index < 0:
        raise RuntimeConfigError("dataset shard_index must be >= 0")
    if shard_index >= num_shards:
        raise RuntimeConfigError("dataset shard_index must be < num_shards")
    start_abs = int(offset) + (max(int(step), 1) - 1) * int(batch_size)
    out = []
    for absolute_idx in range(start_abs, start_abs + int(batch_size)):
        sharded_absolute_idx = int(shard_index) + absolute_idx * int(num_shards)
        epoch = sharded_absolute_idx // int(length)
        position = sharded_absolute_idx % int(length)
        if shuffle:
            order = list(range(int(length)))
            rng = random.Random(int(seed) + epoch)
            rng.shuffle(order)
            out.append(order[position])
        else:
            out.append(position)
    return out
