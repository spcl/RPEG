# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Workload:
    """
    Data class describing the workload.

    B: Batch size (prompts per step).
    K: Rollouts per prompt.
    S: Prompt length.
    T: Generation length.
    gen_quantiles: Generation quantiles. Defaults to None.
    """
    B: int
    K: int
    S: int
    T: int
    gen_quantiles: Optional[Tuple[Tuple[float, int], ...]] = None  # ((cum_frac, token_len), ...)
