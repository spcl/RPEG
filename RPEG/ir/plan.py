# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt

from dataclasses import dataclass
from typing import Dict, Tuple

from core.types import TemplateVertex, Allocation


@dataclass(frozen=True)
class ExecutionPlan:
    """
    Execution Plan P = (sigma, pi, mb) where
    - sigma: V -> D (device allocation function)
    - pi: V -> N^3 (3D parallel strategy: dp x pp x tp)
    - mb: V -> N (microbatching if pp > 1)
    """
    sigma: Dict[TemplateVertex, Allocation]
    pi: Dict[TemplateVertex, Tuple[int, int, int]]
    mb: Dict[TemplateVertex, int]
