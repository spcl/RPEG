# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BuiltModelInfo:
    backend: str
    family: str
    parameter_count: int
    dtype: str
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    data_parallel_size: int
    rope_scaling: bool = False
    rope_scaling_factor: float = 1.0
