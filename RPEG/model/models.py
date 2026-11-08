# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt
#
# Contributions: Pierre Pang

from dataclasses import dataclass
from typing import Dict, Optional

from core.types import Model


@dataclass(frozen=True)
class ModelSpec:
    """
    Data class representing a model's specification.

    L: Number of Transformer layers.
    d: Hidden dimension.
    d_ff: FFN dimension.
    V: Vocabulary size.
    dtype_bytes: Number of Bytes per scalar. Defaults to 2.
    d_head: Attention head dimension. Defaults to 128.
    """
    L: int
    d: int
    d_ff: int
    V: int
    dtype_bytes: int = 2
    d_head: int = 128
    parameter_count: Optional[int] = None


def base_transformer_params(spec: ModelSpec) -> int:
    """
    Compute the number of basic Transformer parameters of the model.

    :param spec: Model specification.
    :type spec: ModelSpec
    :return: Transformer parameter count.
    :rtype: int
    """
    if spec.parameter_count is not None:
        return int(spec.parameter_count)
    return spec.L * (4 * spec.d**2 + 2 * spec.d * spec.d_ff) + spec.V * spec.d


def transformer_parameter_count(spec: ModelSpec, m: Model) -> int:
    """
    Compute the number of Transformer parameters (including any additional attention heads if it is
    a critic, reward model or weight-sharing is applied) of the model.

    :param spec: Model specification.
    :type spec: ModelSpec
    :return: Transformer parameter count.
    :rtype: int
    """
    extra_head = (m in {Model.CRITIC, Model.REWARD, Model.AC_SHARE}) * spec.d
    return base_transformer_params(spec) + extra_head 


@dataclass(frozen=True)
class ModelSuite:
    """
    Data class representing a model and its specification.
    """
    specs: Dict[Model, ModelSpec]

    def n_params(self, m: Model) -> int:
        """
        Compute the number of Transformer parameters of the model.

        :param m: Input model.
        :type m: Model
        :return: Transformer parameter count.
        :rtype: int
        """
        return transformer_parameter_count(self.specs[m], m)
