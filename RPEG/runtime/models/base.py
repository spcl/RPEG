# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime.hf import load_hf_config


@dataclass(frozen=True)
class MegatronGPTArchitecture:
    family: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_query_groups: int | None
    ffn_hidden_size: int
    kv_channels: int
    vocab_size: int
    max_position_embeddings: int
    layernorm_epsilon: float
    normalization: str
    gated_linear_unit: bool
    activation: str
    add_bias_linear: bool
    add_qkv_bias: bool
    attention_dropout: float
    hidden_dropout: float
    position_embedding_type: str
    rotary_base: int
    rotary_percent: float
    share_embeddings_and_output_weights: bool
    rope_scaling: bool = False
    rope_scaling_factor: float = 1.0


class ModelAdapter:
    family: str

    @classmethod
    def matches(cls, hf_config: dict[str, Any]) -> bool:
        raise NotImplementedError

    @classmethod
    def to_megatron_architecture(cls, hf_config: dict[str, Any]) -> MegatronGPTArchitecture:
        raise NotImplementedError


def load_hf_architecture(path: str | Path, adapter: type[ModelAdapter]) -> MegatronGPTArchitecture:
    return adapter.to_megatron_architecture(load_hf_config(path))
