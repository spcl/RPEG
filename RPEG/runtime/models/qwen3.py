# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

from typing import Any

from runtime.errors import RuntimeConfigError
from runtime.models.base import MegatronGPTArchitecture, ModelAdapter


class Qwen3Adapter(ModelAdapter):
    family = "qwen3"

    @classmethod
    def matches(cls, hf_config: dict[str, Any]) -> bool:
        return hf_config.get("model_type") == "qwen3"

    @classmethod
    def to_megatron_architecture(cls, hf_config: dict[str, Any]) -> MegatronGPTArchitecture:
        hidden = int(hf_config["hidden_size"])
        heads = int(hf_config["num_attention_heads"])
        head_dim = int(hf_config.get("head_dim") or hidden // heads)
        return MegatronGPTArchitecture(
            family=cls.family,
            num_layers=int(hf_config["num_hidden_layers"]),
            hidden_size=hidden,
            num_attention_heads=heads,
            num_query_groups=int(hf_config.get("num_key_value_heads", heads)),
            ffn_hidden_size=int(hf_config["intermediate_size"]),
            kv_channels=head_dim,
            vocab_size=int(hf_config["vocab_size"]),
            max_position_embeddings=int(hf_config.get("max_position_embeddings", 40960)),
            layernorm_epsilon=float(hf_config.get("rms_norm_eps", 1e-6)),
            normalization="RMSNorm",
            gated_linear_unit=True,
            activation=str(hf_config.get("hidden_act", "silu")),
            add_bias_linear=bool(hf_config.get("attention_bias", False)),
            add_qkv_bias=bool(hf_config.get("attention_bias", False)),
            attention_dropout=float(hf_config.get("attention_dropout", 0.0)),
            hidden_dropout=0.0,
            position_embedding_type="rope",
            rotary_base=int(hf_config.get("rope_theta", 1000000)),
            rotary_percent=1.0,
            share_embeddings_and_output_weights=bool(hf_config.get("tie_word_embeddings", False)),
        )
