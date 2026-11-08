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


class LlamaAdapter(ModelAdapter):
    family = "llama"

    @classmethod
    def matches(cls, hf_config: dict[str, Any]) -> bool:
        return hf_config.get("model_type") == "llama"

    @classmethod
    def to_megatron_architecture(cls, hf_config: dict[str, Any]) -> MegatronGPTArchitecture:
        hidden = int(hf_config["hidden_size"])
        heads = int(hf_config["num_attention_heads"])
        head_dim = int(hf_config.get("head_dim") or hidden // heads)
        rope_scaling = _llama3_rope_scaling(hf_config)
        return MegatronGPTArchitecture(
            family=cls.family,
            num_layers=int(hf_config["num_hidden_layers"]),
            hidden_size=hidden,
            num_attention_heads=heads,
            num_query_groups=int(hf_config.get("num_key_value_heads", heads)),
            ffn_hidden_size=int(hf_config["intermediate_size"]),
            kv_channels=head_dim,
            vocab_size=int(hf_config["vocab_size"]),
            max_position_embeddings=int(hf_config.get("max_position_embeddings", 8192)),
            layernorm_epsilon=float(hf_config.get("rms_norm_eps", 1e-5)),
            normalization="RMSNorm",
            gated_linear_unit=True,
            activation=str(hf_config.get("hidden_act", "silu")),
            add_bias_linear=False,
            add_qkv_bias=bool(hf_config.get("attention_bias", False)),
            attention_dropout=float(hf_config.get("attention_dropout", 0.0)),
            hidden_dropout=0.0,
            position_embedding_type="rope",
            rotary_base=int(hf_config.get("rope_theta", 500000)),
            rotary_percent=1.0,
            share_embeddings_and_output_weights=bool(hf_config.get("tie_word_embeddings", False)),
            rope_scaling=rope_scaling[0],
            rope_scaling_factor=rope_scaling[1],
        )


def _llama3_rope_scaling(hf_config: dict[str, Any]) -> tuple[bool, float]:
    raw = hf_config.get("rope_scaling")
    if raw is None:
        return False, 1.0
    if not isinstance(raw, dict):
        raise RuntimeConfigError("Llama rope_scaling must be a JSON object when provided")
    rope_type = raw.get("rope_type", raw.get("type"))
    if rope_type != "llama3":
        raise RuntimeConfigError(
            f"unsupported Llama rope_scaling type {rope_type!r}; Megatron runtime supports llama3 scaling"
        )
    low = float(raw.get("low_freq_factor", 1.0))
    high = float(raw.get("high_freq_factor", 4.0))
    original = int(raw.get("original_max_position_embeddings", 8192))
    if low != 1.0 or high != 4.0 or original != 8192:
        raise RuntimeConfigError(
            "unsupported Llama rope_scaling factors; Megatron runtime supports "
            "low_freq_factor=1, high_freq_factor=4, original_max_position_embeddings=8192"
        )
    factor = float(raw.get("factor", 1.0))
    if factor <= 1.0:
        return False, 1.0
    return True, factor
