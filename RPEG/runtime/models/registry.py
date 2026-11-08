# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

from pathlib import Path

from runtime.errors import RuntimeConfigError
from runtime.hf import load_hf_config
from runtime.models.base import MegatronGPTArchitecture
from runtime.models.llama import LlamaAdapter
from runtime.models.qwen3 import Qwen3Adapter


_ADAPTERS = (Qwen3Adapter, LlamaAdapter)


def architecture_from_hf_config(hf_config: dict) -> MegatronGPTArchitecture:
    expert_fields = {
        key: hf_config.get(key)
        for key in (
            "num_experts",
            "num_local_experts",
            "n_routed_experts",
            "moe_intermediate_size",
        )
        if hf_config.get(key)
    }
    if expert_fields:
        raise RuntimeConfigError(
            "The RPEG validation runtime supports dense decoder-only models; "
            f"MoE configuration requires expert parallelism: {expert_fields}"
        )
    for adapter in _ADAPTERS:
        if adapter.matches(hf_config):
            return adapter.to_megatron_architecture(hf_config)
    model_type = hf_config.get("model_type", "<missing>")
    raise RuntimeConfigError(f"Unsupported HF model_type for Megatron runtime: {model_type!r}")


def architecture_from_hf_path(path: str | Path) -> MegatronGPTArchitecture:
    return architecture_from_hf_config(load_hf_config(path))
