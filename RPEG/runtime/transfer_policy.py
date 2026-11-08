# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

from collections.abc import Mapping

from core.types import TransferResourcePolicy


def transfer_resource_policy_for_transports(
    *,
    activation_transport: str,
    weight_transport: str,
    direct_source_overlap: bool,
) -> TransferResourcePolicy:
    overlap = bool(direct_source_overlap)
    return TransferResourcePolicy(
        weight=(
            "destination_only"
            if overlap and str(weight_transport).lower() == "ucx_direct"
            else "both_endpoints"
        ),
        activation=(
            "destination_only"
            if overlap and str(activation_transport).lower() in {"ucx_direct", "tcp_direct"}
            else "both_endpoints"
        ),
    )


def transfer_resource_policy_from_runtime(runtime: object) -> TransferResourcePolicy:
    return transfer_resource_policy_for_transports(
        activation_transport=str(getattr(runtime, "activation_transport")),
        weight_transport=str(getattr(runtime, "weight_transport")),
        direct_source_overlap=bool(getattr(runtime, "direct_source_overlap")),
    )


def transfer_resource_policy_from_job_config(config: Mapping[str, object]) -> TransferResourcePolicy:
    raw = config.get("runtime")
    runtime = raw if isinstance(raw, Mapping) else {}
    return transfer_resource_policy_for_transports(
        activation_transport=str(runtime.get("activation_transport", "auto")),
        weight_transport=str(runtime.get("weight_transport", "auto")),
        direct_source_overlap=bool(runtime.get("direct_source_overlap", False)),
    )
