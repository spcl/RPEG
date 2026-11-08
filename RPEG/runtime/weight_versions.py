# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

from typing import Any

from runtime.compiler import CompiledRuntimePlan


_RPEG_MODEL_TO_RUNTIME_MODEL = {
    "Pi": "actor",
    "V": "critic",
    "Ref": "reference",
    "R": "reward",
    "AC": "ac_share",
}


def consumer_weight_version(compiled: CompiledRuntimePlan, model_name: str, step: int) -> int:
    tau = int((compiled.optimized_plan.structure.get("tau") or {}).get(model_name, 0))
    return max(0, int(step) - 1 - tau)


def uses_shadow_copies(compiled: CompiledRuntimePlan, model_name: str) -> bool:
    tau = int((compiled.optimized_plan.structure.get("tau") or {}).get(model_name, 0))
    if tau <= 0:
        return False
    worker_by_vertex = {
        vertex_id: worker
        for worker in compiled.workers
        for vertex_id in worker.owned_vertex_ids
    }
    for edge in compiled.config.rpeg.pub_edges:
        edge_model = _RPEG_MODEL_TO_RUNTIME_MODEL.get(edge.src.kind.model.value, edge.src.kind.model.value)
        if edge_model != model_name:
            continue
        source = worker_by_vertex.get(edge.src.id)
        destination = worker_by_vertex.get(edge.dst.id)
        if source is None or destination is None:
            continue
        source_devices = {(dev.hostname, dev.local_cuda_index) for dev in source.devices}
        destination_devices = {(dev.hostname, dev.local_cuda_index) for dev in destination.devices}
        if source_devices.isdisjoint(destination_devices):
            return True
    return False


def snapshot_module_state(module) -> dict[str, Any]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def load_module_state(module, state: dict[str, Any]) -> None:
    module.load_state_dict(state, strict=True)
