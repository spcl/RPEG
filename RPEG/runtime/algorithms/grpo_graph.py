# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import json

from pathlib import Path

from runtime.algorithms.ppo_graph import PPOTrainResult, TorchPPOGraphHandler
from runtime.compiler import CompiledRuntimePlan
from runtime.containers import ContainerStore
from runtime.errors import RuntimeConfigError
from runtime.executor import RuntimeExecutor
from runtime.graph import build_runtime_graph
from runtime.metrics import write_runtime_summary_for_run
from runtime.trace import RuntimeTraceWriter


def run_graph_grpo(compiled: CompiledRuntimePlan) -> PPOTrainResult:
    if compiled.run_config.training.max_steps < 1:
        raise RuntimeConfigError("GRPO graph training requires training.max_steps >= 1")
    if compiled.run_config.runtime.default_backend == "megatron":
        from runtime.algorithms.megatron_ppo_graph import run_megatron_graph_ppo

        return run_megatron_graph_ppo(compiled)
    if compiled.run_config.runtime.default_backend not in {"torch_grpo", "torch", "local_torch"}:
        raise RuntimeConfigError(
            "GRPO graph training requires runtime.default_backend=megatron/torch_grpo/torch/local_torch"
        )
    graph = compiled.runtime_graph()
    trace = RuntimeTraceWriter(compiled.run_config.runtime.trace_path, compiled.run_config.run_id)
    store_root = Path(compiled.run_config.runtime.store_dir or Path(compiled.run_config.checkpoints.output_dir) / "runtime_store")
    store = ContainerStore(store_root / "containers")
    handler = TorchPPOGraphHandler(compiled, step=1)
    executor = RuntimeExecutor(
        graph,
        run_id=compiled.run_config.run_id,
        handler=handler,
        trace=trace,
        store=store,
    )
    executor.run()
    handler.save_if_needed(compiled.run_config.training.max_steps)
    last_actor_loss = handler.last_actor_loss
    if compiled.run_config.training.max_steps % compiled.run_config.training.log_every_steps == 0:
        print(
            json.dumps(
                {
                    "step": compiled.run_config.training.max_steps,
                    "actor_loss": last_actor_loss,
                    "value_loss": None,
                }
            )
        )
    if compiled.run_config.checkpoints.save_final:
        output_dir = handler.save_final()
    else:
        output_dir = str(Path(compiled.run_config.checkpoints.output_dir))
    write_runtime_summary_for_run(
        compiled.run_config,
        extra={
            "algorithm": "grpo",
            "steps": compiled.run_config.training.max_steps,
            "output_dir": output_dir,
            "last_actor_loss": last_actor_loss,
            "last_value_loss": None,
        },
    )
    return PPOTrainResult(
        steps=compiled.run_config.training.max_steps,
        last_actor_loss=last_actor_loss,
        last_value_loss=None,
        output_dir=output_dir,
    )
