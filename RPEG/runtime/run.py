# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

from runtime.algorithms.dpo_graph import run_graph_dpo
from runtime.algorithms.grpo_graph import run_graph_grpo
from runtime.algorithms.ppo_graph import run_graph_ppo
from runtime.compiler import compile_runtime_plan
from runtime.errors import RuntimeConfigError, RuntimeDependencyError, RuntimeExecutionError
from runtime.executor import MetadataOnlyOperationHandler, RuntimeExecutor
from runtime.graph import build_runtime_graph
from runtime.trace import RuntimeTraceWriter
from runtime.verification import verify_runtime_trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or validate an RPEG runtime plan")
    parser.add_argument("--config", required=True, help="Path to rpeg.run.v1 config")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print worker plan without training")
    parser.add_argument(
        "--execute-graph-metadata",
        action="store_true",
        help="Execute the lowered RPEG graph with metadata-only operations for scheduler validation",
    )
    args = parser.parse_args(argv)

    try:
        compiled = compile_runtime_plan(args.config, dry_run=args.dry_run or None)
        if args.execute_graph_metadata or compiled.run_config.runtime.default_backend == "metadata":
            graph = compiled.runtime_graph()
            trace_path = Path(compiled.run_config.runtime.trace_path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text("")
            trace = RuntimeTraceWriter(trace_path, compiled.run_config.run_id)
            executor = RuntimeExecutor(
                graph,
                run_id=compiled.run_config.run_id,
                handler=MetadataOnlyOperationHandler(compiled),
                trace=trace,
            )
            context = executor.run()
            events = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
            trace_verification = verify_runtime_trace(compiled, graph, events)
            print(
                json.dumps(
                    {
                        "run_id": compiled.run_config.run_id,
                        "executed_nodes": len(context.completed),
                        "trace_path": compiled.run_config.runtime.trace_path,
                        "trace_verification": trace_verification,
                    },
                    indent=2,
                )
            )
            return 0
        if args.dry_run or compiled.run_config.runtime.dry_run:
            print(json.dumps(compiled.summary(), indent=2))
            return 0
        if compiled.run_config.algorithm.name == "dpo":
            result = run_graph_dpo(compiled)
            print(json.dumps(result.__dict__, indent=2))
            return 0
        if compiled.run_config.algorithm.name == "ppo":
            result = run_graph_ppo(compiled)
            print(json.dumps(result.__dict__, indent=2))
            return 0
        if compiled.run_config.algorithm.name == "grpo":
            result = run_graph_grpo(compiled)
            print(json.dumps(result.__dict__, indent=2))
            return 0
        raise RuntimeConfigError(
            f"Execution for {compiled.run_config.algorithm.name.upper()} starts in later phases; "
            "available graph executors are DPO, PPO, and GRPO."
        )
    except (RuntimeConfigError, RuntimeDependencyError, RuntimeExecutionError) as exc:
        print(f"runtime error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
