# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Authors: Leonard Schmidt
#          Pierre Pang

"""
Serialization of an optimized Configuration (RPEG + ExecutionPlan) to JSON or YAML.

The output schema is:

  device_space:
    racks: int
    nodes_per_rack: int
    gpus_per_node: int

  structure:
    ac_share: bool            # actor-critic weight sharing
    tau:                      # staleness budget per trainable model
      actor:  int
      critic: int
    trainable: [str, ...]     # which models are trained

  plan:
    <vertex_id>:              # e.g. "Pi", "V", "Ref", "R"
      model: str              # human-readable model name
      role:  str              # GEN | INF | TRAIN
      placement:
        origin: [r, n, g]
        extent: [R, N, G]
      parallelism:
        dp: int
        pp: int
        tp: int
      microbatch: int
      refinement: str | null  # "batch" | "sample" | null
"""

import hashlib
import json

from typing import Optional

from core.types import Model
from ir.rpeg import RPEG
from model.workload import Workload
from schedule.trace import Trace
from search.evaluate import Configuration


_MODEL_NAME = {
    Model.ACTOR:     "actor",
    Model.CRITIC:    "critic",
    Model.REFERENCE: "reference",
    Model.REWARD:    "reward",
    Model.AC_SHARE:  "ac_share",
}


def plan_to_dict(config: Configuration) -> dict:
    """
    Return a plain dict representation of config suitable for JSON/YAML serialization.

    :param config: Input configuration.
    :type config: Configuration
    :return: Dictionary representation of the configuration.
    :type: dict
    """
    rpeg = config.rpeg
    plan = config.plan
    ds   = config.device_space

    # Structure (inter-model decisions)
    structure = {
        "ac_share":  Model.AC_SHARE in rpeg.trainable,
        "intra_stage_fusion": bool(config.intra_stage_fusion),
        "tau":       {_MODEL_NAME[m]: t for m, t in rpeg.tau.items()},
        "trainable": [_MODEL_NAME[m] for m in sorted(rpeg.trainable, key=lambda m: m.value)],
    }
    if config.transfer_resource_policy is not None:
        structure["transfer_resource_policy"] = config.transfer_resource_policy.to_dict()

    # Per-vertex plan
    vertices: dict = {}
    for v in rpeg.vertices:
        sigma = plan.sigma[v]
        pi    = plan.pi[v]
        mb    = plan.mb[v]
        vertices[v.id] = {
            "model":       _MODEL_NAME[v.kind.model],
            "role":        v.kind.role.value,
            "placement": {
                "origin": list(sigma.origin),
                "extent": list(sigma.extent),
            },
            "parallelism": {
                "dp": pi[0],
                "pp": pi[1],
                "tp": pi[2],
            },
            "microbatch":  mb,
        }

    result: dict = {
        "device_space": {
            "racks":          ds.R,
            "nodes_per_rack": ds.N,
            "gpus_per_node":  ds.G,
        },
        "structure": structure,
        "plan":      vertices,
    }

    return result


def load_plan(path: str, rpeg: RPEG):
    """
    Load an ExecutionPlan from a YAML/JSON file, matching vertices to the given RPEG.

    :param path: Path to the input execution plan.
    :type path: str
    :param rpeg: RLM Pipeline Execution Graph.
    :type rpeg: RPEG
    :return: Execution plan.
    :rtype: ExecutionPlan
    """
    import yaml

    from core.types import Allocation
    from ir.plan import ExecutionPlan

    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            d = yaml.safe_load(f)
        else:
            d = json.load(f)

    plan_d = d["plan"]
    # Build vertex lookup by id
    v_by_id = {v.id: v for v in rpeg.vertices}

    sigma, pi, mb = {}, {}, {}
    for vid, info in plan_d.items():
        v = v_by_id.get(vid)
        if v is None:
            continue
        p = info["placement"]
        sigma[v] = Allocation(origin=tuple(p["origin"]), extent=tuple(p["extent"]))
        par = info["parallelism"]
        pi[v] = (par["dp"], par["pp"], par["tp"])
        mb[v] = info["microbatch"]

    return ExecutionPlan(sigma, pi, mb)


def save_plan(config: Configuration, path: str) -> None:
    """
    Serialize configuration and write it to path.

    Format is inferred from the file extension:
      .yaml / .yml  → YAML
      anything else → JSON (default)

    :param config: Configuration to store.
    :type config: Configuration
    :param path: Path to the output file.
    :type path: str
    """
    d = plan_to_dict(config)

    if path.endswith((".yaml", ".yml")):
        import yaml
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    else:
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    print(f"  Saved plan to -> {path}")


def optimized_plan_to_dict(
    config: Configuration,
    algorithm: str,
    workload: Workload,
    source_config_hash: Optional[str] = None,
    execution_schedule: Optional[dict] = None,
) -> dict:
    """
    Return a plain dictionary representation of an optimized plan including its configuration
    suitable for JSON/YAML serialization.

    :param config: Input configuration.
    :type config: Configuration
    :param algorithm: Name of the policy optimization algorithm.
    :type algorithm: str
    :param workload: Workload settings.
    :type workload: Workload
    :param source_config_hash: Hash values for original configuration. Defaults to None.
    :type source_config_hash: Optional[str]
    :param execution_schedule: Dictionary representing the execution schedule. Defaults to None.
    :type execution_schedule: Optional[dict]
    :return: Dictionary representation of the optimized plan.
    :type: dict
    """
    d = plan_to_dict(config)
    d.update(
        {
            "schema_version": "rpeg.optimized_plan.v1",
            "source_config_hash": source_config_hash,
            "algorithm": algorithm.lower(),
            "workload": {
                "batch_size": workload.B,
                "samples_per_prompt": workload.K,
                "context_length": workload.S,
                "generation_length": workload.T,
                "gen_quantiles": [list(x) for x in workload.gen_quantiles] if workload.gen_quantiles else None,
            },
        }
    )
    if execution_schedule is not None:
        d["execution_schedule"] = execution_schedule
    canonical = json.dumps({k: v for k, v in d.items() if k != "plan_hash"}, sort_keys=True, separators=(",", ":"))
    d["plan_hash"] = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    ordered = {
        "schema_version": d["schema_version"],
        "plan_hash": d["plan_hash"],
        "source_config_hash": d["source_config_hash"],
        "algorithm": d["algorithm"],
        "workload": d["workload"],
        "device_space": d["device_space"],
        "structure": d["structure"],
        "plan": d["plan"],
    }
    if execution_schedule is not None:
        ordered["execution_schedule"] = d["execution_schedule"]
    return ordered


def execution_schedule_to_dict(trace: Trace, *, horizon: int) -> dict:
    """
    Return a plain dictionary representation of an execution schedule suitable for JSON/YAML
    serialization.

    :param trace: Input execution schedule.
    :type trace: Trace
    :param horizon: Number of iterations/steps to unroll. Should be a non-negative integer.
    :type horizon: int
    :return: Dictionary representation of the execution schedule.
    :type: dict
    :raises ValueError: If horizon is zero or negative.
    """
    if horizon < 1:
        raise ValueError("execution schedule horizon must be >= 1")
    nodes = []
    for dispatch_index, node in enumerate(trace.start):
        nodes.append(
            {
                "node_id": node.id,
                "dispatch_index": dispatch_index,
                "predicted_start_s": float(trace.start[node]),
                "predicted_finish_s": float(trace.finish[node]),
            }
        )
    return {
        "policy": "rpeg_critical_path_list_v1",
        "horizon": horizon,
        "predicted_makespan_s": float(trace.makespan()),
        "nodes": nodes,
    }


def save_optimized_plan(
    config: Configuration,
    path: str,
    algorithm: str,
    workload: Workload,
    source_config_hash: Optional[str] = None,
    execution_schedule: Optional[dict] = None,
) -> None:
    """
    Serialize an versioned optimized plan artifact and write it to path.

    Format is inferred from the file extension:
      .yaml / .yml  → YAML
      anything else → JSON (default)

    :param config: Input configuration.
    :type config: Configuration
    :param path: Path to the output file.
    :type path: str
    :param algorithm: Name of the policy optimization algorithm.
    :type algorithm: str
    :param workload: Workload settings.
    :type workload: Workload
    :param source_config_hash: Hash values for original configuration. Defaults to None.
    :type source_config_hash: Optional[str]
    :param execution_schedule: Dictionary representing the execution schedule. Defaults to None.
    :type execution_schedule: Optional[dict]
    """
    d = optimized_plan_to_dict(
        config,
        algorithm,
        workload,
        source_config_hash,
        execution_schedule=execution_schedule,
    )
    if path.endswith((".yaml", ".yml")):
        import yaml
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    else:
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
            f.write("\n")


def load_optimized_plan(path: str) -> dict:
    """
    Load an versioned optimized plan artifact from a YAML/JSON file into a plain dictionary.

    :param path: Path to the input execution plan.
    :type path: str
    :return: Dictionary containing the versioned optimized plan artifact.
    :rtype: dict
    :raises ValueError: If schema version is not supported.
    """
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            import yaml
            d = yaml.safe_load(f)
        else:
            d = json.load(f)
    if d.get("schema_version") != "rpeg.optimized_plan.v1":
        raise ValueError("optimized plan must use schema_version rpeg.optimized_plan.v1")
    return d
