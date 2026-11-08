# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import argparse
import csv
import json

from pathlib import Path
from typing import Any


_SUMMARY_FIELDS = (
    "steps",
    "wall_time_s",
    "generation_tokens",
    "generation_tokens_per_s",
    "generation_node_tokens_per_s",
    "semantic_transfer_count",
    "semantic_transfer_bytes",
    "weight_transfer_count",
    "weight_transfer_bytes",
    "checkpoint_recovery_count",
    "zero2_reduced_bytes",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _one_result(row: dict[str, str]) -> dict[str, Any]:
    root = Path(row["result_root"])
    result: dict[str, Any] = {
        "case": row["case"],
        "nodes": int(row["nodes"]),
        "repetition": int(row["repetition"]),
        "job_id": row["job_id"],
        "result_root": str(root),
        "status": "missing",
    }
    collective = root / "collective_benchmark.json"
    summaries = sorted(root.glob("metrics/*.summary.json"))
    alerts = sorted(root.glob("metrics/*.alerts.json"))

    if collective.exists():
        report = _load_json(collective)
        measurements = report.get("results") or []
        largest = measurements[-1] if measurements else {}
        result.update(
            {
                "status": "ok",
                "world_size": report.get("world_size"),
                "collective_size_bytes": largest.get("size_bytes"),
                "collective_algorithmic_bandwidth_gbps": largest.get(
                    "algorithmic_bandwidth_gbps"
                ),
                "collective_bus_bandwidth_gbps": largest.get("bus_bandwidth_gbps"),
            }
        )
        return result

    if not summaries:
        return result
    summary = _load_json(summaries[-1])
    alert = _load_json(alerts[-1]) if alerts else {}
    result["status"] = str(alert.get("status") or summary.get("alert_status") or "unknown")
    result["algorithm"] = summary.get("algorithm")
    result["failed_node_count"] = len(summary.get("failed_nodes") or [])
    result["alert_count"] = int(alert.get("alert_count") or len(summary.get("alerts") or []))
    for field in _SUMMARY_FIELDS:
        result[field] = summary.get(field)
    return result


def build_suite_report(submission_root: str | Path) -> dict[str, Any]:
    root = Path(submission_root).resolve()
    manifest = root / "submitted.tsv"
    if not manifest.exists():
        raise FileNotFoundError(f"submission manifest not found: {manifest}")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"case", "nodes", "repetition", "job_id", "result_root"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"{manifest} does not contain required columns {sorted(required)}")
    results = [_one_result(row) for row in rows]
    return {
        "schema_version": "rpeg.clariden_suite_report.v1",
        "submission_root": str(root),
        "job_count": len(results),
        "ok_count": sum(item["status"] == "ok" for item in results),
        "critical_count": sum(item["status"] == "critical" for item in results),
        "missing_count": sum(item["status"] == "missing" for item in results),
        "results": results,
    }


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for result in results:
        for field in result:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate one suite submission")
    parser.add_argument("submission_root")
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    parser.add_argument("--strict", action="store_true", help="fail for missing, critical, or unknown results")
    args = parser.parse_args(argv)

    report = build_suite_report(args.submission_root)
    root = Path(args.submission_root)
    json_path = Path(args.output_json) if args.output_json else root / "suite_report.json"
    csv_path = Path(args.output_csv) if args.output_csv else root / "suite_report.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_csv(csv_path, report["results"])
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and any(item["status"] != "ok" for item in report["results"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
