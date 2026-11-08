# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys

from typing import Any


_REQUIRED_DISTRIBUTIONS = {
    "torch": "2.6",
    "transformers": "4.51",
    "safetensors": "0.5",
    "huggingface-hub": "0.30",
    "PyYAML": "6.0.2",
    "megatron-core": "0.17.0",
    "pytest": "9.0",
    "absl-py": "2.0",
    "psutil": "5.9",
}


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in value.split("."):
        digits = ""
        for char in part:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _distribution_versions() -> tuple[dict[str, str | None], list[str]]:
    versions: dict[str, str | None] = {}
    errors: list[str] = []
    for name, minimum in _REQUIRED_DISTRIBUTIONS.items():
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
            errors.append(f"missing required distribution {name}")
            continue
        versions[name] = version
        if _version_tuple(version) < _version_tuple(minimum):
            errors.append(f"{name} {version} is below required minimum {minimum}")
    if versions.get("megatron-core") not in {None, "0.17.0"}:
        errors.append(
            f"megatron-core {versions['megatron-core']} does not match the validated API version 0.17.0"
        )
    transformers_version = versions.get("transformers")
    if transformers_version is not None and _version_tuple(transformers_version) >= (6,):
        errors.append(f"transformers {transformers_version} is outside the supported <6 range")
    return versions, errors


def _import_status(module: str) -> tuple[bool, str | None]:
    try:
        importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - report binary and ABI import errors verbatim.
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def check_environment(
    *,
    require_cuda: bool,
    require_ucx: bool,
    require_transformer_engine: bool = False,
) -> dict[str, Any]:
    versions, errors = _distribution_versions()
    imports = {}
    modules = [
        "torch",
        "yaml",
        "transformers",
        "safetensors",
        "huggingface_hub",
        "megatron.core",
    ]
    if require_transformer_engine:
        modules.append("transformer_engine.pytorch")
        try:
            versions["transformer-engine"] = importlib.metadata.version("transformer-engine")
        except importlib.metadata.PackageNotFoundError:
            versions["transformer-engine"] = None
            errors.append("missing required distribution transformer-engine")
    for module in modules:
        ok, error = _import_status(module)
        imports[module] = {"ok": ok, "error": error}
        if not ok:
            errors.append(f"failed to import {module}: {error}")

    ucx_module = None
    ucx_errors = {}
    for module in ("ucxx", "ucp"):
        ok, error = _import_status(module)
        ucx_errors[module] = error
        if ok:
            ucx_module = module
            break
    if require_ucx and ucx_module is None:
        errors.append(
            "ucx_direct requires an importable ucxx or ucp Python binding; "
            f"import failures: {ucx_errors}"
        )

    cuda: dict[str, Any] = {}
    if imports["torch"]["ok"]:
        import torch

        cuda = {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "distributed_available": bool(torch.distributed.is_available()),
            "nccl_available": bool(torch.distributed.is_nccl_available()),
            "nccl_version": (
                torch.cuda.nccl.version()
                if torch.cuda.is_available() and hasattr(torch.cuda, "nccl")
                else None
            ),
        }
        if require_cuda and not cuda["cuda_available"]:
            errors.append("torch.cuda.is_available() is false")
        if require_cuda and not cuda["nccl_available"]:
            errors.append("torch.distributed NCCL backend is unavailable")

    python_ok = sys.version_info >= (3, 10)
    if not python_ok:
        errors.append(f"Python {platform.python_version()} is below required minimum 3.10")
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        errors.append("SCRATCH is not set")
    elif not os.path.isdir(scratch) or not os.access(scratch, os.W_OK):
        errors.append(f"SCRATCH is not a writable directory: {scratch}")

    return {
        "schema_version": "rpeg.environment_check.v1",
        "ok": not errors,
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "hostname": platform.node(),
        "distributions": versions,
        "imports": imports,
        "cuda": cuda,
        "ucx_binding": ucx_module,
        "require_cuda": require_cuda,
        "require_ucx": require_ucx,
        "require_transformer_engine": require_transformer_engine,
        "scratch": scratch,
        "fabric_environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_HOME",
                "CUDA_PATH",
                "RPEG_NVIDIA_SITE",
                "NCCL_NET",
                "NCCL_NET_GDR_LEVEL",
                "NCCL_CROSS_NIC",
                "NCCL_PROTO",
                "FI_CXI_DEFAULT_CQ_SIZE",
                "FI_CXI_DEFAULT_TX_SIZE",
                "FI_CXI_DISABLE_HOST_REGISTER",
                "FI_CXI_RX_MATCH_MODE",
                "FI_MR_CACHE_MONITOR",
                "MPICH_GPU_SUPPORT_ENABLED",
                "CUDA_CACHE_DISABLE",
                "TRITON_HOME",
            )
        },
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the RPEG runtime Python/CUDA environment")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-ucx", action="store_true")
    parser.add_argument("--require-transformer-engine", action="store_true")
    args = parser.parse_args(argv)
    report = check_environment(
        require_cuda=args.require_cuda,
        require_ucx=args.require_ucx,
        require_transformer_engine=args.require_transformer_engine,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
