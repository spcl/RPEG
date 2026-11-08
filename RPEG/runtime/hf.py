# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import json
import math
import re
import struct

from pathlib import Path
from typing import Any

from model.models import ModelSpec
from runtime.errors import RuntimeConfigError, RuntimeDependencyError


_HF_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _require_pinned_revision(repo: str, revision: str | None) -> str:
    if not revision:
        raise RuntimeConfigError(
            f"huggingface model {repo!r} requires a pinned revision commit SHA; "
            "branches and floating tags are not accepted by the runtime"
        )
    revision = str(revision)
    if not _HF_COMMIT_RE.fullmatch(revision):
        raise RuntimeConfigError(
            f"huggingface model {repo!r} revision must be a 40-character commit SHA, got {revision!r}"
        )
    return revision


def download_hf_snapshot(repo: str, revision: str, cache_dir: str | None = None) -> Path:
    """
    Download or locate an immutable Hugging Face snapshot.

    The runtime requires commit-SHA revisions for reproducibility. Local snapshots should be
    supplied with checkpoint_path and bypass this helper.
    """
    revision = _require_pinned_revision(repo, revision)
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeDependencyError(
            "huggingface_hub is required to download Hugging Face model snapshots"
        ) from exc
    path = snapshot_download(
        repo_id=repo,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=False,
    )
    return Path(path)


def resolve_model_path(
    source: str,
    repo: str | None,
    checkpoint_path: str | None,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> Path:
    if source == "local":
        if not checkpoint_path:
            raise RuntimeConfigError("local model source requires checkpoint_path")
        return Path(checkpoint_path)
    if source == "huggingface":
        if checkpoint_path:
            return Path(checkpoint_path)
        if not repo:
            raise RuntimeConfigError("huggingface model source requires repo or checkpoint_path")
        return download_hf_snapshot(repo, _require_pinned_revision(repo, revision), cache_dir=cache_dir)
    raise RuntimeConfigError(f"Unsupported model source {source!r}")


def load_hf_config(model_dir: str | Path) -> dict[str, Any]:
    path = Path(model_dir) / "config.json"
    if not path.exists():
        raise RuntimeConfigError(f"Missing Hugging Face config: {path}")
    with path.open() as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise RuntimeConfigError(f"{path} must contain a JSON object")
    return cfg


def model_spec_from_hf_config(cfg: dict[str, Any]) -> ModelSpec:
    try:
        layers = int(cfg["num_hidden_layers"])
        hidden = int(cfg["hidden_size"])
        vocab = int(cfg["vocab_size"])
        intermediate = int(cfg["intermediate_size"])
    except KeyError as exc:
        raise RuntimeConfigError(f"HF config missing required key: {exc.args[0]}") from exc
    head_dim = int(cfg.get("head_dim") or hidden // int(cfg.get("num_attention_heads", 1)))
    return ModelSpec(L=layers, d=hidden, d_ff=intermediate, V=vocab, dtype_bytes=2, d_head=head_dim)


def infer_model_spec(model_dir: str | Path) -> ModelSpec:
    config = load_hf_config(model_dir)
    spec = model_spec_from_hf_config(config)
    parameter_count = safetensors_parameter_count(
        model_dir,
        tied_embeddings=bool(config.get("tie_word_embeddings", False)),
    )
    if parameter_count is None:
        return spec
    return ModelSpec(
        L=spec.L,
        d=spec.d,
        d_ff=spec.d_ff,
        V=spec.V,
        dtype_bytes=spec.dtype_bytes,
        d_head=spec.d_head,
        parameter_count=parameter_count,
    )


def safetensors_parameter_count(
    model_dir: str | Path,
    *,
    tied_embeddings: bool = False,
) -> int | None:
    """
    Count checkpoint tensors from safetensors headers without loading weights.
    """
    root = Path(model_dir)
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeConfigError(f"{index_path} has no safetensors weight_map")
        files = sorted({root / str(name) for name in weight_map.values()})
    else:
        consolidated = root / "model.safetensors"
        files = [consolidated] if consolidated.exists() else sorted(root.glob("model-*.safetensors"))
    if not files:
        return None

    total = 0
    seen_names: set[str] = set()
    tensor_sizes: dict[str, int] = {}
    for path in files:
        if not path.exists():
            raise RuntimeConfigError(f"Missing safetensors shard referenced by checkpoint: {path}")
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise RuntimeConfigError(f"Truncated safetensors header: {path}")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size <= 0 or header_size > 256 * 1024 * 1024:
                raise RuntimeConfigError(f"Invalid safetensors header size {header_size}: {path}")
            payload = handle.read(header_size)
        try:
            header = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeConfigError(f"Invalid safetensors header JSON: {path}") from exc
        if not isinstance(header, dict):
            raise RuntimeConfigError(f"Safetensors header must be an object: {path}")
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if name in seen_names:
                raise RuntimeConfigError(f"Duplicate safetensors tensor {name!r} in {path}")
            if not isinstance(entry, dict) or not isinstance(entry.get("shape"), list):
                raise RuntimeConfigError(f"Invalid safetensors tensor entry {name!r} in {path}")
            shape = entry["shape"]
            if any(not isinstance(dim, int) or dim < 0 for dim in shape):
                raise RuntimeConfigError(f"Invalid safetensors shape for {name!r} in {path}")
            seen_names.add(name)
            tensor_sizes[name] = math.prod(shape)
            total += tensor_sizes[name]
    if tied_embeddings:
        embedding_names = [
            name
            for name in tensor_sizes
            if name.endswith("embed_tokens.weight") or name.endswith("word_embeddings.weight")
        ]
        output_names = [
            name
            for name in tensor_sizes
            if name == "lm_head.weight" or name.endswith(".lm_head.weight")
        ]
        if embedding_names and output_names:
            embedding_size = tensor_sizes[embedding_names[0]]
            output_size = tensor_sizes[output_names[0]]
            if embedding_size != output_size:
                raise RuntimeConfigError(
                    "tied HF input/output embedding tensors have different parameter counts"
                )
            # Some safetensors exports materialize both aliases even though the runtime model shares
            # one Parameter object.
            total -= output_size
    if total <= 0:
        raise RuntimeConfigError(f"No tensors found in safetensors checkpoint {root}")
    return total
