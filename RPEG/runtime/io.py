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
from typing import Any

from runtime.errors import RuntimeConfigError


def load_mapping(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() in {".json"}:
        data = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeConfigError(
                f"Cannot read YAML file {p}: PyYAML is not installed. "
                "Install pyyaml, or use JSON for this Phase 0 environment."
            ) from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise RuntimeConfigError(f"{p} must contain a mapping at the top level")
    return data


def dump_mapping(data: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".json":
        p.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
        return
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeConfigError(
            f"Cannot write YAML file {p}: PyYAML is not installed. "
            "Install pyyaml, or write to a .json path."
        ) from exc
    with p.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def sha256_file(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()
