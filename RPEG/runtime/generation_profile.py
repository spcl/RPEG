# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import argparse
import hashlib
import json
import math

from collections import Counter
from pathlib import Path
from typing import Iterable

from runtime.errors import RuntimeConfigError, RuntimeDependencyError
from runtime.io import dump_mapping


def fit_generation_quantiles(
    lengths: Iterable[int],
    fractions: Iterable[float],
    *,
    max_new_tokens: int,
) -> tuple[tuple[float, int], ...]:
    observed = sorted(max(1, min(int(length), int(max_new_tokens))) for length in lengths)
    requested = tuple(float(fraction) for fraction in fractions)
    if not observed:
        raise RuntimeConfigError("generation profiling produced no response lengths")
    if not requested or any(not 0.0 < fraction <= 1.0 for fraction in requested):
        raise RuntimeConfigError("profile quantile fractions must be in (0, 1]")
    if any(left >= right for left, right in zip(requested, requested[1:])):
        raise RuntimeConfigError("profile quantile fractions must be strictly increasing")
    if requested[-1] != 1.0:
        raise RuntimeConfigError("profile quantile fractions must end at 1.0")

    fitted: list[tuple[float, int]] = []
    for fraction in requested[:-1]:
        index = max(0, min(len(observed) - 1, math.ceil(fraction * len(observed)) - 1))
        token_count = observed[index]
        if token_count >= max_new_tokens:
            continue
        empirical_fraction = sum(length <= token_count for length in observed) / len(observed)
        point = (float(empirical_fraction), int(token_count))
        if fitted and point[1] == fitted[-1][1]:
            fitted[-1] = point
        elif not fitted or point[0] > fitted[-1][0]:
            fitted.append(point)
    fitted.append((1.0, int(max_new_tokens)))
    return tuple(fitted)


def _load_prompts(path: Path, sample_size: int) -> list[str]:
    prompts = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if len(prompts) >= sample_size:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "prompt" not in row:
                raise RuntimeConfigError(f"{path}:{line_number} missing prompt")
            prompts.append(str(row["prompt"]))
    if not prompts:
        raise RuntimeConfigError(f"prompt profile source is empty: {path}")
    return prompts


def profile_generation(args: argparse.Namespace) -> dict:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeDependencyError("generation profiling requires torch and transformers") from exc

    model_path = Path(args.model_path).resolve()
    prompt_path = Path(args.prompts).resolve()
    prompts = _load_prompts(prompt_path, args.sample_size)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    ).to(device)
    model.eval()
    lengths: list[int] = []
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start:start + args.batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.context_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generation_kwargs = {
                **encoded,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if args.do_sample:
                generation_kwargs.update(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
            generated = model.generate(**generation_kwargs)
        responses = generated[:, encoded["input_ids"].shape[1]:]
        for response in responses:
            length = int(response.numel())
            if tokenizer.eos_token_id is not None:
                eos = (response == int(tokenizer.eos_token_id)).nonzero(as_tuple=False)
                if eos.numel():
                    length = int(eos[0, 0]) + 1
            lengths.append(max(1, min(length, args.max_new_tokens)))

    fractions = tuple(float(value) for value in args.quantiles.split(",") if value.strip())
    quantiles = fit_generation_quantiles(lengths, fractions, max_new_tokens=args.max_new_tokens)
    config_path = model_path / "config.json"
    model_config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.exists() else None
    return {
        "schema_version": "rpeg.generation_profile.v1",
        "model_path": str(model_path),
        "model_config_sha256": model_config_sha256,
        "prompt_path": str(prompt_path),
        "sample_size": len(lengths),
        "seed": args.seed,
        "context_length": args.context_length,
        "max_new_tokens": args.max_new_tokens,
        "generation": {
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
        "length_histogram": {str(length): count for length, count in sorted(Counter(lengths).items())},
        "mean_length": sum(lengths) / len(lengths),
        "quantiles": [list(point) for point in quantiles],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile empirical response lengths for RPEG lowering")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--quantiles", default="0.25,0.5,0.85,1.0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)
    if min(args.sample_size, args.batch_size, args.context_length, args.max_new_tokens) <= 0:
        parser.error("sample-size, batch-size, context-length, and max-new-tokens must be positive")
    result = profile_generation(args)
    dump_mapping(result, Path(args.output))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
