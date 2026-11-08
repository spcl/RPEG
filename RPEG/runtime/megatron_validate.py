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
import os
import time

from pathlib import Path

from runtime.control import Heartbeat, HeartbeatStore
from runtime.worker_entry import _rank_context
from runtime.workers.megatron_worker import MegatronBuildConfig, MegatronWorker


def _heartbeat(args, state: str, rank: int | None, world_size: int | None, local_rank: int | None, message: str | None = None) -> None:
    store = HeartbeatStore(Path(args.heartbeat_path).parent)
    store.write(
        Heartbeat(
            worker_id=args.worker_id,
            run_id=args.run_id,
            state=state,
            timestamp=time.time(),
            pid=os.getpid(),
            rank=rank if world_size and world_size > 1 else None,
            world_size=world_size,
            local_rank=local_rank,
            message=message,
            metadata={
                "vertex_id": args.vertex_id,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "hostname": os.uname().nodename,
                "master_addr": os.environ.get("MASTER_ADDR", ""),
                "master_port": os.environ.get("MASTER_PORT", ""),
            },
        )
    )


def _auxiliary_heads_for_vertex(vertex_id: str) -> tuple[str, ...]:
    if vertex_id in {"i_V", "t_V"}:
        return ("value",)
    if vertex_id == "i_R":
        return ("reward",)
    if vertex_id in {"t_ac", "t_ac_share"}:
        return ("value",)
    return ()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Distributed Megatron validation worker")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--vertex-id", required=True)
    parser.add_argument("--heartbeat-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dp", type=int, required=True)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=16)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--no-transformer-engine", action="store_true")
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--load-hf-checkpoint", action="store_true")
    parser.add_argument("--exercise-ppo", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    rank, world_size, local_rank = _rank_context()
    _heartbeat(args, "starting", rank, world_size, local_rank)
    worker = None
    try:
        import torch

        worker = MegatronWorker()
        worker.build_model(
            MegatronBuildConfig(
                model_path=args.model_path,
                parallelism=(args.dp, args.pp, args.tp),
                dtype=args.dtype,
                use_transformer_engine=not args.no_transformer_engine,
                perform_initialization=True,
                max_sequence_length=args.seq_length,
                activation_checkpointing=args.activation_checkpointing,
                microbatch=args.microbatch,
                auxiliary_heads=_auxiliary_heads_for_vertex(args.vertex_id),
            )
        )
        if args.load_hf_checkpoint:
            worker.load_hf_checkpoint(args.model_path, strict=False)
        input_ids = torch.randint(0, worker.architecture.vocab_size, (args.batch_size, args.seq_length))
        loss = worker.training_step(input_ids)
        ppo = None
        if args.exercise_ppo:
            generation_length = max(1, min(2, args.seq_length - 1))
            prompt_length = max(1, args.seq_length - generation_length)
            prompt_ids = torch.randint(0, worker.architecture.vocab_size, (args.batch_size, prompt_length))
            completion_ids, old_logps, response_mask, prompt_lens = worker.greedy_generate_with_logps(
                prompt_ids,
                max_new_tokens=generation_length,
            )
            ref_logps, _ = worker.response_token_logps(
                completion_ids,
                prompt_lens,
                generation_length,
                grad=False,
            )
            rewards = (ref_logps.float() * response_mask).sum(dim=1) / response_mask.sum(dim=1).clamp(min=1.0)
            values = ref_logps.detach().float()
            advantages = (rewards.unsqueeze(1) - values) * response_mask
            returns = (advantages + values) * response_mask
            actor_loss = worker.ppo_actor_step(
                completion_ids,
                prompt_lens,
                old_logps,
                advantages,
                response_mask,
                max_new_tokens=generation_length,
                clip_range=0.2,
                entropy_coef=0.0,
            )
            critic_loss = None
            if worker.value_head is not None:
                critic_loss = worker.ppo_critic_step(
                    completion_ids,
                    prompt_lens,
                    returns,
                    response_mask,
                    max_new_tokens=generation_length,
                )
            ppo = {
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "generated_shape": list(completion_ids.shape),
                "logps_shape": list(ref_logps.shape),
            }
        ckpt = None
        if args.checkpoint_dir:
            ckpt = str(worker.save_distributed_checkpoint(args.checkpoint_dir, step=1))
            worker.load_distributed_checkpoint(ckpt, strict=True)
        info = worker.built_model_info()
        print(
            json.dumps(
                {
                    "rank": rank,
                    "world_size": world_size,
                    "local_rank": local_rank,
                    "loss": loss,
                    "ppo": ppo,
                    "checkpoint_dir": ckpt,
                    "model": info.__dict__,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        worker.shutdown()
    except Exception as exc:
        _heartbeat(args, "error", rank, world_size, local_rank, str(exc))
        if worker is not None:
            worker.shutdown()
        raise
    _heartbeat(args, "finished", rank, world_size, local_rank)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
