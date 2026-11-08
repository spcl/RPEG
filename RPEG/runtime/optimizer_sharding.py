# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

from __future__ import annotations

import os
from collections.abc import Iterable
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any


def balanced_parameter_owners(parameters: Iterable[Any], data_parallel_size: int) -> tuple[int, ...]:
    """
    Assign whole parameters to DP ranks with deterministic greedy load balancing.
    """
    dp = int(data_parallel_size)
    if dp < 1:
        raise ValueError(f"data_parallel_size must be >= 1, got {dp}")
    sizes = [int(parameter.numel()) for parameter in parameters]
    loads = [0] * dp
    owners = [0] * len(sizes)
    for index in sorted(range(len(sizes)), key=lambda item: (-sizes[item], item)):
        owner = min(range(dp), key=lambda rank: (loads[rank], rank))
        owners[index] = owner
        loads[owner] += sizes[index]
    return tuple(owners)


@dataclass(frozen=True)
class FlatParameterSegment:
    parameter: Any
    parameter_offset: int
    global_offset: int
    numel: int


class FlatParameterLayout:
    """
    Contiguous padded parameter layout partitioned evenly across DP ranks.
    """
    def __init__(
        self,
        parameters: Iterable[Any],
        *,
        data_parallel_size: int,
        data_parallel_rank: int,
    ) -> None:
        params = tuple(parameters)
        dp = int(data_parallel_size)
        rank = int(data_parallel_rank)
        if dp < 1:
            raise ValueError(f"data_parallel_size must be >= 1, got {dp}")
        if rank < 0 or rank >= dp:
            raise ValueError(f"data_parallel_rank must be in [0,{dp}), got {rank}")
        self.parameters = params
        self.data_parallel_size = dp
        self.data_parallel_rank = rank
        self.parameter_offsets: dict[Any, tuple[int, int]] = {}
        offset = 0
        for parameter in params:
            end = offset + int(parameter.numel())
            self.parameter_offsets[parameter] = (offset, end)
            offset = end
        self.total_numel = offset
        self.shard_numel = (offset + dp - 1) // dp
        self.padded_numel = self.shard_numel * dp
        self.local_start = rank * self.shard_numel
        self.local_end = self.local_start + self.shard_numel

    def segments_for_global_range(self, start: int, end: int) -> tuple[FlatParameterSegment, ...]:
        if start < 0 or end < start or end > self.padded_numel:
            raise ValueError(f"invalid flat parameter range [{start},{end}) for padded size {self.padded_numel}")
        out = []
        valid_end = min(end, self.total_numel)
        for parameter in self.parameters:
            parameter_start, parameter_end = self.parameter_offsets[parameter]
            overlap_start = max(start, parameter_start)
            overlap_end = min(valid_end, parameter_end)
            if overlap_start >= overlap_end:
                continue
            out.append(FlatParameterSegment(
                parameter=parameter,
                parameter_offset=overlap_start - parameter_start,
                global_offset=overlap_start,
                numel=overlap_end - overlap_start,
            ))
        return tuple(out)

    def local_segments(self) -> tuple[FlatParameterSegment, ...]:
        return self.segments_for_global_range(self.local_start, self.local_end)


class _ModelParameterStateView(MutableMapping):
    def __init__(self, optimizer: "FP32MasterAdamW") -> None:
        self._optimizer = optimizer

    def __getitem__(self, model_parameter):
        master = self._optimizer.master_parameter(model_parameter)
        return {
            **self._optimizer._optimizer.state[master],
            "_rpeg_fp32_master": master.data,
        }

    def __setitem__(self, model_parameter, value) -> None:
        master = self._optimizer.master_parameter(model_parameter)
        master_value = value.get("_rpeg_fp32_master")
        if master_value is not None:
            master.data.copy_(master_value.to(device=master.device, dtype=master.dtype))
        self._optimizer._optimizer.state[master] = {
            key: (
                tensor.to(
                    device=master.device,
                    dtype=master.dtype if getattr(tensor, "ndim", 0) > 0 else tensor.dtype,
                )
                if hasattr(tensor, "to")
                else tensor
            )
            for key, tensor in value.items()
            if key != "_rpeg_fp32_master"
        }

    def __delitem__(self, model_parameter) -> None:
        del self._optimizer._optimizer.state[self._optimizer.master_parameter(model_parameter)]

    def __iter__(self):
        state = self._optimizer._optimizer.state
        for model_parameter, master in self._optimizer._master_by_model.items():
            if master in state:
                yield model_parameter

    def __len__(self) -> int:
        return len(self._optimizer._optimizer.state)

    def clear(self) -> None:
        self._optimizer._optimizer.state.clear()


class FP32MasterAdamW:
    """
    AdamW with FP32 master parameters and model-parameter keyed state.

    Only parameters owned by the current data-parallel rank receive master storage in ZeRO-1/2
    modes. The public state view remains keyed by the BF16 model parameters, so checkpointing and
    named optimizer resharding do not depend on the internal master tensors.
    """
    def __init__(
        self,
        model_parameters: Iterable[Any],
        *,
        lr: float,
        weight_decay: float,
        data_parallel_size: int = 1,
        data_parallel_rank: int = 0,
        shard_state: bool = False,
        flat_shard: bool = False,
        fused: bool | None = None,
    ) -> None:
        import torch

        parameters = tuple(model_parameters)
        self._flat_layout = (
            FlatParameterLayout(
                parameters,
                data_parallel_size=data_parallel_size,
                data_parallel_rank=data_parallel_rank,
            )
            if flat_shard and shard_state
            else None
        )
        self._flat_data_parallel_size = int(data_parallel_size)
        self._flat_data_parallel_rank = int(data_parallel_rank)
        self._flat_master = None
        if self._flat_layout is not None:
            flat_master = torch.zeros(
                self._flat_layout.shard_numel,
                dtype=torch.float32,
                device=parameters[0].device,
            )
            for segment in self._flat_layout.local_segments():
                destination_offset = segment.global_offset - self._flat_layout.local_start
                flat_master.narrow(0, destination_offset, segment.numel).copy_(
                    segment.parameter.detach().reshape(-1).narrow(
                        0,
                        segment.parameter_offset,
                        segment.numel,
                    )
                )
            self._flat_master = torch.nn.Parameter(flat_master, requires_grad=True)
            owned = ()
        elif shard_state:
            owners = balanced_parameter_owners(parameters, data_parallel_size)
            owned = tuple(
                parameter
                for parameter, owner in zip(parameters, owners, strict=True)
                if int(owner) == int(data_parallel_rank)
            )
        else:
            owned = parameters
        self._model_parameters = parameters
        self._master_by_model = {
            parameter: torch.nn.Parameter(
                parameter.detach().to(dtype=torch.float32, copy=True),
                requires_grad=True,
            )
            for parameter in owned
        }
        self._model_by_master = {
            master: parameter for parameter, master in self._master_by_model.items()
        }
        optimizer_parameters = (
            (self._flat_master,)
            if self._flat_master is not None
            else tuple(self._master_by_model.values())
        )
        fused_enabled = (
            bool(fused)
            if fused is not None
            else bool(optimizer_parameters)
            and all(parameter.device.type == "cuda" for parameter in optimizer_parameters)
            and os.environ.get("RPEG_FUSED_ADAMW", "1") == "1"
        )
        self._optimizer = torch.optim.AdamW(
            optimizer_parameters,
            lr=float(lr),
            weight_decay=float(weight_decay),
            fused=fused_enabled,
        )
        self.state = (
            _FlatModelParameterStateView(self)
            if self._flat_layout is not None
            else _ModelParameterStateView(self)
        )
        if self._flat_layout is not None:
            self._ensure_flat_state()

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    @property
    def defaults(self):
        return self._optimizer.defaults

    @property
    def fused(self) -> bool:
        return bool(self._optimizer.defaults.get("fused", False))

    def master_parameter(self, model_parameter):
        if self._flat_layout is not None:
            raise KeyError("flat-sharded optimizer parameters can span multiple model parameters")
        try:
            return self._master_by_model[model_parameter]
        except KeyError as exc:
            raise KeyError("model parameter is not owned by this optimizer rank") from exc

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._optimizer.zero_grad(set_to_none=set_to_none)
        for parameter in self._model_parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    def step(self, closure=None):
        import torch

        if self._flat_master is not None:
            if self._flat_master.grad is None:
                raise RuntimeError("flat-sharded AdamW requires a reduce-scattered gradient before step")
            return self._optimizer.step(closure=closure)
        for model_parameter, master in self._master_by_model.items():
            grad = model_parameter.grad
            master.grad = None if grad is None else grad.detach().to(dtype=torch.float32)
        loss = self._optimizer.step(closure=closure)
        for model_parameter, master in self._master_by_model.items():
            model_parameter.data.copy_(master.data)
        return loss

    def state_dict(self) -> dict[str, Any]:
        state = dict(self._optimizer.state_dict())
        state["rpeg_schema_version"] = "rpeg.fp32_master_adamw.v1"
        if self._flat_master is not None:
            state["rpeg_flat_master"] = self._flat_master.detach().clone()
        else:
            state["rpeg_master_parameters"] = [
                master.detach().clone()
                for master in self._master_by_model.values()
            ]
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        optimizer_state = {
            key: value
            for key, value in state_dict.items()
            if key in {"state", "param_groups"}
        }
        self._optimizer.load_state_dict(optimizer_state)
        if self._flat_master is not None:
            master = state_dict.get("rpeg_flat_master")
            if master is not None:
                self._flat_master.data.copy_(
                    master.to(device=self._flat_master.device, dtype=self._flat_master.dtype)
                )
            return
        masters = state_dict.get("rpeg_master_parameters")
        if masters is None:
            return
        local_masters = tuple(self._master_by_model.values())
        if len(masters) != len(local_masters):
            raise ValueError(
                f"checkpoint has {len(masters)} FP32 masters, expected {len(local_masters)}"
            )
        for destination, source in zip(local_masters, masters, strict=True):
            destination.data.copy_(source.to(device=destination.device, dtype=destination.dtype))

    @property
    def flat_layout(self) -> FlatParameterLayout | None:
        return self._flat_layout

    @property
    def is_flat_sharded(self) -> bool:
        return self._flat_master is not None

    @property
    def model_parameters_bound(self) -> bool:
        return bool(self._model_parameters)

    def unbind_model_parameters(self) -> None:
        """
        Drop all BF16 model references while retaining flat FP32 optimizer state.
        """
        if self._flat_master is None:
            raise RuntimeError("only a flat-sharded optimizer can be unbound")
        self._model_parameters = ()
        self._flat_layout = None

    def bind_model_parameters(self, model_parameters: Iterable[Any]) -> None:
        """
        Attach a rebuilt model with the canonical training parameter order.
        """
        if self._flat_master is None:
            raise RuntimeError("only a flat-sharded optimizer can be rebound")
        parameters = tuple(model_parameters)
        layout = FlatParameterLayout(
            parameters,
            data_parallel_size=self._flat_data_parallel_size,
            data_parallel_rank=self._flat_data_parallel_rank,
        )
        if int(layout.shard_numel) != int(self._flat_master.numel()):
            raise ValueError(
                f"rebuilt model flat shard has {layout.shard_numel} elements, "
                f"expected {self._flat_master.numel()}"
            )
        self._model_parameters = parameters
        self._flat_layout = layout

    def set_sharded_gradient(self, gradient) -> None:
        if self._flat_master is None:
            raise RuntimeError("optimizer is not flat sharded")
        if int(gradient.numel()) != int(self._flat_master.numel()):
            raise ValueError(
                f"gradient shard has {gradient.numel()} elements, expected {self._flat_master.numel()}"
            )
        self._flat_master.grad = gradient.detach().to(dtype=self._flat_master.dtype)

    def local_model_parameter_shard(self, *, dtype) -> Any:
        if self._flat_master is None:
            raise RuntimeError("optimizer is not flat sharded")
        return self._flat_master.detach().to(dtype=dtype)

    def set_local_model_gradients(self, *, dtype) -> Any:
        """
        Flatten the bound model gradients into this rank's optimizer shard.

        This is the DP=1 (degenerate ZeRO-2) counterpart of reduce-scatter. Keeping the same flat
        optimizer representation is important even when there is only one DP rank: RPEG can
        temporarily switch the worker to a different PP/TP inference layout and later rebind the
        canonical train layout without retaining the old BF16 module.
        """
        import torch

        if self._flat_master is None or self._flat_layout is None:
            raise RuntimeError("optimizer is not a bound flat shard")
        gradient = torch.zeros(
            self._flat_layout.shard_numel,
            dtype=dtype,
            device=self._flat_master.device,
        )
        for segment in self._flat_layout.local_segments():
            parameter_gradient = segment.parameter.grad
            if parameter_gradient is None:
                continue
            destination_offset = segment.global_offset - self._flat_layout.local_start
            gradient.narrow(0, destination_offset, segment.numel).copy_(
                parameter_gradient.detach().reshape(-1).narrow(
                    0,
                    segment.parameter_offset,
                    segment.numel,
                )
            )
            segment.parameter.grad = None
        self.set_sharded_gradient(gradient)
        return gradient

    def copy_flat_master_parameters_to_model(self) -> None:
        """
        Materialize this rank's complete local model from a DP=1 flat master.
        """
        if self._flat_master is None or self._flat_layout is None:
            raise RuntimeError("optimizer is not a bound flat shard")
        if self._flat_layout.data_parallel_size != 1:
            raise RuntimeError("local flat-master materialization is only valid for DP=1")
        for segment in self._flat_layout.local_segments():
            source_offset = segment.global_offset - self._flat_layout.local_start
            segment.parameter.data.reshape(-1).narrow(
                0,
                segment.parameter_offset,
                segment.numel,
            ).copy_(
                self._flat_master.data.narrow(0, source_offset, segment.numel).to(
                    dtype=segment.parameter.dtype
                )
            )

    def copy_master_parameters_to_model(self) -> None:
        """
        Restore locally owned model parameters from their FP32 masters.
        """
        if self._flat_master is not None:
            raise RuntimeError("flat-sharded optimizer parameters require distributed materialization")
        for model_parameter, master in self._master_by_model.items():
            if int(model_parameter.numel()) != int(master.numel()):
                raise RuntimeError(
                    "model parameter storage must be materialized before restoring optimizer masters"
                )
            model_parameter.data.copy_(
                master.data.to(device=model_parameter.device, dtype=model_parameter.dtype)
            )

    def _ensure_flat_state(self) -> dict[str, Any]:
        import torch

        if self._flat_master is None:
            raise RuntimeError("optimizer is not flat sharded")
        state = self._optimizer.state[self._flat_master]
        if not state:
            state.update({
                "step": torch.tensor(
                    0.0,
                    device=self._flat_master.device if self.fused else None,
                ),
                "exp_avg": torch.zeros_like(self._flat_master),
                "exp_avg_sq": torch.zeros_like(self._flat_master),
            })
        return state


class _FlatModelParameterStateView(MutableMapping):
    def __init__(self, optimizer: FP32MasterAdamW) -> None:
        self._optimizer = optimizer

    def _segment(self, model_parameter) -> FlatParameterSegment:
        layout = self._optimizer.flat_layout
        if layout is None:
            raise KeyError("optimizer is not flat sharded")
        for segment in layout.local_segments():
            if segment.parameter is model_parameter:
                return segment
        raise KeyError("model parameter does not intersect this optimizer shard")

    def __getitem__(self, model_parameter):
        segment = self._segment(model_parameter)
        layout = self._optimizer.flat_layout
        assert layout is not None
        state = self._optimizer._optimizer.state.get(self._optimizer._flat_master)
        if not state:
            raise KeyError(model_parameter)
        local_offset = segment.global_offset - layout.local_start
        return {
            "step": state["step"],
            "exp_avg": state["exp_avg"].narrow(0, local_offset, segment.numel),
            "exp_avg_sq": state["exp_avg_sq"].narrow(0, local_offset, segment.numel),
            "_rpeg_fp32_master": self._optimizer._flat_master.data.narrow(
                0,
                local_offset,
                segment.numel,
            ),
            "_rpeg_flat_shard": {
                "start": int(segment.parameter_offset),
                "end": int(segment.parameter_offset + segment.numel),
                "numel": int(model_parameter.numel()),
                "shape": tuple(int(x) for x in model_parameter.shape),
            },
        }

    def __setitem__(self, model_parameter, value) -> None:
        segment = self._segment(model_parameter)
        layout = self._optimizer.flat_layout
        assert layout is not None
        state = self._optimizer._ensure_flat_state()
        metadata = dict(value.get("_rpeg_flat_shard") or {})
        source_start = int(metadata.get("start", 0))
        source_end = int(metadata.get("end", model_parameter.numel()))
        local_start = int(segment.parameter_offset)
        local_end = local_start + int(segment.numel)
        overlap_start = max(source_start, local_start)
        overlap_end = min(source_end, local_end)
        if overlap_start >= overlap_end:
            return
        local_destination = segment.global_offset - layout.local_start + overlap_start - local_start
        source_offset = overlap_start - source_start
        length = overlap_end - overlap_start
        for key in ("exp_avg", "exp_avg_sq", "_rpeg_fp32_master"):
            if key not in value:
                continue
            source = value[key].reshape(-1).narrow(0, source_offset, length)
            destination = (
                self._optimizer._flat_master.data
                if key == "_rpeg_fp32_master"
                else state[key]
            )
            destination.narrow(0, local_destination, length).copy_(
                source.to(device=destination.device, dtype=destination.dtype)
            )
        incoming_step = value.get("step")
        if incoming_step is not None:
            state["step"].copy_(
                incoming_step.to(device=state["step"].device, dtype=state["step"].dtype)
                if hasattr(incoming_step, "to")
                else state["step"].new_tensor(incoming_step)
            )

    def __delitem__(self, model_parameter) -> None:
        del model_parameter
        raise NotImplementedError("individual flat optimizer state segments cannot be deleted")

    def __iter__(self):
        state = self._optimizer._optimizer.state.get(self._optimizer._flat_master)
        if not state:
            return
        layout = self._optimizer.flat_layout
        assert layout is not None
        for segment in layout.local_segments():
            yield segment.parameter

    def __len__(self) -> int:
        state = self._optimizer._optimizer.state.get(self._optimizer._flat_master)
        if not state:
            return 0
        layout = self._optimizer.flat_layout
        assert layout is not None
        return len(layout.local_segments())

    def clear(self) -> None:
        self._optimizer._optimizer.state.clear()
