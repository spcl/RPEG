# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt

"""
core.log — structured logging for RPEG.

Sets up a clean stdout logger (no timestamps, no level prefixes) and
provides:
  - init_logging(path)   — tee all output (print + logger) to a log file
  - print_run_header()   — dump the full run configuration at startup
"""

import atexit
import logging
import os
import sys

from datetime import datetime
from typing import Any, Optional, Set


class _Tee:
    """
    The _Tee class wraps sys.stdout, so that every write goes to both the terminal and a log file.
    """
    def __init__(self, real_stdout, log_file) -> None:
        """
        Initialize class with the file handlers to stdout and the log file.

        :param real_stdout: stdout file handler.
        :param log_file: Log file handler.
        """
        self._stdout  = real_stdout
        self._file    = log_file

    def write(self, data: str) -> int:
        """
        Write data string to stdout and if possible the log file.

        :param data: String to write.
        :type data: str
        :return: Number of characters in the written data.
        :rtype: int
        """
        self._stdout.write(data)
        if not self._file.closed:
            self._file.write(data)
        return len(data)

    def flush(self) -> None:
        """
        Force any temporarily cached data to be written to stdout and if possible the log file.
        """
        self._stdout.flush()
        if not self._file.closed:
            self._file.flush()

    def fileno(self) -> int:
        """
        Returns the file descriptor of stdout.

        Necessary for some tools that call fileno().

        :return: stdout file descriptor.
        :rtype: int
        """
        return self._stdout.fileno()

    def __getattr__(self, name) -> Any:
        """
        Get value of the key name from the stdout object.

        Forwards isatty(), encoding, etc.

        :return: Value for the given key.
        :rtype: Any
        """
        return getattr(self._stdout, name)


def init_logging(log_path: Optional[str] = None) -> str:
    """
    Redirect all stdout output (print + logger) to both the terminal and log_path. If log_path is
    None, a timestamped file is created inside a 'logs/' directory in the current working directory.

    Returns the resolved log file path.

    :param log_path: Path to the log file. Defaults to None.
    :type log_path: Optional[str]
    """
    if log_path is None:
        os.makedirs("logs", exist_ok=True)
        stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join("logs", f"run_{stamp}.log")

    log_file   = open(log_path, "w", buffering=1, encoding="utf-8")
    tee        = _Tee(sys.__stdout__, log_file)
    sys.stdout = tee

    # The module-level StreamHandler captured the original sys.stdout at import
    # time — redirect it to the Tee so logger.info() calls are also captured.
    _handler.stream = tee

    atexit.register(log_file.close)
    print(f"Logging to: {log_path}\n")
    return log_path


# ── Logger setup ──────────────────────────────────────────────────────────────

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))

logger = logging.getLogger("RPEG")
logger.setLevel(logging.INFO)
logger.addHandler(_handler)
logger.propagate = False   # don't double-print if root logger is configured


# ── Internal helpers ───────────────────────────────────────────────────────────

_W = 68   # rule width

def _rule(char: str = "═") -> str:
    """
    Repeat a string N times.

    :param char: String to be repeated, normally a single character. Defaults to '='.
    :type char: str
    :return: Repeated string.
    :rtype: str
    """
    return char * _W

def _h(title: str) -> str:
    """
    Format the input string as a title on a new line and with an indentation.

    :param title: Input string.
    :type title: str
    :return: Formatted string.
    :rtype: str
    """
    return f"\n  {title}"


# ── Public API ─────────────────────────────────────────────────────────────────

def print_run_header(suite, workload, hw, device_space, optim, t_cand, trainable, K_eval) -> None:
    """
    Print a structured summary of all run hyperparameters.

    t_cand is a Dict[Model, Set[int]] mapping each trainable model to its staleness candidate set
    (e.g., {Model.ACTOR: {0, 1}, Model.CRITIC: {0}}).

    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param workload: Workload settings.
    :type: Workload
    :param hw: Hardware settings.
    :type hw: Dict
    :param device_space: Device space settings.
    :type device_space: Dict
    :param optim: Optimizer hyperparameters.
    :type optim: Dict
    :param t_cand: Maps each trainable model to its staleness candidate set.
    :type t_cand: Dict[Model, Set[int]]
    :param trainable: List of trainable models.
    :type trainable: List
    :param K_eval: Unroll factor.
    :type K_eval: int
    """
    from core.types import Model
    from model.models import transformer_parameter_count

    _MODEL_NAMES = {
        Model.ACTOR:     "Actor",
        Model.CRITIC:    "Critic",
        Model.REFERENCE: "Reference",
        Model.REWARD:    "Reward",
        Model.AC_SHARE:  "AC-Share",
    }

    log = logger.info
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log(_rule())
    log(f"  RPEG — Pipeline Run  ({stamp})")
    log(_rule())

    # Device space
    total = device_space.R * device_space.N * device_space.G
    log(_h("Device Space"))
    log(f"    R={device_space.R} racks  ×  N={device_space.N} nodes/rack  ×  G={device_space.G} GPUs/node"
        f"  →  {total} GPUs total")

    # Workload
    tokens = workload.B * workload.K * (workload.S + workload.T)
    log(_h("Workload"))
    log(f"    B={workload.B} prompts  K={workload.K} samples/prompt"
        f"  S={workload.S} ctx tokens  T={workload.T} gen tokens")
    log(f"    tokens/step = {workload.B} × {workload.K} × ({workload.S}+{workload.T})"
        f" = {tokens:,}")

    # Models
    log(_h("Models"))
    log(f"    {'model':12s}  {'L':>4s}  {'d_model':>7s}  {'d_ff':>6s}  {'vocab':>6s}  {'dtype':>5s}  {'params':>8s}  role")
    log(f"    {'─'*12}  {'─'*4}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*20}")
    for m, spec in suite.specs.items():
        n_params = transformer_parameter_count(spec, m)
        is_train = m in trainable
        tau_set  = t_cand.get(m) if t_cand else None
        if is_train:
            tau_str = ",".join(str(v) for v in sorted(tau_set)) if tau_set else "0"
            role_str = f"trainable  τ={{{tau_str}}}"
        else:
            role_str = "frozen"
        log(f"    {_MODEL_NAMES.get(m, m.value):12s}  {spec.L:>4d}  {spec.d:>7d}  {spec.d_ff:>6d}  {spec.V:>6d}  {spec.dtype_bytes*8:>4d}b  {n_params/1e9:>7.2f} B  {role_str}")

    # Hardware
    log(_h("Hardware"))
    log(f"    compute     {hw.flops_per_gpu/1e12:.1f} TFLOP/s"
        f"  ·  {hw.memory_per_gpu/1e9:.1f} GB HBM  per GPU")
    log(f"    mem BW      {hw.memory_bandwidth/1e12:.2f} TB/s  per GPU")
    log(f"    coll intra   α={hw.alpha_coll_g*1e6:.1f} µs   BW={1/hw.beta_coll_g/1e9:.0f} GB/s")
    log(f"    coll inter-n α={hw.alpha_coll_n*1e6:.1f} µs   BW={1/hw.beta_coll_n/1e9:.0f} GB/s")
    log(f"    p2p  intra   α={hw.alpha_p2p_g*1e6:.1f} µs   BW={1/hw.beta_p2p_g/1e9:.0f} GB/s")
    log(f"    p2p  inter-n α={hw.alpha_p2p_n*1e6:.1f} µs   BW={1/hw.beta_p2p_n/1e9:.0f} GB/s")

    # Optimizer
    log(_h("Optimizer"))
    log(f"    budget     E_max={optim.E_max}  E_struct={optim.E_struct}  R_max={optim.R_max}  I_max={optim.I_max}")
    log(f"    SA         T_init={optim.T_init}  T_warm={optim.T_warm}  T_min={optim.T_min}  T_cool={optim.T_cool:.4f}")
    log(f"    move gen   K_x={optim.K_x}  T_max={optim.T_max}")
    log(f"    scoring    lambda_ac={optim.lambda_ac}  score_eps={optim.score_eps}")
    log(f"    eval       K_eval={K_eval}")

    log(f"\n{_rule()}\n")
