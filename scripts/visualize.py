# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import shutil
import subprocess

from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


matplotlib.rcParams['pdf.fonttype'] = 42 # TrueType fonts in PDF (vector, LaTeX-compatible)
matplotlib.rcParams['ps.fonttype'] = 42

# GANTT (schedule)
def trace_to_gantt(
    executable,
    trace,
    device_space,
    path: str,
    title: str = "Schedule Gantt Chart",
    show_comm: bool = True,
    show_reshard: bool = True,
    show_labels: bool = True,
    group_by: str = "gpu",
    t_max: float = None,
) -> None:
    """
    Generate a Gantt-style bar chart of the schedule.

    Y-axis: hardware, organized as rack > node > GPU
    X-axis: wall-clock time
    Bar color: model (one color per model, same for GEN/INF/TRAIN)
      Idle: light gray bars wherever a GPU has no scheduled operation
      COMM: gray bars (data communication)
      RESHARD: lavender bars (parameter reallocation)
    Bar text: node id (e.g. g_pi@3) on bars wide enough to fit

    :param executable: Graph to execute.
    :type executable: ExecutableRPEG
    :param trace: Execution trace.
    :type trace: Trace
    :param device_space: Available devices and their hierarchy. (R × N × G)
    :type device_space: DeviceSpace
    :param path: Path to the output file.
    :type path: str
    :param title: Title of the plot. Defaults to 'Schedule Gantt Chart'.
    :type title: str
    :param show_comm: Flag to indicate, whether COMM nodes should be rendered. Defaults to True.
    :type show_comm: bool
    :param show_reshard: Flag to indicate, whether RESHARD nodes should be rendered. Defaults to
                         True.
    :type show_reshard: bool
    :param show_labels: Flag to indicate, whether overlay node-id text on bars wide enough to fit
                        should be displayed. Defaults to True.
    :type show_labels: bool
    :param group_by: Indicate whether the data should be grouped. Possible values: 'gpu' (one row
                     per physical GPU for more details) or 'node' (one row per node for a more
                     compact representation for larger clusters). Defaults to 'gpu'.
    :type group_by: str
    :param t_max: Upper limit for the x axis, which is ignored if the trace is longer. Defaults to
                  None.
    :type t_max: float
    """
    matplotlib.use("Agg")

    R, N, G = device_space.R, device_space.N, device_space.G

    # Row index
    # row 0 = top of chart. We invert when placing matplotlib bars.
    if group_by == "node":
        row_of = {}
        row_labels_list = []
        for r in range(R):
            for n in range(N):
                idx = r * N + n
                row_of[(r, n)] = idx
                row_labels_list.append(f"Node {idx}")
        num_rows = R * N

        def _alloc_rows(alloc) -> Set:
            """
            Return the rows of the allocation, i.e., the bars to plot.

            :param alloc: Input allocation.
            :type alloc: Allocation
            :return: Set of row indices.
            :rtype: Set
            """
            or_, on, og = alloc.origin
            er, en, eg = alloc.extent
            out = set()
            for dr in range(er):
                for dn in range(en):
                    key = (or_ + dr, on + dn)
                    if key in row_of:
                        out.add(row_of[key])
            return out

    else: # 'gpu'
        row_of = {}
        row_labels_list = []
        for r in range(R):
            for n in range(N):
                for g in range(G):
                    idx = r * N * G + n * G + g
                    row_of[(r, n, g)] = idx
                    row_labels_list.append(f"GPU {idx}")
        num_rows = R * N * G

        def _alloc_rows(alloc) -> Set:
            """
            Return the rows of the allocation, i.e., the bars to plot.

            :param alloc: Input allocation.
            :type alloc: Allocation
            :return: Set of row indices.
            :rtype: Set
            """
            or_, on, og = alloc.origin
            er, en, eg = alloc.extent
            out = set()
            for dr in range(er):
                for dn in range(en):
                    for dg in range(eg):
                        key = (or_ + dr, on + dn, og + dg)
                        if key in row_of:
                            out.add(row_of[key])
            return out

    def _node_rows(node) -> Set:
        """
        Return the set of row indices of the allocations associated with this node.

        :param node: Node to process.
        :type node: TemplateVertex
        :return: Set of row indices.
        :rtype: Set
        """
        # COMM/RESHARD only block destination devices (allocs[1]),
        # matching the scheduler's non-blocking-send semantics.
        if node.kind.value in ("comm", "reshard") and len(node.allocs) >= 2:
            return _alloc_rows(node.allocs[1])
        out = set()
        for alloc in node.allocs:
            out |= _alloc_rows(alloc)
        return out

    # Color palette
    # Consistent across Gantt and DAG graph.
    # One color per model — same shade regardless of role (GEN/INF/TRAIN).
    _MODEL_COLORS = {
        "Pi": "#6CB4EE", # Actor – vivid sky blue
        "V": "#FF8080", # Critic – bright coral
        "Ref": "#FFB347", # Reference – bright orange
        "R": "#66E680", # Reward – bright green
        "AC": "#D9A0F0", # AC_Share – bright purple
    }
    _COMM_COLOR = "#D0D0D0" # neutral gray (data communication)
    _RESHARD_COLOR = "#E0B0E0" # light plum (parameter reallocation)
    _IDLE_COLOR = "#F0F0F0" # very light gray (idle)
    _FALLBACK = "#888888"

    _MODEL_LONG = {
        "Pi": "Actor",
        "V": "Critic",
        "Ref": "Ref",
        "R": "Reward",
        "AC": "AC-Share",
    }

    def _bar_color(node) -> Tuple[str, str, float]:
        """
        Return (facecolor, edgecolor, linewidth) for a node bar.

        :param node: Input node.
        :type node: TemplateVertex
        :return: Tuple of the facecolor, edgecolor, and linewidth for the bar.
        :rtype: Tuple[str, str, float]
        """
        _EC = "#333333"
        _LW = 0.5
        if node.kind.value == "comm":
            return _COMM_COLOR, _EC, _LW
        if node.kind.value == "reshard":
            return _RESHARD_COLOR, _EC, _LW
        color = _MODEL_COLORS.get(node.model.value if node.model else "", _FALLBACK)
        return color, _EC, _LW

    def _node_id_label(node) -> str:
        """
        Label printed on the bar.

        OP nodes: id@step on the first line, (dp,pp,tp,mb) on the second.
        RESHARD: 'RSRD' + model name on second line.
        COMM/other: node id only.

        :param node: Input node.
        :type node: TemplateVertex
        :return: Label string.
        :rtype: str
        """
        if node.kind.value == "reshard":
            if node.model:
                m = _MODEL_LONG.get(node.model.value, node.model.value)
                return f"RSRD\n{m}"
            return "RSRD"
        if node.kind.value == "op" and node.pi is not None:
            dp, pp, tp = node.pi
            mb = node.mb if node.mb is not None else "?"
            return f"{node.id}\n({dp},{pp},{tp},{mb})"
        return node.id   # comm nodes — id only

    def _legend_key(node) -> str:
        """
        One legend entry per model (no role subdivision).

        COMM and RESHARD each get one fixed entry.

        :param node: Input node.
        :type node: TemplateVertex
        :return: Legend string for that node.
        :rtype: str
        """
        if node.kind.value == "comm":
            return "Data Comm"
        if node.kind.value == "reshard":
            return "Reshard"
        m = node.model.value if node.model else ""
        return _MODEL_LONG.get(m, m or "Unknown")

    # Plot
    row_px = 0.38 # inches per row
    fig_h = max(5.0, num_rows * row_px + 2.5)
    fig_w = 22.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    BAR_H = 0.68
    BAR_OFF = (1.0 - BAR_H) / 2

    # Total time span — use t_max if provided (shared axis across charts)
    if trace.finish:
        t_end = max(trace.finish.values())
        t_start = min(trace.start.values())
    else:
        t_start = 0.0
        t_end = 1.0
    if t_max is not None:
        t_end = max(t_end, t_max)
    t_span = max(t_end - t_start, 1e-12)

    # Idle-time gray background (drawn first, zorder=1)
    # Paint every row fully gray; operation bars (zorder=2) will paint over it, leaving gray visible
    # only in genuinely idle gaps.
    for row_idx in range(num_rows):
        y = (num_rows - 1 - row_idx) + BAR_OFF
        ax.broken_barh(
            [(t_start, t_span)],
            (y, BAR_H),
            facecolors = _IDLE_COLOR,
            edgecolors = "none",
            linewidth = 0,
            alpha = 0.45,
            zorder = 1,
        )

    # Draw operation bars (zorder=2)
    legend_handles: dict = {}
    label_items: list = [] # (x_center, y_center, text, fontsize)

    for node in executable.nodes:
        if node not in trace.start:
            continue
        if node.kind.value == "comm" and not show_comm:
            continue
        if node.kind.value == "reshard" and not show_reshard:
            continue

        s = trace.start[node]
        f = trace.finish[node]
        dur = f - s
        if dur <= 0:
            continue

        color, ec, lw = _bar_color(node)
        node_rows = _node_rows(node)
        if not node_rows:
            continue

        # RESHARD bars sit above COMM bars (zorder 2.5 vs 2.0), so they are never hidden when they
        # overlap on the same rows.
        bar_zorder = 2.5 if node.kind.value == "reshard" else 2.0

        for row_idx in node_rows:
            y = (num_rows - 1 - row_idx) + BAR_OFF
            ax.broken_barh(
                [(s, dur)],
                (y, BAR_H),
                facecolors = color,
                edgecolors = ec,
                linewidth = lw,
                zorder = bar_zorder,
            )

        leg_key = _legend_key(node)
        if leg_key not in legend_handles:
            legend_handles[leg_key] = mpatches.Patch(
                facecolor = color,
                alpha = 1.0,
                edgecolor = "#333333",
                linewidth = 0.5,
                label = leg_key,
            )

        if show_labels and dur / t_span > 0.01:
            mid_row = sorted(node_rows)[len(node_rows) // 2]
            y_center = (num_rows - 1 - mid_row) + 0.5
            lbl = _node_id_label(node)
            if lbl:
                fs = max(3.5, min(6.5, 6.5 * (dur / t_span) ** 0.25))
                label_items.append((s + dur / 2, y_center, lbl, fs))

    # Add idle to legend
    legend_handles["Idle"] = mpatches.Patch(
        facecolor = _IDLE_COLOR,
        alpha = 0.50,
        edgecolor = "#333333",
        linewidth = 0.5,
        label = "Idle",
    )

    # Text labels (third pass, highest zorder)
    for x_c, y_c, txt, fs in label_items:
        ax.text(
            x_c,
            y_c,
            txt,
            ha = "center",
            va = "center",
            fontsize = fs,
            color = "#1A1A1A",
            fontweight = "bold",
            clip_on = True,
            zorder = 3,
        )

    # Hardware-label typographic hierarchy
    # One monochromatic scale: darker / heavier = higher level in the hierarchy.
    #   Rack : bold,   8.5 pt,  near-black  (#1A1A1A)
    #   Node : normal, 7.0 pt,  mid-gray    (#555555)
    #   GPU  : italic, 6.0 pt,  light-gray  (#AAAAAA)
    _LBL_RACK = dict(fontsize=8.5, color="#1A1A1A", fontweight="bold")
    _LBL_NODE = dict(fontsize=7.0, color="#555555")
    _LBL_GPU_FS  = 6.0
    _LBL_GPU_CLR = "#AAAAAA"

    # Y-axis (GPU tick labels — level 3)
    ytick_pos = [num_rows - 1 - i + 0.5 for i in range(num_rows)]
    fs = min(_LBL_GPU_FS, max(4.0, 8.0 - 0.04 * num_rows))
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(row_labels_list, fontsize=fs)

    # Apply GPU tick-label style (italic + light color) uniformly.
    for lbl in ax.get_yticklabels():
        lbl.set_color(_LBL_GPU_CLR)
        lbl.set_style("italic")

    ax.set_ylim(0, num_rows)

    # Separator lines
    if group_by == "gpu":
        rows_per_node = G
        rows_per_rack = N * G
    else:
        rows_per_node = 1
        rows_per_rack = N

    for row_idx in range(num_rows + 1):
        y = num_rows - row_idx
        if row_idx % rows_per_rack == 0:
            ax.axhline(y=y, color="#222222", linewidth=1.2, zorder=5)
        elif row_idx % rows_per_node == 0:
            ax.axhline(y=y, color="#777777", linewidth=0.6, linestyle="--", zorder=5)
        else:
            ax.axhline(y=y, color="#CCCCCC", linewidth=0.25, linestyle=":", zorder=5)

    # Rack background bands
    _RACK_COLORS = ["#E8F4FD", "#FEF9E7", "#E9F7EF", "#FDF2F8"]
    rows_per_rack_val = (N * G) if group_by == "gpu" else N
    for r in range(R):
        y_hi = num_rows - r * rows_per_rack_val
        y_lo = y_hi - rows_per_rack_val
        c = _RACK_COLORS[r % len(_RACK_COLORS)]
        ax.axhspan(y_lo, y_hi, facecolor=c, alpha=0.35, zorder=0)

    # Node sub-bands within each rack (GPU mode, alternating subtle shading)
    if group_by == "gpu" and N > 1:
        for r in range(R):
            for n in range(N):
                if n % 2 == 1: # odd nodes get a faint dark overlay
                    y_hi = num_rows - (r * N * G + n * G)
                    y_lo = y_hi - G
                    ax.axhspan(y_lo, y_hi, facecolor="#000000", alpha=0.05, zorder=0.05)

    # Rack annotation — level 1 (bold, near-black, outer column)
    for r in range(R):
        y_hi = num_rows - r * rows_per_rack_val
        y_lo = y_hi - rows_per_rack_val
        ax.text(
            -0.032,
            (y_lo + y_hi) / 2,
            f"Rack {r}",
            transform = ax.get_yaxis_transform(),
            ha = "right",
            va = "center",
            **_LBL_RACK,
        )

    # Node annotation — level 2 (normal, mid-gray, inner column)
    if group_by == "gpu" and N > 1:
        for r in range(R):
            for n in range(N):
                node_linear = r * N + n
                y_hi = num_rows - (r * N * G + n * G)
                y_lo = y_hi - G
                ax.text(
                    -0.014,
                    (y_lo + y_hi) / 2,
                    f"Node {node_linear}",
                    transform = ax.get_yaxis_transform(),
                    ha = "right",
                    va = "center",
                    **_LBL_NODE,
                )

    # X-axis
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="x", which="major", color="#DDDDDD", linewidth=0.5, zorder=0)
    ax.grid(axis="x", which="minor", color="#EEEEEE", linewidth=0.3, zorder=0)
    ax.set_xlim(left=0, right=t_end if t_max is not None else None)

    # Legend + tuple explanation
    # Sort: models first (Actor, Critic, …), then Idle, then COMM/RESHARD
    _LEG_ORDER = ["Actor", "Critic", "Reference", "Reward", "AC-Share", "Idle", "RSRD", "Data"]

    def _leg_sort_key(label: str) -> int:
        """
        Return the position of the label according to _LEG_ORDER.

        :param label: Input label.
        :type label: str
        :return: Index of the label in _LEG_ORDER or the length of _LEG_ORDER if the label is not
                 present.
        :rtype: int
        """
        for i, prefix in enumerate(_LEG_ORDER):
            if label.startswith(prefix):
                return i
        return len(_LEG_ORDER)

    sorted_handles = sorted(legend_handles.values(), key=lambda h: _leg_sort_key(h.get_label()))

    # Layout strategy: tight_layout leaves a fixed bottom margin (12 % of the figure height). The
    # x-axis label "Time (s)" lives just below the axes inside that margin. The legend and the
    # tuple-format note are placed further down inside the same reserved strip, using figure
    # coordinates so the positioning is independent of how many GPU rows the chart has.
    _BOTTOM_FRAC = 0.12 # fraction of figure height reserved below axes
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    # left=0.08 reserves extra left margin for the two-column Rack/Node labels.
    fig.tight_layout(rect=[0.08, _BOTTOM_FRAC, 1, 1])

    # Legend: centred horizontally, placed in the upper portion of the strip.
    leg = fig.legend(
        handles = sorted_handles,
        loc = "upper center",
        bbox_to_anchor = (0.5, _BOTTOM_FRAC - 0.01),
        ncol = len(sorted_handles), # always one row
        fontsize = 8,
        framealpha = 0.92,
        title = (
            "bar text: id@step / (dp, pp, tp, mb)\n"
            "dp = data-parallel  ·  pp = pipeline-parallel  ·  "
            "tp = tensor-parallel  ·  mb = microbatch count"
        ),
        title_fontsize = 7.5,
        borderaxespad = 0.4,
    )
    leg.get_title().set_ha("center")
    leg.get_title().set_linespacing(1.6)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fmt_alloc(a) -> str:
    """
    Returns a formated string for an allocation.

    :param a: Input allocation.
    :type a: Allocation
    :return: Formatted string for the allocation.
    :rtype: str
    """
    return f"o={a.origin}, e={a.extent}"


# EXECUTABLE
def exec_to_dot(executable, path: str) -> None:
    """
    Generate a publication-quality Graphviz DOT file for a lowered executable RPEG.

    OP nodes are ellipses coloured by model; COMM and RESHARD are small boxes.
    Strategy (dp,pp,tp), microbatch count, and placement are shown on OP nodes.

    :param executable: Graph to execute.
    :type executable: ExecutableRPEG
    :param path: Path to the output file.
    :type path: str
    """
    # Color palette — per-model, identical to gantt _MODEL_COLORS
    _MODEL_COLORS = {
        "Pi": "#6CB4EE", # Actor – vivid sky blue
        "V": "#FF8080", # Critic – bright coral
        "Ref": "#FFB347", # Reference – bright orange
        "R": "#66E680", # Reward – bright green
        "AC": "#D9A0F0", # AC_Share – bright purple
    }
    _MODEL_FULL = {
        "Pi": "Actor",
        "V": "Critic",
        "Ref": "Reference",
        "R": "Reward",
        "AC": "AC-Share",
    }
    _COMM_HEX    = "#D0D0D0" # neutral gray
    _RESHARD_HEX = "#E0B0E0" # light plum

    # Helpers
    def _esc(s: str) -> str:
        """
        Replace certain symbols ('&', '<' and '>') in the string with their html equivalent.

        :param s: Input string.
        :type s: str
        :return: String with certain symbols replaced.
        :rtype: str
        """
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _rw_row(prefix: str, containers) -> str:
        """
        Returns container names of read and write accesses as a formatted string.

        :param prefix: Prefix string indicating the type of access.
        :type prefix: str
        :param containers: Container names to format.
        :type containers: Tuple[Container, ...]
        :return: Formatted string.
        :rtype: str
        """
        if not containers:
            body = "&#8709;"
        else:
            names = [_esc(c.name) for c in containers]
            lines = []
            for i in range(0, len(names), 3):
                lines.append(", ".join(names[i:i+3]))
            body = ",<BR/>    ".join(lines)
        return (
            f'<TR><TD ALIGN="LEFT" CELLPADDING="2">'
            f'<FONT FACE="Liberation Mono" POINT-SIZE="12">'
            f'<I>{prefix}</I>: &#123;{body}&#125;'
            f'</FONT></TD></TR>'
        )

    def _info_row(text: str) -> str:
        """
        Return text as a centered monospace info row.

        :param text: Text to format/display.
        :type text: str
        :return: Formatted text.
        :rtype: str
        """
        return (
            f'<TR><TD ALIGN="CENTER" CELLPADDING="2">'
            f'<FONT FACE="Liberation Mono" POINT-SIZE="12">{_esc(text)}</FONT>'
            f'</TD></TR>'
        )

    def _fmt_alloc(a) -> str:
        """
        Returns a formated string for an allocation.

        :param a: Input allocation.
        :type a: Allocation
        :return: Formatted string for the allocation.
        :rtype: str
        """
        return f"o={a.origin}, e={a.extent}"

    def _node_html(n) -> str:
        """
        Convert node information into a formatted html string.

        :param n: Node to format/display.
        :type n: ExecNode
        :return: Formatted html string.
        :rtype: str
        """
        kind = n.kind.value

        if kind == "op" and n.op is not None:
            tv = n.op.template_vertex
            role = tv.kind.role.value
            model_key = tv.kind.model.value
            model = _MODEL_FULL.get(model_key, model_key)
            fill_hex = _MODEL_COLORS.get(model_key, "#FFFFFF")
            sub_str = f":{n.op.i}" if n.op.i is not None else ""
            rows = [
                f'<TR><TD ALIGN="CENTER" CELLPADDING="4">'
                f'<B><FONT FACE="Liberation Sans" POINT-SIZE="15">'
                f'{_esc(model)} [{role}]  k={n.op.k}{sub_str}'
                f'</FONT></B></TD></TR>',
            ]
            rows.append(_rw_row("R", tv.kind.read))
            rows.append(_rw_row("W", tv.kind.write))
            if n.pi is not None:
                dp, pp, tp = n.pi
                mb_str = f", mb={n.mb}" if n.mb is not None else ""
                rows.append(_info_row(f"\u03c0=({dp},{pp},{tp}){mb_str}"))
            if n.allocs:
                rows.append(_info_row(f"\u03c3: {_fmt_alloc(n.allocs[0])}"))
            table = (
                f'<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" BGCOLOR="{fill_hex}">'
                + "".join(f"\n  {r}" for r in rows)
                + "\n</TABLE>"
            )

        elif kind == "reshard":
            model_str = (_MODEL_FULL.get(n.model.value, n.model.value) + " ") if n.model else ""
            rows = [
                f'<TR><TD ALIGN="CENTER" CELLPADDING="4">'
                f'<B><FONT FACE="Liberation Sans" POINT-SIZE="14">'
                f'{_esc(model_str)}[RESHARD]'
                f'</FONT></B></TD></TR>',
            ]
            if n.label:
                rows.append(_info_row(n.label))
            table = (
                f'<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" BGCOLOR="{_RESHARD_HEX}">'
                + "".join(f"\n  {r}" for r in rows)
                + "\n</TABLE>"
            )

        else: # comm
            rows = [
                f'<TR><TD ALIGN="CENTER" CELLPADDING="4">'
                f'<B><FONT FACE="Liberation Sans" POINT-SIZE="14">[COMM]</FONT></B>'
                f'</TD></TR>',
            ]
            if n.label:
                rows.append(_info_row(n.label))
            table = (
                f'<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" BGCOLOR="{_COMM_HEX}">'
                + "".join(f"\n  {r}" for r in rows)
                + "\n</TABLE>"
            )

        return f"<{table}>"

    def _node_style(n) -> Tuple[str, str, str]:
        """
        Returns (shape, fillcolor, color) for the node depending on its type.

        :param n: Input node.
        :type n: ExecNode
        :return: Tuple of shape, fillcolor, and edge color.
        :rtype: Tuple[str, str, str]
        """
        if n.kind.value == "op":
            if n.op and n.op.template_vertex:
                model_key = n.op.template_vertex.kind.model.value
                fill = _MODEL_COLORS.get(model_key, "white")
                return ("ellipse", fill, "black")
            return ("ellipse", "white", "black")
        elif n.kind.value == "comm":
            return ("box", _COMM_HEX, "#555555")
        elif n.kind.value == "reshard":
            return ("box", _RESHARD_HEX, "#555555")
        return ("box", "white", "#555555")

    with open(path, "w") as f:
        f.write("digraph G {\n")
        f.write("  rankdir=LR;\n")
        f.write(
            '  graph [fontname="Liberation Sans", fontsize=11, nodesep=0.8, ranksep=1.4,'
            ' splines=true, bgcolor=white, margin=0.3];\n'
        )
        f.write(
            '  node  [fontname="Liberation Sans", fontsize=10, style=filled, penwidth=1.5];\n'
        )
        f.write('  edge  [fontname="Liberation Sans", fontsize=12, color="#34495E", penwidth=1.2, arrowsize=0.8];\n\n')

        for n in executable.nodes:
            shape, fillcolor, color = _node_style(n)
            if n.kind.value == "op":
                size_attrs = ' fixedsize=true, width=3.4, height=3.4,'
            else:
                size_attrs = ' fixedsize=false, width=0, height=0, margin="0.10,0.06",'
            f.write(
                f'  "{n.id}" [label={_node_html(n)}, shape={shape},'
                f'{size_attrs} fillcolor="{fillcolor}", color="{color}"];\n'
            )
        f.write('\n')

        for e in executable.edges:
            f.write(f'  "{e.src.id}" -> "{e.dst.id}";\n')

        f.write("}\n")


# RPEG
def rpeg_to_dot(rpeg, path: str, show_containers: bool = True) -> None:
    """
    Generate a publication-quality Graphviz DOT file for an RPEG graph.

    Encodes A_in / A_out as boundary port nodes and τ per-model on each vertex. Designed for
    publication-class vector-graphics figures.

    :param rpeg: RLM Pipeline Execution Graph.
    :type rpeg: RPEG
    :param path: Path to the output file.
    :type path: str
    :param show_containers: Flag indicating whether containers should be shown. Defaults to True.
    :type show_containers: bool
    """
    from collections import deque, defaultdict

    # Color palette — per-model, identical to gantt _MODEL_COLORS
    _MODEL_COLORS = {
        "Pi": "#6CB4EE", # Actor – vivid sky blue
        "V": "#FF8080", # Critic – bright coral
        "Ref": "#FFB347", # Reference – bright orange
        "R": "#66E680", # Reward – bright green
        "AC": "#D9A0F0", # AC_Share – bright purple
    }
    _MODEL_FULL = {
        "Pi": "Actor",
        "V": "Critic",
        "Ref": "Reference",
        "R": "Reward",
        "AC": "AC-Share",
    }

    # Helpers
    def _esc(s: str) -> str:
        """
        Replace certain symbols ('&', '<' and '>') in the string with their html equivalent.

        :param s: Input string.
        :type s: str
        :return: String with certain symbols replaced.
        :rtype: str
        """
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _rw_row(prefix: str, containers) -> str:
        """
        Returns a compact set notation (container names of read and write accesses) with line
        wrapping for long sets.

        :param prefix: Prefix string indicating the type of access.
        :type prefix: str
        :param containers: Container names to format.
        :type containers: Tuple[Container, ...]
        :return: Formatted string.
        :rtype: str
        """
        if not containers:
            body = "&#8709;"
        else:
            names = [_esc(c.name) for c in containers]
            # Wrap into lines of ≤ 2 items for readability inside ellipses
            lines = []
            for i in range(0, len(names), 2):
                lines.append(", ".join(names[i:i+2]))
            body = ",<BR/>    ".join(lines)
        return (
            f'<TR><TD ALIGN="LEFT" CELLPADDING="2">'
            f'<FONT FACE="Liberation Mono" POINT-SIZE="22">'
            f'<I>{prefix}</I>: &#123;{body}&#125;'
            f'</FONT></TD></TR>'
        )

    def _node_html_label(v) -> str:
        """
        Convert node information into a formatted html string to be used as a label.

        :param v: Node to format/display.
        :type v: ExecNode
        :return: Formatted html string to be used as a label.
        :rtype: str
        """
        role = v.kind.role.value
        model_key = v.kind.model.value
        model = _MODEL_FULL.get(model_key, model_key)
        fill_hex = _MODEL_COLORS.get(model_key, "#FFFFFF")
        # τ annotation — only for trainable models
        tau_val = rpeg.tau.get(v.kind.model, None)
        tau_str = f"  &#964;={tau_val}" if (tau_val is not None and v.kind.model in rpeg.trainable) else ""
        rows = [
            f'<TR><TD ALIGN="CENTER" CELLPADDING="4">'
            f'<B><FONT FACE="Liberation Sans" POINT-SIZE="28">'
            f'{_esc(model)} [{role}]{tau_str}'
            f'</FONT></B></TD></TR>'
        ]
        if show_containers:
            rows.append(_rw_row("R", v.kind.read))
            rows.append(_rw_row("W", v.kind.write))
        table = (
            f'<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" BGCOLOR="{fill_hex}">'
            + "".join(f"\n  {r}" for r in rows)
            + "\n</TABLE>"
        )
        return f"<{table}>"

    def _edge_label(e) -> str:
        """
        Return the name of the data container associated with the edge.

        :param e: Edge for which to retrieve the container name.
        :type e: TemplateEdge
        :return: Name of the data container.
        :rtype: str
        """
        return f'"{_esc(e.container.name)}"'

    def _bfs_levels(vertices, data_edges) -> Dict[int, List[TemplateVertex]]:
        """
        BFS from GEN-role roots over data edges → rank groups.

        :param vertices: Vertices in the RPEG.
        :type vertices: Tuple[TemplateVertex, ...]
        :param data_edges: Data edges in the RPEG.
        :type data_egdes: Tuple[TemplateEdge, ...]
        :return: Dictionary of depth levels and vertices in that depth level.
        :rtype: Dict[int, List[TemplateVertex]]
        """
        adj, in_deg = defaultdict(list), {v: 0 for v in vertices}
        for e in data_edges:
            adj[e.src].append(e.dst)
            in_deg[e.dst] += 1
        roots = ([v for v in vertices if v.kind.role.value == "GEN"]
                 or [v for v in vertices if in_deg[v] == 0]
                 or [vertices[0]])
        levels = {r: 0 for r in roots}
        queue  = deque(roots)
        while queue:
            v = queue.popleft()
            for w in adj[v]:
                if w not in levels:
                    levels[w] = levels[v] + 1
                    queue.append(w)
        max_lvl = max(levels.values(), default=0)
        for v in vertices:
            if v not in levels:
                levels[v] = max_lvl + 1
        groups = defaultdict(list)
        for v, lvl in levels.items():
            groups[lvl].append(v)
        return groups

    # Layout
    level_groups = _bfs_levels(rpeg.vertices, rpeg.data_edges)
    max_level = max(level_groups.keys(), default=0)
    train_nodes = level_groups.get(max_level, [])

    # Write DOT
    with open(path, "w") as f:
        f.write("digraph G {\n")
        f.write("  rankdir=TB;\n")
        f.write(
            '  graph [fontname="Liberation Sans", fontsize=22, nodesep=1.4, ranksep=1.6,'
            ' splines=true, bgcolor=white, margin=0.2];\n'
        )
        f.write(
            '  node  [fontname="Liberation Sans", fontsize=24, shape=ellipse,'
            ' style=filled, penwidth=1.5, fixedsize=true, width=5.0, height=3.2];\n'
        )
        f.write('  edge  [fontname="Liberation Sans", fontsize=26, penwidth=1.4];\n\n')

        # Rank constraints
        for level in sorted(level_groups.keys()):
            f.write(f'  {{ rank=same;')
            for v in level_groups[level]:
                f.write(f' "{v.id}";')
            f.write(' }\n')

        f.write('\n')

        # Centering: pull GEN above middle INF node
        gen_nodes = level_groups.get(0, [])
        inf_nodes = level_groups.get(1, [])
        if len(gen_nodes) == 1 and len(inf_nodes) >= 3:
            # High-weight invisible edges to all INF nodes for balanced centering
            for inf_v in inf_nodes:
                f.write(f'  "{gen_nodes[0].id}" -> "{inf_v.id}" [style=invis, weight=10];\n')
        f.write('\n')

        # Visible nodes
        for v in rpeg.vertices:
            model_key = v.kind.model.value
            f.write(
                f'  "{v.id}" [label={_node_html_label(v)},'
                f' fillcolor="{_MODEL_COLORS.get(model_key, "white")}",'
                f' color="black"];\n'
            )
        f.write('\n')

        # Data edges — solid, charcoal — leave south, enter north
        for e in rpeg.data_edges:
            f.write(
                f'  "{e.src.id}":s -> "{e.dst.id}":n'
                f' [label={_edge_label(e)}, color="#34495E", fontcolor="#34495E",'
                f' arrowsize=0.9, weight=10];\n'
            )
        # Publication edges — dashed dark red — leave north, enter north
        for e in rpeg.pub_edges:
            f.write(
                f'  "{e.src.id}":e -> "{e.dst.id}":e'
                f' [label={_edge_label(e)}, style=dashed, color="#B22222", penwidth=2.0,'
                f' arrowsize=0.9, fontcolor="#B22222"];\n'
            )

        f.write("}\n")


# UNROLLED
def unrolled_to_dot(unrolled, path: str, show_containers: bool = True, rpeg = None) -> None:
    """
    Generate a publication-quality Graphviz DOT file for an unrolled RPEG.

    Shows per-step instantiation of the template RPEG with step index k on each vertex. If rpeg is
    provided, τ is shown on trainable-model vertices.

    :param unrolled: Unrolled RPEG.
    :type unrolled: UnrolledRPEG
    :param path: Path to the output file.
    :type path: str
    :param show_containers: Flag indicating whether containers should be shown. Defaults to True.
    :type show_containers: bool
    :param rpeg: RLM Pipeline Execution Graph. Defaults to None.
    :type rpeg: Optional[RPEG]
    """
    # Color palette — per-model, identical to gantt _MODEL_COLORS
    _MODEL_COLORS = {
        "Pi": "#6CB4EE", # Actor – vivid sky blue
        "V": "#FF8080", # Critic – bright coral
        "Ref": "#FFB347", # Reference – bright orange
        "R": "#66E680", # Reward – bright green
        "AC": "#D9A0F0", # AC_Share – bright purple
    }
    _MODEL_FULL = {
        "Pi": "Actor",
        "V": "Critic",
        "Ref": "Reference",
        "R": "Reward",
        "AC": "AC-Share",
    }

    trainable = rpeg.trainable if rpeg is not None else set()
    tau_map = rpeg.tau if rpeg is not None else {}

    # Helpers
    def _esc(s: str) -> str:
        """
        Replace certain symbols ('&', '<' and '>') in the string with their html equivalent.

        :param s: Input string.
        :type s: str
        :return: String with certain symbols replaced.
        :rtype: str
        """
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _rw_row(prefix: str, containers) -> str:
        """
        Returns container names of read and write accesses as a formatted string.

        :param prefix: Prefix string indicating the type of access.
        :type prefix: str
        :param containers: Container names to format.
        :type containers: Tuple[Container, ...]
        :return: Formatted string.
        :rtype: str
        """
        if not containers:
            body = "&#8709;"
        else:
            names = [_esc(c.name) for c in containers]
            lines = []
            for i in range(0, len(names), 3):
                lines.append(", ".join(names[i:i+3]))
            body = ",<BR/>    ".join(lines)
        return (
            f'<TR><TD ALIGN="LEFT" CELLPADDING="2">'
            f'<FONT FACE="Liberation Mono" POINT-SIZE="11">'
            f'<I>{prefix}</I>: &#123;{body}&#125;'
            f'</FONT></TD></TR>'
        )

    def _node_id(x) -> str:
        """
        Returns a formatted string of the node ID.

        :param x: Input node.
        :type x: ExecNode
        :return: Formatted node ID.
        :rtype: str
        """
        return f"{x.template_vertex.id}@{x.k}" if x.i is None else f"{x.template_vertex.id}@{x.k}:{x.i}"

    def _node_label(x) -> str:
        """
        Convert node information into a formatted html string to be used as a label.

        :param x: Node to format/display.
        :type x: ExecNode
        :return: Formatted html string.
        :rtype: str
        """
        tv = x.template_vertex
        role = tv.kind.role.value
        model_key = tv.kind.model.value
        model = _MODEL_FULL.get(model_key, model_key)
        fill_hex = _MODEL_COLORS.get(model_key, "#FFFFFF")
        # τ — only for trainable models
        tau_val = tau_map.get(tv.kind.model, None)
        tau_str = f"  &#964;={tau_val}" if (tau_val is not None and tv.kind.model in trainable) else ""
        # Sub-index for generation quantile splits
        sub_str = f":{x.i}" if x.i is not None else ""
        rows = [
            f'<TR><TD ALIGN="CENTER" CELLPADDING="4">'
            f'<B><FONT FACE="Liberation Sans" POINT-SIZE="14">'
            f'{_esc(model)} [{role}]  k={x.k}{sub_str}{tau_str}'
            f'</FONT></B></TD></TR>'
        ]
        if show_containers:
            rows.append(_rw_row("R", tv.kind.read))
            rows.append(_rw_row("W", tv.kind.write))
        table = (
            f'<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" BGCOLOR="{fill_hex}">'
            + "".join(f"\n  {r}" for r in rows)
            + "\n</TABLE>"
        )
        return f"<{table}>"

    def _edge_label(e) -> str:
        """
        Return the name of the data container associated with the edge.

        :param e: Edge for which to retrieve the container name.
        :type e: TemplateEdge
        :return: Name of the data container.
        :rtype: str
        """
        return f'"{_esc(e.container.name)}"'

    # Group nodes by step and role for rank constraints
    from collections import defaultdict
    _ROLE_ORDER = {"GEN": 0, "INF": 1, "TRAIN": 2}
    phase_groups = defaultdict(list) # (k, role_order) → [node_id, ...]
    for x in unrolled.nodes:
        role = x.template_vertex.kind.role.value
        phase_groups[(x.k, _ROLE_ORDER.get(role, 1))].append(_node_id(x))

    # Write DOT
    with open(path, "w") as f:
        f.write("digraph G {\n")
        f.write("  rankdir=LR;\n")
        f.write(
            '  graph [fontname="Liberation Sans", fontsize=11, nodesep=0.6, ranksep=1.4,'
            ' splines=true, bgcolor=white, margin=0.3, ordering=out, newrank=true];\n'
        )
        f.write(
            '  node  [fontname="Liberation Sans", fontsize=12, shape=ellipse,'
            ' style=filled, penwidth=1.5, fixedsize=true, width=3.4, height=3.4];\n'
        )
        f.write('  edge  [fontname="Liberation Sans", fontsize=12, penwidth=1.2];\n\n')

        # Group each step k into a cluster subgraph to prevent vertical drift.
        # Within each cluster, use rank=same for each phase (GEN/INF/TRAIN).
        K = max(x.k for x in unrolled.nodes) + 1
        nodes_by_k = defaultdict(list)
        for x in unrolled.nodes:
            nodes_by_k[x.k].append(x)

        for k in sorted(nodes_by_k.keys()):
            f.write(f'  subgraph cluster_{k} {{\n')
            f.write(f'    style=invis;\n') # invisible cluster border

            # Rank groups within this step
            step_phases = defaultdict(list)
            for x in nodes_by_k[k]:
                role = x.template_vertex.kind.role.value
                step_phases[_ROLE_ORDER.get(role, 1)].append(_node_id(x))

            for phase in sorted(step_phases.keys()):
                ids = step_phases[phase]
                f.write(f'    {{ rank=same;')
                for nid in ids:
                    f.write(f' "{nid}";')
                f.write(' }\n')

            # Node definitions inside cluster
            for x in nodes_by_k[k]:
                model_key = x.template_vertex.kind.model.value
                f.write(
                    f'    "{_node_id(x)}" [label={_node_label(x)},'
                    f' fillcolor="{_MODEL_COLORS.get(model_key, "white")}",'
                    f' color="black"];\n'
                )
            f.write(f'  }}\n\n')

        # Invisible edges between clusters to enforce step ordering
        train_by_k = {}
        gen_by_k = {}
        for x in unrolled.nodes:
            role = x.template_vertex.kind.role.value
            if role == "TRAIN" and x.k not in train_by_k:
                train_by_k[x.k] = _node_id(x)
            if role == "GEN" and x.k not in gen_by_k:
                gen_by_k[x.k] = _node_id(x)
        for k in range(K - 1):
            if k in train_by_k and (k + 1) in gen_by_k:
                f.write(f'  "{train_by_k[k]}" -> "{gen_by_k[k+1]}" [style=invis, weight=10];\n')
        f.write('\n')

        # Data edges — solid, charcoal
        for e in unrolled.data_edges:
            f.write(
                f'  "{_node_id(e.src)}" -> "{_node_id(e.dst)}"'
                f' [label={_edge_label(e)}, color="#34495E", fontcolor="#34495E", arrowsize=0.9];\n'
            )
        # Publication edges — dashed dark red (recurrent weight updates)
        # constraint=false so they don't distort rank layout
        for e in unrolled.pub_edges:
            f.write(
                f'  "{_node_id(e.src)}" -> "{_node_id(e.dst)}"'
                f' [label={_edge_label(e)}, style=dashed, color="#B22222", penwidth=1.8,'
                f' arrowsize=0.9, fontcolor="#B22222", constraint=false];\n'
            )
        f.write("}\n")


# Conversion from DOT to image
_FORMAT_FROM_EXT = {
    ".png": "png", ".pdf": "pdf", ".svg": "svg", ".eps": "eps",
}

def dot_to_img(dot_out_pairs: Iterable[Tuple[Path, Path]]) -> None:
    """
    Convert .dot files to images via Graphviz's dot.

    Output format is inferred from the output file extension (.png, .pdf, .svg, .eps).

    :param dot_out_pairs: List of tuples (path to input dot file, path to output image file).
    :type dot_out_pairs: Iterable[Tuple[Path, Path]]
    :raises FileNotFoundError: If one of the input files does not exist.
    :raises RuntimeError: If Graphviz is not found.
    :raises ValueError: If one of the input dot files does not end with '.dot' or the format for the
                        output file is not supported.
    """
    dot_bin = shutil.which("dot")
    if dot_bin is None:
        raise RuntimeError("Graphviz 'dot' not found on PATH (install graphviz / load module).")

    for dot_path, out_path in dot_out_pairs:
        dot_path = Path(dot_path)
        out_path = Path(out_path)

        if dot_path.suffix != ".dot":
            raise ValueError(f"Expected a .dot file, got: {dot_path}")
        if not dot_path.exists():
            raise FileNotFoundError(f"Missing .dot file: {dot_path.resolve()}")

        fmt = _FORMAT_FROM_EXT.get(out_path.suffix.lower())
        if fmt is None:
            raise ValueError(
                f"Unsupported output format '{out_path.suffix}'. "
                f"Supported: {', '.join(_FORMAT_FROM_EXT.keys())}"
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [dot_bin, f"-T{fmt}", str(dot_path), "-o", str(out_path)],
            check=True,
        )


def safe_unlink(paths: Iterable[Path]) -> None:
    """
    Safely remove a list of temporary files.

    :param paths: List of paths to file that should be deleted.
    :type paths: Iterable[Path]
    """
    for p in paths:
        Path(p).unlink(missing_ok=True)
