# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt

"""
Unified matplotlib style for paper figures.
"""

import matplotlib
import matplotlib.pyplot as plt

RCPARAMS = {
    # Font: Liberation Sans
    'font.family': 'sans-serif',
    'font.sans-serif': ['Liberation Sans', 'Helvetica', 'DejaVu Sans', 'Arial'],
    'mathtext.fontset': 'dejavusans',

    # Vector PDF output (TrueType, no rasterization)
    'pdf.fonttype': 42,
    'ps.fonttype': 42,

    # Font sizes (8pt minimum at print)
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,

    # Figure output
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,

    # Axes and ticks
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.4,
    'ytick.major.width': 0.4,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'in',
    'ytick.direction': 'in',

    # Grid
    'axes.grid': True,
    'grid.alpha': 0.15,
    'grid.linewidth': 0.3,

    # Spines: hide top/right for cleaner look
    'axes.spines.top': False,
    'axes.spines.right': False,
}


def apply_style():
    """
    Apply the unified style to matplotlib.
    """
    matplotlib.rcParams.update(RCPARAMS)
