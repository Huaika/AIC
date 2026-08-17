#!/usr/bin/env python
"""Shared plot-chrome palette -- the single source of truth for the ink/grid/muted
colours used across every figure.

It lives at the top level (not in ``view``) so BOTH the view layer and the
controller-side plotting drivers (``controller/heatwave/compare``, ``spectrum``,
``controller/casestudy/plots``) can import it without a controller->view
dependency. ``view.plotting`` re-exports these names for back-compat.

Model colours (blue=NeuralGCM, red=GraphCast) are a separate concern and live with
the sources in ``controller/eval/sources`` (``MODEL_COLORS`` / ``model_color``).
"""
from __future__ import annotations

import matplotlib as _mpl

INK = "#222222"     # primary text / axis ink
GRID = "#dddddd"    # grid lines
MUTED = "#666666"   # secondary text / spines / ticks

# Base font sizes for every figure (single source of truth). Elements that set an
# explicit fontsize= scale up from these in the plotting helpers; everything else
# (ticks, axis + colourbar labels, legends) inherits these rcParams.
_mpl.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})
