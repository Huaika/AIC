#!/usr/bin/env python
"""Percentile climatologies on the model grid -- the threshold each heat-wave
definition compares against. Two flavours:

  * ``doy_percentile``    -- a per-calendar-day threshold with a +/-window (ours /
                             mixture / ECMWF).
  * ``season_percentile`` -- a SINGLE per-cell threshold from one season's values
                             (EURO-CORDEX: the 99th pctile of May-Sep daily maxima).
"""
from __future__ import annotations

import numpy as np


def doy_percentile(vals, doy, window, q, ndoy=366):
    """Per-calendar-day q-quantile of `vals` (T, Y, X), pooling a circular +/-window
    of neighbouring calendar days. Returns thr[ndoy, Y, X] (index = doy-1)."""
    out = np.full((ndoy,) + vals.shape[1:], np.nan, "float32")
    for d in range(1, ndoy + 1):
        dist = np.abs(doy - d)
        dist = np.minimum(dist, ndoy - dist)
        sel = vals[dist <= window]
        if sel.shape[0]:
            out[d - 1] = np.quantile(sel, q, axis=0)
    return out


def season_percentile(vals, months, season, q):
    """Single per-cell q-quantile of `vals` (T, Y, X) over the calendar months in
    ``season`` (inclusive ``(m_start, m_end)``, e.g. ``(5, 9)`` = May-Sep). Returns a
    2-D ``thr[Y, X]`` -- one fixed threshold per cell, no day-of-year dependence
    (the EURO-CORDEX heat-wave threshold)."""
    m0, m1 = season
    months = np.asarray(months)
    sel = vals[(months >= m0) & (months <= m1)]
    return np.quantile(sel, q, axis=0).astype("float32")
