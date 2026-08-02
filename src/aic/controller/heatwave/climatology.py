#!/usr/bin/env python
"""Day-of-year percentile climatology on the model grid -- shared by every
definition (the 95th-percentile threshold for each daily statistic)."""
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
