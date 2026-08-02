#!/usr/bin/env python
"""The heat-wave DEFINITIONS compared in this project, as data.

Each Definition names a variable (which daily-statistics file to read) and which
daily statistic(s) must exceed their day-of-year percentile threshold for a day to
count as "hot". A heat wave is >= MIN_DUR consecutive hot days (handled downstream).
Keeping the three definitions as a single list here means the detection + plotting
code is written ONCE and simply iterates over definitions (no per-definition code).

  ours    : our definition -- T850 at 00 UTC (a single daily value) > 95th pctile.
  mixture : the ECMWF definition applied at 850 hPa -- daily T850 MIN and MAX both
            > their 95th pctiles.
  ecmwf   : the ECMWF definition -- daily 2 m temperature MIN and MAX both > their
            95th pctiles.

Daily-statistics files (see controller/heatwave/staging_daily): one per
(variable, year), <tag>_daily_<year>.nc, with variables <tag>_tmin / <tag>_tmax /
<tag>_tval (00 UTC value). tag = 't850' | 't2m'.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# Detection percentile, shared by all three definitions. Configurable so the whole
# comparison can be re-run at a stricter threshold (e.g. HW_PCT=0.99) and saved
# alongside the default. PTAG is the filename/label token (p95, p99, ...).
PCT = float(os.environ.get("HW_PCT", "0.95"))
PP = round(PCT * 100)
PTAG = f"p{PP}"


@dataclass(frozen=True)
class Definition:
    name: str                     # short id (also the file/label token)
    tag: str                      # daily-stats file prefix: 't850' | 't2m'
    stats: tuple                  # daily stats that must ALL exceed the pctile: subset
                                  # of ('tmin', 'tmax', 'tval')
    pct: float = PCT              # percentile threshold (highest 5% -> 0.95)
    label: str = ""               # human-readable label for plots

    def var(self, stat: str) -> str:
        """Variable name of a stat inside the daily-stats NetCDF (e.g. t850_tmin)."""
        return f"{self.tag}_{stat}"


OURS = Definition(
    "ours", "t850", ("tval",), PCT, f"ours: T$_{{850}}$ 00 UTC > {PTAG}")  # tval = 00 UTC value
MIXTURE = Definition(
    "mixture", "t850", ("tmin", "tmax"), PCT, f"mixture: T$_{{850}}$ min & max > {PTAG}")
ECMWF = Definition(
    "ecmwf", "t2m", ("tmin", "tmax"), PCT, f"ECMWF: T$_{{2m}}$ min & max > {PTAG}")

# order = increasing strictness of the condition (single value -> both min&max),
# 'mixture' sits between 'ours' and 'ecmwf' (same min&max rule as ECMWF but at 850 hPa).
DEFINITIONS = [OURS, MIXTURE, ECMWF]
BY_NAME = {d.name: d for d in DEFINITIONS}

# consistent colours for the three definitions on overlay plots
COLORS = {"ours": "#1f77b4", "mixture": "#9467bd", "ecmwf": "#d62728"}


def selected(names=None):
    """Return the Definitions named in `names` (list/str), or all three."""
    if not names:
        return list(DEFINITIONS)
    if isinstance(names, str):
        names = names.replace(",", " ").split()
    return [BY_NAME[n] for n in names]
