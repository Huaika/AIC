#!/usr/bin/env python
"""The heatwave DEFINITIONS compared in this project, as data.

Each Definition names a variable (which daily-statistics file to read) and which
daily statistic(s) must exceed their day-of-year percentile threshold for a day to
count as "hot". A heatwave is >= MIN_DUR consecutive hot days (handled downstream).
Keeping the three definitions as a single list here means the detection + plotting
code is written ONCE and simply iterates over definitions (no per-definition code).

  ours    : our definition -- T850 at 00 UTC (a single daily value) > 95th pctile.
  mixture : the ECMWF definition applied at 850 hPa -- daily T850 MIN and MAX both
            > their 95th pctiles.
  ecmwf   : the ECMWF definition -- daily 2 m temperature MIN and MAX both > their
            95th pctiles.
  cordex  : the EURO-CORDEX definition -- daily 2 m MAX > the 99th pctile of the
            May-Sep daily maxima of the 1971-2000 control period. Unlike the others
            this is a SINGLE seasonal threshold (kind="season"), fixed at the 99th
            percentile and referenced to 1971-2000, so it is window-independent.

The three windowed definitions use a day-of-year +/-window percentile (kind="doy")
over the 1991-2020 reference; EURO-CORDEX uses a single May-Sep seasonal percentile.

Daily-statistics files (see controller/heatwave/staging_daily): one per
(variable, year), <tag>_daily_<year>.nc, with variables <tag>_tmin / <tag>_tmax /
<tag>_tval (00 UTC value). tag = 't850' | 't2m'.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from aic import config

# Detection percentile for the WINDOWED (doy) definitions. Configurable so the whole
# comparison can be re-run at a stricter threshold (e.g. HW_PCT=0.99) and saved
# alongside the default. PTAG is the filename/label token (p95, p99, ...).
PCT = config.env_float("HW_PCT", 0.95)
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
    kind: str = "doy"             # threshold kind: "doy" (day-of-year +/-window) or
                                  # "season" (single seasonal percentile)
    season: tuple | None = None   # inclusive month range (m0, m1) when kind=="season"
    ref_years: tuple = (1991, 2020)   # inclusive reference-period bounds

    def var(self, stat: str) -> str:
        """Variable name of a stat inside the daily-stats NetCDF (e.g. t850_tmin)."""
        return f"{self.tag}_{stat}"

    @property
    def ref_range(self) -> range:
        """The reference years, as a range (end-inclusive bounds)."""
        return range(self.ref_years[0], self.ref_years[1] + 1)


OURS = Definition(
    "ours", "t850", ("tval",), PCT, f"ours: T$_{{850}}$ 00 UTC > {PTAG}")  # tval = 00 UTC value
MIXTURE = Definition(
    "mixture", "t850", ("tmin", "tmax"), PCT, f"mixture: T$_{{850}}$ min & max > {PTAG}")
ECMWF = Definition(
    "ecmwf", "t2m", ("tmin", "tmax"), PCT, f"ECMWF: T$_{{2m}}$ min & max > {PTAG}")
# EURO-CORDEX: single seasonal (May-Sep) 99th-pctile threshold, 1971-2000 control
# period -- fixed regardless of the HW_PCT sweep, and window-independent.
CORDEX = Definition(
    "cordex", "t2m", ("tmax",), 0.99,
    "EURO-CORDEX: T$_{2m}$ max > May–Sep p99 (1971–2000)",
    kind="season", season=(5, 9), ref_years=(1971, 2000))

# order = increasing strictness of the condition (single value -> both min&max);
# 'mixture' sits between 'ours' and 'ecmwf'; EURO-CORDEX (seasonal) sits apart.
DEFINITIONS = [OURS, MIXTURE, ECMWF, CORDEX]
BY_NAME = {d.name: d for d in DEFINITIONS}

# consistent colours for the definitions on overlay plots (highly distinguishable).
# Copernicus (ecmwf)=green, Copernicus Adjusted (mixture)=violet, EURO-CORDEX (cordex)=orange.
COLORS = {"ours": "#1f77b4", "mixture": "#9467bd", "ecmwf": "#2ca02c",
          "cordex": "#ff7f0e"}


def selected(names=None):
    """Return the Definitions named in `names` (list/str), or all three."""
    if not names:
        return list(DEFINITIONS)
    if isinstance(names, str):
        names = names.replace(",", " ").split()
    return [BY_NAME[n] for n in names]
