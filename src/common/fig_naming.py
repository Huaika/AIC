#!/usr/bin/env python
"""SINGLE SOURCE OF TRUTH for diagnostic-figure names, shared by every model
(NeuralGCM, GraphCast) and every plot type so the scheme is changed in ONE place.

Filename scheme (fields joined by '_'; region & variable are 2-4 letter abbrevs;
other fields hyphenate any internal spaces so the name stays parseable):

    <model>_<dataset>_<region>_<variable>_<level>_<year>_<months>_<plottype>.<ext>

e.g.  neuralgcm_era5_wld_temp_L0850_2023_entire-year_spaghetti.png
      graphcast_era5_wld_temp_L0850_2023_entire-year_drift-rmse-bias.png

The folder tree is unchanged (figures/<dataset><year>/<months>/<region>/<variable>/
<plottype>/); only the filename carries the full self-describing scheme.
"""
from __future__ import annotations

# 2-4 letter region abbreviations
REGION_ABBR = {
    "world": "wld", "africa": "afr", "europe": "eur", "asia": "asi",
    "north_america": "nam", "south_america": "sam", "oceania": "oce",
    "antarctica": "ant",
}

# 2-4 letter variable abbreviations
VAR_ABBR = {
    "temperature": "temp", "geopotential": "geo", "specific_humidity": "shum",
    "u_component_of_wind": "uwnd", "v_component_of_wind": "vwnd",
    "specific_cloud_ice_water_content": "ciwc",
    "specific_cloud_liquid_water_content": "clwc",
    "wind_speed": "wspd",   # derived: sqrt(u^2 + v^2)
}

# plot-type token per plot family
PLOTTYPE = {
    "spaghetti": "spaghetti",
    "drift_stats": "drift-rmse-bias",
    "drift_maps": "drift-map",
}

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]


def months_token(period: int) -> str:
    """0 -> 'entire-year'; 1..12 -> month name (e.g. 'june')."""
    return "entire-year" if period == 0 else MONTHS[period - 1]


def level_token(level) -> str:
    """Accept an int level (850) or a preformatted 'L0850' -> 'L0850'."""
    if isinstance(level, str):
        return level if level.startswith("L") else f"L{int(level):04d}"
    return f"L{int(level):04d}"


def figure_name(model: str, dataset: str, region: str, variable: str, level,
                year, months: str, plottype: str, ext: str = "png") -> str:
    """Build the unified figure filename. ``region``/``variable`` may be the full
    canonical name (abbreviated here) or an already-short code; ``months`` is a
    token from months_token(); ``plottype`` is a key of PLOTTYPE (or a literal)."""
    return "_".join([
        model,
        dataset,
        REGION_ABBR.get(region, region),
        VAR_ABBR.get(variable, variable),
        level_token(level),
        str(year),
        months,
        PLOTTYPE.get(plottype, plottype),
    ]) + f".{ext}"
