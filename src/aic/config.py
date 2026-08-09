#!/usr/bin/env python
"""Central configuration for the aic package: cluster paths + typed env accessors.

The ONE place the package reads its environment. Every ``os.environ`` read in the
first-party code (controllers + views; the vendored GraphCast fork excepted) goes
through here, and every absolute ``/pfs/...`` path lives here -- so there is one
place to see the knobs and one place to repoint the data.

1. **Cluster paths in one place.** Absolute workspace paths default to the BW
   UniCluster layout but are overridable, so a teammate on a different cluster
   repoints the code without editing source::

       AIC_WORKSPACE        base scratch workspace (default /pfs/work9/workspace/scratch)
       AIC_DATA_ROOT        this project's data root (default <WS>/ka_dm9435-ai-climate)
       AIC_COAST_ZARR       land-sea-mask zarr for coastline overlays
       AIC_NEXTGEMS_ROOT    NextGEMS source dir
       AIC_ERA5_INPUTS_ROOT staged ERA5 rollout inputs/predictions root

   The named subdirs (``ERA5_HEATWAVE``, ``HEATWAVE_CLIM``, ...) derive from
   ``DATA_ROOT``; ``era5_inputs_dir()`` / ``nextgems_dir()`` build the per-year paths.

2. **Typed env accessors.** ``env_str`` / ``env_int`` / ``env_float`` / ``env_bool``
   / ``env_list`` / ``env_required`` replace the scattered, inconsistent
   ``int(os.environ.get(...))`` idioms with one validated path.

Importing this module has NO side effects (no network, no filesystem, nothing that
raises), so it is safe to import anywhere -- including tests and the GIF scripts,
unlike ``eval_common`` which resolves a run at import time.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# typed env accessors
# --------------------------------------------------------------------------- #
def env_str(key: str, default: str | None = None) -> str | None:
    v = os.environ.get(key)
    return v if v is not None and v != "" else default


def env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    return int(v) if v not in (None, "") else default


def env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    return float(v) if v not in (None, "") else default


def env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key)
    if v in (None, ""):
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def env_list(key: str, default=()) -> list[str]:
    """Comma- or whitespace-separated list; empty/unset -> list(default)."""
    v = os.environ.get(key)
    if v in (None, ""):
        return list(default)
    return v.replace(",", " ").split()


def env_required(key: str) -> str:
    """A required env var (no default); a clear error if it is unset/empty."""
    v = os.environ.get(key)
    if not v:
        raise SystemExit(f"{key} is required (set it in the environment)")
    return v


# --------------------------------------------------------------------------- #
# cluster paths (overridable; defaults = BW UniCluster workspace layout)
# --------------------------------------------------------------------------- #
WORKSPACE = env_str("AIC_WORKSPACE", "/pfs/work9/workspace/scratch")
DATA_ROOT = env_str("AIC_DATA_ROOT", f"{WORKSPACE}/ka_dm9435-ai-climate")

# this project's data subdirs (under DATA_ROOT) -- the previously-duplicated literals
ERA5_HEATWAVE       = f"{DATA_ROOT}/era5_heatwave"          # 6-hourly T850 etc.
ERA5_HEATWAVE_DAILY = f"{DATA_ROOT}/era5_heatwave_daily"    # daily tmin/tmax/t00
HEATWAVE_CLIM       = f"{DATA_ROOT}/heatwave_clim"          # 2.8deg regridded stats cache
HEATWAVE_FIGURES    = f"{DATA_ROOT}/heatwave_figures"
HEATWAVE_GIFS       = f"{DATA_ROOT}/heatwave_gifs"
HEATWAVE_CATALOG    = f"{DATA_ROOT}/heatwave_catalog"
NEXTGEMS_2049_PRED  = f"{DATA_ROOT}/nextgems_2049/predictions"

# colleagues' shared inputs (NOT under DATA_ROOT), each overridable
# land-sea mask (grid-independent backdrop for coastline overlays)
COAST_ZARR    = env_str("AIC_COAST_ZARR", f"{WORKSPACE}/ka_je2428-nextgems_2049/constant_fields.zarr")
NEXTGEMS_ROOT = env_str("AIC_NEXTGEMS_ROOT", f"{WORKSPACE}/ka_je2428-nextgems_2049")  # NextGEMS source
ERA5_INPUTS_ROOT = env_str("AIC_ERA5_INPUTS_ROOT", f"{WORKSPACE}/ka_hc5935-ai-climate")  # era5_<year>/{inputs,predictions}


def nextgems_dir(year: int | str) -> str:
    """NextGEMS source dir for a year (``ka_je2428-nextgems_<year>``); 2049 -> NEXTGEMS_ROOT."""
    return NEXTGEMS_ROOT if str(year) == "2049" else f"{WORKSPACE}/ka_je2428-nextgems_{year}"


def era5_inputs_dir(year: int | str, kind: str = "inputs") -> str:
    """Staged ERA5 rollout dir: ``ka_hc5935-ai-climate/era5_<year>/<kind>`` (inputs|predictions)."""
    return f"{ERA5_INPUTS_ROOT}/era5_{year}/{kind}"
