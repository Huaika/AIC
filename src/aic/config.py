#!/usr/bin/env python
"""Central configuration for the aic package: cluster paths + typed env accessors.

TWO jobs, both aimed at the config sprawl (~130 env vars read across the code) and
the hardcoded ``/pfs/...`` paths:

1. **Cluster paths in one place.** Every absolute workspace path defaults to the BW
   UniCluster layout but is overridable via an env var, so a teammate on a different
   cluster points the code elsewhere without editing source::

       AIC_WORKSPACE   base scratch workspace  (default /pfs/work9/workspace/scratch)
       AIC_DATA_ROOT   this project's data root (default <WS>/ka_dm9435-ai-climate)
       AIC_COAST_ZARR  land-sea-mask zarr for the coastline overlay

2. **Typed env accessors.** ``env_int`` / ``env_float`` / ``env_bool`` / ``env_list``
   replace the scattered, inconsistent ``int(os.environ.get(...))`` idioms with one
   validated path, so the knobs are read the same way everywhere.

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


# --------------------------------------------------------------------------- #
# cluster paths (overridable; defaults = BW UniCluster workspace layout)
# --------------------------------------------------------------------------- #
WORKSPACE = env_str("AIC_WORKSPACE", "/pfs/work9/workspace/scratch")
DATA_ROOT = env_str("AIC_DATA_ROOT", f"{WORKSPACE}/ka_dm9435-ai-climate")
# land-sea mask (a colleague's NextGEMS constant-fields zarr; grid-independent backdrop)
COAST_ZARR = env_str("AIC_COAST_ZARR",
                     f"{WORKSPACE}/ka_je2428-nextgems_2049/constant_fields.zarr")
