#!/usr/bin/env python
"""Heat-wave (grid cell, day) masks + episodes on the NeuralGCM 2.8deg grid.

The bridge between the definition-driven heat-wave DETECTOR
(``controller/heatwave``) and the rollout EVAL layer (``controller/eval``): it
turns a heat-wave definition into a boolean ``(time, lat, lon)`` mask on the model
grid, and splits it into temporally connected EPISODES with a spatial FOOTPRINT
each -- the "set of grid points" the case-study plots analyse.

It reads the cheap pre-regridded daily-stats caches
(``heatwave_clim/<tag>_daily_<year>_2p8deg.nc``) directly, so there is NO GCS /
model load: only the same ``hot_mask`` -> ``active_mask`` chain ``compare.py``
uses. Everything is returned as xarray objects on the model grid, so the mask
aligns by COORDINATES with the rollout predictions and truth caches (all three are
the identical 128x64 / 2.8deg grid) with no regridding.

The active definition + percentile come from ``controller/heatwave/definitions``
(the ``HW_PCT`` env var, read at import): set ``HW_PCT=0.99`` and pass
``definitions.MIXTURE`` for the mixture-at-p99 case study.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from aic.controller.heatwave import detect as DET
from aic.regions import region_mask

CLIM_DIR = Path(os.environ.get(
    "HW_CACHE_DIR", "/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/heatwave_clim"))
DEFAULT_WINDOW = int(os.environ.get("HW_WINDOW", "5"))   # +/- day-of-year window
DEFAULT_COVER = float(os.environ.get("HW_CS_COVER", "0.02"))  # region area fraction
DEFAULT_GAP = int(os.environ.get("HW_EPISODE_GAP", "2"))      # merge gap (days)
DEFAULT_MIN_DAYS = int(os.environ.get("HW_CS_MIN_DAYS", "3"))  # min episode length


def _load_stats(tag: str, year: int) -> xr.Dataset:
    """Regridded daily-stats Dataset for one year from the 2.8deg cache."""
    f = CLIM_DIR / f"{tag}_daily_{year}_2p8deg.nc"
    if not f.exists():
        raise SystemExit(f"[casestudy] missing regridded daily-stats cache: {f}")
    return xr.open_dataset(f)


def active_mask_da(defn, year: int, window: int = DEFAULT_WINDOW,
                   region: str = "europe") -> xr.DataArray:
    """Boolean ``(time, latitude, longitude)`` mask on the model grid: True where a
    cell is inside a >=3-day heat-wave spell under ``defn`` in ``year``, restricted
    to ``region`` (cells outside the region box are False). Coordinates match the
    NeuralGCM 2.8deg grid, so it aligns with the rollouts/truth by coordinates."""
    tag, stats = defn.tag, list(defn.stats)
    refs = [_load_stats(tag, y) for y in defn.ref_range]   # per-definition ref period
    ref = xr.concat(refs, dim="time")
    tgt = _load_stats(tag, year)
    lat = tgt["latitude"].values
    lon = tgt["longitude"].values

    ref_stats = {s: ref[f"{tag}_{s}"].values for s in stats}
    tgt_stats = {s: tgt[f"{tag}_{s}"].values for s in stats}
    ref_t = pd.to_datetime(ref["time"].values)
    ref_doy = ref_t.dayofyear.values
    tgt_doy = pd.to_datetime(tgt["time"].values).dayofyear.values

    hot = DET.hot_mask(defn, ref_stats, ref_doy, tgt_stats, tgt_doy, window,
                       ref_months=ref_t.month.values)
    active = DET.active_mask(hot, DET.MIN_DUR)           # (T, Y, X) bool

    latm, lonm = region_mask(lat, lon, region)
    reg2d = latm[:, None] & lonm[None, :]
    active = active & reg2d[None, :, :]

    da = xr.DataArray(
        active, dims=("time", "latitude", "longitude"),
        coords={"time": tgt["time"].values, "latitude": lat, "longitude": lon},
        name="hw_active",
        attrs=dict(definition=defn.name, pct=float(defn.pct), window=int(window),
                   region=region, year=int(year)))
    ref.close(); tgt.close()
    for d in refs:
        d.close()
    return da


def region_coverage(active_da: xr.DataArray, region: str = "europe") -> np.ndarray:
    """cos(lat)-weighted fraction of the region's cells that are active, per day."""
    lat = active_da["latitude"].values
    lon = active_da["longitude"].values
    latm, lonm = region_mask(lat, lon, region)
    w = np.cos(np.deg2rad(lat))
    aw = (np.where(latm, w, 0.0)[:, None] * lonm[None, :].astype(float))  # (Y, X)
    denom = aw.sum()
    num = (active_da.values * aw[None, :, :]).sum(axis=(1, 2))
    return num / denom


@dataclass
class Episode:
    """One temporally connected heat-wave episode (mixture/p99 etc.)."""
    idx: int                       # 1-based episode number within the year
    start: pd.Timestamp            # first day of the episode span
    end: pd.Timestamp              # last day of the episode span
    n_days: int
    peak_date: pd.Timestamp        # day of maximum region coverage
    peak_cover: float              # max cos-lat area fraction of the region
    span: tuple = field(repr=False)          # (i0, i1) inclusive time-index span
    footprint: xr.DataArray = field(repr=False)  # 2-D bool (lat,lon): union of cells
    n_cells: int = 0               # number of cells in the footprint

    @property
    def tag(self) -> str:
        return f"ep{self.idx:02d}"

    @property
    def label(self) -> str:
        return f"{self.start:%Y-%m-%d}..{self.end:%Y-%m-%d}"


def episodes(active_da: xr.DataArray, region: str = "europe",
             cover_thr: float = DEFAULT_COVER, gap: int = DEFAULT_GAP,
             min_days: int = DEFAULT_MIN_DAYS) -> list[Episode]:
    """Split the active mask into episodes: runs of days whose region coverage
    exceeds ``cover_thr``, merged across gaps of <= ``gap`` days, kept if the span
    is >= ``min_days``. Each episode's FOOTPRINT is the union of cells active on any
    day of its (gap-filled, contiguous) span."""
    cov = region_coverage(active_da, region)
    times = pd.to_datetime(active_da["time"].values)
    qual = np.where(cov >= cover_thr)[0]
    if qual.size == 0:
        return []
    groups = [[int(qual[0])]]
    for d in qual[1:]:
        (groups[-1].append(int(d)) if d - groups[-1][-1] <= gap
         else groups.append([int(d)]))
    eps: list[Episode] = []
    for g in groups:
        i0, i1 = g[0], g[-1]
        if (i1 - i0 + 1) < min_days:
            continue
        sl = slice(i0, i1 + 1)
        fp = active_da.isel(time=sl).any("time")         # 2-D bool union
        cwin = cov[i0:i1 + 1]
        pk = i0 + int(np.argmax(cwin))
        eps.append(Episode(
            idx=len(eps) + 1, start=times[i0], end=times[i1], n_days=i1 - i0 + 1,
            peak_date=times[pk], peak_cover=float(cov[pk]), span=(i0, i1),
            footprint=fp, n_cells=int(fp.values.sum())))
    return eps


def episodes_dataframe(eps: list[Episode]) -> pd.DataFrame:
    return pd.DataFrame([dict(
        episode=e.idx, start=e.start.date(), end=e.end.date(), n_days=e.n_days,
        peak_date=e.peak_date.date(), peak_cover=round(e.peak_cover, 4),
        n_cells=e.n_cells) for e in eps])


def _main() -> None:
    """Enumerate episodes for a definition/percentile/year/region (text only)."""
    from aic.controller.heatwave import definitions as D

    defn_name = os.environ.get("HW_CS_DEF", "mixture")
    defn = D.BY_NAME[defn_name]
    year = int(os.environ.get("HW_YEAR", "2023"))
    region = os.environ.get("HW_SPEC_REGION", "europe").strip().lower()
    window = DEFAULT_WINDOW
    print(f"[casestudy] def={defn.name} pct={defn.pct} ({D.PTAG}) window=+/-{window}d "
          f"region={region} year={year}", flush=True)
    da = active_mask_da(defn, year, window, region)
    cov = region_coverage(da, region)
    print(f"[casestudy] {da.sizes['time']} days; peak region coverage "
          f"{cov.max():.3f}; days with any active cell: {int((cov > 0).sum())}",
          flush=True)
    eps = episodes(da, region)
    df = episodes_dataframe(eps)
    print(f"[casestudy] {len(eps)} episode(s) "
          f"(cover>={DEFAULT_COVER}, gap<={DEFAULT_GAP}, min_days>={DEFAULT_MIN_DAYS}):",
          flush=True)
    print(df.to_string(index=False) if len(eps) else "  (none)", flush=True)


if __name__ == "__main__":
    _main()
