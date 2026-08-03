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
# cells farther apart than this Manhattan distance (in grid cells) are NOT the same
# heat wave -- unless a later day bridges them within this reach (spatio-temporal
# connectivity). 0 falls back to the old time-only merging.
DEFAULT_MANHATTAN = int(os.environ.get("HW_MANHATTAN", "4"))


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
    tgt_t = pd.to_datetime(tgt["time"].values)
    ref_doy = ref_t.dayofyear.values
    tgt_doy = tgt_t.dayofyear.values

    hot = DET.hot_mask(defn, ref_stats, ref_doy, tgt_stats, tgt_doy, window,
                       ref_months=ref_t.month.values, tgt_months=tgt_t.month.values,
                       lat=lat)
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
    """One spatio-temporally connected heat-wave event (a single physical heat wave:
    cells linked in space (Manhattan <= max) and time (gap), possibly merging later)."""
    idx: int                       # 1-based episode number within the year
    start: pd.Timestamp            # first day of the event
    end: pd.Timestamp              # last day of the event
    n_days: int                    # number of distinct active days
    peak_date: pd.Timestamp        # day of maximum coverage by THIS event's cells
    peak_cover: float              # max cos-lat area fraction of the region
    span: tuple = field(repr=False)          # (i0, i1) inclusive time-index span
    footprint: xr.DataArray = field(repr=False)  # 2-D bool (lat,lon): union of cells
    n_cells: int = 0               # number of cells in the footprint
    mask: object = field(default=None, repr=False)  # (T,Y,X) bool: this event's cells

    @property
    def tag(self) -> str:
        return f"ep{self.idx:02d}"

    @property
    def label(self) -> str:
        return f"{self.start:%Y-%m-%d}..{self.end:%Y-%m-%d}"


def _st_components(active, gap, max_manhattan):
    """Spatio-temporal connected components of the active cells via union-find: two
    active cells join if within `max_manhattan` grid cells (Manhattan) AND within
    `gap` days. A later bridging day merges earlier-separate blobs ("connected
    later"). Returns (pts[N,3] (t,y,x), list of member-index lists)."""
    pts = np.argwhere(active)
    n = len(pts)
    if n == 0:
        return pts, []
    parent = list(range(n))

    def find(a):
        r = a
        while parent[r] != r:
            r = parent[r]
        while parent[a] != r:
            parent[a], a = r, parent[a]
        return r

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    T = active.shape[0]
    by_day = [dict() for _ in range(T)]
    for i, (t, y, x) in enumerate(pts):
        by_day[t][(int(y), int(x))] = i
    m = max_manhattan
    offs = [(dy, dx) for dy in range(-m, m + 1)
            for dx in range(-(m - abs(dy)), (m - abs(dy)) + 1)]
    for i, (t, y, x) in enumerate(pts):
        t, y, x = int(t), int(y), int(x)
        for dt in range(0, gap + 1):
            d = t + dt
            if d >= T:
                break
            bd = by_day[d]
            if not bd:
                continue
            for dy, dx in offs:
                j = bd.get((y + dy, x + dx))
                if j is not None:
                    union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return pts, list(groups.values())


def episodes(active_da: xr.DataArray, region: str = "europe",
             cover_thr: float = DEFAULT_COVER, gap: int = DEFAULT_GAP,
             min_days: int = DEFAULT_MIN_DAYS,
             max_manhattan: int = DEFAULT_MANHATTAN) -> list[Episode]:
    """Split the active mask into distinct heat waves by SPATIO-TEMPORAL connectivity:
    active cells belong to the same event when they are within `max_manhattan` grid
    cells (Manhattan) and within `gap` days -- so spatially separated heat waves are
    kept apart unless a later day bridges them. An event is kept if it spans
    >= `min_days` distinct days and reaches >= `cover_thr` of the region at its peak.
    Each event carries its 2-D FOOTPRINT (union of its cells) and its (T,Y,X) mask."""
    active = active_da.values
    T, Y, X = active.shape
    times = pd.to_datetime(active_da["time"].values)
    lat = active_da["latitude"].values
    lon = active_da["longitude"].values
    latm, lonm = region_mask(lat, lon, region)
    aw = np.where(latm, np.cos(np.deg2rad(lat)), 0.0)[:, None] * lonm[None, :].astype(float)
    region_area = aw.sum()

    pts, comps = _st_components(active, gap, max_manhattan)
    raw = []
    for members in comps:
        P = pts[members]                                  # (M, 3): t, y, x
        days = P[:, 0]
        if len(np.unique(days)) < min_days:
            continue
        cm = np.zeros((T, Y, X), bool)
        cm[P[:, 0], P[:, 1], P[:, 2]] = True
        cov = (cm * aw[None]).sum(axis=(1, 2)) / region_area
        pk = int(np.argmax(cov))
        if float(cov[pk]) < cover_thr:
            continue
        fp2d = cm.any(0)
        raw.append((int(days.min()), int(days.max()), int(len(np.unique(days))), pk,
                    float(cov[pk]), fp2d, cm))
    raw.sort(key=lambda r: (r[0], -r[4]))
    eps: list[Episode] = []
    for k, (i0, i1, ndays, pk, pcov, fp2d, cm) in enumerate(raw):
        fp = xr.DataArray(fp2d, dims=("latitude", "longitude"),
                          coords={"latitude": lat, "longitude": lon})
        eps.append(Episode(
            idx=k + 1, start=times[i0], end=times[i1], n_days=ndays,
            peak_date=times[pk], peak_cover=pcov, span=(i0, i1),
            footprint=fp, n_cells=int(fp2d.sum()), mask=cm))
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
