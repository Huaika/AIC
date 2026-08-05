#!/usr/bin/env python
"""Multi-source abstraction for the unified plotters (MVC View).

A ``Source`` is one thing a plot can draw: a *forecast* model's rollouts
(NeuralGCM or GraphCast) scored against a shared *reference* (ERA5 / NextGEMS
truth). The plotters take a LIST of sources and either draw one alone (the
historical single-RUN behaviour) or overlay several at once
(``EVAL_SOURCES=neuralgcm,graphcast`` + ``EVAL_YEAR=2023``):

    spaghetti / drift : one coloured curve (or line-bundle) per forecast source,
                        on shared axes, over a single reference line.
    drift maps        : one field panel per forecast source + one shared
                        reference panel + one drift panel per source.

This layer is ADDITIVE: it reuses ``eval_common`` (the single-RUN engine +
stateless helpers) untouched and reads only artifacts it already built (the
per-run prediction files and the ``truth_modelgrid_<short>_<run>.nc`` caches).
It never rebuilds a truth cache -- if one is missing it errors, exactly as the
EVAL_REQUIRE_CACHE guard does.

Selection:
  EVAL_SOURCES  comma/space list of forecast MODELS (neuralgcm, graphcast).
                Unset -> single source resolved from EVAL_RUN (backward compat).
  EVAL_YEAR     the year to compare (required when EVAL_SOURCES is set).
  EVAL_DATASET  forcing/truth dataset (default 'era5'; 'nextgems' for the 2049 run).

Each (model, dataset, year) maps to exactly one RUN key in eval_common.RUNS.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import xarray as xr

from aic.controller.eval import eval_common as C
from aic.controller.eval import gridpoints as GP

# Pretty model names + a stable, colour-blind-friendly palette (fallback for a 3rd+
# model). The reference line is always drawn black.
MODEL_PRETTY = {"neuralgcm": "NeuralGCM", "graphcast": "GraphCast"}
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

# SINGLE SOURCE OF TRUTH for model line colours, shared by every view script
# (spaghetti / drift / drift maps / case study) so a model is the same colour
# everywhere. The reference (ERA5 / NextGEMS) line is always REF_COLOR.
MODEL_COLORS = {"neuralgcm": "#1f77b4", "graphcast": "#d62728"}
REF_COLOR = "black"
# line styles to tell apart YEARS / rollout-windows of the SAME model on one axis
# (the multi-year OOD spaghetti): solid, dashed, dotted, dash-dot, ...
YEAR_LINESTYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]

FINAL_DAY_LEAD_MIN = 216  # >= 216 h == the day-10 slab (drift maps)


def _model_of(run: str) -> str:
    return "graphcast" if run.startswith("graphcast") else "neuralgcm"


def model_color(model: str) -> str:
    """The shared colour for a model (fallback to the palette for a 3rd+ model)."""
    return MODEL_COLORS.get(model, PALETTE[0])


def year_linestyle(index: int):
    """A line style distinguishing the ``index``-th year/window of one model."""
    return YEAR_LINESTYLES[index % len(YEAR_LINESTYLES)]


def run_for(model: str, dataset: str, year: int) -> str | None:
    """The RUN key whose (model, truth_kind, year) matches, or None."""
    for run, cfg in C.RUNS.items():
        if (_model_of(run) == model and cfg["truth_kind"] == dataset
                and int(cfg["year"]) == int(year)):
            return run
    return None


@dataclass
class Source:
    """One forecast model's rollouts for a (dataset, year), plus its own-grid
    reference. Wraps a single eval_common RUN; all cache paths match the
    single-RUN layout so existing caches are reused verbatim."""
    run: str
    model: str
    dataset: str
    year: int
    ref_label: str
    pred_dir: Path
    outdir: Path
    color: str
    kind: str = "forecast"
    _levels: list | None = field(default=None, repr=False)
    _grid: tuple | None = field(default=None, repr=False)
    _mgrid: tuple | None = field(default=None, repr=False)
    _truth: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_run(cls, run: str, color: str | None = None) -> "Source":
        if run not in C.RUNS:
            raise SystemExit(f"unknown run {run!r}; choose from {list(C.RUNS)}")
        cfg = C.RUNS[run]
        return cls(
            run=run, model=_model_of(run), dataset=cfg["truth_kind"],
            year=int(cfg["year"]), ref_label=cfg["ref_label"],
            pred_dir=Path(cfg["pred_dir"]),
            outdir=C.RESULTS_ROOT / f"results_eval_{run}",
            color=color if color is not None else model_color(_model_of(run)))

    @property
    def pretty(self) -> str:
        return MODEL_PRETTY.get(self.model, self.model)

    # -- prediction grid / levels (from this source's own pred files) -------- #
    def pred_files(self) -> list[Path]:
        fs = sorted(self.pred_dir.glob(f"pred_{self.year}_*.nc"))
        if not fs:
            raise SystemExit(f"[source {self.run}] no pred files in {self.pred_dir}")
        return fs

    def prediction_levels(self) -> list[int]:
        if self._levels is None:
            with xr.open_dataset(self.pred_files()[0]) as ds:
                self._levels = [int(x) for x in ds.level.values]
        return self._levels

    def native_grid(self):
        """The grid the prediction files are actually stored on (2.8deg for
        NeuralGCM, 0.25deg for GraphCast)."""
        if self._grid is None:
            with xr.open_dataset(self.pred_files()[0]) as ds:
                self._grid = (ds.latitude.values.copy(), ds.longitude.values.copy())
        return self._grid

    def model_grid(self):
        """The common 2.8deg NeuralGCM grid everything is scored on (read from this
        run's 2.8deg truth cache -- identical to the heat-wave / NeuralGCM grid)."""
        if self._mgrid is None:
            with xr.open_dataset(self._truth_nc("temperature")) as ds:
                self._mgrid = (ds.latitude.values.copy(), ds.longitude.values.copy())
        return self._mgrid

    @property
    def needs_regrid(self) -> bool:
        """GraphCast rollouts (0.25deg) are interpolated onto the 2.8deg model grid;
        NeuralGCM rollouts are already on it."""
        return self.model == "graphcast"

    def prediction_grid(self):
        """The grid used for ALL downstream analysis: the common 2.8deg model grid
        (so GraphCast and NeuralGCM, their truth and the heat-wave footprint are on
        one grid). Equals the native grid for NeuralGCM."""
        return self.model_grid() if self.needs_regrid else self.native_grid()

    def regrid_field(self, da: xr.DataArray) -> xr.DataArray:
        """Bilinearly interpolate a ``(..., latitude, longitude)`` field onto the
        2.8deg model grid when this source is on a finer native grid (GraphCast
        0.25deg); a no-op for NeuralGCM. Bilinear = linear in latitude and longitude,
        following WeatherBench (Rasp et al. 2020), which regrids fields to the coarse
        (~2.8deg) evaluation grid by bilinear interpolation (WeatherBench-2 likewise
        regrids all forecasts + truth to one common grid before scoring). Idempotent:
        a field already on the model grid (e.g. from the 2.8deg prediction cache) is
        returned unchanged."""
        if not self.needs_regrid:
            return da
        tlat, tlon = self.model_grid()
        if da.sizes.get("latitude") == len(tlat) and da.sizes.get("longitude") == len(tlon):
            return da
        da = da.sortby("latitude").sortby("longitude")   # interp needs ascending
        return da.interp(latitude=tlat, longitude=tlon, method="linear")

    @property
    def pred_cache_dir(self) -> Path:
        """Where 2.8deg-regridded copies of GraphCast predictions are cached, so the
        0.25deg files are regridded ONCE instead of on every read."""
        return self.pred_dir.parent / f"{self.pred_dir.name}_2p8deg"

    def open_pred(self, f) -> xr.Dataset:
        """Open a prediction file. For GraphCast, return the cached 2.8deg-regridded
        copy when it exists (a tiny 128x64 read); otherwise the native 0.25deg file
        (regrid_field then regrids downstream). For NeuralGCM, just the native file."""
        if self.needs_regrid:
            cache = self.pred_cache_dir / Path(f).name
            if cache.exists():
                return xr.open_dataset(cache)
        return xr.open_dataset(f)

    def build_pred_cache(self, variables: list[str], overwrite: bool = False,
                         shard: tuple[int, int] = (0, 1)) -> int:
        """Regrid every prediction file's 3-D fields onto the 2.8deg model grid and
        cache them (GraphCast only). Run once; afterwards open_pred() reads the cache.
        ``shard=(k, n)`` processes only files with ``index % n == k`` (stride sharding
        for parallel array tasks). Skips already-cached files, so it resumes.
        Returns the number of files written."""
        if not self.needs_regrid:
            return 0
        k, n = shard
        tlat, tlon = self.model_grid()
        self.pred_cache_dir.mkdir(parents=True, exist_ok=True)
        want = [v for v in variables]
        written = 0
        files = [f for i, f in enumerate(self.pred_files()) if i % n == k]
        for i, f in enumerate(files):
            cache = self.pred_cache_dir / f.name
            if cache.exists() and not overwrite:
                continue
            ds = xr.open_dataset(f)
            keep = [v for v in want if v in ds.data_vars]
            sub = (ds[keep].sortby("latitude").sortby("longitude")
                   .interp(latitude=tlat, longitude=tlon, method="linear"))
            for c in ("lead_hours", "valid_time", "time", "level"):
                if c in ds.variables and c not in sub.coords and c not in sub:
                    sub[c] = ds[c]
            sub.attrs.update(ds.attrs)
            enc = {v: {"dtype": "float32", "zlib": True, "complevel": 4} for v in keep}
            tmp = cache.with_suffix(".tmp.nc")
            sub.to_netcdf(tmp, encoding=enc)
            tmp.rename(cache)
            ds.close()
            written += 1
            if i % 25 == 0 or i == len(files) - 1:
                print(f"[predcache {self.run}] {i + 1}/{len(files)}", flush=True)
        return written

    # -- reference (truth) on THIS source's grid ----------------------------- #
    def _truth_nc(self, var: str) -> Path:
        return self.outdir / f"truth_modelgrid_{C.VARIABLES[var]['short']}_{self.run}.nc"

    def truth_at_levels(self, var: str, levels: list[int]) -> xr.DataArray:
        key = (var, tuple(levels))
        if key in self._truth:
            return self._truth[key]
        nc = self._truth_nc(var)
        if not nc.exists():
            raise SystemExit(
                f"[source {self.run}] truth cache missing for {var}: {nc}\n"
                f"  build it via the truth-chunk + finalize jobs (EVAL_RUN={self.run}).")
        native = xr.open_dataarray(nc)
        out = native.interp(level=list(levels), method="linear",
                            kwargs={"fill_value": "extrapolate"}).load()
        native.close()
        clat, clon = self.prediction_grid()
        # prediction_grid is the 2.8deg model grid for every source (GraphCast is
        # regridded to it via regrid_field), and the truth cache is already on that
        # grid, so this reindex is an exact alignment.
        out = out.reindex(latitude=clat, longitude=clon, method="nearest")
        self._truth[key] = out
        return out

    # -- grid-point-set selection ------------------------------------------- #
    def as_points(self, regions) -> list[GP.GridPoints]:
        """Normalise a mixed list of region-name strings and/or GridPoints into
        GridPoints on THIS source's prediction grid. Region names become the cells
        inside their box (the global/regional/OOD path); GridPoints (e.g. a
        heat-wave footprint) pass through unchanged."""
        lat, lon = self.prediction_grid()
        out = []
        for r in regions:
            out.append(r if isinstance(r, GP.GridPoints)
                       else GP.GridPoints.from_region(lat, lon, r))
        return out

    # -- per-source data builders (cache paths identical to the single-RUN ones) #
    def rollout_gmean(self, var, short, levels, regions, files=None,
                      cache=True) -> pd.DataFrame:
        """Area-mean rollout time series per (region, level, init), cached CSV.

        ``regions`` may be region names and/or GridPoints (arbitrary footprints);
        the row ``region`` column carries each selector's ``key``. ``files``
        restricts the rollouts (e.g. a case-study episode window); when given, or
        ``cache=False``, the CSV cache is bypassed (ad-hoc footprints are not
        persisted)."""
        points = self.as_points(regions)
        keys = [p.key for p in points]
        csv = self.outdir / f"{self.run}_rollout_gmean_{short}_{C.level_tag()}.csv"
        use_cache = cache and files is None
        if use_cache and csv.exists():
            df = pd.read_csv(csv, parse_dates=["init_date"])
            if "region" in df.columns and set(keys) <= set(df["region"].unique()):
                print(f"[{self.run}] rollout cached {csv}")
                return df
            print(f"[{self.run}] rollout cache missing regions -> recompute")
        files = files if files is not None else self.pred_files()
        print(f"[{self.run}] area-mean {var} @ {len(levels)} lev, "
              f"{len(keys)} selector(s), {len(files)} files")
        rows = []
        for i, f in enumerate(files):
            ds = self.open_pred(f)
            init = pd.to_datetime(ds.attrs.get(
                "init_date", f.stem.replace(f"pred_{self.year}_", "")))
            da = self.regrid_field(ds[var].sel(level=levels)).compute()
            lead_h = ds["lead_hours"].values.astype(int)
            for gp in points:
                gm = GP.masked_area_mean(da, gp)
                for lev in levels:
                    rows.append(pd.DataFrame({
                        "init_date": init, "lead_hours": lead_h, "level": lev,
                        "region": gp.key, "pred_gmean": gm.sel(level=lev).values}))
            ds.close()
            if i % 50 == 0 or i == len(files) - 1:
                print(f"  {i + 1}/{len(files)}")
        df = pd.concat(rows, ignore_index=True)
        if use_cache:
            df.to_csv(csv, index=False)
            print(f"[{self.run}] wrote {csv}")
        return df

    def drift_per_init(self, var, short, levels, regions, files=None,
                       cache=True) -> pd.DataFrame:
        """Per-(region, level, init, lead) mse + bias vs this source's truth.

        ``regions`` may be region names and/or GridPoints; ``files`` restricts the
        rollouts (case-study episode window) and, like ``cache=False``, bypasses the
        CSV cache."""
        import numpy as np
        points = self.as_points(regions)
        keys = [p.key for p in points]
        csv = self.outdir / f"{self.run}_drift_per_init_{short}_{C.level_tag()}.csv"
        use_cache = cache and files is None
        if use_cache and csv.exists():
            df = pd.read_csv(csv, parse_dates=["init_date"])
            if "region" in df.columns and set(keys) <= set(df["region"].unique()):
                print(f"[{self.run}] drift cached {csv}")
                return df
            print(f"[{self.run}] drift cache missing regions -> recompute")
        truth = self.truth_at_levels(var, levels)
        files = files if files is not None else self.pred_files()
        print(f"[{self.run}] scoring {len(files)} rollouts vs {self.ref_label} "
              f"({var}) @ {len(levels)} lev, {len(keys)} selector(s)")
        rows = []
        for i, f in enumerate(files):
            ds = self.open_pred(f)
            init = pd.to_datetime(ds.attrs.get(
                "init_date", f.stem.replace(f"pred_{self.year}_", "")))
            pred = self.regrid_field(ds[var].sel(level=levels))
            tru = truth.sel(time=ds["valid_time"].values, method="nearest")
            tru = tru.assign_coords(time=pred["time"].values)
            diff = (pred - tru).compute()
            lead_h = ds["lead_hours"].values.astype(int)
            for gp in points:
                mse = GP.masked_area_mean(diff ** 2, gp)
                bias = GP.masked_area_mean(diff, gp)
                for lev in levels:
                    rows.append(pd.DataFrame({
                        "init_date": init, "lead_hours": lead_h, "level": lev,
                        "region": gp.key,
                        "mse": np.asarray(mse.sel(level=lev).values, float),
                        "bias": np.asarray(bias.sel(level=lev).values, float)}))
            ds.close()
            if i % 50 == 0 or i == len(files) - 1:
                print(f"  {i + 1}/{len(files)}")
        df = pd.concat(rows, ignore_index=True)
        if use_cache:
            df.to_csv(csv, index=False)
            print(f"[{self.run}] wrote {csv}")
        return df

    def day10_fields(self, var, short, levels, period) -> xr.Dataset | None:
        """Day-10 clim / reference clim / drift fields for a period, cached NC.
        None if the period has no init-days."""
        truth = self.truth_at_levels(var, levels)
        suffix = "" if period == 0 else f"_{period:02d}"
        nc = self.outdir / f"{self.run}_drift_maps_{short}{suffix}_{C.level_tag()}.nc"
        if nc.exists():
            print(f"[{self.run}] maps cached {nc}")
            ds = xr.open_dataset(nc)
            # pre-multi-source caches used ngcm_day10_clim / ref_annual_clim
            rn = {old: new for old, new in
                  {"ngcm_day10_clim": "fc_day10_clim",
                   "ref_annual_clim": "ref_clim"}.items() if old in ds}
            return ds.rename(rn) if rn else ds
        files = self.pred_files()
        if period == 0:
            ref_clim = truth.mean("time")
        else:
            files = [f for f in files if C.pred_init_month(f) == period]
            mask = (truth["time"].dt.month == period).values
            ref_clim = truth.isel(time=mask).mean("time")
        if not files:
            print(f"[{self.run}] no init-days for {C.period_dir_name(period)}; skip")
            return None
        ref_clim = ref_clim.transpose("level", "latitude", "longitude")
        print(f"[{self.run}] {C.period_dir_name(period)}: end-of-forecast mean over "
              f"{len(files)} rollouts ({var}), {len(levels)} lev")
        acc, n = None, 0
        for i, f in enumerate(files):
            ds = self.open_pred(f)
            t = self.regrid_field(ds[var].sel(level=levels))
            end = t.where(ds["lead_hours"] >= FINAL_DAY_LEAD_MIN, drop=True).mean("time")
            end = end.transpose("level", "latitude", "longitude")
            acc = end if acc is None else acc + end
            n += 1
            ds.close()
            if i % 50 == 0 or i == len(files) - 1:
                print(f"  {i + 1}/{len(files)}")
        fc = acc / n
        out = xr.Dataset({"fc_day10_clim": fc, "ref_clim": ref_clim,
                          "drift": fc - ref_clim})
        out.attrs.update(n_inits=n, final_day_lead_min_h=FINAL_DAY_LEAD_MIN,
                         variable=var, period=C.period_dir_name(period), run=self.run,
                         drift_def=f"mean(end-of-10day-forecast) - {self.ref_label} mean")
        out.to_netcdf(nc)
        print(f"[{self.run}] wrote {nc}")
        return out


# --------------------------------------------------------------------------- #
# Resolving the source list + shared naming/path context
# --------------------------------------------------------------------------- #
def resolve_sources() -> list[Source]:
    """EVAL_SOURCES (models) + EVAL_YEAR (+ EVAL_DATASET) -> [Source]; or, if
    EVAL_SOURCES is unset, a single Source from EVAL_RUN (backward compatible)."""
    # EVAL_RUNS: explicit run-key list (any model/dataset/year mix on one plot, e.g.
    # the multi-year OOD spaghetti). Colour by model; years are told apart downstream.
    env_runs = os.environ.get("EVAL_RUNS", "").strip()
    if env_runs:
        runs = [r.strip() for r in env_runs.replace(",", " ").split() if r.strip()]
        srcs = [Source.from_run(r, color=model_color(_model_of(r))) for r in runs]
        print(f"[sources] {len(srcs)} explicit run(s): "
              + ", ".join(f"{s.pretty}({s.run})" for s in srcs))
        return srcs
    env = os.environ.get("EVAL_SOURCES", "").strip()
    if not env:
        return [Source.from_run(C.RUN, color=PALETTE[0])]
    models = [m.strip() for m in env.replace(",", " ").split() if m.strip()]
    dataset = os.environ.get("EVAL_DATASET", "era5").strip()
    yr = os.environ.get("EVAL_YEAR", "").strip()
    if not yr:
        raise SystemExit("EVAL_SOURCES set -> EVAL_YEAR is required")
    year = int(yr)
    srcs = []
    for i, m in enumerate(models):
        run = run_for(m, dataset, year)
        if run is None:
            raise SystemExit(
                f"no run for model={m!r} dataset={dataset!r} year={year}; "
                f"known runs: {list(C.RUNS)}")
        srcs.append(Source.from_run(run, color=model_color(m)))
    labels = ", ".join(f"{s.pretty}({s.run})" for s in srcs)
    print(f"[sources] {len(srcs)}: {labels}")
    return srcs


def is_multi(sources) -> bool:
    return len(sources) > 1


def model_token(sources) -> str:
    """Figure-name <model> field: the single model, or models joined with '-'."""
    if len(sources) == 1:
        return sources[0].model
    seen = list(dict.fromkeys(s.model for s in sources))
    return "-".join(seen)


def run_label(sources) -> str:
    """Top figure folder under FIG_ROOT: the run for a single source, or a
    'compare_<dataset>_<year>' bucket for an overlay (so comparisons never
    overwrite single-source figures)."""
    if len(sources) == 1:
        return sources[0].run
    s = sources[0]
    return f"compare_{s.dataset}_{s.year}"


def figure_dir(sources, period, region, variable, kind) -> Path:
    """<FIG_ROOT>/<run-or-compare>/<period>/<region>/<variable>/<kind>/."""
    d = (C.FIG_ROOT / run_label(sources) / C.period_dir_name(period)
         / region / variable / kind)
    d.mkdir(parents=True, exist_ok=True)
    return d


def requested_levels(sources) -> list[int]:
    """Env-requested levels (NG_LEVELS / NG_LEVEL_INTERVAL etc.) intersected with
    the levels present in EVERY source's prediction grid."""
    explicit = os.environ.get("NG_LEVELS", "").strip()
    if explicit:
        req = [int(float(x)) for x in explicit.replace(",", " ").split()]
    else:
        interval = int(os.environ.get("NG_LEVEL_INTERVAL", "50"))
        lo = int(os.environ.get("NG_LEVEL_MIN", str(interval)))
        hi = int(os.environ.get("NG_LEVEL_MAX", "1000"))
        req = list(range(lo, hi + 1, interval))
    req = [l for l in req if 0 < l <= 1000]
    avail = set(sources[0].prediction_levels())
    for s in sources[1:]:
        avail &= set(s.prediction_levels())
    kept = [l for l in req if l in avail]
    dropped = [l for l in req if l not in avail]
    if dropped:
        print(f"[levels] skipping {dropped} -- not in all source grids")
    print(f"[levels] {len(kept)} levels: {kept}")
    return kept
