#!/usr/bin/env python
"""Global 10-day drift maps (Rackow et al. 2024 Fig. 3 style) -- RUN/VARIABLE-
agnostic, per requested pressure level.

drift = annual-mean of the model's day-10 fields minus the reference annual mean:
  ngcm_day10_clim = mean over inits of the rollout's final-day mean (216..240 h)
  ref_clim        = reference annual-mean field
  drift           = ngcm_day10_clim - ref_clim   (model grid)
Three panels per (variable, level): NeuralGCM day-10 clim, reference, drift.
Run via EVAL_RUN, variable set via EVAL_VARS.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import eval_common as C

YEAR = C.YEAR
FINAL_DAY_LEAD_MIN = 216


def build_fields(var, short, levels, truth) -> xr.Dataset:
    nc = C.OUTDIR / f"{C.RUN}_drift_maps_{short}_{C.level_tag()}.nc"
    if nc.exists():
        print(f"[maps] cached {nc}")
        return xr.open_dataset(nc)

    ref_clim = truth.mean("time").transpose("level", "latitude", "longitude")
    files = sorted(C.PRED_DIR.glob(f"pred_{YEAR}_*.nc"))
    print(f"[maps] end-of-forecast mean over {len(files)} rollouts "
          f"({var}), {len(levels)} levels")
    acc, n = None, 0
    for i, f in enumerate(files):
        ds = xr.open_dataset(f)
        t = ds[var].sel(level=levels)
        end = t.where(ds["lead_hours"] >= FINAL_DAY_LEAD_MIN, drop=True).mean("time")
        end = end.transpose("level", "latitude", "longitude")
        acc = end if acc is None else acc + end
        n += 1
        ds.close()
        if i % 50 == 0 or i == len(files) - 1:
            print(f"  {i + 1}/{len(files)}")
    ngcm = acc / n
    out = xr.Dataset({"ngcm_day10_clim": ngcm, "ref_annual_clim": ref_clim,
                      "drift": ngcm - ref_clim})
    out.attrs.update(n_inits=n, final_day_lead_min_h=FINAL_DAY_LEAD_MIN,
                     variable=var,
                     drift_def=f"mean(end-of-10day-forecast) - {C.REF_LABEL} annual mean")
    out.to_netcdf(nc)
    print(f"[maps] wrote {nc}")
    return out


def plot_variable(var, levels):
    meta = C.VARIABLES[var]
    short, units, label, fcmap = (meta["short"], meta["units"],
                                  meta["label"], meta["cmap"])
    print(f"=== drift maps: {var} ({short}) ===")
    figdir = C.figure_dir(var, "drift_maps")
    truth = C.truth_at_levels(var, levels)
    out = build_fields(var, short, levels, truth)

    for lev in levels:
        ng = C.to_world(out["ngcm_day10_clim"].sel(level=lev))
        rf = C.to_world(out["ref_annual_clim"].sel(level=lev))
        dr = C.to_world(out["drift"].sel(level=lev))
        lon, lat = ng.longitude, ng.latitude
        vmin = float(min(ng.min(), rf.min())); vmax = float(max(ng.max(), rf.max()))
        dlim = float(np.nanpercentile(np.abs(dr.values), 99)) or 1.0

        fig, axes = plt.subplots(1, 3, figsize=(19, 4.3))
        m0 = axes[0].pcolormesh(lon, lat, ng, cmap=fcmap, vmin=vmin, vmax=vmax,
                                shading="auto")
        axes[0].set_title(f"NeuralGCM day-10 climatology\n(mean of {out.attrs['n_inits']} forecasts)")
        fig.colorbar(m0, ax=axes[0], shrink=0.8, label=f"{label} [{units}]")
        m1 = axes[1].pcolormesh(lon, lat, rf, cmap=fcmap, vmin=vmin, vmax=vmax,
                                shading="auto")
        axes[1].set_title(f"{C.REF_LABEL} annual mean\n(reference)")
        fig.colorbar(m1, ax=axes[1], shrink=0.8, label=f"{label} [{units}]")
        m2 = axes[2].pcolormesh(lon, lat, dr, cmap="RdBu_r", vmin=-dlim, vmax=dlim,
                                shading="auto")
        gm = float(dr.weighted(np.cos(np.deg2rad(lat))).mean(["longitude", "latitude"]))
        axes[2].set_title(f"10-day drift = NeuralGCM − {C.REF_LABEL}\n(global mean {gm:+.4g} {units})")
        fig.colorbar(m2, ax=axes[2], shrink=0.8, label=f"drift [{units}]")
        for ax in axes:
            C.draw_coastlines(ax)
            ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
            ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); ax.grid(alpha=0.2)
        fig.suptitle(f"{C.REF_LABEL} — mean 10-day {label}@{lev} hPa drift "
                     f"(Rackow et al. 2024, Fig. 3 style)", y=1.04, fontsize=13)
        fig.tight_layout()
        out_png = figdir / f"{C.RUN}_driftmap_L{lev:04d}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"{var} L{lev}: drift global mean {gm:+.4g} {units}, "
              f"range {float(dr.min()):+.4g}..{float(dr.max()):+.4g} {units}")


def main():
    levels = C.requested_levels()
    for var in C.selected_variables():
        plot_variable(var, levels)
    print(f"done -> {C.FIGROOT}/<variable>/drift_maps/")


if __name__ == "__main__":
    main()
