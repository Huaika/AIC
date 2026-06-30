#!/usr/bin/env python
"""Global 10-day temperature drift maps for NextGEMS-2049, Rackow et al. (2024)
Fig. 3 style (arXiv:2409.18529), for every requested pressure level.

Fig. 3 definition: drift = annual-mean of the data-driven model's end-of-forecast
(day-10) fields minus the annual-mean of the reference simulation.
  ngcm_day10_clim = mean over inits of the rollout's final-day mean (lead 216..240 h)
  ref_clim        = NextGEMS-2049 annual-mean field
  drift           = ngcm_day10_clim - ref_clim   (model grid)

Three panels per level (NeuralGCM day-10 climatology, NextGEMS reference, drift),
saved under figures/drift_maps/.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import ng2049_common as C

YEAR = C.YEAR
FINAL_DAY_LEAD_MIN = 216
FIGDIR = C.figure_dir("drift_maps")


def build_fields(levels, truth) -> xr.Dataset:
    nc = C.OUTDIR / f"nextgems_{YEAR}_drift_maps_{C.level_tag()}.nc"
    if nc.exists():
        print(f"[maps] cached {nc}")
        return xr.open_dataset(nc)

    ref_clim = truth.mean("time").transpose("level", "latitude", "longitude")
    files = sorted(C.PRED_DIR.glob(f"pred_{YEAR}_*.nc"))
    print(f"[maps] end-of-forecast mean over {len(files)} rollouts, {len(levels)} levels")
    acc, n = None, 0
    for i, f in enumerate(files):
        ds = xr.open_dataset(f)
        t = ds["temperature"].sel(level=levels)
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
                     drift_def="mean(end-of-10day-forecast) - NextGEMS-2049 annual mean")
    out.to_netcdf(nc)
    print(f"[maps] wrote {nc}")
    return out


def main():
    levels = C.requested_levels()
    truth = C.truth_at_levels(levels)
    out = build_fields(levels, truth)

    for lev in levels:
        ng = C.to_world(out["ngcm_day10_clim"].sel(level=lev))
        rf = C.to_world(out["ref_annual_clim"].sel(level=lev))
        dr = C.to_world(out["drift"].sel(level=lev))
        lon, lat = ng.longitude, ng.latitude
        tmin = float(min(ng.min(), rf.min())); tmax = float(max(ng.max(), rf.max()))
        dlim = float(np.nanpercentile(np.abs(dr.values), 99)) or 1.0

        fig, axes = plt.subplots(1, 3, figsize=(19, 4.3))
        m0 = axes[0].pcolormesh(lon, lat, ng, cmap="RdYlBu_r", vmin=tmin, vmax=tmax,
                                shading="auto")
        axes[0].set_title(f"NeuralGCM day-10 climatology\n(mean of {out.attrs['n_inits']} forecasts)")
        fig.colorbar(m0, ax=axes[0], shrink=0.8, label="T [K]")
        m1 = axes[1].pcolormesh(lon, lat, rf, cmap="RdYlBu_r", vmin=tmin, vmax=tmax,
                                shading="auto")
        axes[1].set_title(f"NextGEMS-{YEAR} annual mean\n(reference)")
        fig.colorbar(m1, ax=axes[1], shrink=0.8, label="T [K]")
        m2 = axes[2].pcolormesh(lon, lat, dr, cmap="RdBu_r", vmin=-dlim, vmax=dlim,
                                shading="auto")
        gm = float(dr.weighted(np.cos(np.deg2rad(lat))).mean(["longitude", "latitude"]))
        axes[2].set_title(f"10-day drift = NeuralGCM − NextGEMS\n(global mean {gm:+.2f} K)")
        fig.colorbar(m2, ax=axes[2], shrink=0.8, label="drift [K]")
        for ax in axes:
            C.draw_coastlines(ax)
            ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
            ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); ax.grid(alpha=0.2)
        fig.suptitle(f"NextGEMS-{YEAR} — mean 10-day T@{lev} hPa drift "
                     f"(Rackow et al. 2024, Fig. 3 style)", y=1.04, fontsize=13)
        fig.tight_layout()
        out_png = FIGDIR / f"nextgems{YEAR}_driftmap_T_L{lev:04d}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"L{lev}: drift global mean {gm:+.3f} K, "
              f"range {float(dr.min()):+.2f}..{float(dr.max()):+.2f} K")
    print(f"done -> {FIGDIR}/")


if __name__ == "__main__":
    main()
