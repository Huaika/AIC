#!/usr/bin/env python
"""A "set of grid points" the eval plotters derive their data from, plus the
helper that builds one out of lat/lon rectangles.

Every analysis reduces a field over some set of grid cells: the global analysis
over the whole globe, the regional analysis over a continent box, the
out-of-distribution (NextGEMS) analysis over the same boxes, and the heat-wave
case study over an arbitrary per-cell footprint. Instead of hard-wiring a
rectangular crop, the plotters take a ``GridPoints`` -- a named boolean mask on
the model grid -- and reduce through ONE masked, cos(lat)-weighted mean
(``masked_area_mean``). ``boxes_to_points`` turns rectangles into such a set, so
the box-based analyses (global / regional / OOD) go through the exact same path as
the case study; the heat-wave masks supply their footprint directly.

A box-derived ``GridPoints`` reduces to the identical cells and weights as the old
``lat_weighted_mean(select_region(da, region))``, so existing figures are
unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from aic.regions import REGIONS, region_extent


def boxes_to_points(lat, lon, boxes) -> np.ndarray:
    """2-D boolean mask ``(len(lat), len(lon))`` True inside ANY rectangle.

    ``boxes`` is an iterable of ``(lat_south, lat_north, lon_west, lon_east)`` in
    the -180..180 convention (as in ``aic.regions.REGIONS``); ``lon`` may be in any
    convention (normalised internally). A box with ``lon_west > lon_east`` wraps the
    antimeridian, exactly like ``region_mask``.
    """
    lat = np.asarray(lat, float)
    lon2 = ((np.asarray(lon, float) + 180) % 360) - 180
    out = np.zeros((lat.size, lon.size), bool)
    for s, n, w, e in boxes:
        latm = (lat >= s) & (lat <= n)
        lonm = ((lon2 >= w) & (lon2 <= e)) if w <= e else ((lon2 >= w) | (lon2 <= e))
        out |= latm[:, None] & lonm[None, :]
    return out


@dataclass
class GridPoints:
    """A named set of grid cells to analyse: a 2-D boolean mask on the model grid
    (coords aligned to the predictions/truth), a ``key`` token for CSV rows /
    filenames, and a map ``extent`` ``(w, e, s, n)`` for axis limits."""
    key: str
    mask: xr.DataArray                 # 2-D bool (latitude, longitude)
    extent: tuple                      # (lon_west, lon_east, lat_south, lat_north)

    @classmethod
    def from_region(cls, lat, lon, region: str) -> "GridPoints":
        """The grid cells inside a named region box (world -> whole grid)."""
        m = boxes_to_points(lat, lon, [REGIONS[region]])
        da = xr.DataArray(m, dims=("latitude", "longitude"),
                          coords={"latitude": np.asarray(lat),
                                  "longitude": np.asarray(lon)})
        return cls(region, da, region_extent(region))

    @classmethod
    def from_mask(cls, key: str, mask2d: xr.DataArray, extent) -> "GridPoints":
        """An arbitrary per-cell footprint (e.g. a heat-wave episode)."""
        return cls(key, mask2d, tuple(extent))

    @property
    def n_cells(self) -> int:
        return int(self.mask.values.sum())


def masked_area_mean(da: xr.DataArray, points) -> xr.DataArray:
    """cos(lat)-weighted mean of ``da`` over the cells where ``points`` is True.

    ``points`` is a boolean ``GridPoints`` / DataArray, 2-D ``(lat, lon)`` or 3-D
    ``(time, lat, lon)`` aligned by coordinates. Cells outside the set become NaN
    and are dropped from both the value sum and the weight sum, so the result is the
    area-weighted mean over exactly the selected cells (empty set -> NaN).
    """
    mask = points.mask if isinstance(points, GridPoints) else points
    w = np.cos(np.deg2rad(da.latitude))
    return da.where(mask).weighted(w).mean(["latitude", "longitude"])
