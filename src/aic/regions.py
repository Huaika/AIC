#!/usr/bin/env python
"""Single source of truth for the named lat/lon regions used across the project.

ONE definition, imported by the eval plotters (spaghetti / drift / drift-maps via
eval_common), the heatwave analysis, and the ERA5 staging code -- so a region box
is changed in exactly one place. Each region is
(lat_south, lat_north, lon_west, lon_east) in the -180..180 longitude convention;
'world' is the whole globe (the default). The continent boxes are approximate
bounding boxes and may overlap (e.g. europe/asia) -- that is fine.

No import-time side effects (no env reads; only numpy), so every module can import
it safely, including the standalone staging jobs.
"""
from __future__ import annotations

import numpy as np

REGIONS = {
    "world":         (-90.0,  90.0, -180.0, 180.0),
    "africa":        (-37.0,  38.0,  -20.0,  55.0),
    "europe":        ( 34.0,  72.0,  -25.0,  45.0),
    "asia":          (  5.0,  78.0,   25.0, 180.0),
    "north_america": (  7.0,  84.0, -170.0, -52.0),
    "south_america": (-57.0,  14.0,  -82.0, -34.0),
    "oceania":       (-50.0,   0.0,  110.0, 180.0),
    "antarctica":    (-90.0, -60.0, -180.0, 180.0),
}
DEFAULT_REGIONS = ["world"]
CONTINENTS = [r for r in REGIONS if r != "world"]


def wrap180(lon):
    """Longitude values (any convention) -> the -180..180 convention. Numpy in,
    numpy out; for masking / box tests where only the values matter, not order."""
    return ((np.asarray(lon, float) + 180) % 360) - 180


def to_lon180(da):
    """An xarray field -> the -180..180 longitude convention, sorted ascending in
    both latitude and longitude, so it aligns with ``region_extent`` boxes for
    cropping and plotting. Model-grid fields are stored on 0..360; without this the
    western hemisphere (real -180..0) sits past the right edge of a -180..180 extent
    and silently vanishes. Single source for the ``((lon+180)%360)-180`` idiom."""
    da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180))
    da = da.sortby("longitude")
    return da.sortby("latitude") if "latitude" in da.coords else da


def select_region(da, region):
    """Crop an xarray DataArray (latitude/longitude dims) to the region's box
    (no-op footprint for 'world'). Normalizes longitude to -180..180 first; handles
    a box that wraps the antimeridian (lon_west > lon_east)."""
    s, n, w, e = REGIONS[region]
    da = to_lon180(da)
    if w <= e:
        da = da.sel(longitude=slice(w, e))
    else:  # box crosses the antimeridian (e.g. 150 .. -150)
        da = da.sel(longitude=(da.longitude >= w) | (da.longitude <= e))
    return da.sel(latitude=slice(s, n))


def region_extent(region):
    """(lon_west, lon_east, lat_south, lat_north) for map axis limits."""
    s, n, w, e = REGIONS[region]
    return w, e, s, n


def region_mask(lat, lon, region):
    """Boolean (lat_mask, lon_mask) selecting the region box from 1-D numpy lat/lon
    coordinate arrays (lon in any convention) -- for gridded/array workflows that
    do not go through the xarray-based select_region()."""
    s, n, w, e = REGIONS[region]
    lat = np.asarray(lat)
    lon2 = wrap180(lon)
    latm = (lat >= s) & (lat <= n)
    lonm = ((lon2 >= w) & (lon2 <= e)) if w <= e else ((lon2 >= w) | (lon2 <= e))
    return latm, lonm
