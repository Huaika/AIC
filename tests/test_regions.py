"""Tests for the shared region/longitude helpers (aic.regions)."""
import numpy as np
import xarray as xr

from aic.regions import (
    REGIONS, region_extent, region_mask, select_region, to_lon180, wrap180,
)


def test_wrap180_scalars_and_arrays():
    assert wrap180(350) == -10
    assert wrap180(190) == -170
    assert wrap180(0) == 0
    assert wrap180(180) == -180          # boundary maps to -180
    got = wrap180(np.array([0.0, 90.0, 200.0, 359.0]))
    np.testing.assert_allclose(got, [0.0, 90.0, -160.0, -1.0])


def test_to_lon180_sorts_and_relabels():
    da = xr.DataArray(
        np.arange(4.0), dims="longitude",
        coords={"longitude": [0.0, 90.0, 200.0, 350.0]},
    )
    out = to_lon180(da)
    # longitudes now ascending in -180..180
    np.testing.assert_allclose(out.longitude.values, [-160.0, -10.0, 0.0, 90.0])
    # values follow their coordinate (200->-160 carried value 2, 350->-10 carried 3)
    assert out.sel(longitude=-160.0).item() == 2.0
    assert out.sel(longitude=-10.0).item() == 3.0


def test_region_extent_matches_box():
    s, n, w, e = REGIONS["europe"]
    assert region_extent("europe") == (w, e, s, n)


def test_region_mask_selects_box():
    lat = np.array([-40.0, 0.0, 50.0, 80.0])
    lon = np.array([0.0, 30.0, 350.0])     # 350 == -10 in -180..180
    latm, lonm = region_mask(lat, lon, "europe")   # europe: lat 34..72, lon -25..45
    assert list(latm) == [False, False, True, False]
    assert list(lonm) == [True, True, True]         # 0, 30, -10 all within -25..45


def test_select_region_crops():
    lat = np.linspace(-80, 80, 9)
    lon = np.arange(0, 360, 40.0)
    da = xr.DataArray(np.ones((lat.size, lon.size)), dims=("latitude", "longitude"),
                      coords={"latitude": lat, "longitude": lon})
    cropped = select_region(da, "europe")
    assert cropped.latitude.min() >= 34 and cropped.latitude.max() <= 72
    assert cropped.longitude.min() >= -25 and cropped.longitude.max() <= 45


def test_select_region_world_is_full_grid():
    lat = np.linspace(-80, 80, 9)
    lon = np.arange(0, 360, 40.0)
    da = xr.DataArray(np.ones((lat.size, lon.size)), dims=("latitude", "longitude"),
                      coords={"latitude": lat, "longitude": lon})
    assert select_region(da, "world").size == da.size
