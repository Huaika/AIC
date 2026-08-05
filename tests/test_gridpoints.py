"""Tests for the grid-point-set abstraction (aic.controller.eval.gridpoints).

The key invariant: a box-derived GridPoints reduces to the identical cells and
weights as the old ``lat_weighted_mean(select_region(da, region))`` -- so switching
the analyses onto GridPoints leaves the figures unchanged.
"""
import numpy as np
import xarray as xr

from aic.controller.eval import gridpoints as GP
from aic.regions import select_region


def _grid():
    lat = np.linspace(-87.0, 87.0, 32)
    lon = np.arange(0.0, 360.0, 360.0 / 64)
    da = xr.DataArray(
        np.random.default_rng(0).normal(size=(lat.size, lon.size)),
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
    )
    return lat, lon, da


def _crop_mean(da, region):
    c = select_region(da, region)
    w = np.cos(np.deg2rad(c.latitude))
    return float(c.weighted(w).mean(["latitude", "longitude"]))


def test_boxes_to_points_shape_and_membership():
    lat = np.array([0.0, 50.0])
    lon = np.array([10.0, 350.0])            # 350 == -10
    m = GP.boxes_to_points(lat, lon, [(34.0, 72.0, -25.0, 45.0)])  # europe box
    assert m.shape == (2, 2)
    assert m[1, 0] and m[1, 1]               # lat 50 in-box for both lons
    assert not m[0, 0] and not m[0, 1]       # lat 0 out of box


def test_masked_area_mean_full_mask_equals_weighted_mean():
    _, _, da = _grid()
    gp = GP.GridPoints.from_region(da.latitude.values, da.longitude.values, "world")
    w = np.cos(np.deg2rad(da.latitude))
    direct = float(da.weighted(w).mean(["latitude", "longitude"]))
    assert gp.n_cells == da.size
    np.testing.assert_allclose(float(GP.masked_area_mean(da, gp)), direct, rtol=1e-12)


def test_masked_area_mean_matches_crop_mean_for_regions():
    _, _, da = _grid()
    for region in ("europe", "north_america", "oceania", "asia"):
        gp = GP.GridPoints.from_region(da.latitude.values, da.longitude.values, region)
        np.testing.assert_allclose(
            float(GP.masked_area_mean(da, gp)), _crop_mean(da, region), rtol=1e-10)


def test_masked_area_mean_empty_set_is_nan():
    _, _, da = _grid()
    empty = xr.zeros_like(da).astype(bool)
    assert np.isnan(float(GP.masked_area_mean(da, empty)))
