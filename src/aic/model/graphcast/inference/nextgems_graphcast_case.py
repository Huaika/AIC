from __future__ import annotations

import contextlib
import dataclasses
import logging
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_NEXTGEMS_ROOT_TEMPLATE = (
    "/pfs/work9/workspace/scratch/ka_je2428-nextgems_{year}"
)
GRAVITY = 9.80665

NEXTGEMS_RENAMES = {
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "t": "temperature",
    "q": "specific_humidity",
    "z": "geopotential",
    "t2m": "2m_temperature",
    "2m_t": "2m_temperature",
    "2t": "2m_temperature",
    "u10": "10m_u_component_of_wind",
    "10m_u": "10m_u_component_of_wind",
    "10u": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "10m_v": "10m_v_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "mean_sea_level_pressure": "mean_sea_level_pressure",
    "oro": "orography",
    "orog": "orography",
    "land_sea_mask": "land_sea_mask",
    "ci": "sea_ice_cover",
    "sst": "sea_surface_temperature",
    "ciwc": "specific_cloud_ice_water_content",
    "clwc": "specific_cloud_liquid_water_content",
    "lat": "lat",
    "latitude": "lat",
    "lon": "lon",
    "longitude": "lon",
    "lev": "level",
    "plev": "level",
    "pressure_level": "level",
}

ATMOSPHERIC_VARIABLES = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
)
SURFACE_VARIABLES = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
)
STATIC_VARIABLES = (
    "geopotential_at_surface",
    "land_sea_mask",
)
PRECIPITATION_ALIASES = (
    "total_precipitation_6hr",
    "total_precipitation",
    "tp",
)


@dataclasses.dataclass(frozen=True)
class NextGemsPaths:
  root: Path
  atmospheric: Path
  surface: Path
  sst: Path
  constants: Path


@contextlib.contextmanager
def log_duration(message: str):
  start = time.monotonic()
  logging.info("%s ...", message)
  try:
    yield
  finally:
    logging.info("%s done in %.1f s", message, time.monotonic() - start)


def nextgems_paths(year: int, root: Path | None = None) -> NextGemsPaths:
  root = root or Path(DEFAULT_NEXTGEMS_ROOT_TEMPLATE.format(year=year))
  return NextGemsPaths(
      root=root,
      atmospheric=root / f"3D_nextgems_{year}_6hourly_0.25deg_lat-lon.nc",
      surface=root / f"surface_nextgems_{year}_6hourly_0.25deg_lat-lon.nc",
      sst=root / f"surface_nextgems_{year}_6hourly_0.25deg_ci_SSTs_lat-lon.nc",
      constants=root / "constant_fields.zarr",
  )


def safe_time_label(init_time: str) -> str:
  timestamp = pd.Timestamp(init_time)
  return timestamp.strftime("%Y%m%d_%H")


def default_case_path(
    case_cache_dir: Path,
    year: int,
    init_time: str,
    rollout_steps: int,
    step_hours: int,
) -> Path:
  return (
      case_cache_dir
      / f"graphcast_nextgems_{year}_{safe_time_label(init_time)}"
      f"_steps{rollout_steps}_h{step_hours}.nc"
  )


def load_nextgems_year(
    year: int,
    root: Path | None = None,
    chunks: dict[str, int] | None = None,
) -> xr.Dataset:
  paths = nextgems_paths(year, root)
  chunks = chunks or {"time": 1}
  atmospheric = xr.open_dataset(paths.atmospheric, decode_times=True, chunks=chunks)
  surface = xr.merge(
      [
          xr.open_dataset(paths.surface, decode_times=True, chunks=chunks),
          xr.open_dataset(paths.sst, decode_times=True, chunks=chunks),
      ],
      compat="no_conflicts",
  )
  constants = xr.open_zarr(paths.constants, consolidated=True)
  return xr.merge(
      [atmospheric, surface, constants],
      compat="no_conflicts",
      join="outer",
  )


def rename_nextgems_for_graphcast(ds: xr.Dataset) -> xr.Dataset:
  rename: dict[str, str] = {}
  for source_name, target_name in NEXTGEMS_RENAMES.items():
    if source_name == target_name:
      continue
    if source_name not in ds and source_name not in ds.coords and source_name not in ds.dims:
      continue
    if target_name in ds or target_name in ds.coords or target_name in ds.dims:
      continue
    rename[source_name] = target_name
  return ds.rename(rename)


def normalize_dims_and_coords(ds: xr.Dataset) -> xr.Dataset:
  ds = rename_nextgems_for_graphcast(ds)

  if "lat" in ds.coords and ds["lat"].values[0] > ds["lat"].values[-1]:
    ds = ds.isel(lat=slice(None, None, -1))
  if "lon" in ds.coords and np.any(ds["lon"].values < 0):
    ds = ds.assign_coords(lon=np.mod(ds["lon"], 360.0)).sortby("lon")
  if "level" in ds.coords:
    level_values = np.asarray(ds["level"].values, dtype=np.float64)
    if np.nanmax(level_values) > 2000:
      level_values = level_values / 100.0
    ds = ds.assign_coords(level=level_values).sortby("level")
  return ds


def add_surface_geopotential(ds: xr.Dataset) -> xr.Dataset:
  if "geopotential_at_surface" in ds:
    return ds
  if "orography" not in ds:
    return ds

  surface_geopotential = ds["orography"] * GRAVITY
  surface_geopotential.attrs.update({
      "long_name": "geopotential at surface",
      "units": "m2 s-2",
      "source": "orography multiplied by standard gravity",
  })
  return ds.assign(geopotential_at_surface=surface_geopotential)


def pressure_level_variables(ds: xr.Dataset) -> list[str]:
  return [name for name, variable in ds.data_vars.items() if "level" in variable.dims]


def validate_pressure_level_coverage(ds: xr.Dataset, target_levels: np.ndarray) -> None:
  if "level" not in ds.coords:
    raise ValueError("Cannot interpolate pressure levels: dataset has no 'level' coordinate.")
  source_levels = np.asarray(ds["level"].values, dtype=np.float64)
  source_min = np.nanmin(source_levels)
  source_max = np.nanmax(source_levels)
  target_min = np.nanmin(target_levels)
  target_max = np.nanmax(target_levels)
  if target_min < source_min or target_max > source_max:
    raise ValueError(
        "Requested GraphCast pressure levels are outside the NextGEMS level range: "
        f"requested {target_min:g}-{target_max:g} hPa, "
        f"available {source_min:g}-{source_max:g} hPa."
    )


def interpolate_to_pressure_levels(
    ds: xr.Dataset,
    pressure_levels: Iterable[int],
) -> xr.Dataset:
  ds = normalize_dims_and_coords(ds)
  target_levels = np.asarray(tuple(int(level) for level in pressure_levels), dtype=np.float64)
  validate_pressure_level_coverage(ds, target_levels)
  level_variables = pressure_level_variables(ds)
  interpolated = ds.drop_vars(level_variables).drop_vars("level", errors="ignore")
  for name in level_variables:
    interpolated[name] = ds[name].interp(level=target_levels)
  return interpolated.assign_coords(level=target_levels.astype(np.int32))


def prepare_nextgems_for_graphcast(
    ds: xr.Dataset,
    pressure_levels: Iterable[int],
) -> xr.Dataset:
  ds = normalize_dims_and_coords(ds)
  ds = add_surface_geopotential(ds)
  return interpolate_to_pressure_levels(ds, pressure_levels)


def requested_datetimes(
    init_time: str,
    rollout_steps: int,
    step_hours: int,
) -> np.ndarray:
  init = np.datetime64(pd.Timestamp(init_time).to_datetime64(), "ns")
  step = np.timedelta64(step_hours, "h")
  return init - step + np.arange(rollout_steps + 2) * step


def select_case_window(
    ds: xr.Dataset,
    init_time: str,
    rollout_steps: int,
    step_hours: int,
) -> xr.Dataset:
  times = requested_datetimes(init_time, rollout_steps, step_hours)
  try:
    return ds.sel(time=times)
  except KeyError as exc:
    raise ValueError(
        "NextGEMS data is missing one or more requested 6-hourly times between "
        f"{times[0]} and {times[-1]}."
    ) from exc


def missing_graphcast_variables(ds: xr.Dataset) -> list[str]:
  required = ATMOSPHERIC_VARIABLES + SURFACE_VARIABLES + STATIC_VARIABLES
  return sorted(name for name in required if name not in ds)


def add_batch_and_time_coords(
    data_array: xr.DataArray,
    time_coord: np.ndarray,
    dims: tuple[str, ...],
) -> xr.DataArray:
  data_array = data_array.transpose(*dims).astype(np.float32)
  data_array = data_array.assign_coords(time=time_coord)
  data_array = data_array.expand_dims(batch=[0], axis=0)
  return data_array.transpose("batch", *dims)


def total_precipitation_6hr(ds: xr.Dataset, times: np.ndarray) -> xr.DataArray:
  for name in PRECIPITATION_ALIASES:
    if name not in ds:
      continue
    precip = ds[name]
    if "time" in precip.dims:
      precip = precip.sel(time=times)
    return precip

  logging.warning(
      "NextGEMS dataset has no total_precipitation_6hr/total_precipitation/tp. "
      "Using a zero placeholder so GraphCast can build the target template."
  )
  template = ds["2m_temperature"].sel(time=times)
  return xr.zeros_like(template).rename("total_precipitation_6hr")


def build_graphcast_case_from_nextgems(
    nextgems: xr.Dataset,
    init_time: str,
    rollout_steps: int,
    step_hours: int,
    pressure_levels: tuple[int, ...],
) -> xr.Dataset:
  times = requested_datetimes(init_time, rollout_steps, step_hours)
  time_coord = np.arange(len(times)) * np.timedelta64(step_hours, "h")
  logging.info(
      "Building GraphCast case from NextGEMS for init %s: %s to %s (%d frames)",
      init_time,
      times[0],
      times[-1],
      len(times),
  )

  prepared = prepare_nextgems_for_graphcast(nextgems, pressure_levels)
  missing = missing_graphcast_variables(prepared)
  if missing:
    raise KeyError(
        "NextGEMS dataset is missing GraphCast-required variables after "
        f"renaming/preparation: {missing}"
    )

  source_names = set(ATMOSPHERIC_VARIABLES) | set(SURFACE_VARIABLES) | set(STATIC_VARIABLES)
  source_names |= {name for name in PRECIPITATION_ALIASES if name in prepared}
  window = select_case_window(prepared[list(source_names)], init_time, rollout_steps, step_hours)

  data_vars: dict[str, xr.DataArray] = {}
  for name in ATMOSPHERIC_VARIABLES:
    logging.info("Mapping NextGEMS pressure variable: %s", name)
    data_vars[name] = add_batch_and_time_coords(
        window[name],
        time_coord,
        ("time", "lat", "lon", "level"),
    )

  for name in SURFACE_VARIABLES:
    logging.info("Mapping NextGEMS surface variable: %s", name)
    data_vars[name] = add_batch_and_time_coords(
        window[name],
        time_coord,
        ("time", "lat", "lon"),
    )

  data_vars["total_precipitation_6hr"] = add_batch_and_time_coords(
      total_precipitation_6hr(prepared, times),
      time_coord,
      ("time", "lat", "lon"),
  )

  for name in STATIC_VARIABLES:
    static = window[name]
    if "time" in static.dims:
      static = static.isel(time=0).drop_vars("time", errors="ignore")
    logging.info("Mapping NextGEMS static variable: %s", name)
    data_vars[name] = static.transpose("lat", "lon").astype(np.float32)

  dataset = xr.Dataset(data_vars)
  dataset = dataset.assign_coords(
      batch=np.asarray([0], dtype=np.int32),
      time=time_coord,
      datetime=(("batch", "time"), times[None, :]),
      lat=window["lat"].astype(np.float32),
      lon=window["lon"].astype(np.float32),
      level=np.asarray(pressure_levels, dtype=np.int32),
  )
  logging.info("Constructed NextGEMS GraphCast case dimensions: %s", dict(dataset.sizes))
  logging.info(
      "Constructed NextGEMS GraphCast case variables: %s",
      ", ".join(sorted(dataset.data_vars)),
  )
  return dataset


def build_encoding(dataset: xr.Dataset, compression_level: int) -> dict[str, dict[str, object]]:
  encoding: dict[str, dict[str, object]] = {}
  for name, variable in dataset.data_vars.items():
    if not np.issubdtype(variable.dtype, np.floating):
      continue
    variable_encoding: dict[str, object] = {"dtype": "float32"}
    if compression_level > 0:
      variable_encoding.update({
          "zlib": True,
          "complevel": compression_level,
          "shuffle": True,
      })
      if variable.dims == ("batch", "time", "lat", "lon", "level"):
        variable_encoding["chunksizes"] = (
            1, 1, variable.sizes["lat"], variable.sizes["lon"], 1)
      elif variable.dims == ("batch", "time", "lat", "lon"):
        variable_encoding["chunksizes"] = (
            1, 1, variable.sizes["lat"], variable.sizes["lon"])
      elif variable.dims == ("lat", "lon"):
        variable_encoding["chunksizes"] = (
            variable.sizes["lat"], variable.sizes["lon"])
    encoding[name] = variable_encoding
  return encoding


def remove_decode_conflicting_time_attrs(path: Path) -> None:
  from netCDF4 import Dataset  # pylint: disable=import-outside-toplevel

  with Dataset(path, mode="a") as netcdf:
    if "time" not in netcdf.variables:
      return
    time_var = netcdf.variables["time"]
    if "dtype" in time_var.ncattrs():
      time_var.delncattr("dtype")


def ensure_nextgems_graphcast_case(
    case_cache_dir: Path,
    init_time: str,
    rollout_steps: int,
    step_hours: int,
    pressure_levels: tuple[int, ...],
    *,
    year: int = 2049,
    root: Path | None = None,
    output_path: Path | None = None,
    overwrite: bool = False,
    compression_level: int = 1,
) -> Path:
  output_path = output_path or default_case_path(
      case_cache_dir, year, init_time, rollout_steps, step_hours)
  output_path = output_path.expanduser()
  if output_path.exists() and not overwrite:
    logging.info("Reusing cached NextGEMS GraphCast case: %s", output_path)
    return output_path

  output_path.parent.mkdir(parents=True, exist_ok=True)
  logging.info("NextGEMS GraphCast case cache path: %s", output_path)
  nextgems = load_nextgems_year(year, root)
  with log_duration("Building lazy GraphCast case from NextGEMS selection"):
    dataset = build_graphcast_case_from_nextgems(
        nextgems,
        init_time,
        rollout_steps,
        step_hours,
        tuple(int(level) for level in pressure_levels),
    )
  logging.info("NextGEMS case dimensions: %s", dict(dataset.sizes))
  logging.info("This write triggers the actual NextGEMS data reads.")
  with log_duration(f"Writing NextGEMS GraphCast case NetCDF to {output_path}"):
    dataset.to_netcdf(
        output_path,
        engine="netcdf4",
        encoding=build_encoding(dataset, compression_level),
    )
  remove_decode_conflicting_time_attrs(output_path)
  logging.info("Finished cached NextGEMS GraphCast case: %s", output_path)
  return output_path
