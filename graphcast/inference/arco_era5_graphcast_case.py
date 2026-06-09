from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_ARCO_ERA5_PATH = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)

ATMOSPHERIC_ALIASES = {
    "temperature": ("temperature", "t"),
    "geopotential": ("geopotential", "z"),
    "u_component_of_wind": ("u_component_of_wind", "u"),
    "v_component_of_wind": ("v_component_of_wind", "v"),
    "vertical_velocity": ("vertical_velocity", "w"),
    "specific_humidity": ("specific_humidity", "q"),
}
SURFACE_ALIASES = {
    "2m_temperature": ("2m_temperature", "t2m"),
    "mean_sea_level_pressure": ("mean_sea_level_pressure", "msl"),
    "10m_u_component_of_wind": ("10m_u_component_of_wind", "u10"),
    "10m_v_component_of_wind": ("10m_v_component_of_wind", "v10"),
}
STATIC_ALIASES = {
    "geopotential_at_surface": ("geopotential_at_surface", "surface_geopotential"),
    "land_sea_mask": ("land_sea_mask", "lsm"),
}
PRECIPITATION_ALIASES = ("total_precipitation_6hr", "total_precipitation", "tp")
TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def env_flag(name: str, default: bool = False) -> bool:
  value = os.environ.get(name)
  if value is None:
    return default
  return value.lower() in TRUE_ENV_VALUES


def arco_chunks_setting():
  value = os.environ.get("GRAPHCAST_ARCO_CHUNKS", "none").lower()
  if value in {"none", "false", "0"}:
    return None
  if value == "native":
    return {}
  if value == "auto":
    return "auto"
  raise ValueError(
      "GRAPHCAST_ARCO_CHUNKS must be one of: none, native, auto."
  )


@contextlib.contextmanager
def log_duration(message: str):
  start = time.monotonic()
  logging.info("%s ...", message)
  try:
    yield
  finally:
    logging.info("%s done in %.1f s", message, time.monotonic() - start)


def requested_datetimes(
    init_time: str,
    rollout_steps: int,
    step_hours: int,
) -> np.ndarray:
  init = np.datetime64(pd.Timestamp(init_time).to_datetime64(), "ns")
  step = np.timedelta64(step_hours, "h")
  return init - step + np.arange(rollout_steps + 2) * step


def safe_time_label(init_time: str) -> str:
  timestamp = pd.Timestamp(init_time)
  return timestamp.strftime("%Y%m%d_%H")


def default_case_path(
    case_cache_dir: Path,
    init_time: str,
    rollout_steps: int,
    step_hours: int,
) -> Path:
  return (
      case_cache_dir
      / f"graphcast_arco_{safe_time_label(init_time)}"
      f"_steps{rollout_steps}_h{step_hours}.nc"
  )


def open_arco_era5(arco_path: str = DEFAULT_ARCO_ERA5_PATH) -> xr.Dataset:
  logging.info("Opening ARCO ERA5: %s", arco_path)
  consolidated = env_flag("GRAPHCAST_ARCO_CONSOLIDATED", default=False)
  chunks = arco_chunks_setting()
  logging.info(
      "ARCO open settings: consolidated=%s, chunks=%s",
      consolidated,
      "none" if chunks is None else chunks,
  )
  storage_options = {"token": "anon"} if arco_path.startswith("gs://") else None
  if storage_options is not None:
    try:
      import gcsfs  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError as exc:
      raise RuntimeError(
          "ARCO ERA5 over gs:// requires gcsfs. Install it in graphcast_env with "
          "`python -m pip install gcsfs`."
      ) from exc

  try:
    with log_duration("Opening ARCO ERA5 Zarr metadata with xarray.open_zarr"):
      dataset = xr.open_zarr(
          arco_path,
          chunks=chunks,
          consolidated=consolidated,
          storage_options=storage_options,
      )
    logging.info("ARCO ERA5 opened. Dimensions: %s", dict(dataset.sizes))
    logging.info("ARCO ERA5 data variables: %d", len(dataset.data_vars))
    return dataset
  except Exception as open_zarr_error:  # pylint: disable=broad-exception-caught
    logging.warning("xarray.open_zarr failed; trying xarray.open_dataset(engine='zarr')")
    backend_kwargs = {"consolidated": consolidated}
    if storage_options is not None:
      backend_kwargs["storage_options"] = storage_options
    try:
      with log_duration("Opening ARCO ERA5 Zarr metadata with xarray.open_dataset"):
        dataset = xr.open_dataset(
            arco_path,
            engine="zarr",
            chunks=chunks,
            backend_kwargs=backend_kwargs,
        )
      logging.info("ARCO ERA5 opened. Dimensions: %s", dict(dataset.sizes))
      logging.info("ARCO ERA5 data variables: %d", len(dataset.data_vars))
      return dataset
    except Exception as open_dataset_error:  # pylint: disable=broad-exception-caught
      raise RuntimeError(
          f"Could not open ARCO ERA5 Zarr store {arco_path!r}."
      ) from open_zarr_error


def normalize_dims_and_coords(ds: xr.Dataset) -> xr.Dataset:
  rename = {
      "valid_time": "time",
      "latitude": "lat",
      "longitude": "lon",
      "pressure_level": "level",
  }
  ds = ds.rename({old: new for old, new in rename.items()
                  if old in ds.dims or old in ds.coords})
  ds = ds.drop_vars([name for name in ("number", "expver") if name in ds],
                    errors="ignore")

  if "lat" in ds.coords and ds["lat"].values[0] > ds["lat"].values[-1]:
    ds = ds.isel(lat=slice(None, None, -1))
  if "lon" in ds.coords and np.any(ds["lon"].values < 0):
    ds = ds.assign_coords(lon=np.mod(ds["lon"], 360.0)).sortby("lon")
  return ds


def find_variable(ds: xr.Dataset, aliases: Iterable[str], output_name: str) -> str:
  for name in aliases:
    if name in ds:
      return name
  raise KeyError(
      f"Could not find ARCO ERA5 variable for {output_name!r}. Tried: "
      f"{', '.join(aliases)}"
  )


def select_times(ds: xr.Dataset, times: np.ndarray) -> xr.Dataset:
  try:
    return ds.sel(time=times)
  except KeyError as exc:
    raise ValueError(
        f"ARCO ERA5 is missing one or more requested times between "
        f"{times[0]} and {times[-1]}."
    ) from exc


def add_batch_and_time_coords(
    data_array: xr.DataArray,
    time_coord: np.ndarray,
    dims: tuple[str, ...],
) -> xr.DataArray:
  data_array = data_array.transpose(*dims).astype(np.float32)
  data_array = data_array.assign_coords(time=time_coord)
  data_array = data_array.expand_dims(batch=[0], axis=0)
  return data_array.transpose("batch", *dims)


def precipitation_6hr(
    ds: xr.Dataset,
    times: np.ndarray,
    step_hours: int,
) -> xr.DataArray:
  source_name = find_variable(ds, PRECIPITATION_ALIASES, "total_precipitation_6hr")
  logging.info("Using ARCO precipitation source variable: %s", source_name)
  if source_name == "total_precipitation_6hr":
    logging.info("Selecting existing total_precipitation_6hr at target times")
    return select_times(ds[[source_name]], times)[source_name]

  hour = np.timedelta64(1, "h")
  window_hours = int(step_hours)
  hourly_times = np.arange(
      times[0] - (window_hours - 1) * hour,
      times[-1] + hour,
      hour,
      dtype="datetime64[ns]",
  )
  logging.info(
      "Selecting hourly precipitation window: %s to %s (%d frames)",
      hourly_times[0],
      hourly_times[-1],
      len(hourly_times),
  )
  hourly = select_times(ds[[source_name]], hourly_times)[source_name]
  logging.info("Building %d-hour rolling precipitation accumulation", window_hours)
  return hourly.rolling(time=window_hours, min_periods=window_hours).sum().sel(time=times)


def build_graphcast_case_from_arco(
    arco: xr.Dataset,
    init_time: str,
    rollout_steps: int,
    step_hours: int,
    pressure_levels: tuple[int, ...],
) -> xr.Dataset:
  times = requested_datetimes(init_time, rollout_steps, step_hours)
  time_coord = np.arange(len(times)) * np.timedelta64(step_hours, "h")
  logging.info(
      "Building GraphCast case for init %s: %s to %s (%d frames)",
      init_time,
      times[0],
      times[-1],
      len(times),
  )
  logging.info("Requested pressure levels: %s", list(pressure_levels))
  arco = normalize_dims_and_coords(arco)
  logging.info("Normalized ARCO dimensions: %s", dict(arco.sizes))

  atmospheric_sources = {
      graphcast_name: find_variable(arco, aliases, graphcast_name)
      for graphcast_name, aliases in ATMOSPHERIC_ALIASES.items()
  }
  surface_sources = {
      graphcast_name: find_variable(arco, aliases, graphcast_name)
      for graphcast_name, aliases in SURFACE_ALIASES.items()
  }
  static_sources = {
      graphcast_name: find_variable(arco, aliases, graphcast_name)
      for graphcast_name, aliases in STATIC_ALIASES.items()
  }
  logging.info("ARCO atmospheric variable mapping: %s", atmospheric_sources)
  logging.info("ARCO surface variable mapping: %s", surface_sources)
  logging.info("ARCO static variable mapping: %s", static_sources)

  source_names = set(atmospheric_sources.values()) | set(surface_sources.values())
  source_names |= set(static_sources.values())
  logging.info("Selecting %d ARCO variables at %d target/input times",
               len(source_names), len(times))
  dynamic = select_times(arco[list(source_names)], times)
  if "level" in dynamic.coords:
    logging.info("Selecting %d pressure levels", len(pressure_levels))
    dynamic = dynamic.sel(level=list(pressure_levels))
    dynamic = dynamic.assign_coords(
        level=np.asarray(pressure_levels, dtype=np.int32))
  logging.info("Selected dynamic ARCO dimensions: %s", dict(dynamic.sizes))

  data_vars: dict[str, xr.DataArray] = {}
  for graphcast_name, source_name in atmospheric_sources.items():
    logging.info("Mapping ARCO pressure variable %s -> %s",
                 source_name, graphcast_name)
    data_vars[graphcast_name] = add_batch_and_time_coords(
        dynamic[source_name],
        time_coord,
        ("time", "lat", "lon", "level"),
    )

  for graphcast_name, source_name in surface_sources.items():
    logging.info("Mapping ARCO surface variable %s -> %s",
                 source_name, graphcast_name)
    data_vars[graphcast_name] = add_batch_and_time_coords(
        dynamic[source_name],
        time_coord,
        ("time", "lat", "lon"),
    )

  logging.info("Deriving total_precipitation_6hr from ARCO precipitation")
  data_vars["total_precipitation_6hr"] = add_batch_and_time_coords(
      precipitation_6hr(arco, times, step_hours),
      time_coord,
      ("time", "lat", "lon"),
  )

  for graphcast_name, source_name in static_sources.items():
    logging.info("Mapping ARCO static variable %s -> %s",
                 source_name, graphcast_name)
    static = dynamic[source_name]
    if "time" in static.dims:
      static = static.isel(time=0).drop_vars("time", errors="ignore")
    data_vars[graphcast_name] = static.transpose("lat", "lon").astype(np.float32)

  dataset = xr.Dataset(data_vars)
  dataset = dataset.assign_coords(
      batch=np.asarray([0], dtype=np.int32),
      time=time_coord,
      datetime=(("batch", "time"), times[None, :]),
      lat=dynamic["lat"].astype(np.float32),
      lon=dynamic["lon"].astype(np.float32),
      level=np.asarray(pressure_levels, dtype=np.int32),
  )
  logging.info("Constructed lazy GraphCast case dimensions: %s", dict(dataset.sizes))
  logging.info("Constructed GraphCast case variables: %s", ", ".join(sorted(dataset.data_vars)))
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
    time = netcdf.variables["time"]
    if "dtype" in time.ncattrs():
      time.delncattr("dtype")


def ensure_arco_graphcast_case(
    case_cache_dir: Path,
    init_time: str,
    rollout_steps: int,
    step_hours: int,
    pressure_levels: tuple[int, ...],
    *,
    arco_path: str = DEFAULT_ARCO_ERA5_PATH,
    output_path: Path | None = None,
    overwrite: bool = False,
    compression_level: int = 1,
) -> Path:
  output_path = output_path or default_case_path(
      case_cache_dir, init_time, rollout_steps, step_hours)
  output_path = output_path.expanduser()
  if output_path.exists() and not overwrite:
    logging.info("Reusing cached ARCO GraphCast case: %s", output_path)
    return output_path

  logging.info("ARCO GraphCast case cache path: %s", output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  arco = open_arco_era5(arco_path)
  with log_duration("Building lazy GraphCast case from ARCO selection"):
    dataset = build_graphcast_case_from_arco(
        arco,
        init_time,
        rollout_steps,
        step_hours,
        tuple(int(level) for level in pressure_levels),
    )
  logging.info("ARCO case dimensions: %s", dict(dataset.sizes))
  logging.info("This write triggers the actual ARCO data reads.")
  with log_duration(f"Writing ARCO GraphCast case NetCDF to {output_path}"):
    dataset.to_netcdf(
        output_path,
        engine="netcdf4",
        encoding=build_encoding(dataset, compression_level),
    )
  remove_decode_conflicting_time_attrs(output_path)
  logging.info("Finished cached ARCO GraphCast case: %s", output_path)
  return output_path
