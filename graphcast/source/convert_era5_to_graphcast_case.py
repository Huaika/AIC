#!/usr/bin/env python3
"""Convert raw CDS ERA5 files into one GraphCast-ready xarray Dataset.

The CDS retrieval script writes separate pressure-level and single-level ERA5
files. DeepMind's GraphCast data utilities expect one Dataset with GraphCast
variable names, a batch dimension, timedeltas on the time coordinate, and a
separate datetime coordinate for solar/progress forcings.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "graphcast" / "data" / "graphcast" / "dataset" / "pre_industrial"
DEFAULT_OUTPUT_NAME = "graphcast_1955_init_19550101_06.nc"

PRESSURE_LEVELS_13 = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

ATMOSPHERIC_RENAMES = {
    "t": "temperature",
    "z": "geopotential",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "q": "specific_humidity",
}

SURFACE_RENAMES = {
    "t2m": "2m_temperature",
    "msl": "mean_sea_level_pressure",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
}

STATIC_RENAMES = {
    "z": "geopotential_at_surface",
    "lsm": "land_sea_mask",
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Build a GraphCast-ready NetCDF case from raw CDS ERA5 files."
  )
  parser.add_argument(
      "--raw-dir",
      type=Path,
      default=DEFAULT_RAW_DIR,
      help="Directory containing the raw ERA5 pressure/single-level downloads.",
  )
  parser.add_argument(
      "--pressure-file",
      type=Path,
      help="Raw pressure-level ERA5 NetCDF file.",
  )
  parser.add_argument(
      "--single-level-file",
      type=Path,
      help=(
          "Raw single-level ERA5 download. This may be the CDS zip file even "
          "if it has a .nc suffix."
      ),
  )
  parser.add_argument(
      "--single-instant-file",
      type=Path,
      help="Extracted single-level instant NetCDF file.",
  )
  parser.add_argument(
      "--single-accum-file",
      type=Path,
      help="Extracted single-level accumulation NetCDF file containing tp.",
  )
  parser.add_argument(
      "--output",
      type=Path,
      help="Output GraphCast-ready NetCDF path.",
  )
  parser.add_argument(
      "--init-time",
      default="1955-01-01T06:00",
      help="Forecast initialization time. The first input is init-time - 6h.",
  )
  parser.add_argument(
      "--rollout-steps",
      type=int,
      default=40,
      help="Number of 6-hour target frames to include after initialization.",
  )
  parser.add_argument(
      "--step-hours",
      type=int,
      default=6,
      help="Cadence of the ERA5 files and GraphCast rollout.",
  )
  parser.add_argument(
      "--compression-level",
      type=int,
      default=1,
      choices=range(0, 10),
      metavar="[0-9]",
      help="NetCDF zlib compression level. Use 0 for uncompressed output.",
  )
  parser.add_argument(
      "--overwrite",
      action="store_true",
      help="Replace the output file if it already exists.",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Open inputs and print the planned output without writing it.",
  )
  parser.add_argument(
      "--log-level",
      default="INFO",
      help="Python logging level.",
  )
  return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None, Path]:
  raw_dir = args.raw_dir.expanduser().resolve()
  pressure_file = (
      args.pressure_file.expanduser().resolve()
      if args.pressure_file
      else raw_dir / "era5_pressure_levels_graphcast_raw.nc"
  )
  output = (
      args.output.expanduser().resolve()
      if args.output
      else raw_dir / DEFAULT_OUTPUT_NAME
  )

  if args.single_instant_file:
    instant_file = args.single_instant_file.expanduser().resolve()
  else:
    instant_file = raw_dir / "single_levels_unzipped" / "data_stream-oper_stepType-instant.nc"

  if args.single_accum_file:
    accum_file = args.single_accum_file.expanduser().resolve()
  else:
    accum_file = raw_dir / "single_levels_unzipped" / "data_stream-oper_stepType-accum.nc"

  single_level_file = (
      args.single_level_file.expanduser().resolve()
      if args.single_level_file
      else raw_dir / "era5_single_levels_graphcast_raw.nc"
  )

  if not instant_file.exists() or not accum_file.exists():
    instant_file, accum_file = extract_single_level_zip(single_level_file, raw_dir)

  return pressure_file, instant_file, accum_file, output


def extract_single_level_zip(single_level_file: Path, raw_dir: Path) -> tuple[Path, Path | None]:
  if not single_level_file.exists():
    raise FileNotFoundError(
        "Could not find extracted single-level files or raw single-level file: "
        f"{single_level_file}"
    )

  if not zipfile.is_zipfile(single_level_file):
    return single_level_file, None

  extract_dir = raw_dir / "single_levels_unzipped"
  extract_dir.mkdir(parents=True, exist_ok=True)
  logging.info("Extracting %s into %s", single_level_file, extract_dir)
  with zipfile.ZipFile(single_level_file) as archive:
    archive.extractall(extract_dir)

  instant_matches = sorted(extract_dir.glob("*instant*.nc"))
  accum_matches = sorted(extract_dir.glob("*accum*.nc"))
  if not instant_matches:
    raise FileNotFoundError(f"No instant NetCDF found after extracting {single_level_file}")
  return instant_matches[0], accum_matches[0] if accum_matches else None


def requested_datetimes(
    init_time: str,
    rollout_steps: int,
    step_hours: int,
) -> np.ndarray:
  init = np.datetime64(pd.Timestamp(init_time).to_datetime64(), "ns")
  step = np.timedelta64(step_hours, "h")
  n_times = rollout_steps + 2
  return init - step + np.arange(n_times) * step


def open_era5(path: Path) -> xr.Dataset:
  if not path.exists():
    raise FileNotFoundError(path)
  logging.info("Opening %s", path)
  try:
    import dask.array  # pylint: disable=unused-import,import-outside-toplevel
  except ImportError:
    logging.warning("dask is not installed; xarray will use the default backend arrays.")
    return xr.open_dataset(path)
  return xr.open_dataset(path, chunks={})


def rename_dims_and_subset(ds: xr.Dataset, times: np.ndarray) -> xr.Dataset:
  missing = sorted(set(times) - set(ds["valid_time"].values.astype("datetime64[ns]")))
  if missing:
    preview = ", ".join(str(t) for t in missing[:5])
    raise ValueError(f"{len(missing)} requested times are missing. First missing: {preview}")

  ds = ds.sel(valid_time=times)
  rename = {
      "valid_time": "time",
      "latitude": "lat",
      "longitude": "lon",
      "pressure_level": "level",
  }
  ds = ds.rename({old: new for old, new in rename.items() if old in ds.dims or old in ds.coords})
  ds = ds.drop_vars([name for name in ("number", "expver") if name in ds], errors="ignore")

  if "lat" in ds.coords and ds["lat"].values[0] > ds["lat"].values[-1]:
    ds = ds.isel(lat=slice(None, None, -1))
  if "lon" in ds.coords and np.any(ds["lon"].values < 0):
    ds = ds.assign_coords(lon=np.mod(ds["lon"], 360.0)).sortby("lon")
  return ds


def add_batch_and_time_coords(
    data_array: xr.DataArray,
    time_coord: np.ndarray,
    dims: tuple[str, ...],
) -> xr.DataArray:
  data_array = data_array.transpose(*dims)
  data_array = data_array.assign_coords(time=time_coord)
  data_array = data_array.expand_dims(batch=[0], axis=0)
  return data_array.transpose("batch", *dims)


def build_graphcast_dataset(
    pressure: xr.Dataset,
    instant: xr.Dataset,
    accum: xr.Dataset | None,
    times: np.ndarray,
    step_hours: int,
) -> xr.Dataset:
  n_times = len(times)
  time_coord = np.arange(n_times) * np.timedelta64(step_hours, "h")

  pressure = rename_dims_and_subset(pressure, times)
  pressure = pressure.sel(level=list(PRESSURE_LEVELS_13))
  pressure = pressure.assign_coords(level=np.asarray(PRESSURE_LEVELS_13, dtype=np.int32))

  instant = rename_dims_and_subset(instant, times)
  if accum is not None:
    accum = rename_dims_and_subset(accum, times)

  data_vars: dict[str, xr.DataArray] = {}

  for source_name, graphcast_name in ATMOSPHERIC_RENAMES.items():
    if source_name not in pressure:
      raise KeyError(f"Missing pressure-level variable {source_name!r}")
    logging.info("Mapping pressure variable %s -> %s", source_name, graphcast_name)
    data_vars[graphcast_name] = add_batch_and_time_coords(
        pressure[source_name],
        time_coord,
        ("time", "lat", "lon", "level"),
    )

  for source_name, graphcast_name in SURFACE_RENAMES.items():
    if source_name not in instant:
      raise KeyError(f"Missing single-level variable {source_name!r}")
    logging.info("Mapping surface variable %s -> %s", source_name, graphcast_name)
    data_vars[graphcast_name] = add_batch_and_time_coords(
        instant[source_name],
        time_coord,
        ("time", "lat", "lon"),
    )

  if accum is not None and "tp" in accum:
    logging.info("Mapping accumulation variable tp -> total_precipitation_6hr")
    precipitation = accum["tp"]
  else:
    logging.warning("No tp accumulation file found; writing zero precipitation target.")
    precipitation = xr.zeros_like(instant["t2m"], dtype=np.float32)
  data_vars["total_precipitation_6hr"] = add_batch_and_time_coords(
      precipitation,
      time_coord,
      ("time", "lat", "lon"),
  )

  for source_name, graphcast_name in STATIC_RENAMES.items():
    if source_name not in instant:
      raise KeyError(f"Missing static source variable {source_name!r}")
    logging.info("Mapping static variable %s -> %s", source_name, graphcast_name)
    data_vars[graphcast_name] = (
        instant[source_name]
        .isel(time=0)
        .drop_vars("time", errors="ignore")
        .transpose("lat", "lon")
    )

  dataset = xr.Dataset(data_vars)
  dataset = dataset.assign_coords(
      batch=np.asarray([0], dtype=np.int32),
      time=time_coord,
      datetime=(("batch", "time"), times[None, :]),
      lat=instant["lat"].astype(np.float32),
      lon=instant["lon"].astype(np.float32),
      level=np.asarray(PRESSURE_LEVELS_13, dtype=np.int32),
  )
  return dataset


def build_encoding(dataset: xr.Dataset, compression_level: int) -> dict[str, dict[str, object]]:
  encoding: dict[str, dict[str, object]] = {}
  for name, variable in dataset.data_vars.items():
    if not np.issubdtype(variable.dtype, np.floating):
      continue
    variable_encoding: dict[str, object] = {"dtype": "float32"}
    if compression_level > 0:
      variable_encoding.update({"zlib": True, "complevel": compression_level, "shuffle": True})
      if variable.dims == ("batch", "time", "lat", "lon", "level"):
        variable_encoding["chunksizes"] = (1, 1, variable.sizes["lat"], variable.sizes["lon"], 1)
      elif variable.dims == ("batch", "time", "lat", "lon"):
        variable_encoding["chunksizes"] = (1, 1, variable.sizes["lat"], variable.sizes["lon"])
      elif variable.dims == ("lat", "lon"):
        variable_encoding["chunksizes"] = (variable.sizes["lat"], variable.sizes["lon"])
    encoding[name] = variable_encoding
  return encoding


def main() -> None:
  args = parse_args()
  logging.basicConfig(
      level=getattr(logging, args.log_level.upper()),
      format="%(asctime)s %(levelname)s %(message)s",
  )

  pressure_file, instant_file, accum_file, output = resolve_paths(args)
  if output.exists() and not args.overwrite and not args.dry_run:
    raise FileExistsError(f"{output} already exists. Pass --overwrite to replace it.")

  times = requested_datetimes(args.init_time, args.rollout_steps, args.step_hours)
  logging.info("GraphCast case window: %s to %s", times[0], times[-1])
  logging.info("Frames: %d (%d inputs + %d targets)", len(times), 2, args.rollout_steps)

  pressure = open_era5(pressure_file)
  instant = open_era5(instant_file)
  accum = open_era5(accum_file) if accum_file is not None and accum_file.exists() else None
  dataset = build_graphcast_dataset(pressure, instant, accum, times, args.step_hours)

  logging.info("Output dimensions: %s", dict(dataset.sizes))
  logging.info("Output variables: %s", ", ".join(sorted(dataset.data_vars)))
  if args.dry_run:
    print(dataset)
    return

  output.parent.mkdir(parents=True, exist_ok=True)
  logging.info("Writing %s", output)
  dataset.to_netcdf(
      output,
      engine="netcdf4",
      encoding=build_encoding(dataset, args.compression_level),
  )
  logging.info("Done: %s", output)


if __name__ == "__main__":
  main()
