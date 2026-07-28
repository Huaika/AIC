#!/usr/bin/env python3
"""Retrieve ERA5 fields needed to build a GraphCast case.

This downloads the raw ERA5 pressure-level and single-level NetCDF files for one
GraphCast forecast window. It does not yet convert them into the final GraphCast
xarray layout; it creates the two raw inputs used by that conversion step.

Example:

  python retrieve_era5_graphcast_case.py \
      --init-time 1955-01-01T06:00 \
      --output-dir /scratch/$USER/era5_graphcast_1955

For a 10-day rollout, the script requests 42 six-hourly analysis times:
t-6h, t, and t+6h ... t+240h.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path


PRESSURE_LEVEL_VARIABLES = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]

GRAPHCAST_13_PRESSURE_LEVELS = [
    "50",
    "100",
    "150",
    "200",
    "250",
    "300",
    "400",
    "500",
    "600",
    "700",
    "850",
    "925",
    "1000",
]

SINGLE_LEVEL_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "geopotential",
    "land_sea_mask",
    "mean_sea_level_pressure",
    "total_precipitation",
]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Download raw ERA5 fields for a GraphCast forecast case."
  )
  parser.add_argument(
      "--init-time",
      required=True,
      help=(
          "Forecast initialization time, e.g. 1955-01-01T06:00. The request "
          "starts 6 hours before this time."
      ),
  )
  parser.add_argument(
      "--lead-days",
      type=int,
      default=10,
      help="Forecast lead time in days. Default: 10.",
  )
  parser.add_argument(
      "--step-hours",
      type=int,
      default=6,
      help="Temporal spacing to request. GraphCast uses 6 hours.",
  )
  parser.add_argument(
      "--output-dir",
      type=Path,
      required=True,
      help="Directory for downloaded ERA5 NetCDF files.",
  )
  parser.add_argument(
      "--pressure-output",
      default="era5_pressure_levels_graphcast_raw.nc",
      help="Output filename for pressure-level data.",
  )
  parser.add_argument(
      "--single-output",
      default="era5_single_levels_graphcast_raw.nc",
      help="Output filename for single-level data.",
  )
  parser.add_argument(
      "--data-format",
      choices=("netcdf", "grib"),
      default="netcdf",
      help="CDS data_format. NetCDF is easiest for xarray conversion.",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Print the CDS requests without downloading.",
  )
  return parser.parse_args()


def parse_init_time(value: str) -> datetime:
  return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def requested_datetimes(
    init_time: datetime, lead_days: int, step_hours: int
) -> list[datetime]:
  first_time = init_time - timedelta(hours=step_hours)
  num_future_steps = lead_days * 24 // step_hours
  num_total_steps = num_future_steps + 2
  return [
      first_time + i * timedelta(hours=step_hours)
      for i in range(num_total_steps)
  ]


def cds_date_fields(times: list[datetime]) -> dict[str, list[str]]:
  return {
      "year": sorted({f"{t.year:04d}" for t in times}),
      "month": sorted({f"{t.month:02d}" for t in times}),
      "day": sorted({f"{t.day:02d}" for t in times}),
      "time": sorted({f"{t.hour:02d}:00" for t in times}),
  }


def pressure_level_request(times: list[datetime], data_format: str) -> dict:
  request = {
      "product_type": ["reanalysis"],
      "variable": PRESSURE_LEVEL_VARIABLES,
      "pressure_level": GRAPHCAST_13_PRESSURE_LEVELS,
      **cds_date_fields(times),
      "data_format": data_format,
      "download_format": "unarchived",
  }
  return request


def single_level_request(times: list[datetime], data_format: str) -> dict:
  request = {
      "product_type": ["reanalysis"],
      "variable": SINGLE_LEVEL_VARIABLES,
      **cds_date_fields(times),
      "data_format": data_format,
      "download_format": "unarchived",
  }
  return request


def download(dataset: str, request: dict, target: Path) -> None:
  if target.exists():
    print(f"Skipping existing file: {target}")
    return
  print(f"Retrieving {dataset} -> {target}")
  import cdsapi  # pylint: disable=import-outside-toplevel

  client = cdsapi.Client()
  client.retrieve(dataset, request).download(str(target))


def main() -> None:
  args = parse_args()
  init_time = parse_init_time(args.init_time)
  times = requested_datetimes(init_time, args.lead_days, args.step_hours)
  args.output_dir.mkdir(parents=True, exist_ok=True)

  pressure_target = args.output_dir / args.pressure_output
  single_target = args.output_dir / args.single_output

  print(f"Initialization time: {init_time:%Y-%m-%dT%H:%M}")
  print(f"Requested window: {times[0]:%Y-%m-%dT%H:%M} to {times[-1]:%Y-%m-%dT%H:%M}")
  print(f"Number of times: {len(times)}")

  pressure_request = pressure_level_request(times, args.data_format)
  single_request = single_level_request(times, args.data_format)

  if args.dry_run:
    print("\nPressure-level request:")
    print(json.dumps(pressure_request, indent=2))
    print("\nSingle-level request:")
    print(json.dumps(single_request, indent=2))
    return

  download(
      "reanalysis-era5-pressure-levels",
      pressure_request,
      pressure_target,
  )
  download(
      "reanalysis-era5-single-levels",
      single_request,
      single_target,
  )


if __name__ == "__main__":
  main()
