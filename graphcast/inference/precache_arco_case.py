from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import xarray as xr

if __package__:
  from .arco_era5_graphcast_case import (
      DEFAULT_ARCO_ERA5_PATH,
      default_case_path,
      ensure_arco_graphcast_case,
  )
else:
  from arco_era5_graphcast_case import (
      DEFAULT_ARCO_ERA5_PATH,
      default_case_path,
      ensure_arco_graphcast_case,
  )


DEFAULT_PRESSURE_LEVELS = (
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
)
REQUIRED_VARIABLES = {
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "geopotential",
    "geopotential_at_surface",
    "land_sea_mask",
    "mean_sea_level_pressure",
    "specific_humidity",
    "temperature",
    "total_precipitation_6hr",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
}


def default_cache_dir() -> Path:
  repo_root = Path(__file__).resolve().parents[2]
  fallback = repo_root / "graphcast" / "results" / "spaghetti" / "arco_cases"
  return Path(os.environ.get("GRAPHCAST_ARCO_CASE_CACHE_DIR", fallback)).expanduser()


def parse_pressure_levels(value: str) -> tuple[int, ...]:
  levels = tuple(int(item.strip()) for item in value.split(",") if item.strip())
  if not levels:
    raise argparse.ArgumentTypeError("at least one pressure level is required")
  return levels


def validation_error(
    path: Path,
    rollout_steps: int,
    pressure_levels: tuple[int, ...],
) -> str | None:
  try:
    dataset = xr.open_dataset(path)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    return f"could not open NetCDF: {exc}"

  try:
    expected_time = rollout_steps + 2
    if dataset.sizes.get("time") != expected_time:
      return (
          f"time dimension is {dataset.sizes.get('time')}, "
          f"expected {expected_time}"
      )
    if dataset.sizes.get("level") != len(pressure_levels):
      return (
          f"level dimension is {dataset.sizes.get('level')}, "
          f"expected {len(pressure_levels)}"
      )
    missing = sorted(REQUIRED_VARIABLES - set(dataset.data_vars))
    if missing:
      return f"missing variables: {', '.join(missing)}"
  finally:
    dataset.close()

  return None


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Build one cached ARCO ERA5 GraphCast case NetCDF."
  )
  parser.add_argument("--init-time", required=True, help="Initialization time, e.g. 2023-01-01T06:00.")
  parser.add_argument("--rollout-steps", type=int, default=40)
  parser.add_argument("--step-hours", type=int, default=6)
  parser.add_argument("--case-cache-dir", type=Path, default=default_cache_dir())
  parser.add_argument(
      "--arco-era5-path",
      default=os.environ.get("GRAPHCAST_ARCO_ERA5_PATH", DEFAULT_ARCO_ERA5_PATH),
  )
  parser.add_argument(
      "--pressure-levels",
      type=parse_pressure_levels,
      default=DEFAULT_PRESSURE_LEVELS,
      help="Comma-separated pressure levels in hPa.",
  )
  parser.add_argument(
      "--compression-level",
      type=int,
      default=int(os.environ.get("GRAPHCAST_ARCO_COMPRESSION_LEVEL", "1")),
  )
  parser.add_argument("--overwrite", action="store_true")
  return parser.parse_args()


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
  args = parse_args()
  case_cache_dir = args.case_cache_dir.expanduser()
  case_path = default_case_path(
      case_cache_dir,
      args.init_time,
      args.rollout_steps,
      args.step_hours,
  )

  overwrite = args.overwrite
  if case_path.exists() and not overwrite:
    error = validation_error(case_path, args.rollout_steps, args.pressure_levels)
    if error is None:
      logging.info("Valid cached ARCO GraphCast case already exists: %s", case_path)
      print(case_path)
      return
    logging.warning("Cached ARCO case is invalid and will be rebuilt: %s (%s)", case_path, error)
    overwrite = True

  path = ensure_arco_graphcast_case(
      case_cache_dir,
      args.init_time,
      args.rollout_steps,
      args.step_hours,
      tuple(int(level) for level in args.pressure_levels),
      arco_path=args.arco_era5_path,
      overwrite=overwrite,
      compression_level=args.compression_level,
  )
  error = validation_error(path, args.rollout_steps, args.pressure_levels)
  if error is not None:
    raise RuntimeError(f"Cached ARCO case failed validation: {path} ({error})")
  print(path)


if __name__ == "__main__":
  main()
