from __future__ import annotations

import dataclasses
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

import haiku as hk  # pylint: disable=import-outside-toplevel
import jax  # pylint: disable=import-outside-toplevel
import pandas as pd  # pylint: disable=import-outside-toplevel
from google.api_core import exceptions as google_exceptions  # pylint: disable=import-outside-toplevel
from google.cloud import storage  # pylint: disable=import-outside-toplevel
from graphcast import autoregressive  # pylint: disable=import-outside-toplevel
from graphcast import casting  # pylint: disable=import-outside-toplevel
from graphcast import checkpoint  # pylint: disable=import-outside-toplevel
from graphcast import data_utils  # pylint: disable=import-outside-toplevel
from graphcast import graphcast  # pylint: disable=import-outside-toplevel
from graphcast import normalization  # pylint: disable=import-outside-toplevel
from graphcast import rollout  # pylint: disable=import-outside-toplevel,unused-import
from graphcast import xarray_jax  # pylint: disable=import-outside-toplevel,unused-import

if __package__:
  from .metrics import (
      compute_global_mean_evolution,
      compute_metric_summary,
      select_metric_field,
  )
  from .plots import plot_global_mean_evolution, plot_metrics
else:
  from metrics import (
      compute_global_mean_evolution,
      compute_metric_summary,
      select_metric_field,
  )
  from plots import plot_global_mean_evolution, plot_metrics


GCS_BUCKET = "dm_graphcast"
GRAPHCAST_PREFIX = "graphcast/"
PARAMS_PREFIX = GRAPHCAST_PREFIX + "params/"
STATS_PREFIX = GRAPHCAST_PREFIX + "stats/"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_ERA5 = (
    REPO_ROOT
    / "graphcast"
    / "data"
    / "graphcast"
    / "dataset"
    / "pre_industrial"
    / "graphcast_1955_init_19550101_06.nc"
)
DEFAULT_ROLLOUT_STEPS = 40
DEFAULT_STEP_HOURS = 6
DEFAULT_SEED = 1
METRICS_FREQUENCY = os.environ.get("GRAPHCAST_METRICS_FREQUENCY", "daily")
REFERENCE_MODE = os.environ.get("GRAPHCAST_REFERENCE_MODE", "auto")
METRIC_VARIABLE = os.environ.get("GRAPHCAST_METRIC_VARIABLE", "2m_temperature")
METRIC_LEVEL = os.environ.get("GRAPHCAST_METRIC_LEVEL")
METRIC_LEVEL = int(METRIC_LEVEL) if METRIC_LEVEL else None
METRIC_ID = (
    f"{METRIC_VARIABLE}_{METRIC_LEVEL}hPa"
    if METRIC_LEVEL is not None
    else METRIC_VARIABLE
).replace(" ", "_")
OUTPUT_CSV = Path(os.environ.get("GRAPHCAST_OUTPUT_CSV", f"{METRIC_ID}_metrics.csv"))
PLOT_PATH = Path(os.environ.get("GRAPHCAST_PLOT_PATH", f"{METRIC_ID}_metrics.png"))
EVOLUTION_CSV = Path(
    os.environ.get("GRAPHCAST_EVOLUTION_CSV", f"{METRIC_ID}_global_mean_evolution.csv")
)
EVOLUTION_PLOT_PATH = Path(
    os.environ.get(
        "GRAPHCAST_EVOLUTION_PLOT_PATH",
        f"{METRIC_ID}_global_mean_evolution.png",
    )
)
WRITE_PLOT = os.environ.get("GRAPHCAST_NO_PLOT", "").lower() not in {
    "1",
    "true",
    "yes",
}

DEFAULT_OPERATIONAL_CHECKPOINT = (
    "GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - "
    "pressure levels 13 - mesh 2to6 - precipitation output only.npz"
)

CHECKPOINT_CANDIDATES = {
    "operational": (
        DEFAULT_OPERATIONAL_CHECKPOINT,
        "GraphCast_operational - ERA5-HRES 1979-2021-reso 0.25 - "
        "precipitation input and output.npz",
    ),
    "small": (
        "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - "
        "pressure levels 13 - mesh 2to5 - precipitation input and output.npz",
    ),
    "era5_0p25": (
        "GraphCast - ERA5 1979-2017 - resolution 0.25 - "
        "pressure levels 37 - mesh 2to6 - precipitation input and output.npz",
    ),
}

STATS_FILES = (
    "diffs_stddev_by_level.nc",
    "mean_by_level.nc",
    "stddev_by_level.nc",
)
DERIVED_FORCING_VARIABLES = {
    "year_progress",
    "year_progress_sin",
    "year_progress_cos",
    "day_progress",
    "day_progress_sin",
    "day_progress_cos",
}
TISR = "toa_incident_solar_radiation"
TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def env_flag(name: str, default: bool = False) -> bool:
  value = os.environ.get(name)
  if value is None:
    return default
  return value.lower() in TRUE_ENV_VALUES

def anonymous_bucket(storage_module: Any) -> Any:
  """Return the public GraphCast Cloud Storage bucket using anonymous access."""
  client = storage_module.Client.create_anonymous_client()
  return client.bucket(GCS_BUCKET)

def cache_blob(bucket: Any, object_name: str, cache_dir: Path) -> Path:
  """Download a GCS object into the local cache if needed and return its path."""
  local_path = cache_dir / object_name
  local_path.parent.mkdir(parents=True, exist_ok=True)
  if local_path.exists():
    logging.info("Using cached gs://%s/%s", GCS_BUCKET, object_name)
    return local_path

  logging.info("Downloading gs://%s/%s", GCS_BUCKET, object_name)
  bucket.blob(object_name).download_to_filename(local_path)
  return local_path


def list_gcs_names(bucket: Any, prefix: str) -> list[str]:
  names = []
  for blob in bucket.list_blobs(prefix=prefix):
    name = blob.name.removeprefix(prefix)
    if name:
      names.append(name)
  return sorted(names)


def checkpoint_matches_alias(name: str, alias: str) -> bool:
  lowered = name.lower()
  if alias == "operational":
    return (
        "graphcast_operational" in lowered
        and "era5-hres" in lowered
        and ("0.25" in lowered or "0p25" in lowered)
    )
  if alias == "small":
    return "graphcast_small" in lowered and ("1.0" in lowered or "1p0" in lowered)
  if alias == "era5_0p25":
    return (
        lowered.startswith("graphcast - era5")
        and ("0.25" in lowered or "0p25" in lowered)
        and "37" in lowered
    )
  raise ValueError(f"Unknown checkpoint alias: {alias}")


def resolve_checkpoint_name(bucket: Any, model_alias: str, checkpoint_name: str | None) -> str:
  if checkpoint_name:
    return checkpoint_name

  candidates = CHECKPOINT_CANDIDATES[model_alias]
  available = list_gcs_names(bucket, PARAMS_PREFIX)
  for candidate in candidates:
    if candidate in available:
      return candidate

  matches = [name for name in available if checkpoint_matches_alias(name, model_alias)]
  if matches:
    logging.info("Resolved model %s to checkpoint %s", model_alias, matches[0])
    return matches[0]

  msg = "\n".join(available) if available else "(no checkpoint names listed)"
  raise FileNotFoundError(
      f"Could not resolve checkpoint for model {model_alias}. "
      f"Available checkpoints under gs://{GCS_BUCKET}/{PARAMS_PREFIX}:\n{msg}"
  )

def load_checkpoint(
    bucket: Any,
    checkpoint_module: Any,
    graphcast_module: Any,
    cache_dir: Path,
    model_alias: str,
    checkpoint_name: str | None,
) -> tuple[Any, str]:
  """Load a GraphCast checkpoint from cache or the public GCS bucket."""
  candidates = (checkpoint_name,) if checkpoint_name else CHECKPOINT_CANDIDATES[model_alias]
  missing: list[str] = []
  last_not_found: google_exceptions.NotFound | None = None
  for resolved_name in candidates:
    try:
      path = cache_blob(bucket, PARAMS_PREFIX + resolved_name, cache_dir)
    except google_exceptions.NotFound as exc:
      logging.warning("Checkpoint not found, trying next candidate: %s", resolved_name)
      missing.append(resolved_name)
      last_not_found = exc
      continue

    with path.open("rb") as f:
      ckpt = checkpoint_module.load(f, graphcast_module.CheckPoint)

    logging.info("Loaded checkpoint: %s", resolved_name)
    logging.info("Model description: %s", ckpt.description)
    return ckpt, resolved_name

  raise FileNotFoundError(
      "Could not find a usable checkpoint. Tried:\n" + "\n".join(missing)
  ) from last_not_found


def load_normalization_stats(bucket: Any, cache_dir: Path) -> dict[str, xr.Dataset]:
  """Load GraphCast normalization statistics from cache or public GCS."""
  stats = {}
  for file_name in STATS_FILES:
    path = cache_blob(bucket, STATS_PREFIX + file_name, cache_dir)
    stats[file_name.removesuffix(".nc")] = xr.load_dataset(path).compute()
  return stats


def reference_era5_uri() -> str:
  """Return the configured GraphCast-ready ERA5 reference dataset path/URI."""
  return (
      os.environ.get("GRAPHCAST_REFERENCE_ERA5")
      or os.environ.get("GRAPHCAST_DATASET")
      or str(DEFAULT_REFERENCE_ERA5)
  )


def configured_reference_era5_uri(
    cache_dir: Path,
    task_config: Any,
    rollout_steps: int,
    step_hours: int,
    init_time: str | None = None,
) -> str:
  if init_time is None:
    init_time = os.environ.get("GRAPHCAST_INIT_TIME")
  if env_flag("GRAPHCAST_USE_NEXTGEMS"):
    if not init_time:
      raise ValueError(
          "GRAPHCAST_USE_NEXTGEMS is enabled, but GRAPHCAST_INIT_TIME is unset."
      )

    from nextgems_graphcast_case import (  # pylint: disable=import-outside-toplevel
        ensure_nextgems_graphcast_case,
    )

    case_cache_dir = Path(
        os.environ.get(
            "GRAPHCAST_NEXTGEMS_CASE_CACHE_DIR",
            str(cache_dir / "nextgems_cases"),
        )
    ).expanduser()
    output_path = os.environ.get("GRAPHCAST_NEXTGEMS_CASE_PATH")
    root = os.environ.get("GRAPHCAST_NEXTGEMS_ROOT")
    case_path = ensure_nextgems_graphcast_case(
        case_cache_dir,
        init_time,
        rollout_steps,
        step_hours,
        tuple(int(level) for level in task_config.pressure_levels),
        year=int(os.environ.get("GRAPHCAST_NEXTGEMS_YEAR", "2049")),
        root=Path(root).expanduser() if root else None,
        output_path=Path(output_path).expanduser() if output_path else None,
        overwrite=env_flag("GRAPHCAST_NEXTGEMS_OVERWRITE"),
        compression_level=int(os.environ.get("GRAPHCAST_NEXTGEMS_COMPRESSION_LEVEL", "1")),
    )
    return str(case_path)

  if not env_flag("GRAPHCAST_USE_ARCO", default=bool(init_time)):
    return reference_era5_uri()
  if not init_time:
    raise ValueError("GRAPHCAST_USE_ARCO is enabled, but GRAPHCAST_INIT_TIME is unset.")

  from arco_era5_graphcast_case import (  # pylint: disable=import-outside-toplevel
      DEFAULT_ARCO_ERA5_PATH,
      ensure_arco_graphcast_case,
  )

  case_cache_dir = Path(
      os.environ.get("GRAPHCAST_ARCO_CASE_CACHE_DIR", str(cache_dir / "arco_cases"))
  ).expanduser()
  output_path = os.environ.get("GRAPHCAST_ARCO_CASE_PATH")
  case_path = ensure_arco_graphcast_case(
      case_cache_dir,
      init_time,
      rollout_steps,
      step_hours,
      tuple(int(level) for level in task_config.pressure_levels),
      arco_path=os.environ.get("GRAPHCAST_ARCO_ERA5_PATH", DEFAULT_ARCO_ERA5_PATH),
      output_path=Path(output_path).expanduser() if output_path else None,
      overwrite=env_flag("GRAPHCAST_ARCO_OVERWRITE"),
      compression_level=int(os.environ.get("GRAPHCAST_ARCO_COMPRESSION_LEVEL", "1")),
  )
  return str(case_path)


def open_reference_era5(
    bucket: Any,
    cache_dir: Path,
    reference_uri: str | None = None,
) -> tuple[xr.Dataset, str]:
  """Open the GraphCast-ready ERA5 reference dataset without eagerly loading it."""
  uri = reference_uri or reference_era5_uri()
  if uri.startswith("gs://"):
    bucket_name, object_name = uri.removeprefix("gs://").split("/", 1)
    if bucket_name != GCS_BUCKET:
      raise ValueError(f"Expected gs://{GCS_BUCKET}/..., got {uri!r}")
    path = cache_blob(bucket, object_name, cache_dir)
    label = uri
  else:
    path = Path(uri).expanduser()
    label = str(path)

  try:
    import dask.array  # pylint: disable=unused-import,import-outside-toplevel
  except ImportError:
    dataset = xr.open_dataset(path, decode_timedelta=True)
  else:
    dataset = xr.open_dataset(path, decode_timedelta=True, chunks={})
  return dataset, label


def relative_input_times(input_duration: Any, step_hours: int) -> np.ndarray:
  step = pd.Timedelta(hours=step_hours)
  input_steps = int(round(pd.Timedelta(input_duration) / step))
  if input_steps < 1:
    raise ValueError(f"Input duration {input_duration!r} is shorter than one step.")
  offsets = np.arange(-(input_steps - 1), 1, dtype=np.int32)
  return offsets * np.timedelta64(step_hours, "h")


def relative_target_times(rollout_steps: int, step_hours: int) -> np.ndarray:
  return np.arange(1, rollout_steps + 1, dtype=np.int32) * np.timedelta64(
      step_hours, "h"
  )


def lazy_template_array(
    dims: tuple[str, ...],
    coords: dict[str, Any],
    chunks: tuple[int, ...],
) -> xr.DataArray:
  try:
    import dask.array as da  # pylint: disable=import-outside-toplevel
  except ImportError as exc:
    raise RuntimeError(
        "GRAPHCAST_ARCO_DIRECT requires dask so target templates can stay lazy."
    ) from exc

  shape = tuple(len(coords[dim]) for dim in dims)
  data = da.full(shape, np.nan, dtype=np.float32, chunks=chunks)
  return xr.DataArray(data, dims=dims, coords={dim: coords[dim] for dim in dims})


def build_direct_targets_template(
    target_variables: tuple[str, ...],
    pressure_variables: set[str],
    surface_variables: set[str],
    lat: np.ndarray,
    lon: np.ndarray,
    levels: np.ndarray,
    target_times: np.ndarray,
) -> xr.Dataset:
  coords: dict[str, Any] = {
      "batch": np.asarray([0], dtype=np.int32),
      "time": target_times,
      "lat": lat,
      "lon": lon,
      "level": levels,
  }
  data_vars: dict[str, xr.DataArray] = {}
  for variable in target_variables:
    if variable in pressure_variables:
      dims = ("batch", "time", "lat", "lon", "level")
      chunks = (1, 1, len(lat), len(lon), 1)
    elif variable in surface_variables or variable == "total_precipitation_6hr":
      dims = ("batch", "time", "lat", "lon")
      chunks = (1, 1, len(lat), len(lon))
    else:
      raise KeyError(f"Cannot build ARCO direct target template for {variable!r}.")
    data_vars[variable] = lazy_template_array(dims, coords, chunks)
  return xr.Dataset(data_vars)


def read_direct_arco_era5(
    task_config: Any,
    rollout_steps: int,
    step_hours: int,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset, str]:
  """Build inputs/templates/truth directly from ARCO without writing a case file."""
  init_time = os.environ.get("GRAPHCAST_INIT_TIME")
  if not init_time:
    raise ValueError("GRAPHCAST_ARCO_DIRECT requires GRAPHCAST_INIT_TIME.")

  from arco_era5_graphcast_case import (  # pylint: disable=import-outside-toplevel
      ATMOSPHERIC_ALIASES,
      DEFAULT_ARCO_ERA5_PATH,
      STATIC_ALIASES,
      SURFACE_ALIASES,
      add_batch_and_time_coords,
      find_variable,
      normalize_dims_and_coords,
      open_arco_era5,
      precipitation_6hr,
      select_times,
  )

  init_datetime = np.datetime64(init_time, "ns")
  input_time = relative_input_times(task_config.input_duration, step_hours)
  target_time = relative_target_times(rollout_steps, step_hours)
  input_datetimes = init_datetime + input_time.astype("timedelta64[ns]")
  target_datetimes = init_datetime + target_time.astype("timedelta64[ns]")
  pressure_levels = np.asarray(
      tuple(int(level) for level in task_config.pressure_levels), dtype=np.int32)
  pressure_variables = set(ATMOSPHERIC_ALIASES)
  surface_variables = set(SURFACE_ALIASES)
  static_variables = set(STATIC_ALIASES)

  logging.info("Using direct ARCO streaming mode; no per-date NetCDF will be written.")
  logging.info(
      "Direct ARCO init %s: %d input frames, %d target frames",
      init_time,
      len(input_datetimes),
      len(target_datetimes),
  )
  arco = normalize_dims_and_coords(
      open_arco_era5(os.environ.get("GRAPHCAST_ARCO_ERA5_PATH", DEFAULT_ARCO_ERA5_PATH))
  )
  lat = np.asarray(arco["lat"].values, dtype=np.float32)
  lon = np.asarray(arco["lon"].values, dtype=np.float32)

  atmospheric_sources = {
      name: find_variable(arco, aliases, name)
      for name, aliases in ATMOSPHERIC_ALIASES.items()
  }
  surface_sources = {
      name: find_variable(arco, aliases, name)
      for name, aliases in SURFACE_ALIASES.items()
  }
  static_sources = {
      name: find_variable(arco, aliases, name)
      for name, aliases in STATIC_ALIASES.items()
  }

  input_variables = set(task_config.input_variables)
  input_source_names = {
      atmospheric_sources[name]
      for name in input_variables & pressure_variables
  } | {
      surface_sources[name]
      for name in input_variables & surface_variables
  }
  static_input_names = input_variables & static_variables

  logging.info("Selecting direct ARCO input variables: %s", sorted(input_variables))
  dynamic_inputs = select_times(arco[list(input_source_names)], input_datetimes)
  if "level" in dynamic_inputs.coords:
    dynamic_inputs = dynamic_inputs.sel(level=pressure_levels.tolist())
    dynamic_inputs = dynamic_inputs.assign_coords(level=pressure_levels)

  input_data_vars: dict[str, xr.DataArray] = {}
  for name in input_variables & pressure_variables:
    logging.info("Direct ARCO input pressure variable: %s", name)
    input_data_vars[name] = add_batch_and_time_coords(
        dynamic_inputs[atmospheric_sources[name]],
        input_time,
        ("time", "lat", "lon", "level"),
    )
  for name in input_variables & surface_variables:
    logging.info("Direct ARCO input surface variable: %s", name)
    input_data_vars[name] = add_batch_and_time_coords(
        dynamic_inputs[surface_sources[name]],
        input_time,
        ("time", "lat", "lon"),
    )
  if "total_precipitation_6hr" in input_variables:
    logging.info("Direct ARCO input precipitation variable: total_precipitation_6hr")
    input_data_vars["total_precipitation_6hr"] = add_batch_and_time_coords(
        precipitation_6hr(arco, input_datetimes, step_hours),
        input_time,
        ("time", "lat", "lon"),
    )
  for name in static_input_names:
    logging.info("Direct ARCO static input variable: %s", name)
    static = arco[static_sources[name]]
    if "time" in static.dims:
      static = select_times(arco[[static_sources[name]]], input_datetimes)[
          static_sources[name]].isel(time=0).drop_vars("time", errors="ignore")
    input_data_vars[name] = static.transpose("lat", "lon").astype(np.float32)

  inputs = xr.Dataset(input_data_vars).assign_coords(
      batch=np.asarray([0], dtype=np.int32),
      time=input_time,
      datetime=(("batch", "time"), input_datetimes[None, :]),
      lat=lat,
      lon=lon,
      level=pressure_levels,
  )
  if input_variables & DERIVED_FORCING_VARIABLES:
    data_utils.add_derived_vars(inputs)
  if TISR in input_variables:
    data_utils.add_tisr_var(inputs)
  inputs = inputs.drop_vars("datetime", errors="ignore")
  inputs = inputs[list(task_config.input_variables)]

  targets_template = build_direct_targets_template(
      tuple(task_config.target_variables),
      pressure_variables,
      surface_variables,
      lat,
      lon,
      pressure_levels,
      target_time,
  )
  reference_context = xr.Dataset(coords={
      "batch": np.asarray([0], dtype=np.int32),
      "time": input_time,
      "datetime": (("batch", "time"), input_datetimes[None, :]),
      "lat": lat,
      "lon": lon,
  })
  forcings = make_forcings(
      reference_context,
      inputs,
      targets_template.coords["time"],
      task_config.forcing_variables,
      step_hours,
  )

  truth_vars: dict[str, xr.DataArray] = {}
  if METRIC_VARIABLE in pressure_variables:
    source = atmospheric_sources[METRIC_VARIABLE]
    metric_levels = [METRIC_LEVEL] if METRIC_LEVEL is not None else pressure_levels.tolist()
    logging.info(
        "Direct ARCO truth variable: %s at levels %s",
        METRIC_VARIABLE,
        metric_levels,
    )
    truth = select_times(arco[[source]], target_datetimes)[source].sel(level=metric_levels)
    truth = truth.assign_coords(level=np.asarray(metric_levels, dtype=np.int32))
    truth_vars[METRIC_VARIABLE] = add_batch_and_time_coords(
        truth,
        target_time,
        ("time", "lat", "lon", "level"),
    )
  elif METRIC_VARIABLE in surface_variables:
    source = surface_sources[METRIC_VARIABLE]
    logging.info("Direct ARCO truth variable: %s", METRIC_VARIABLE)
    truth_vars[METRIC_VARIABLE] = add_batch_and_time_coords(
        select_times(arco[[source]], target_datetimes)[source],
        target_time,
        ("time", "lat", "lon"),
    )
  elif METRIC_VARIABLE == "total_precipitation_6hr":
    logging.info("Direct ARCO truth variable: total_precipitation_6hr")
    truth_vars[METRIC_VARIABLE] = add_batch_and_time_coords(
        precipitation_6hr(arco, target_datetimes, step_hours),
        target_time,
        ("time", "lat", "lon"),
    )
  else:
    logging.warning("No direct ARCO truth source for metric variable %s", METRIC_VARIABLE)
  truth = xr.Dataset(truth_vars)

  label = f"direct ARCO ERA5 stream ({init_time})"
  logging.info("Direct ARCO inputs dims: %s", dict(inputs.sizes))
  logging.info("Direct ARCO target template dims: %s", dict(targets_template.sizes))
  logging.info("Direct ARCO forcings dims: %s", dict(forcings.sizes))
  logging.info("Direct ARCO truth dims: %s", dict(truth.sizes))
  return inputs, targets_template, forcings, truth, label


def materialize_coords(dataset: xr.Dataset) -> xr.Dataset:
  """Return dataset with eager coordinates while preserving lazy data variables."""
  coords = {
      name: xr.Variable(coord.dims, np.asarray(coord.values), coord.attrs)
      for name, coord in dataset.coords.items()
  }
  return dataset.assign_coords(coords)


def forecast_start_datetime(reference: xr.Dataset) -> np.ndarray:
  if "datetime" in reference.coords:
    datetime = reference.coords["datetime"]
    if "time" in datetime.dims:
      datetime = datetime.isel(time=1)
    values = datetime.values
    if "batch" in datetime.dims:
      return np.asarray(values, dtype="datetime64[ns]")
    return np.asarray([values], dtype="datetime64[ns]")

  time = reference.coords["time"].values
  if np.issubdtype(time.dtype, np.datetime64):
    return np.asarray([time[1]], dtype="datetime64[ns]")

  raise ValueError("Reference ERA5 needs a datetime coordinate to extend forcings.")


def make_forcings(
    reference: xr.Dataset,
    inputs: xr.Dataset,
    target_times: xr.DataArray,
    forcing_variables: tuple[str, ...],
    step_hours: int,
) -> xr.Dataset:
  start_datetime = forecast_start_datetime(reference)
  lead_hours = np.arange(1, target_times.sizes["time"] + 1) * step_hours
  target_datetimes = (
      start_datetime[:, None] + lead_hours[None, :].astype("timedelta64[h]")
  )

  coords: dict[str, Any] = {
      "time": target_times.values,
      "lat": inputs.coords["lat"].values,
      "lon": inputs.coords["lon"].values,
  }
  if "batch" in inputs.dims:
    coords["batch"] = inputs.coords["batch"].values
    coords["datetime"] = (("batch", "time"), target_datetimes)
  else:
    coords["datetime"] = ("time", target_datetimes[0])

  forcings = xr.Dataset(coords=coords)
  if set(forcing_variables) & DERIVED_FORCING_VARIABLES:
    data_utils.add_derived_vars(forcings)
  if TISR in forcing_variables:
    data_utils.add_tisr_var(forcings)
  if "datetime" in forcings:
    forcings = forcings.drop_vars("datetime")
  return forcings[list(forcing_variables)].astype(np.float32)


def read_reference_era5(
    bucket: Any,
    cache_dir: Path,
    task_config: Any,
    rollout_steps: int = DEFAULT_ROLLOUT_STEPS,
    step_hours: int = DEFAULT_STEP_HOURS,
    init_time: str | None = None,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset, str]:
  """Read reference ERA5 and extract GraphCast inputs, template, forcings, truth."""
  init_time = os.environ.get("GRAPHCAST_INIT_TIME")
  if (
      env_flag("GRAPHCAST_USE_ARCO", default=bool(init_time))
      and env_flag("GRAPHCAST_ARCO_DIRECT")
      and not env_flag("GRAPHCAST_USE_NEXTGEMS")
  ):
    return read_direct_arco_era5(task_config, rollout_steps, step_hours)

  reference_uri = configured_reference_era5_uri(
      cache_dir, task_config, rollout_steps, step_hours, init_time)
  reference, label = open_reference_era5(bucket, cache_dir, reference_uri)
  if reference.sizes["time"] < 3:
    raise ValueError("Reference ERA5 needs at least 3 timesteps: 2 inputs + 1 target.")

  available_target_steps = reference.sizes["time"] - 2
  target_steps = min(rollout_steps, available_target_steps)
  reference_window = materialize_coords(
      reference.isel(time=slice(0, target_steps + 2)))
  inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
      reference_window,
      target_lead_times=slice(f"{step_hours}h", f"{target_steps * step_hours}h"),
      **dataclasses.asdict(task_config),
  )

  if rollout_steps > target_steps:
    one_step_template = targets.isel(time=slice(0, 1))
    targets_template = rollout.extend_targets_template(one_step_template, rollout_steps)
    forcings = make_forcings(
        reference_window,
        inputs,
        targets_template.coords["time"],
        task_config.forcing_variables,
        step_hours,
    )
  else:
    targets_template = targets

  targets_template = targets_template * np.nan
  return inputs, targets_template, forcings, targets, label

def build_jitted_predictor(
    modules: dict[str, Any],
    params: dict[str, Any],
    model_config: Any,
    task_config: Any,
    stats: dict[str, xr.Dataset],
    gradient_checkpointing: bool = False,
):
  """Build a jitted GraphCast predictor that unrolls the whole trajectory.

  The one-step GraphCast is wrapped in ``autoregressive.Predictor`` so a single
  jitted call runs the full rollout via ``hk.scan`` on-device. This is the
  GraphCast analogue of NeuralGCM's ``model.unroll``: it removes the per-step
  Python loop and the per-step host<->device round-trips of the previous
  implementation.
  """
  hk = modules["hk"]
  jax = modules["jax"]
  graphcast_module = modules["graphcast"]
  casting_module = modules["casting"]
  normalization_module = modules["normalization"]
  autoregressive_module = modules["autoregressive"]

  def construct_wrapped_graphcast():
    predictor = graphcast_module.GraphCast(model_config, task_config)
    predictor = casting_module.Bfloat16Cast(predictor)
    predictor = normalization_module.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=stats["diffs_stddev_by_level"],
        mean_by_level=stats["mean_by_level"],
        stddev_by_level=stats["stddev_by_level"],
    )
    predictor = autoregressive_module.Predictor(
        predictor, gradient_checkpointing=gradient_checkpointing)
    return predictor

  @hk.transform_with_state
  def run_forward(inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast()
    return predictor(inputs, targets_template=targets_template, forcings=forcings)

  state = {}

  def predictor(rng, inputs, targets_template, forcings):
    predictions, _ = run_forward.apply(
        params, state, rng, inputs, targets_template, forcings)
    return predictions

  return jax.jit(predictor)


def run_fused_rollout(
    modules: dict[str, Any],
    predictor: Any,
    rng_seed: int,
    inputs: xr.Dataset,
    targets_template: xr.Dataset,
    forcings: xr.Dataset,
) -> xr.Dataset:
  """Run the entire rollout in a single jitted call and pull results off-device once.

  Unlike the previous step-by-step host loop, the autoregressive feedback happens
  inside ``hk.scan`` on the accelerator; here we only do a single ``device_get``
  of the full predicted trajectory. Returns the full prediction Dataset (all
  variables, all lead times), with the target lead-time coordinates.
  """
  jax = modules["jax"]
  rng = jax.random.PRNGKey(rng_seed)
  predictions = predictor(
      rng,
      inputs.compute(),
      targets_template.compute(),
      forcings.compute(),
  )
  predictions = jax.device_get(predictions)
  return predictions.assign_coords(time=targets_template.coords["time"])


# --------------------------------------------------------------------------- #
# Batch-over-init-days driver (NeuralGCM-style: load model once, loop days)
# --------------------------------------------------------------------------- #
def safe_time_label(init_time: str) -> str:
  return pd.Timestamp(init_time).strftime("%Y%m%d_%H")


def init_times_for_batch(rollout_steps: int, step_hours: int) -> list[str | None]:
  """Return the init-times this process should run (year sweep or single).

  With GRAPHCAST_YEAR set, generate one init-time per GRAPHCAST_INIT_STRIDE_DAYS
  at GRAPHCAST_INIT_HOUR, bounded so the whole rollout stays inside the year.
  Otherwise fall back to a single init (GRAPHCAST_INIT_TIME, possibly None ->
  local reference file). GRAPHCAST_INIT_START/_COUNT then select this process's
  slice (batching, exactly like ERA5_INIT_START/_COUNT in era5_rollout.py).
  """
  year = os.environ.get("GRAPHCAST_YEAR")
  if year:
    hour = int(os.environ.get("GRAPHCAST_INIT_HOUR", "6"))
    stride = int(os.environ.get("GRAPHCAST_INIT_STRIDE_DAYS", "1"))
    lead = pd.Timedelta(hours=rollout_steps * step_hours)
    year_end = pd.Timestamp(f"{year}-12-31T23:00")
    starts = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq=f"{stride}D")
    all_inits = [
        s + pd.Timedelta(hours=hour)
        for s in starts
        if (s + pd.Timedelta(hours=hour) + lead) <= year_end
    ]
    # For a partial / current year, GRAPHCAST_INIT_END_DATE caps the LAST init to
    # the ERA5 data front (the rollout needs real ERA5 only at the 2 input steps
    # <= init; the forecast itself uses astronomically-computed forcings, so it
    # may run past the front). Mirrors era5_rollout's ERA5_END_DATE. No-op if unset.
    end = os.environ.get("GRAPHCAST_INIT_END_DATE", "").strip()
    if end:
        cutoff = pd.Timestamp(end) + pd.Timedelta(days=1)   # inclusive of the day
        all_inits = [t for t in all_inits if t < cutoff]
    inits: list[str | None] = [t.strftime("%Y-%m-%dT%H:00") for t in all_inits]
  else:
    inits = [os.environ.get("GRAPHCAST_INIT_TIME")]

  start = int(os.environ.get("GRAPHCAST_INIT_START", "0"))
  count = int(os.environ.get("GRAPHCAST_INIT_COUNT", str(len(inits))))
  return inits[start:start + count]


def output_paths(init_time: str | None, rollout_steps: int, step_hours: int) -> dict[str, Path]:
  """Per-init output paths. Batch layout under GRAPHCAST_OUT_DIR, else legacy env."""
  out_dir = os.environ.get("GRAPHCAST_OUT_DIR")
  if out_dir is None:
    # Legacy single-init mode: honor the explicit env CSV/plot paths.
    return {
        "evolution_csv": EVOLUTION_CSV,
        "metrics_csv": OUTPUT_CSV,
        "evolution_png": EVOLUTION_PLOT_PATH,
        "metrics_png": PLOT_PATH,
        "pred_nc": Path(os.environ["GRAPHCAST_PRED_NC"])
        if os.environ.get("GRAPHCAST_PRED_NC")
        else None,
    }

  # NeuralGCM-compatible layout: flat pred_<year>_<YYYY-MM-DD>.nc directly under
  # GRAPHCAST_OUT_DIR (a .../predictions dir) so eval_common's
  # `glob("pred_<year>_*.nc")` picks them up; metric CSVs go in a metrics/ subdir
  # so they don't pollute that glob. (Coords are renamed to latitude/longitude in
  # prediction_output_dataset.)
  if env_flag("GRAPHCAST_NEURALGCM_LAYOUT") and init_time is not None:
    ts = pd.Timestamp(init_time)
    date = ts.strftime("%Y-%m-%d")
    base = Path(out_dir).expanduser()
    metrics = base / "metrics"
    # Two-stage pipeline: when GRAPHCAST_FINAL_DIR is set, GPU writes the raw
    # (uncompressed) pred here (OUT_DIR = .../predictions_raw) and a CPU stage
    # compresses it into FINAL_DIR (.../predictions). "final_nc" lets the GPU
    # resumability skip a day whose compressed final already exists (so deleting
    # the raw after compression doesn't trigger a needless rebuild).
    final_dir = os.environ.get("GRAPHCAST_FINAL_DIR")
    final_nc = (Path(final_dir).expanduser() / f"pred_{ts.year}_{date}.nc"
                if final_dir else None)
    return {
        "evolution_csv": metrics / f"pred_{ts.year}_{date}_{METRIC_ID}_evolution.csv",
        "metrics_csv": metrics / f"pred_{ts.year}_{date}_{METRIC_ID}_rmse_bias.csv",
        "evolution_png": metrics / f"pred_{ts.year}_{date}_{METRIC_ID}_evolution.png",
        "metrics_png": metrics / f"pred_{ts.year}_{date}_{METRIC_ID}_rmse_bias.png",
        "pred_nc": base / f"pred_{ts.year}_{date}.nc",
        "final_nc": final_nc,
    }

  label = safe_time_label(init_time) if init_time else "reference"
  days = rollout_steps * step_hours // 24
  prefix = os.environ.get("GRAPHCAST_RUN_PREFIX", "operational")
  day_dir = Path(out_dir).expanduser() / f"{prefix}_{label}_{days}d"
  return {
      "evolution_csv": day_dir / f"{METRIC_ID}_global_mean_evolution.csv",
      "metrics_csv": day_dir / f"{METRIC_ID}_rmse_bias_operational_daily.csv",
      "evolution_png": day_dir / f"{METRIC_ID}_global_mean_evolution.png",
      "metrics_png": day_dir / f"{METRIC_ID}_rmse_bias_operational_daily.png",
      "pred_nc": day_dir / f"pred_{label}.nc",
  }


def prediction_output_dataset(
    predictions: xr.Dataset,
    init_time: str | None,
    step_hours: int,
) -> xr.Dataset:
  """Add lead_hours / valid_time coords for a self-describing prediction file."""
  ds = predictions.isel(batch=0, drop=True) if "batch" in predictions.dims else predictions
  lead_h = timedelta_hours_int(ds.coords["time"].values, step_hours)
  ds = ds.assign_coords(lead_hours=("time", lead_h.astype(np.int32)))
  if init_time is not None:
    valid = np.datetime64(pd.Timestamp(init_time)) + lead_h * np.timedelta64(1, "h")
    ds = ds.assign_coords(valid_time=("time", valid))
  ds.attrs.update(
      source="GraphCast fused autoregressive rollout",
      init_time=str(init_time) if init_time else "",
      rollout_steps=predictions.sizes["time"],
      step_hours=step_hours,
  )
  # Match NeuralGCM's coordinate naming so the same eval/figure pipeline applies.
  rename = {old: new for old, new in {"lat": "latitude", "lon": "longitude"}.items()
            if old in ds.coords or old in ds.dims}
  if rename:
    ds = ds.rename(rename)
  return ds


def timedelta_hours_int(values: np.ndarray, step_hours: int) -> np.ndarray:
  if np.issubdtype(values.dtype, np.timedelta64):
    return values.astype("timedelta64[h]").astype(np.int64)
  return np.arange(1, len(values) + 1, dtype=np.int64) * step_hours


def write_prediction_netcdf(ds: xr.Dataset, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  # GRAPHCAST_WRITE_COMPLEVEL=0 -> uncompressed (fast, for the "compress on CPU
  # later" pipeline); >0 -> zlib at that level (default 4).
  complevel = int(os.environ.get("GRAPHCAST_WRITE_COMPLEVEL", "4"))
  if complevel > 0:
    base = {"dtype": "float32", "zlib": True, "complevel": complevel}
  else:
    base = {"dtype": "float32"}
  enc = {v: dict(base) for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.floating)}
  tmp = path.with_suffix(".nc.tmp")
  ds.to_netcdf(tmp, encoding=enc)
  tmp.rename(path)


def process(
    modules: dict[str, Any],
    bucket: Any,
    cache_dir: Path,
    predictor: Any,
    task_config: Any,
    init_time: str | None,
    rollout_steps: int,
    step_hours: int,
    seed: int,
    write_netcdf: bool,
    force: bool,
) -> None:
  # Phase 1 can save ONLY the prediction NetCDFs (metrics CSVs + plots are phase 2).
  netcdf_only = env_flag("GRAPHCAST_NETCDF_ONLY", default=False)
  paths = output_paths(init_time, rollout_steps, step_hours)

  if netcdf_only:
    fin = paths.get("final_nc")
    done = (paths["pred_nc"] is not None and paths["pred_nc"].exists()) \
        or (fin is not None and fin.exists())
  else:
    done = paths["evolution_csv"].exists() and paths["metrics_csv"].exists()
    if write_netcdf and paths["pred_nc"] is not None:
      done = done and paths["pred_nc"].exists()
  if done and not force:
    logging.info("[%s] outputs already exist -> skip", init_time)
    return

  t0 = time.time()
  inputs, targets_template, forcings, reference_targets, label = read_reference_era5(
      bucket, cache_dir, task_config, rollout_steps, step_hours, init_time)
  t_prep = time.time()

  predictions = run_fused_rollout(
      modules, predictor, seed, inputs, targets_template, forcings)
  t_roll = time.time()

  if netcdf_only:
    write_prediction_netcdf(
        prediction_output_dataset(predictions, init_time, step_hours),
        paths["pred_nc"])
    dt = time.time() - t0
    logging.info(
        "[%s] done in %.1fs (prep %.1fs, rollout %.1fs, write %.1fs) -> %s",
        init_time, dt, t_prep - t0, t_roll - t_prep, time.time() - t_roll,
        paths["pred_nc"])
    return

  metric_field = select_metric_field(predictions, METRIC_VARIABLE, METRIC_LEVEL)
  evolution = compute_global_mean_evolution(
      predictions=metric_field, inputs=inputs, truth=reference_targets,
      reference_mode=REFERENCE_MODE, metric_variable=METRIC_VARIABLE,
      metric_level=METRIC_LEVEL, metric_id=METRIC_ID, init_time=init_time,
      scenario_label=os.environ.get("GRAPHCAST_SCENARIO_LABEL"))
  metrics = compute_metric_summary(
      predictions=metric_field, inputs=inputs, truth=reference_targets,
      reference_mode=REFERENCE_MODE, step_hours=step_hours,
      metrics_frequency=METRICS_FREQUENCY, metric_variable=METRIC_VARIABLE,
      metric_level=METRIC_LEVEL, metric_id=METRIC_ID)

  paths["evolution_csv"].parent.mkdir(parents=True, exist_ok=True)
  evolution.to_csv(paths["evolution_csv"], index=False)
  metrics.to_csv(paths["metrics_csv"], index=False)
  if WRITE_PLOT:
    plot_global_mean_evolution(evolution, paths["evolution_png"], METRIC_ID)
    plot_metrics(metrics, paths["metrics_png"], METRIC_ID)
  if write_netcdf and paths["pred_nc"] is not None:
    write_prediction_netcdf(
        prediction_output_dataset(predictions, init_time, step_hours),
        paths["pred_nc"])

  dt = time.time() - t0
  logging.info(
      "[%s] done in %.1fs (prep %.1fs, rollout %.1fs) -> %s | reference=%s",
      init_time, dt, t_prep - t0, t_roll - t_prep,
      paths["evolution_csv"].parent, metrics["reference"].iloc[0])


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
  cache_dir = Path(os.environ.get("GRAPHCAST_CACHE_DIR", "~/.cache/graphcast")).expanduser()
  rollout_steps = int(os.environ.get("GRAPHCAST_ROLLOUT_STEPS", DEFAULT_ROLLOUT_STEPS))
  step_hours = int(os.environ.get("GRAPHCAST_STEP_HOURS", DEFAULT_STEP_HOURS))
  seed = int(os.environ.get("GRAPHCAST_SEED", str(DEFAULT_SEED)))
  write_netcdf = env_flag("GRAPHCAST_WRITE_NETCDF", default=True)
  force = env_flag("GRAPHCAST_FORCE", default=False)
  gradient_checkpointing = env_flag("GRAPHCAST_GRADIENT_CHECKPOINTING", default=False)

  modules = {
      "hk": hk,
      "jax": jax,
      "graphcast": graphcast,
      "casting": casting,
      "normalization": normalization,
      "autoregressive": autoregressive,
  }
  logging.info("JAX devices: %s", jax.devices())

  init_times = init_times_for_batch(rollout_steps, step_hours)
  if not init_times:
    logging.info("No init-times selected for this batch; nothing to do.")
    return
  logging.info(
      "Batch of %d init-time(s): %s", len(init_times),
      [t or "<local-reference>" for t in init_times])

  # Load checkpoint + stats + build the jitted predictor ONCE, reused for all
  # init-days (the lead-time coordinates are constant, so no recompile per day).
  bucket = anonymous_bucket(storage)
  ckpt, checkpoint_name = load_checkpoint(
      bucket, checkpoint, graphcast, cache_dir, "operational",
      os.environ.get("GRAPHCAST_CHECKPOINT_NAME"))
  stats = load_normalization_stats(bucket, cache_dir)
  logging.info("Checkpoint: %s", checkpoint_name)
  logging.info(
      "Metric variable: %s%s | rollout_steps=%d step_hours=%d",
      METRIC_VARIABLE,
      f" at {METRIC_LEVEL} hPa" if METRIC_LEVEL is not None else "",
      rollout_steps, step_hours)

  predictor = build_jitted_predictor(
      modules, ckpt.params, ckpt.model_config, ckpt.task_config, stats,
      gradient_checkpointing=gradient_checkpointing)

  for init_time in init_times:
    process(
        modules, bucket, cache_dir, predictor, ckpt.task_config,
        init_time, rollout_steps, step_hours, seed, write_netcdf, force)
  logging.info("done.")


if __name__ == "__main__":
  main()
