#!/usr/bin/env python3
"""Run a 10-day GraphCast t2m autoregressive drift/RMSE experiment.

This script follows the official DeepMind GraphCast Haiku/JAX wrapping pattern:

  GraphCast -> Bfloat16Cast -> normalization.InputsAndResiduals

It then performs an explicit Python autoregressive rollout loop. Each loop call
predicts one 6-hour step with a jitted one-step predictor, feeds predicted
prognostic variables back into the next input window, and logs progress.

Examples:

  python run_t2m_drift_rollout.py --model small --rollout-steps 40

  python run_t2m_drift_rollout.py \
      --model operational \
      --dataset-name source-hres_date-2022-01-01_res-0.25_levels-13_steps-01.nc

  python run_t2m_drift_rollout.py \
      --checkpoint-name "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz" \
      --dataset-path /path/to/local_example.nc
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


GCS_BUCKET = "dm_graphcast"
GRAPHCAST_PREFIX = "graphcast/"
PARAMS_PREFIX = GRAPHCAST_PREFIX + "params/"
DATASET_PREFIX = GRAPHCAST_PREFIX + "dataset/"
STATS_PREFIX = GRAPHCAST_PREFIX + "stats/"

DEFAULT_OPERATIONAL_CHECKPOINT = (
    "GraphCast_operational - ERA5-HRES 1979-2021-reso 0.25 - "
    "precipitation input and output.npz"
)

CHECKPOINT_CANDIDATES = {
    "operational": (
        DEFAULT_OPERATIONAL_CHECKPOINT,
        "GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - "
        "pressure levels 13 - mesh 2to6 - precipitation output only.npz",
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


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Run an explicit GraphCast autoregressive t2m rollout and write "
          "latitude-weighted RMSE/bias diagnostics."
      )
  )
  parser.add_argument(
      "--model",
      choices=sorted(CHECKPOINT_CANDIDATES),
      default="operational",
      help="Checkpoint preset to use when --checkpoint-name is not supplied.",
  )
  parser.add_argument(
      "--checkpoint-name",
      help=(
          "Exact .npz name under gs://dm_graphcast/graphcast/params/. "
          "Overrides --model."
      ),
  )
  parser.add_argument(
      "--dataset-path",
      help="Local .nc path or gs:// path to an xarray dataset.",
  )
  parser.add_argument(
      "--dataset-name",
      help="Dataset .nc name under gs://dm_graphcast/graphcast/dataset/.",
  )
  parser.add_argument(
      "--dummy-data",
      action="store_true",
      help=(
          "Use synthetic xarray data with the correct GraphCast variable "
          "structure. Useful for plumbing tests, not for scientific metrics."
      ),
  )
  parser.add_argument(
      "--cache-dir",
      type=Path,
      default=Path(os.environ.get("GRAPHCAST_CACHE_DIR", "~/.cache/graphcast")).expanduser(),
      help="Directory used to cache checkpoints, stats, and GCS datasets.",
  )
  parser.add_argument(
      "--rollout-steps",
      type=int,
      default=40,
      help="Number of 6-hour autoregressive steps. 40 steps is 10 days.",
  )
  parser.add_argument(
      "--step-hours",
      type=int,
      default=6,
      help="Forecast step size in hours.",
  )
  parser.add_argument(
      "--metrics-frequency",
      choices=("daily", "6hourly"),
      default="daily",
      help="Rows to keep in the output CSV.",
  )
  parser.add_argument(
      "--reference",
      choices=("auto", "ground_truth", "initial_state"),
      default="auto",
      help=(
          "Reference for RMSE/bias. auto uses ground truth when the dataset has "
          "all requested target steps, otherwise the initial t2m state."
      ),
  )
  parser.add_argument(
      "--output-csv",
      type=Path,
      default=Path("t2m_drift_metrics.csv"),
      help="CSV path for t2m RMSE/bias metrics.",
  )
  parser.add_argument(
      "--plot-path",
      type=Path,
      default=Path("t2m_drift_metrics.png"),
      help="Matplotlib plot path.",
  )
  parser.add_argument(
      "--no-plot",
      action="store_true",
      help="Skip writing the metrics plot.",
  )
  parser.add_argument(
      "--rng-seed",
      type=int,
      default=0,
      help="JAX PRNG seed.",
  )
  parser.add_argument(
      "--xla-mem-fraction",
      help=(
          "Optional XLA_PYTHON_CLIENT_MEM_FRACTION value. Leave unset to let "
          "JAX choose while still disabling preallocation by default."
      ),
  )
  parser.add_argument(
      "--xla-preallocate",
      action="store_true",
      help="Allow JAX/XLA GPU memory preallocation. Default is disabled.",
  )
  parser.add_argument(
      "--keep-inputs-on-device",
      action="store_true",
      help=(
          "Avoid jax.device_get between rollout steps. Faster, but uses more "
          "accelerator memory."
      ),
  )
  parser.add_argument(
      "--log-level",
      default="INFO",
      help="Python logging level.",
  )
  return parser.parse_args()


def configure_jax_memory(args: argparse.Namespace) -> None:
  if not args.xla_preallocate:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
  if args.xla_mem_fraction:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = args.xla_mem_fraction


def import_graphcast_stack() -> dict[str, Any]:
  source_dir = Path(__file__).resolve().parent
  if str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir))

  import haiku as hk  # pylint: disable=import-outside-toplevel
  import jax  # pylint: disable=import-outside-toplevel
  from google.cloud import storage  # pylint: disable=import-outside-toplevel
  from graphcast import casting  # pylint: disable=import-outside-toplevel
  from graphcast import checkpoint  # pylint: disable=import-outside-toplevel
  from graphcast import data_utils  # pylint: disable=import-outside-toplevel
  from graphcast import graphcast  # pylint: disable=import-outside-toplevel
  from graphcast import normalization  # pylint: disable=import-outside-toplevel
  from graphcast import rollout  # pylint: disable=import-outside-toplevel
  from graphcast import xarray_jax  # pylint: disable=import-outside-toplevel,unused-import

  return {
      "hk": hk,
      "jax": jax,
      "storage": storage,
      "casting": casting,
      "checkpoint": checkpoint,
      "data_utils": data_utils,
      "graphcast": graphcast,
      "normalization": normalization,
      "rollout": rollout,
  }


def anonymous_bucket(storage_module: Any) -> Any:
  client = storage_module.Client.create_anonymous_client()
  return client.bucket(GCS_BUCKET)


def parse_file_parts(file_name: str) -> dict[str, str]:
  stem = file_name.removesuffix(".nc").removesuffix(".npz")
  return dict(part.split("-", 1) for part in stem.split("_") if "-" in part)


def list_gcs_names(bucket: Any, prefix: str) -> list[str]:
  names = []
  for blob in bucket.list_blobs(prefix=prefix):
    name = blob.name.removeprefix(prefix)
    if name:
      names.append(name)
  return sorted(names)


def cache_blob(bucket: Any, object_name: str, cache_dir: Path) -> Path:
  local_path = cache_dir / object_name
  local_path.parent.mkdir(parents=True, exist_ok=True)
  if local_path.exists():
    logging.info("Using cached gs://%s/%s", GCS_BUCKET, object_name)
    return local_path

  logging.info("Downloading gs://%s/%s", GCS_BUCKET, object_name)
  bucket.blob(object_name).download_to_filename(local_path)
  return local_path


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
    logging.info("Resolved --model %s to checkpoint %s", model_alias, matches[0])
    return matches[0]

  msg = "\n".join(available) if available else "(no checkpoint names listed)"
  raise FileNotFoundError(
      f"Could not resolve checkpoint for --model {model_alias}. "
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
  resolved_name = resolve_checkpoint_name(bucket, model_alias, checkpoint_name)
  path = cache_blob(bucket, PARAMS_PREFIX + resolved_name, cache_dir)
  with path.open("rb") as f:
    ckpt = checkpoint_module.load(f, graphcast_module.CheckPoint)

  logging.info("Loaded checkpoint: %s", resolved_name)
  logging.info("Model description: %s", ckpt.description)
  return ckpt, resolved_name


def load_normalization_stats(bucket: Any, cache_dir: Path) -> dict[str, xr.Dataset]:
  stats = {}
  for file_name in STATS_FILES:
    path = cache_blob(bucket, STATS_PREFIX + file_name, cache_dir)
    stats[file_name.removesuffix(".nc")] = xr.load_dataset(path).compute()
  return stats


def dataset_valid_for_model(
    file_name: str,
    model_config: Any,
    task_config: Any,
) -> bool:
  try:
    parts = parse_file_parts(file_name)
    resolution = float(parts["res"])
    levels = int(parts["levels"])
    source = parts["source"]
  except (KeyError, ValueError):
    return False

  precipitation_is_input = "total_precipitation_6hr" in task_config.input_variables
  source_ok = (
      source in ("era5", "fake")
      if precipitation_is_input
      else source in ("hres", "fake")
  )
  return (
      model_config.resolution in (0, resolution)
      and len(task_config.pressure_levels) == levels
      and source_ok
  )


def resolve_dataset_name(
    bucket: Any,
    model_config: Any,
    task_config: Any,
    rollout_steps: int,
    dataset_name: str | None,
) -> str:
  if dataset_name:
    return dataset_name

  available = [
      name
      for name in list_gcs_names(bucket, DATASET_PREFIX)
      if dataset_valid_for_model(name, model_config, task_config)
  ]
  if not available:
    raise FileNotFoundError(
        "No matching sample dataset found. Pass --dataset-path, --dataset-name, "
        "or --dummy-data."
    )

  def target_steps(name: str) -> int:
    try:
      return int(parse_file_parts(name)["steps"])
    except (KeyError, ValueError):
      return -1

  enough_truth = [name for name in available if target_steps(name) >= rollout_steps]
  if enough_truth:
    return sorted(enough_truth, key=target_steps, reverse=True)[0]
  return sorted(available, key=target_steps, reverse=True)[0]


def open_dataset_from_path_or_gcs(bucket: Any, path_or_uri: str, cache_dir: Path) -> xr.Dataset:
  if path_or_uri.startswith("gs://"):
    without_scheme = path_or_uri.removeprefix("gs://")
    bucket_name, object_name = without_scheme.split("/", 1)
    if bucket_name != GCS_BUCKET:
      raise ValueError(
          f"This script uses the anonymous {GCS_BUCKET!r} bucket, got {bucket_name!r}."
      )
    path = cache_blob(bucket, object_name, cache_dir)
  else:
    path = Path(path_or_uri).expanduser()

  logging.info("Loading dataset from %s", path)
  return xr.load_dataset(path).compute()


def load_example_dataset(
    bucket: Any,
    cache_dir: Path,
    model_config: Any,
    task_config: Any,
    rollout_steps: int,
    dataset_path: str | None,
    dataset_name: str | None,
) -> tuple[xr.Dataset, str]:
  if dataset_path:
    dataset = open_dataset_from_path_or_gcs(bucket, dataset_path, cache_dir)
    return dataset, dataset_path

  resolved_name = resolve_dataset_name(
      bucket, model_config, task_config, rollout_steps, dataset_name)
  path = cache_blob(bucket, DATASET_PREFIX + resolved_name, cache_dir)
  logging.info("Loading sample dataset: %s", resolved_name)
  return xr.load_dataset(path).compute(), resolved_name


def grid_size_from_resolution(model_config: Any) -> tuple[int, int]:
  if model_config.resolution == 0:
    resolution = 1.0
  else:
    resolution = float(model_config.resolution)
  n_lon = int(round(360.0 / resolution))
  n_lat = int(round(180.0 / resolution)) + 1
  return n_lat, n_lon


def create_dummy_dataset(model_config: Any, task_config: Any, step_hours: int) -> xr.Dataset:
  """Builds a synthetic 3-time dataset sufficient for a 1-step template."""
  n_lat, n_lon = grid_size_from_resolution(model_config)
  levels = np.asarray(task_config.pressure_levels, dtype=np.int32)
  lat = np.linspace(-90.0, 90.0, n_lat, dtype=np.float32)
  lon = np.linspace(0.0, 360.0 - float(360.0 / n_lon), n_lon, dtype=np.float32)
  time = np.arange(3) * np.timedelta64(step_hours, "h")
  datetimes = np.datetime64("2022-01-01T00") + time

  coords = {
      "batch": np.arange(1, dtype=np.int32),
      "time": time,
      "lat": lat,
      "lon": lon,
      "level": levels,
      "datetime": (("batch", "time"), datetimes[None, :]),
  }

  surface_shape = (1, 3, n_lat, n_lon)
  atmos_shape = (1, 3, n_lat, n_lon, len(levels))

  lat_field = lat[None, None, :, None]
  lon_wave = np.sin(np.deg2rad(lon))[None, None, None, :]
  t2m = 280.0 - 35.0 * np.abs(lat_field) / 90.0 + 2.0 * lon_wave
  t2m = np.broadcast_to(t2m, surface_shape).astype(np.float32)

  data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
      "2m_temperature": (("batch", "time", "lat", "lon"), t2m),
      "mean_sea_level_pressure": (
          ("batch", "time", "lat", "lon"),
          np.full(surface_shape, 101325.0, dtype=np.float32),
      ),
      "10m_u_component_of_wind": (
          ("batch", "time", "lat", "lon"),
          np.zeros(surface_shape, dtype=np.float32),
      ),
      "10m_v_component_of_wind": (
          ("batch", "time", "lat", "lon"),
          np.zeros(surface_shape, dtype=np.float32),
      ),
      "total_precipitation_6hr": (
          ("batch", "time", "lat", "lon"),
          np.zeros(surface_shape, dtype=np.float32),
      ),
      "geopotential_at_surface": (
          ("lat", "lon"),
          np.zeros((n_lat, n_lon), dtype=np.float32),
      ),
      "land_sea_mask": (
          ("lat", "lon"),
          np.ones((n_lat, n_lon), dtype=np.float32),
      ),
  }

  atmospheric_defaults = {
      "temperature": 250.0,
      "geopotential": 50000.0,
      "u_component_of_wind": 0.0,
      "v_component_of_wind": 0.0,
      "vertical_velocity": 0.0,
      "specific_humidity": 0.001,
  }
  for name, value in atmospheric_defaults.items():
    data_vars[name] = (
        ("batch", "time", "lat", "lon", "level"),
        np.full(atmos_shape, value, dtype=np.float32),
    )

  logging.warning(
      "Using synthetic dummy data. The rollout can test plumbing, but RMSE/bias "
      "are not scientifically meaningful."
  )
  return xr.Dataset(data_vars=data_vars, coords=coords)


def forecast_start_datetime(example_batch: xr.Dataset) -> np.ndarray:
  if "datetime" in example_batch.coords:
    dt = example_batch.coords["datetime"]
    if "time" in dt.dims:
      dt = dt.isel(time=1)
    values = dt.values
    if "batch" in dt.dims:
      return np.asarray(values, dtype="datetime64[ns]")
    return np.asarray([values], dtype="datetime64[ns]")

  time = example_batch.coords["time"].values
  if np.issubdtype(time.dtype, np.datetime64):
    return np.asarray([time[1]], dtype="datetime64[ns]")

  logging.warning(
      "Dataset has no datetime coordinate; using 2022-01-01T06 for forcings."
  )
  return np.asarray(["2022-01-01T06"], dtype="datetime64[ns]")


def make_forcings(
    data_utils_module: Any,
    example_batch: xr.Dataset,
    inputs: xr.Dataset,
    target_times: xr.DataArray,
    forcing_variables: tuple[str, ...],
    step_hours: int,
) -> xr.Dataset:
  start_dt = forecast_start_datetime(example_batch)
  lead_hours = np.arange(1, target_times.sizes["time"] + 1) * step_hours
  target_datetimes = start_dt[:, None] + lead_hours[None, :].astype("timedelta64[h]")

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
  if set(forcing_variables) & {
      "year_progress",
      "year_progress_sin",
      "year_progress_cos",
      "day_progress",
      "day_progress_sin",
      "day_progress_cos",
  }:
    data_utils_module.add_derived_vars(forcings)
  if "toa_incident_solar_radiation" in forcing_variables:
    data_utils_module.add_tisr_var(forcings)
  if "datetime" in forcings:
    forcings = forcings.drop_vars("datetime")
  return forcings[list(forcing_variables)].astype(np.float32)


def prepare_data(
    modules: dict[str, Any],
    bucket: Any,
    cache_dir: Path,
    model_config: Any,
    task_config: Any,
    rollout_steps: int,
    step_hours: int,
    dataset_path: str | None,
    dataset_name: str | None,
    dummy_data: bool,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset | None, str]:
  if dummy_data:
    example_batch = create_dummy_dataset(model_config, task_config, step_hours)
    dataset_label = "dummy"
  else:
    example_batch, dataset_label = load_example_dataset(
        bucket,
        cache_dir,
        model_config,
        task_config,
        rollout_steps,
        dataset_path,
        dataset_name,
    )

  if example_batch.sizes["time"] < 3:
    raise ValueError("GraphCast examples need at least 3 timesteps: 2 inputs + 1 target.")

  available_target_steps = max(1, example_batch.sizes["time"] - 2)
  extraction_steps = min(rollout_steps, available_target_steps)
  logging.info(
      "Preparing inputs from %s target step(s) available in %s",
      available_target_steps,
      dataset_label,
  )
  example_window = example_batch.isel(time=slice(0, extraction_steps + 2))

  inputs, targets, forcings = modules["data_utils"].extract_inputs_targets_forcings(
      example_window,
      target_lead_times=slice(
          f"{step_hours}h", f"{extraction_steps * step_hours}h"),
      **dataclasses.asdict(task_config),
  )

  if model_config.resolution not in (0, 360.0 / inputs.sizes["lon"]):
    raise ValueError(
        "Model resolution does not match the data resolution. "
        f"model={model_config.resolution}, data={360.0 / inputs.sizes['lon']}"
    )

  if rollout_steps > extraction_steps:
    logging.info(
        "Extending target template/forcings from %d available target step(s) "
        "to %d rollout step(s). Ground-truth RMSE will not be available for "
        "the extended horizon.",
        extraction_steps,
        rollout_steps,
    )
    one_step_template = targets.isel(time=slice(0, 1))
    targets_template = modules["rollout"].extend_targets_template(
        one_step_template, rollout_steps)
    forcings = make_forcings(
        modules["data_utils"],
        example_window,
        inputs,
        targets_template.coords["time"],
        task_config.forcing_variables,
        step_hours,
    )
    truth = None
  else:
    targets_template = targets
    truth = targets

  logging.info("Inputs dims: %s", dict(inputs.sizes))
  logging.info("Targets template dims: %s", dict(targets_template.sizes))
  logging.info("Forcings dims: %s", dict(forcings.sizes))
  return inputs, targets_template * np.nan, forcings, truth, dataset_label


def build_jitted_predictor(
    modules: dict[str, Any],
    params: dict[str, Any],
    model_config: Any,
    task_config: Any,
    stats: dict[str, xr.Dataset],
):
  hk = modules["hk"]
  jax = modules["jax"]
  graphcast_module = modules["graphcast"]
  casting_module = modules["casting"]
  normalization_module = modules["normalization"]

  def construct_wrapped_graphcast():
    predictor = graphcast_module.GraphCast(model_config, task_config)
    predictor = casting_module.Bfloat16Cast(predictor)
    predictor = normalization_module.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=stats["diffs_stddev_by_level"],
        mean_by_level=stats["mean_by_level"],
        stddev_by_level=stats["stddev_by_level"],
    )
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


def next_inputs_from_prediction(prev_inputs: xr.Dataset, next_frame: xr.Dataset) -> xr.Dataset:
  missing = list(set(prev_inputs.keys()) - set(next_frame.keys()))
  if missing and "time" in prev_inputs[missing].dims:
    raise ValueError(
        "Found time-dependent input variables that were neither predicted nor "
        f"forced: {missing}"
    )

  next_keys = list(set(next_frame.keys()).intersection(prev_inputs.keys()))
  next_inputs = next_frame[next_keys]
  return xr.concat(
      [prev_inputs, next_inputs], dim="time", data_vars="different"
  ).tail(time=prev_inputs.sizes["time"])


def run_autoregressive_rollout(
    modules: dict[str, Any],
    predictor: Any,
    rng_seed: int,
    inputs: xr.Dataset,
    targets_template: xr.Dataset,
    forcings: xr.Dataset,
    rollout_steps: int,
    step_hours: int,
    keep_inputs_on_device: bool,
) -> xr.DataArray:
  jax = modules["jax"]
  rng = jax.random.PRNGKey(rng_seed)
  base_target_time = targets_template.coords["time"].isel(time=slice(0, 1))
  one_step_template = targets_template.isel(time=slice(0, 1)).assign_coords(
      time=base_target_time)
  current_inputs = inputs
  input_time_coords = inputs.coords["time"]
  t2m_predictions = []

  for step in range(rollout_steps):
    lead_hours = (step + 1) * step_hours
    logging.info(
        "Rollout step %02d/%02d: lead time +%03d h",
        step + 1,
        rollout_steps,
        lead_hours,
    )

    target_slice = slice(step, step + 1)
    actual_target_time = targets_template.coords["time"].isel(time=target_slice)
    current_forcings = forcings.isel(time=target_slice).assign_coords(
        time=base_target_time).compute()

    rng, step_rng = jax.random.split(rng)
    predictions = predictor(
        step_rng,
        current_inputs,
        one_step_template,
        current_forcings,
    )

    predictions_actual_time = predictions.assign_coords(time=actual_target_time)
    t2m_predictions.append(
        jax.device_get(predictions_actual_time["2m_temperature"]))

    if not keep_inputs_on_device:
      predictions = jax.device_get(predictions)
      current_forcings = jax.device_get(current_forcings)
      current_inputs = jax.device_get(current_inputs)

    if step != rollout_steps - 1:
      next_frame = xr.merge([predictions, current_forcings])
      current_inputs = next_inputs_from_prediction(
          current_inputs, next_frame).assign_coords(time=input_time_coords)

  return xr.concat(t2m_predictions, dim="time")


def lat_weighted_global_mean(field: xr.DataArray) -> xr.DataArray:
  weights = xr.DataArray(
      np.cos(np.deg2rad(field.coords["lat"].values)).astype(np.float32),
      coords={"lat": field.coords["lat"]},
      dims=("lat",),
  )
  result = field.weighted(weights).mean(("lat", "lon"), skipna=True)
  for dim in list(result.dims):
    if dim != "time":
      result = result.mean(dim, skipna=True)
  return result


def selected_reference(
    predictions_t2m: xr.DataArray,
    inputs: xr.Dataset,
    truth: xr.Dataset | None,
    reference_mode: str,
) -> tuple[xr.DataArray, str]:
  has_full_truth = (
      truth is not None
      and "2m_temperature" in truth
      and truth.sizes.get("time", 0) >= predictions_t2m.sizes["time"]
  )

  if reference_mode == "ground_truth" and not has_full_truth:
    raise ValueError(
        "Ground truth requested, but the dataset does not contain all rollout "
        "target steps. Use --reference initial_state or provide a longer dataset."
    )

  if reference_mode == "ground_truth" or (reference_mode == "auto" and has_full_truth):
    ref = truth["2m_temperature"].isel(time=slice(0, predictions_t2m.sizes["time"]))
    ref = ref.assign_coords(time=predictions_t2m.coords["time"])
    return ref, "ground_truth"

  return inputs["2m_temperature"].isel(time=-1), "initial_state"


def timedelta_hours(values: np.ndarray) -> np.ndarray:
  return values.astype("timedelta64[s]").astype(np.float64) / 3600.0


def compute_t2m_metrics(
    predictions_t2m: xr.DataArray,
    inputs: xr.Dataset,
    truth: xr.Dataset | None,
    reference_mode: str,
    step_hours: int,
    metrics_frequency: str,
) -> pd.DataFrame:
  reference_t2m, reference_label = selected_reference(
      predictions_t2m, inputs, truth, reference_mode)
  diff = predictions_t2m - reference_t2m

  rmse = np.sqrt(lat_weighted_global_mean(diff ** 2))
  bias = lat_weighted_global_mean(diff)
  lead_hours = timedelta_hours(predictions_t2m.coords["time"].values)

  frame = pd.DataFrame({
      "step": np.arange(1, predictions_t2m.sizes["time"] + 1, dtype=np.int32),
      "lead_hours": lead_hours,
      "lead_day": lead_hours / 24.0,
      "t2m_rmse_k": np.asarray(rmse.values, dtype=np.float64),
      "t2m_mean_bias_k": np.asarray(bias.values, dtype=np.float64),
      "reference": reference_label,
  })

  if metrics_frequency == "daily":
    steps_per_day = int(round(24 / step_hours))
    frame = frame[frame["step"] % steps_per_day == 0].copy()
    frame["day"] = (frame["step"] // steps_per_day).astype(np.int32)
    cols = [
        "day",
        "step",
        "lead_hours",
        "lead_day",
        "t2m_rmse_k",
        "t2m_mean_bias_k",
        "reference",
    ]
    frame = frame[cols]

  return frame


def plot_metrics(metrics: pd.DataFrame, plot_path: Path) -> None:
  import matplotlib  # pylint: disable=import-outside-toplevel

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

  x_col = "day" if "day" in metrics.columns else "lead_day"
  fig, ax_rmse = plt.subplots(figsize=(8, 4.5))
  ax_bias = ax_rmse.twinx()

  ax_rmse.plot(metrics[x_col], metrics["t2m_rmse_k"], marker="o", color="#1f77b4")
  ax_bias.plot(metrics[x_col], metrics["t2m_mean_bias_k"], marker="s", color="#d62728")

  ax_rmse.set_xlabel("Forecast day")
  ax_rmse.set_ylabel("Latitude-weighted t2m RMSE (K)", color="#1f77b4")
  ax_bias.set_ylabel("Latitude-weighted t2m mean bias (K)", color="#d62728")
  ax_rmse.grid(True, alpha=0.3)
  fig.tight_layout()
  plot_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(plot_path, dpi=160)
  plt.close(fig)


def main() -> None:
  args = parse_args()
  logging.basicConfig(
      level=getattr(logging, args.log_level.upper()),
      format="%(asctime)s %(levelname)s %(message)s",
  )
  configure_jax_memory(args)
  modules = import_graphcast_stack()

  bucket = anonymous_bucket(modules["storage"])
  ckpt, checkpoint_name = load_checkpoint(
      bucket,
      modules["checkpoint"],
      modules["graphcast"],
      args.cache_dir,
      args.model,
      args.checkpoint_name,
  )
  stats = load_normalization_stats(bucket, args.cache_dir)

  inputs, targets_template, forcings, truth, dataset_label = prepare_data(
      modules,
      bucket,
      args.cache_dir,
      ckpt.model_config,
      ckpt.task_config,
      args.rollout_steps,
      args.step_hours,
      args.dataset_path,
      args.dataset_name,
      args.dummy_data,
  )

  predictor = build_jitted_predictor(
      modules,
      ckpt.params,
      ckpt.model_config,
      ckpt.task_config,
      stats,
  )

  logging.info("Starting autoregressive rollout")
  predictions_t2m = run_autoregressive_rollout(
      modules,
      predictor,
      args.rng_seed,
      inputs,
      targets_template,
      forcings,
      args.rollout_steps,
      args.step_hours,
      args.keep_inputs_on_device,
  )

  metrics = compute_t2m_metrics(
      predictions_t2m,
      inputs,
      truth,
      args.reference,
      args.step_hours,
      args.metrics_frequency,
  )
  args.output_csv.parent.mkdir(parents=True, exist_ok=True)
  metrics.to_csv(args.output_csv, index=False)
  logging.info("Wrote metrics CSV: %s", args.output_csv)

  if not args.no_plot:
    plot_metrics(metrics, args.plot_path)
    logging.info("Wrote metrics plot: %s", args.plot_path)

  logging.info("Checkpoint: %s", checkpoint_name)
  logging.info("Dataset: %s", dataset_label)
  logging.info("Reference: %s", metrics["reference"].iloc[0])


if __name__ == "__main__":
  main()
