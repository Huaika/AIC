from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

import haiku as hk  # pylint: disable=import-outside-toplevel
import jax  # pylint: disable=import-outside-toplevel
from google.api_core import exceptions as google_exceptions  # pylint: disable=import-outside-toplevel
from google.cloud import storage  # pylint: disable=import-outside-toplevel
from graphcast import casting  # pylint: disable=import-outside-toplevel
from graphcast import checkpoint  # pylint: disable=import-outside-toplevel
from graphcast import data_utils  # pylint: disable=import-outside-toplevel
from graphcast import graphcast  # pylint: disable=import-outside-toplevel
from graphcast import normalization  # pylint: disable=import-outside-toplevel
from graphcast import rollout  # pylint: disable=import-outside-toplevel
from graphcast import xarray_jax  # pylint: disable=import-outside-toplevel,unused-import


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
KEEP_INPUTS_ON_DEVICE = True
METRICS_FREQUENCY = os.environ.get("GRAPHCAST_METRICS_FREQUENCY", "daily")
REFERENCE_MODE = os.environ.get("GRAPHCAST_REFERENCE_MODE", "auto")
OUTPUT_CSV = Path(os.environ.get("GRAPHCAST_OUTPUT_CSV", "t2m_drift_metrics.csv"))
PLOT_PATH = Path(os.environ.get("GRAPHCAST_PLOT_PATH", "t2m_drift_metrics.png"))
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
) -> str:
  init_time = os.environ.get("GRAPHCAST_INIT_TIME")
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
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset, str]:
  """Read reference ERA5 and extract GraphCast inputs, template, forcings, truth."""
  reference_uri = configured_reference_era5_uri(
      cache_dir, task_config, rollout_steps, step_hours)
  reference, label = open_reference_era5(bucket, cache_dir, reference_uri)
  if reference.sizes["time"] < 3:
    raise ValueError("Reference ERA5 needs at least 3 timesteps: 2 inputs + 1 target.")

  available_target_steps = reference.sizes["time"] - 2
  target_steps = min(rollout_steps, available_target_steps)
  reference_window = reference.isel(time=slice(0, target_steps + 2))
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
):
  """Build a jitted one-step GraphCast predictor with casting and normalization."""
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
        "target steps. Use REFERENCE_MODE='initial_state' or provide a longer "
        "dataset."
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
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
  cache_dir = Path(os.environ.get("GRAPHCAST_CACHE_DIR", "~/.cache/graphcast")).expanduser()
  rollout_steps = int(os.environ.get("GRAPHCAST_ROLLOUT_STEPS", DEFAULT_ROLLOUT_STEPS))
  step_hours = int(os.environ.get("GRAPHCAST_STEP_HOURS", DEFAULT_STEP_HOURS))
  modules = {
      "hk": hk,
      "jax": jax,
      "graphcast": graphcast,
      "casting": casting,
      "normalization": normalization,
  }
  bucket = anonymous_bucket(storage)
  ckpt, checkpoint_name = load_checkpoint(
      bucket,
      checkpoint,
      graphcast,
      cache_dir,
      "operational",
      os.environ.get("GRAPHCAST_CHECKPOINT_NAME"),
  )

  stats = load_normalization_stats(bucket, cache_dir)
  inputs, targets_template, forcings, reference_targets, reference_label = read_reference_era5(
      bucket,
      cache_dir,
      ckpt.task_config,
      rollout_steps,
      step_hours,
  )
  
  logging.info("Prepared reference ERA5: %s", reference_label)
  logging.info("Inputs dims: %s", dict(inputs.sizes))
  logging.info("Targets template dims: %s", dict(targets_template.sizes))
  logging.info("Forcings dims: %s", dict(forcings.sizes))
  logging.info("Reference target dims: %s", dict(reference_targets.sizes))
  logging.info("Checkpoint: %s", checkpoint_name)
  logging.info("Loaded stats: %s", ", ".join(sorted(stats)))
  
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
    DEFAULT_SEED,
    inputs,
    targets_template,
    forcings,
    rollout_steps,
    step_hours,
    KEEP_INPUTS_ON_DEVICE,
  )

  metrics = compute_t2m_metrics(
      predictions_t2m,
      inputs,
      reference_targets,
      REFERENCE_MODE,
      step_hours,
      METRICS_FREQUENCY,
  )
  OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
  metrics.to_csv(OUTPUT_CSV, index=False)
  logging.info("Wrote metrics CSV: %s", OUTPUT_CSV)

  if WRITE_PLOT:
    plot_metrics(metrics, PLOT_PATH)
    logging.info("Wrote metrics plot: %s", PLOT_PATH)

  logging.info("Checkpoint: %s", checkpoint_name)
  logging.info("Dataset: %s", reference_label)
  logging.info("Reference: %s", metrics["reference"].iloc[0])

if __name__ == "__main__":
  main()
