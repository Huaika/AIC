from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


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


def select_metric_field(
    dataset: xr.Dataset,
    variable: str,
    level: int | None,
) -> xr.DataArray:
  if variable not in dataset:
    available = ", ".join(sorted(dataset.data_vars))
    raise KeyError(f"Variable {variable!r} not found. Available variables: {available}")

  field = dataset[variable]
  if "level" in field.dims:
    if level is None:
      raise ValueError(
          f"Variable {variable!r} has a level dimension. Set "
          "GRAPHCAST_METRIC_LEVEL, for example GRAPHCAST_METRIC_LEVEL=850."
      )
    if level not in set(int(value) for value in field.coords["level"].values):
      raise ValueError(
          f"Level {level} is not available for {variable!r}. Available levels: "
          f"{field.coords['level'].values.tolist()}"
      )
    return field.sel(level=level)

  if level is not None:
    raise ValueError(
        f"GRAPHCAST_METRIC_LEVEL={level} was set, but {variable!r} has no "
        "level dimension."
    )
  return field


def selected_reference(
    predictions: xr.DataArray,
    inputs: xr.Dataset,
    truth: xr.Dataset | None,
    reference_mode: str,
    metric_variable: str,
    metric_level: int | None,
) -> tuple[xr.DataArray, str]:
  has_full_truth = (
      truth is not None
      and metric_variable in truth
      and truth.sizes.get("time", 0) >= predictions.sizes["time"]
  )

  if reference_mode == "ground_truth" and not has_full_truth:
    raise ValueError(
        "Ground truth requested, but the dataset does not contain all rollout "
        "target steps. Use REFERENCE_MODE='initial_state' or provide a longer "
        "dataset."
    )

  if reference_mode == "ground_truth" or (reference_mode == "auto" and has_full_truth):
    ref = select_metric_field(truth, metric_variable, metric_level).isel(
        time=slice(0, predictions.sizes["time"]))
    ref = ref.assign_coords(time=predictions.coords["time"])
    return ref, "ground_truth"

  return select_metric_field(inputs, metric_variable, metric_level).isel(
      time=-1), "initial_state"


def timedelta_hours(values: np.ndarray) -> np.ndarray:
  return values.astype("timedelta64[s]").astype(np.float64) / 3600.0


def compute_metric_summary(
    predictions: xr.DataArray,
    inputs: xr.Dataset,
    truth: xr.Dataset | None,
    reference_mode: str,
    step_hours: int,
    metrics_frequency: str,
    metric_variable: str,
    metric_level: int | None,
    metric_id: str,
) -> pd.DataFrame:
  reference_field, reference_label = selected_reference(
      predictions, inputs, truth, reference_mode, metric_variable, metric_level)
  diff = predictions - reference_field

  rmse = np.sqrt(lat_weighted_global_mean(diff ** 2))
  bias = lat_weighted_global_mean(diff)
  lead_hours = timedelta_hours(predictions.coords["time"].values)
  rmse_col = f"{metric_id}_rmse_k"
  bias_col = f"{metric_id}_mean_bias_k"

  frame = pd.DataFrame({
      "step": np.arange(1, predictions.sizes["time"] + 1, dtype=np.int32),
      "lead_hours": lead_hours,
      "lead_day": lead_hours / 24.0,
      rmse_col: np.asarray(rmse.values, dtype=np.float64),
      bias_col: np.asarray(bias.values, dtype=np.float64),
      "metric_variable": metric_variable,
      "metric_level_hpa": metric_level,
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
        rmse_col,
        bias_col,
        "metric_variable",
        "metric_level_hpa",
        "reference",
    ]
    frame = frame[cols]

  return frame
