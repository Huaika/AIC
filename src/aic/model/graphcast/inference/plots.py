from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def configure_matplotlib() -> None:
  config_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"
  os.environ.setdefault("MPLCONFIGDIR", str(config_dir))
  config_dir.mkdir(parents=True, exist_ok=True)


def infer_metric_id(metrics_by_label: dict[str, pd.DataFrame]) -> str:
  for metrics in metrics_by_label.values():
    for column in metrics.columns:
      if column.endswith("_rmse_k"):
        return column.removesuffix("_rmse_k")
  raise ValueError("Could not infer metric id: no '*_rmse_k' column found.")


def read_labeled_metrics_csvs(items: list[str]) -> dict[str, pd.DataFrame]:
  metrics_by_label = {}
  for item in items:
    if "=" not in item:
      raise ValueError(f"Expected LABEL=CSV for metrics series, got {item!r}")
    label, path = item.split("=", 1)
    metrics_by_label[label] = pd.read_csv(Path(path))
  return metrics_by_label


def read_evolution_csvs(paths: list[str | Path]) -> pd.DataFrame:
  frames = [pd.read_csv(Path(path)) for path in paths]
  if not frames:
    raise ValueError("At least one global-mean evolution CSV is required.")
  return pd.concat(frames, ignore_index=True)


def add_spaghetti_time_columns(evolution: pd.DataFrame) -> pd.DataFrame:
  required = {"init_time", "lead_hours", "global_mean_k"}
  missing = required - set(evolution.columns)
  if missing:
    raise KeyError(f"Evolution data is missing columns: {sorted(missing)}")

  evolution = evolution.copy()
  evolution["init_datetime"] = pd.to_datetime(evolution["init_time"])
  evolution["valid_time"] = (
      evolution["init_datetime"]
      + pd.to_timedelta(evolution["lead_hours"], unit="h")
  )
  evolution["init_date"] = evolution["init_datetime"].dt.floor("D")
  evolution["lead_day_idx"] = (evolution["lead_hours"] // 24).astype(int)
  return evolution


def plot_metrics(metrics: pd.DataFrame, plot_path: Path, metric_id: str) -> None:
  configure_matplotlib()
  import matplotlib  # pylint: disable=import-outside-toplevel

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

  x_col = "day" if "day" in metrics.columns else "lead_day"
  rmse_col = f"{metric_id}_rmse_k"
  bias_col = f"{metric_id}_mean_bias_k"
  fig, ax_rmse = plt.subplots(figsize=(8, 4.5))
  ax_bias = ax_rmse.twinx()

  ax_rmse.plot(metrics[x_col], metrics[rmse_col], marker="o", color="#1f77b4")
  ax_bias.plot(metrics[x_col], metrics[bias_col], marker="s", color="#d62728")

  ax_rmse.set_xlabel("Forecast day")
  ax_rmse.set_ylabel(f"Latitude-weighted {metric_id} RMSE (K)", color="#1f77b4")
  ax_bias.set_ylabel(f"Latitude-weighted {metric_id} mean bias (K)", color="#d62728")
  ax_rmse.grid(True, alpha=0.3)
  fig.tight_layout()
  plot_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(plot_path, dpi=160)
  plt.close(fig)


def plot_rmse_bias_comparison(
    metrics_by_label: dict[str, pd.DataFrame],
    plot_path: Path,
    metric_id: str,
) -> None:
  configure_matplotlib()
  import matplotlib  # pylint: disable=import-outside-toplevel

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

  rmse_col = f"{metric_id}_rmse_k"
  bias_col = f"{metric_id}_mean_bias_k"
  colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b")
  fig, ax = plt.subplots(figsize=(8.8, 5.0))

  for index, (label, metrics) in enumerate(metrics_by_label.items()):
    missing = {rmse_col, bias_col} - set(metrics.columns)
    if missing:
      raise KeyError(f"{label!r} metrics CSV is missing columns: {sorted(missing)}")
    x_col = "day" if "day" in metrics.columns else "lead_day"
    metrics = metrics.sort_values(x_col)
    color = colors[index % len(colors)]
    ax.plot(
        metrics[x_col],
        metrics[rmse_col],
        color=color,
        marker="o",
        linewidth=1.6,
        label=f"{label} RMSE",
    )
    ax.plot(
        metrics[x_col],
        metrics[bias_col],
        color=color,
        marker="s",
        linewidth=1.4,
        linestyle="--",
        label=f"{label} mean bias",
    )

  ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.5)
  ax.set_xlabel("Forecast day")
  ax.set_ylabel("Error (K)")
  ax.set_title(f"{metric_id.replace('_', ' ')} RMSE and mean bias")
  ax.grid(True, alpha=0.25)
  ax.legend(frameon=False, ncols=2, fontsize=9)
  fig.tight_layout()
  plot_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(plot_path, dpi=180)
  plt.close(fig)


def plot_rmse_bias_csv_comparison(
    series: list[str],
    plot_path: Path,
    metric_id: str | None = None,
) -> None:
  metrics_by_label = read_labeled_metrics_csvs(series)
  plot_rmse_bias_comparison(
      metrics_by_label,
      plot_path,
      metric_id or infer_metric_id(metrics_by_label),
  )


def plot_global_mean_evolution(
    evolution: pd.DataFrame,
    plot_path: Path,
    metric_id: str,
) -> None:
  configure_matplotlib()
  import matplotlib  # pylint: disable=import-outside-toplevel

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

  metric_label = metric_id.replace("_", " ")
  fig, ax = plt.subplots(figsize=(8.5, 4.8))

  group_col = "init_time" if evolution["init_time"].nunique() > 1 else None
  groups = list(evolution.groupby(group_col, sort=False)) if group_col else [(None, evolution)]

  reference_label_used = False
  for _, group in groups:
    if not group["reference_global_mean_k"].notna().any():
      continue
    reference = group.dropna(subset=["reference_global_mean_k"])
    ax.plot(
        reference["lead_day"],
        reference["reference_global_mean_k"],
        color="#bdbdbd",
        linewidth=1.0,
        alpha=0.75,
        label="Reference analysis" if not reference_label_used else None,
        zorder=1,
    )
    reference_label_used = True

  forecast_label_used = False
  for _, group in groups:
    ax.plot(
        group["lead_day"],
        group["global_mean_k"],
        color="#238b45",
        linewidth=0.8,
        alpha=0.85,
        label="GraphCast forecast" if not forecast_label_used else None,
        zorder=2,
    )
    forecast_label_used = True

  initial_label_used = False
  for _, group in groups:
    initial_values = group["initial_global_mean_k"].dropna()
    if initial_values.empty:
      continue
    ax.axhline(
        initial_values.iloc[0],
        color="black",
        linestyle="--",
        linewidth=0.9,
        alpha=0.8,
        label="Initial global mean" if not initial_label_used else None,
        zorder=3,
    )
    initial_label_used = True

  ax.set_xlabel("Forecast lead time (days)")
  ax.set_ylabel(f"Global-mean {metric_label} (K)")
  ax.set_title("GraphCast global-mean evolution")
  ax.grid(True, alpha=0.25)
  ax.legend(frameon=False, fontsize=9)
  fig.tight_layout()
  plot_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(plot_path, dpi=180)
  plt.close(fig)


def plot_global_mean_spaghetti(
    evolution: pd.DataFrame,
    plot_path: Path,
    year: int,
    *,
    metric_label: str | None = None,
    line_color: str = "#238b45",
    every: int = 1,
) -> None:
  configure_matplotlib()
  import matplotlib  # pylint: disable=import-outside-toplevel

  matplotlib.use("Agg")
  import matplotlib.dates as mdates  # pylint: disable=import-outside-toplevel
  import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

  evolution = add_spaghetti_time_columns(evolution)
  if metric_label is None:
    metric_ids = evolution["metric_id"].dropna() if "metric_id" in evolution else pd.Series()
    metric_label = metric_ids.iloc[0].replace("_", " ") if not metric_ids.empty else "field"

  fig, ax = plt.subplots(figsize=(10, 5.2))

  if "reference_global_mean_k" in evolution and evolution["reference_global_mean_k"].notna().any():
    reference = (
        evolution[["valid_time", "reference_global_mean_k"]]
        .dropna()
        .drop_duplicates("valid_time")
        .copy()
    )
    reference["date"] = reference["valid_time"].dt.floor("D")
    reference_daily = (
        reference.groupby("date", as_index=False)["reference_global_mean_k"].mean()
    )
    ax.plot(
        reference_daily["date"],
        reference_daily["reference_global_mean_k"],
        color="black",
        linewidth=2.2,
        zorder=1,
        label="Reference daily mean",
    )

  linewidth, alpha = (0.5, 0.5) if every == 1 else (0.9, 0.8)
  init_times = sorted(evolution["init_datetime"].dropna().unique())[::every]
  for init_time in init_times:
    group = evolution[evolution["init_datetime"] == init_time]
    daily = (
        group.groupby("lead_day_idx")
        .agg(valid_time=("valid_time", "mean"), global_mean_k=("global_mean_k", "mean"))
        .reset_index()
    )
    ax.plot(
        daily["valid_time"],
        daily["global_mean_k"],
        color=line_color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=2,
    )

  cadence = "every day" if every == 1 else f"every {every}th day"
  model_names = evolution["model"].dropna() if "model" in evolution else pd.Series()
  model_label = model_names.iloc[0] if not model_names.empty else "GraphCast"
  ax.plot(
      [],
      [],
      color=line_color,
      linewidth=1.2,
      alpha=0.9,
      label=f"{model_label} 10-day rollout, daily mean ({cadence})",
  )

  ax.set_title(f"Global-mean {metric_label} - {year}")
  ax.set_ylabel("Global mean [K]")
  ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
  ax.xaxis.set_major_locator(mdates.MonthLocator())
  ax.margins(x=0.01)
  ax.grid(alpha=0.25)
  ax.legend(loc="upper left", framealpha=0.9)
  fig.tight_layout()
  plot_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(plot_path, dpi=180)
  plt.close(fig)


def plot_global_mean_spaghetti_csvs(
    csv_paths: list[str | Path],
    plot_path: Path,
    year: int,
    *,
    metric_label: str | None = None,
    line_color: str = "#238b45",
    every: int = 1,
) -> None:
  plot_global_mean_spaghetti(
      read_evolution_csvs(csv_paths),
      plot_path,
      year,
      metric_label=metric_label,
      line_color=line_color,
      every=every,
  )
