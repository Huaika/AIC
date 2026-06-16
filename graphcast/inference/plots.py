from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_metrics(metrics: pd.DataFrame, plot_path: Path, metric_id: str) -> None:
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


def plot_global_mean_evolution(
    evolution: pd.DataFrame,
    plot_path: Path,
    metric_id: str,
) -> None:
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
