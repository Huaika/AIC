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
