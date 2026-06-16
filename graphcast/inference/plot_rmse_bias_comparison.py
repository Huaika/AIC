from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
  from .plots import plot_rmse_bias_csv_comparison
else:
  from plots import plot_rmse_bias_csv_comparison


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Plot RMSE and mean bias from multiple GraphCast metrics CSVs."
  )
  parser.add_argument(
      "--series",
      action="append",
      required=True,
      help="Series label and CSV path as LABEL=path/to/metrics.csv.",
  )
  parser.add_argument(
      "--metric-id",
      help="Metric id prefix, e.g. temperature_850hPa. Inferred if omitted.",
  )
  parser.add_argument(
      "--output",
      type=Path,
      required=True,
      help="Output PNG path.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  plot_rmse_bias_csv_comparison(args.series, args.output, args.metric_id)
  print(args.output)


if __name__ == "__main__":
  main()
