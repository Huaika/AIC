from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
  from .plots import plot_global_mean_spaghetti_csvs
else:
  from plots import plot_global_mean_spaghetti_csvs


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Plot spaghetti-style global-mean GraphCast rollout evolution."
  )
  parser.add_argument(
      "--csv",
      action="append",
      required=True,
      help="Global-mean evolution CSV. Pass once per rollout or use shell expansion.",
  )
  parser.add_argument(
      "--year",
      type=int,
      required=True,
      help="Year shown in the plot title.",
  )
  parser.add_argument(
      "--output",
      type=Path,
      required=True,
      help="Output PNG path.",
  )
  parser.add_argument(
      "--metric-label",
      help="Human-readable metric label. Inferred from CSV metric_id if omitted.",
  )
  parser.add_argument(
      "--line-color",
      default="#238b45",
      help="Color for forecast rollout lines.",
  )
  parser.add_argument(
      "--every",
      type=int,
      default=1,
      help="Plot every Nth initialization time.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  plot_global_mean_spaghetti_csvs(
      args.csv,
      args.output,
      args.year,
      metric_label=args.metric_label,
      line_color=args.line_color,
      every=args.every,
  )
  print(args.output)


if __name__ == "__main__":
  main()
