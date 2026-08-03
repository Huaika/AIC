# Out-of-distribution (OOD) analysis

Compares **NeuralGCM** and **GraphCast** 10-day rollouts across three climates:
the historical **1955**, present-day **2023**, and the NextGEMS **2049**
future-climate (out-of-distribution) run. Global-mean 850 hPa temperature.

## Figures — `outputs/figures/out_of_distribution/temperature/`

- `drift_rmse/` — per year, **RMSE vs lead**, NeuralGCM + GraphCast overlaid.
- `drift_bias/` — per year, **mean bias vs lead**, both models overlaid (RMSE and
  bias are separate figures for readability).
- `spaghetti/` — a **single** figure overlaying every (model, year) rollout bundle
  plus one reference (truth) line per era: ERA5 for 1955 & 2023, NextGEMS for 2049.
  Colour = model, line style = year, on a day-of-year axis.

## Running

```bash
sbatch ood/run_ood.sbatch
```

## Code

- `src/aic/view/ood.py` — the OOD driver (`skill_year`, `spaghetti_multiyear`).
- `src/aic/view/plotting.py` — shared matplotlib primitives (`draw_rollout_bundle`,
  `draw_skill_metric`, `map_panel`, `doy_axis`, `month_axis`) used by every view.
- Sources are selected with `EVAL_RUNS` (an explicit run-key list mixing models,
  datasets and years); model colours come from `sources.MODEL_COLORS`.

The truth for `graphcast_nextgems_2049` is content-identical to `nextgems2049`
(same NextGEMS source, same 2.8° regrid), so it is copied rather than rebuilt.
