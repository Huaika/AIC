# Heatwave case study — NeuralGCM over European heatwave episodes

Evaluates NeuralGCM 10-day rollouts **only over the grid cells classified as a heat
wave**, for Europe in **2023** and **2026**, using the **mixture** definition at the
**99th percentile** (daily T₈₅₀ minimum *and* maximum both above their day-of-year
p99 threshold, ±5-day climatology window, 1991–2020 reference; ≥3-consecutive-day
spells).

## What it produces

Heatwave activity is split into temporally connected **episodes** (runs of days
where ≥2 % of Europe, cos-lat weighted, is in a heatwave, merged across ≤2-day
gaps). Each episode has a spatial **footprint** = the union of all cells active on
any day of the episode. For every episode, for T₈₅₀ and Z₅₀₀:

| figure | approach | region analysed | time frame |
|---|---|---|---|
| `spaghetti_*` | **Lagrangian** | footprint area-mean | episode ± 10 days |
| `rmse-vs-lead_*` | **Lagrangian** | footprint | rollouts initialised in [start−10 d, end] |
| `drift-map_*` | **Eulerian** | footprint (cells outside left uncoloured) | day-10 valid window |

Plus a per-year `_overview_coverage_<year>.pdf` (Europe heatwave coverage timeline
with the episodes shaded).

- **Eulerian** (drift maps): the whole footprint is analysed over the time frame;
  cells never in the heatwave are left out of the analysis and not coloured.
- **Lagrangian** (spaghetti, RMSE-over-rollout): follows the afflicted footprint over
  its heatwave lifetime plus the roll-out window before and a brief window after.

## Running

```bash
sbatch case_study/run_case_study.sbatch
```

Figures are written to `outputs/figures/case_study/mixture_p99/<year>/<episode>/`.

## Code

- `src/aic/controller/casestudy/heatwave_mask.py` — builds the mixture/p99 active
  mask (`active_mask(hot_mask(...))`) on the 2.8° model grid from the cached daily
  stats, and splits it into `Episode`s with footprints.
- `src/aic/controller/casestudy/plots.py` — per-episode spaghetti / RMSE / drift-map
  rendering.
- `src/aic/controller/eval/gridpoints.py` — the **grid-point-set** primitive
  (`GridPoints`, `masked_area_mean`) and the `boxes_to_points` helper that turns
  rectangles into a grid-point set; the `Source` eval methods reduce through it, so
  the global / regional / out-of-distribution analyses and this case study share one
  masked, cos-lat-weighted reduction path.

## Notes

- NeuralGCM rollouts, the ERA5 truth caches and the heatwave masks are all on the
  identical 128×64 / 2.8° grid, so the footprint mask combines with the rollouts by
  coordinates — no regridding.
- **2026** is a partial year: ERA5/heatwave data end 2026-07-24 and NeuralGCM 2026
  rollouts are initialised only through 2026-06-30, so the long June–July 2026
  episode is covered by rollouts only through its early-July portion.

Definition / percentile: `HW_PCT` + `HW_CS_DEF` (see `heatwave/definitions.py`).
