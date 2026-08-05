# Running the analyses

Each analysis is a module with a `main()` and a matching console entry point
(installed by `pip install -e .`). Both forms are equivalent:

| entry point            | module (`python -m …`)                | what it makes                         |
|------------------------|---------------------------------------|---------------------------------------|
| `aic-spaghetti`        | `aic.view.spaghetti`                  | area-mean rollout spaghetti (one year)|
| `aic-drift`            | `aic.view.drift`                      | RMSE + bias vs lead                    |
| `aic-drift-maps`       | `aic.view.drift_maps`                 | day-10 drift maps                      |
| `aic-ood`              | `aic.view.ood`                        | multi-year out-of-distribution figures |
| `aic-casestudy`        | `aic.controller.casestudy.plots`      | heat-wave case study (per episode + aggregate) |
| `aic-heatwave-compare` | `aic.controller.heatwave.compare`     | compare heat-wave definitions          |

Configuration is entirely via env vars — see [configuration.md](configuration.md).

## Examples

```bash
# NeuralGCM vs GraphCast drift maps, 2023, Europe + world, T@850, to scratch
EVAL_SOURCES=neuralgcm,graphcast EVAL_YEAR=2023 EVAL_REGIONS=world,europe \
  EVAL_VARS=temperature NG_LEVELS=850 EVAL_FIG_ROOT=$SCRATCH/figs \
  aic-drift-maps

# Heat-wave case study, cordex definition, 2023+2026, both models
EVAL_RUN=era5_2023 HW_CS_DEF=cordex HW_PCT=0.99 CS_YEARS="2023 2026" \
  CS_MODELS=neuralgcm,graphcast EVAL_VARS=temperature,geopotential \
  EVAL_FIG_ROOT=$SCRATCH/figs  aic-casestudy
```

## On the cluster (Slurm)

Use the templated submitter instead of copy-pasting `#SBATCH` headers:

```bash
slurm/submit.sh --name drift-maps --time 02:00:00 --mem 48gb --cpus 8 -- \
  env EVAL_SOURCES=neuralgcm,graphcast EVAL_YEAR=2023 EVAL_VARS=temperature \
      NG_LEVELS=850 EVAL_FIG_ROOT=$SCRATCH/figs JAX_PLATFORMS=cpu \
      aic-drift-maps
```

`AIC_PARTITION`, `AIC_LOG_DIR`, `AIC_ACCOUNT` override the submitter's defaults.
Generate figures to a work9 scratch `EVAL_FIG_ROOT`, then copy back and commit from
the login node (the data6 filesystem hangs on many compute nodes).
```
