# Configuration

The pipeline is configured through environment variables. `aic.config` centralizes
the two things that were previously scattered: the **cluster paths** and the
**typed readers** for the env knobs.

## Cluster paths (`aic.config`)

Every absolute workspace path defaults to the BW UniCluster layout but is
overridable, so the code runs elsewhere without edits:

| env var          | default                                               | meaning                         |
|------------------|-------------------------------------------------------|---------------------------------|
| `AIC_WORKSPACE`  | `/pfs/work9/workspace/scratch`                        | base scratch workspace          |
| `AIC_DATA_ROOT`  | `<AIC_WORKSPACE>/ka_dm9435-ai-climate`                | this project's data root        |
| `AIC_COAST_ZARR` | `<AIC_WORKSPACE>/ka_je2428-nextgems_2049/constant_fields.zarr` | land-sea mask for coastlines |

## Typed accessors

Instead of ad-hoc `int(os.environ.get(...))`, read knobs through `aic.config`:

```python
from aic import config
window = config.env_int("HW_WINDOW", 5)
pct    = config.env_float("HW_PCT", 0.95)
force  = config.env_bool("AIC_FORCE_REBUILD", False)
years  = config.env_list("CS_YEARS", ["2023", "2026"])   # comma/space separated
year   = config.env_required("ERA5_YEAR")                # no default -> clear error
```

All first-party env reads go through these (the vendored GraphCast fork excepted);
the only remaining raw `os.environ` calls are the `os.environ.setdefault(...)` JAX
flags in the rollout scripts, which *set* env for a subprocess rather than read it.

## The main knobs by area

Full list is discoverable with `grep -rho 'os\.environ[^)]*' src/aic`. The families:

- **Selection (eval + views):** `EVAL_RUN` / `EVAL_SOURCES`+`EVAL_YEAR` / `EVAL_RUNS`
  (which model+dataset+year), `EVAL_VARS`, `EVAL_REGIONS`, `EVAL_MONTHS`,
  `NG_LEVELS`, `EVAL_FIG_ROOT` (where figures go — point at scratch to avoid the
  data6 write hang).
- **Case study:** `HW_CS_DEF` (mixture|cordex|…), `HW_PCT`, `CS_YEARS`, `CS_MODELS`,
  `CS_BEFORE`/`CS_AFTER`, `CS_ONLY_EP`, `HW_MANHATTAN` (spatio-temporal separation).
- **Heat-wave detection:** `HW_WINDOW(S)`, `HW_Q`, `HW_MIN_DURATION`, `HW_EPISODE_GAP`,
  `HW_SPEC_REGION`, `HW_DEFS`, `HW_CACHE_DIR`, `HW_GIF_DIR`.
- **Caching:** `AIC_FORCE_REBUILD=1` forces derived caches (drift-map day-10 fields)
  to be rebuilt instead of served — they are otherwise reused only when newer than
  their inputs.
- **GraphCast / NeuralGCM inference:** the `GRAPHCAST_*` and `NG_*` families (see
  `model/graphcast/inference` and `controller/rollout`).

## Known cluster caveat

Writing to the data6 filesystem hangs on many compute nodes. Set
`EVAL_FIG_ROOT` to a work9 scratch path when generating figures on compute nodes,
then copy the results back and commit from the login node.
