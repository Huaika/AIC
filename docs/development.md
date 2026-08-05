# Development

## Setup

```bash
# analysis + dev tooling (no CUDA ML stack)
pip install -e .[dev]
pre-commit install

# the heavy inference stack (neuralgcm, jax[cuda12], dinosaur, ...) — cluster only,
# built into the pinned venv:  see requirements.txt / setup_env.sh, or:
pip install -e .[ml]
```

The package imports **without** the `ml` extra: `eval_common` imports `neuralgcm`
and `dinosaur` lazily (inside the three functions that build regridders / truth
caches), so the eval + view modules and the whole test suite import with just the
analysis stack. Only the actual model-loading / truth-staging steps need `ml`.

## Tests

```bash
pytest                      # tests/ — pure functions (regions, gridpoints,
                            #          config, detect, aggregate)
```

Tests cover the layers that don't need GPUs or cluster data. The key invariant test
is `test_gridpoints.py`: a box-derived `GridPoints` area-mean must equal the old
`lat_weighted_mean(select_region(...))`, so the refactor that routed every analysis
through `GridPoints` cannot silently change a figure.

## Linting & formatting

```bash
ruff check .               # lint (import order, unused, bugbear)
ruff format .              # format
mypy                       # typed core (regions, gridpoints); widen over time
```

Config lives in `pyproject.toml`. The vendored GraphCast fork
(`src/aic/model/graphcast/source`) and `src/aic/legacy` are excluded everywhere.

## Architecture contracts

`.importlinter` encodes the MVC dependency rules — the eval/data layer and the
shared utilities (`regions`, `config`, `style`) must **not** import the view layer:

```bash
lint-imports               # run on the cluster / anywhere the `ml` extra is present
```

It is not in CI because building the import graph pulls in the GraphCast/JAX
inference modules, which need the `ml` extra.

## CI

`.github/workflows/ci.yml` runs ruff (lint + format check), mypy and pytest on a
plain Python 3.11 box with only `.[dev]` installed.
