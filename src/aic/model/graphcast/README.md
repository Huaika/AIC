# `aic.model.graphcast`

Two things live here:

- **`inference/`** — this project's GraphCast rollout/inference driver scripts
  (sbatch + Python) that produce the GraphCast predictions the eval layer reads.
- **`source/`** — a **vendored copy of the upstream GraphCast repository**
  (DeepMind, Apache-2.0), including its own `graphcast/` package, notebooks and
  `graphcast.egg-info`.

## Why the fork is vendored

The rollout driver needs the GraphCast model code, and it is pinned in-tree so runs
are reproducible against an exact revision (and so cluster nodes without internet
can import it). See `source/` for upstream's own `README`/`docs`.

## Quarantine status

`source/` is **not part of the `aic` package**: it is excluded from
`pyproject.toml`'s package discovery, from `ruff`/pre-commit, and from the
import-linter contracts. Treat it as a third-party dependency, not first-party code.

## Recommended follow-up

Prefer pinning upstream GraphCast as a real dependency (a git ref in
`pyproject.toml`'s `ml` extra) over vendoring, unless a local patch is required. If
the fork does carry local changes, record the upstream revision + the diff here so
the delta is auditable.
