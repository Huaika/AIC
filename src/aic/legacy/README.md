# `aic.legacy` — retired scripts (quarantined, not deleted)

These modules are the **pre-multi-source** NextGEMS-2049 diagnostics
(`nextgems_2049_spaghetti`, `nextgems_2049_drift`, `nextgems_2049_drift_maps`,
`ng2049_common`, `pretrained_model`). They have been **superseded** by the
multi-source view layer:

| legacy module                 | replacement                    |
|-------------------------------|--------------------------------|
| `nextgems_2049_spaghetti`     | `aic.view.spaghetti` / `aic.view.ood` |
| `nextgems_2049_drift`         | `aic.view.drift`               |
| `nextgems_2049_drift_maps`    | `aic.view.drift_maps`          |
| `ng2049_common`               | `aic.controller.eval.eval_common` + `aic.regions` |

**Status:** nothing in the package imports `aic.legacy`; it is excluded from the
installed package (`pyproject.toml`), from linting (`ruff` / pre-commit) and from
the import-linter contracts. It is kept only so the exact scripts that produced the
older figures remain runnable/reproducible.

**Recommended follow-up:** once you have confirmed no result needs to be
regenerated from these, delete the directory (`git rm -r src/aic/legacy`) — the
history preserves it.
