#!/usr/bin/env python
"""Validate the precached GraphCast ARCO cases for a year.

Checks every expected init-day's case file: exists + correct time/level dims +
all required variables (via precache_arco_case.validation_error). Prints a
summary and the offending dates. Exit 0 only if ALL cases are valid.

Env: YEAR, INIT_HOUR, STRIDE, STEPS, STEP_HOURS, CACHE_DIR, DELETE_INVALID(0/1).
With DELETE_INVALID=1, corrupt/partial files are removed so a precache resubmit
rebuilds them (missing files are simply reported).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

from precache_arco_case import DEFAULT_PRESSURE_LEVELS, validation_error


def main() -> int:
    year = int(os.environ.get("YEAR", "2023"))
    hour = int(os.environ.get("INIT_HOUR", "6"))
    stride = int(os.environ.get("STRIDE", "1"))
    steps = int(os.environ.get("STEPS", "40"))
    step_h = int(os.environ.get("STEP_HOURS", "6"))
    cache = Path(os.environ["CACHE_DIR"])
    levels = tuple(int(x) for x in DEFAULT_PRESSURE_LEVELS)
    delete_invalid = os.environ.get("DELETE_INVALID", "0") == "1"
    mode = os.environ.get("CASE_MODE", "arco")  # arco | nextgems

    if mode == "nextgems":
        from nextgems_graphcast_case import default_case_path as _dcp
        case_path = lambda it: _dcp(cache, year, it, steps, step_h)
    else:
        from arco_era5_graphcast_case import default_case_path as _dcp
        case_path = lambda it: _dcp(cache, it, steps, step_h)

    lead = pd.Timedelta(hours=steps * step_h)
    year_end = pd.Timestamp(f"{year}-12-31T23:00")
    starts = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq=f"{stride}D")
    inits = [s + pd.Timedelta(hours=hour) for s in starts
             if (s + pd.Timedelta(hours=hour) + lead) <= year_end]
    inits = [t.strftime("%Y-%m-%dT%H:00") for t in inits]

    valid = 0
    missing: list[str] = []
    invalid: list[tuple[str, str]] = []
    for it in inits:
        p = case_path(it)
        if not p.exists():
            missing.append(it)
            continue
        err = validation_error(p, steps, levels)
        if err is None:
            valid += 1
        else:
            invalid.append((it, err))
            if delete_invalid:
                try:
                    p.unlink()
                except OSError:
                    pass

    print(f"VALID={valid}/{len(inits)} MISSING={len(missing)} INVALID={len(invalid)}",
          flush=True)
    for it in missing[:30]:
        print("  MISSING", it, flush=True)
    for it, err in invalid[:30]:
        print("  INVALID", it, err, flush=True)
    return 0 if valid == len(inits) else 1


if __name__ == "__main__":
    sys.exit(main())
