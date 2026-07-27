#!/usr/bin/env python
"""Compress one raw (uncompressed) prediction NetCDF into a zlib-compressed one.

The "offload compression to CPU" concept (approach 3): the GPU job writes raw
predictions fast (GRAPHCAST_WRITE_COMPLEVEL=0), and a cheap CPU job runs this to
produce the final compressed .nc. Usage:

    python compress_pred.py <in.nc> <out.nc> [complevel=4]

Prints read/write timing and sizes so we can size the CPU batch.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: compress_pred.py <in.nc> <out.nc> [complevel]", file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    complevel = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    dst.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ds = xr.open_dataset(src).load()  # read raw fully into memory
    t_read = time.time()

    # NaN guard: some rollout raws are all-NaN (e.g. bad nextgems input cases).
    # Don't produce a garbage final or let the caller delete the raw -- flag it.
    probe = next((v for v in ds.data_vars
                  if np.issubdtype(ds[v].dtype, np.floating)), None)
    if probe is not None and not bool(np.isfinite(ds[probe].values).any()):
        print(f"SKIP {src.name}: input is all-NaN/Inf -> not compressing, keeping raw",
              flush=True)
        ds.close()
        return 3

    enc = {
        v: {"dtype": "float32", "zlib": True, "complevel": complevel, "shuffle": True}
        for v in ds.data_vars
        if np.issubdtype(ds[v].dtype, np.floating)
    }
    tmp = dst.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp, encoding=enc)
    tmp.rename(dst)
    t_write = time.time()

    src_gb = src.stat().st_size / 1e9
    dst_gb = dst.stat().st_size / 1e9
    print(f"compress {src.name}: read {t_read - t0:.1f}s, compress+write {t_write - t_read:.1f}s, "
          f"total {t_write - t0:.1f}s | raw {src_gb:.1f}GB -> zlib{complevel} {dst_gb:.1f}GB "
          f"({dst_gb / src_gb * 100:.0f}%)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
