#!/usr/bin/env python
"""Test whether the three 2023 European heat waves are recognised by the framework,
and how the +/-window size affects recognition.

Threshold is built from the 1991-2020 series (per the definition); the 2023 series
is scored against it separately (avoids a 2021-22 gap in a single series). Category
labels are calibrated on the 1991-2020 events at each window. Reported per location
(the Europe box + Southern/Western-Europe points where the events were centred) and
per window half-width.
"""
from __future__ import annotations
import glob
import os

import numpy as np
import pandas as pd

from aic.controller import heatwave as H

DATA = os.environ.get(
    "HW_DATA_DIR", "/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_heatwave")


def _year(f):
    return int(os.path.basename(f).split("_")[-1].split(".")[0])


ALL = sorted(glob.glob(f"{DATA}/t850_24h_world_*.nc"))
REF = [f for f in ALL if 1991 <= _year(f) <= 2020]
F2023 = [f for f in ALL if _year(f) == 2023]

# user-specified 2023 European heat-wave windows
TARGETS = {
    "Apr 24-28": ("2023-04-24", "2023-04-28"),
    "Jul 15-27": ("2023-07-15", "2023-07-27"),
    "Aug 19-25": ("2023-08-19", "2023-08-25"),
}
# locations: the system default box + points near each event's core
LOCS = [
    ("europe_box",        None),
    ("Iberia(40.4,-3.7)", (40.4, -3.7)),   # Apr 2023 Iberian record heat
    ("Italy(41.9,12.5)",  (41.9, 12.5)),   # Jul 2023 "Cerberus"
    ("Greece(38,23.7)",   (38.0, 23.7)),   # Jul 2023
    ("France(46,3)",      (46.0, 3.0)),    # Aug 2023
]
WINDOWS = [1, 2, 3, 5, 7, 10, 15]   # half-widths; +/-5 == the "10-day window"


def overlaps(e, t0, t1):
    return not (e.end < t0 or e.start > t1)


def detect(series, w, ref_series):
    thr = H.doy_threshold(ref_series, q=0.95, window=w, ref_years=(1991, 2020))
    rev = H.find_events("ref", *H.exceedance(ref_series, thr))
    scheme = H.build_scheme(rev)
    ev = H.find_events("2023", *H.exceedance(series, thr))
    H.categorize(ev, scheme)
    return ev


def main():
    if not F2023:
        raise SystemExit("2023 file not staged yet (t850_24h_world_2023.nc)")
    print(f"ref years: {len(REF)} files; target: 2023")
    print(f"windows (half-width, days): {WINDOWS}   [+/-5 == the '10-day window']\n")

    for name, pt in LOCS:
        reg = "world" if pt is not None else "europe"
        ref_series = H.regional_series(REF, region=reg, point=pt)
        tgt_series = H.regional_series(F2023, region=reg, point=pt)
        per_w = {w: detect(tgt_series, w, ref_series) for w in WINDOWS}

        print(f"#### {name} " + "#" * (60 - len(name)))
        hdr = "  target      " + "".join(f"| w=±{w:<2} " for w in WINDOWS)
        print(hdr)
        for tname, (t0, t1) in TARGETS.items():
            cells = []
            for w in WINDOWS:
                m = [e for e in per_w[w] if overlaps(e, t0, t1)]
                if m:
                    e = max(m, key=lambda x: x.cumulative_exceedance_Kdays)
                    cells.append(f"{e.duration_days}d/{e.category[:3]}")
                else:
                    cells.append("  —  ")
            print(f"  {tname:<11}" + "".join(f"| {c:<6}" for c in cells))
        # detail: the exact detected 2023 spans at the baseline window +/-5
        base = per_w[5]
        det = []
        for tname, (t0, t1) in TARGETS.items():
            m = [e for e in base if overlaps(e, t0, t1)]
            if m:
                e = max(m, key=lambda x: x.cumulative_exceedance_Kdays)
                det.append(f"{tname}: {e.start}..{e.end} ({e.duration_days}d, "
                           f"peakT={e.peak_t850_K:.1f}K, cumExc={e.cumulative_exceedance_Kdays:.1f}, "
                           f"{e.category})")
            else:
                det.append(f"{tname}: not detected at ±5")
        print("  [±5 detail] " + "\n               ".join(det) + "\n")


if __name__ == "__main__":
    main()
