"""Heatwave analysis (controller). Two data stagers, a shared analysis core, and
the drivers:

  staging       : ERA5 T850 @00 UTC snapshots (the original 'ours' pipeline).
  staging_daily : ERA5 daily min/max/00 UTC (for the min&max definitions).
  definitions   : the three compared definitions (ours / mixture / ECMWF) as data.
  grid          : conservative regrid onto NeuralGCM's 2.8 deg grid (+ cache).
  climatology   : day-of-year percentile thresholds.
  detect        : ONE definition-driven detection path (hot mask + spells + active).
  compare       : overlay the three definitions across a window sweep (duration +
                  day-of-year timing figures).
  framework / spectrum / window_sweep : the earlier single-definition analyses.
"""
