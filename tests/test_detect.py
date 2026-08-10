"""Tests for the heatwave spell detector (aic.controller.heatwave.detect)."""
import numpy as np

from aic.controller.heatwave.detect import active_mask


def test_active_mask_keeps_long_spells_drops_short():
    # T=5 days, 1 lat, 2 cells. Cell 0: a 3-day run; cell 1: a 2-day run.
    hot = np.zeros((5, 1, 2), bool)
    hot[0:3, 0, 0] = True     # 3 consecutive -> kept (>= MIN_DUR=3)
    hot[0:2, 0, 1] = True     # 2 consecutive -> dropped
    act = active_mask(hot)
    assert list(act[:, 0, 0]) == [True, True, True, False, False]
    assert not act[:, 0, 1].any()


def test_active_mask_respects_min_dur():
    hot = np.zeros((4, 1, 1), bool)
    hot[0:2, 0, 0] = True
    assert active_mask(hot, min_dur=2)[:, 0, 0].tolist() == [True, True, False, False]
    assert not active_mask(hot, min_dur=3)[:, 0, 0].any()


def test_active_mask_two_separate_spells():
    hot = np.zeros((8, 1, 1), bool)
    hot[0:3, 0, 0] = True      # spell A (kept)
    hot[5:8, 0, 0] = True      # spell B (kept), gap in between stays False
    act = active_mask(hot)[:, 0, 0]
    assert act.tolist() == [True, True, True, False, False, True, True, True]
