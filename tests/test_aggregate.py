"""Tests for the drift aggregation reduction (aic.controller.eval.eval_common)."""
import numpy as np
import pandas as pd

from aic.controller.eval.eval_common import aggregate


def _rows():
    # two init-days x one (region, level, lead) group
    return pd.DataFrame({
        "region": ["europe", "europe"],
        "level": [850, 850],
        "lead_hours": [24, 24],
        "mse": [4.0, 16.0],          # mean 10.0 -> rmse sqrt(10)
        "bias": [1.0, -3.0],         # mean -1.0
        "init_date": pd.to_datetime(["2023-06-01", "2023-06-02"]),
    })


def test_aggregate_reduces_over_inits():
    agg = aggregate(_rows())
    assert len(agg) == 1
    row = agg.iloc[0]
    assert row["n_init"] == 2
    np.testing.assert_allclose(row["mse"], 10.0)
    np.testing.assert_allclose(row["rmse"], np.sqrt(10.0))
    np.testing.assert_allclose(row["bias"], -1.0)
    np.testing.assert_allclose(row["lead_day"], 1.0)


def test_aggregate_groups_are_independent():
    df = _rows()
    df2 = df.copy()
    df2["lead_hours"] = 48
    agg = aggregate(pd.concat([df, df2], ignore_index=True)).sort_values("lead_hours")
    assert list(agg["lead_hours"]) == [24, 48]
    assert list(agg["n_init"]) == [2, 2]
    np.testing.assert_allclose(agg["lead_day"].tolist(), [1.0, 2.0])
