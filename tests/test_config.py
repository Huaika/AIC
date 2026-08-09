"""Tests for the typed env accessors (aic.config)."""
import importlib

import aic.config as cfg


def test_env_int(monkeypatch):
    monkeypatch.setenv("AIC_T_INT", "12")
    assert cfg.env_int("AIC_T_INT", 3) == 12
    monkeypatch.delenv("AIC_T_INT", raising=False)
    assert cfg.env_int("AIC_T_INT", 3) == 3
    monkeypatch.setenv("AIC_T_INT", "")          # empty -> default
    assert cfg.env_int("AIC_T_INT", 3) == 3


def test_env_float(monkeypatch):
    monkeypatch.setenv("AIC_T_F", "0.99")
    assert cfg.env_float("AIC_T_F", 0.5) == 0.99
    monkeypatch.delenv("AIC_T_F", raising=False)
    assert cfg.env_float("AIC_T_F", 0.5) == 0.5


def test_env_bool(monkeypatch):
    for v in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("AIC_T_B", v)
        assert cfg.env_bool("AIC_T_B", False) is True
    for v in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("AIC_T_B", v)
        assert cfg.env_bool("AIC_T_B", True) is (v == "")  # "" -> default True
    monkeypatch.delenv("AIC_T_B", raising=False)
    assert cfg.env_bool("AIC_T_B", True) is True


def test_env_list(monkeypatch):
    monkeypatch.setenv("AIC_T_L", "a, b  c,d")
    assert cfg.env_list("AIC_T_L") == ["a", "b", "c", "d"]
    monkeypatch.delenv("AIC_T_L", raising=False)
    assert cfg.env_list("AIC_T_L", ["x"]) == ["x"]


def test_env_required(monkeypatch):
    import pytest
    monkeypatch.setenv("AIC_T_REQ", "hi")
    assert cfg.env_required("AIC_T_REQ") == "hi"
    monkeypatch.delenv("AIC_T_REQ", raising=False)
    with pytest.raises(SystemExit):
        cfg.env_required("AIC_T_REQ")


def test_paths_override(monkeypatch):
    monkeypatch.setenv("AIC_WORKSPACE", "/tmp/ws")
    for k in ("AIC_DATA_ROOT", "AIC_COAST_ZARR", "AIC_NEXTGEMS_ROOT",
              "AIC_ERA5_INPUTS_ROOT"):
        monkeypatch.delenv(k, raising=False)
    reloaded = importlib.reload(cfg)
    # data subdirs derive from DATA_ROOT which derives from WORKSPACE
    assert reloaded.DATA_ROOT == "/tmp/ws/ka_dm9435-ai-climate"
    assert reloaded.HEATWAVE_CLIM == "/tmp/ws/ka_dm9435-ai-climate/heatwave_clim"
    assert reloaded.ERA5_HEATWAVE_DAILY.endswith("/era5_heatwave_daily")
    assert reloaded.COAST_ZARR.startswith("/tmp/ws/")
    # path helpers
    assert reloaded.era5_inputs_dir(2023, "inputs").endswith("/era5_2023/inputs")
    assert reloaded.nextgems_dir(2049) == reloaded.NEXTGEMS_ROOT
    assert reloaded.nextgems_dir(2050).endswith("ka_je2428-nextgems_2050")
    importlib.reload(cfg)  # restore defaults for other tests
