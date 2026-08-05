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


def test_paths_override(monkeypatch):
    monkeypatch.setenv("AIC_WORKSPACE", "/tmp/ws")
    monkeypatch.delenv("AIC_DATA_ROOT", raising=False)
    monkeypatch.delenv("AIC_COAST_ZARR", raising=False)
    reloaded = importlib.reload(cfg)
    assert reloaded.WORKSPACE == "/tmp/ws"
    assert reloaded.DATA_ROOT.startswith("/tmp/ws/")
    assert reloaded.COAST_ZARR.startswith("/tmp/ws/")
    importlib.reload(cfg)  # restore defaults for other tests
