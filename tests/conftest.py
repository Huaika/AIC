"""Shared pytest setup.

Some modules under test (``controller.eval.eval_common``) resolve an eval RUN at
import time and ``raise SystemExit`` if none is set, so we provide a valid default
run for the whole test session. (Removing that import-time requirement is tracked
as the config-centralization work; until then this keeps the pure functions in
those modules testable.)
"""
import os

os.environ.setdefault("EVAL_RUN", "era5_2023")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
