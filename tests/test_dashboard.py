"""Dashboard smoke tests: every screen must render without exceptions
against the generated dataset (spec §12 — a pretty dashboard with broken
screens is a failed project). Skipped automatically if the dataset has
not been generated yet."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic" / "daily_operations.parquet"

SCREENS = ["overview", "unit_economics", "forecasting",
           "scenarios", "expansion", "recommendations"]


@pytest.mark.skipif(not DATA.exists(), reason="run bootstrap.py first")
@pytest.mark.parametrize("screen", SCREENS)
def test_screen_renders(screen):
    from streamlit.testing.v1 import AppTest

    src = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
        "import streamlit as st\n"
        f"from dashboard import {screen}\n"
        f"{screen}.render()\n"
    )
    at = AppTest.from_string(src, default_timeout=300)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
