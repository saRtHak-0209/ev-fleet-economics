"""Revenue forecast view (spec §Module 5A)."""
from __future__ import annotations

import pandas as pd

from forecasting.engine import ModelResult


def revenue_forecast(result: ModelResult, period: str = "annual") -> pd.DataFrame:
    df = result.annual_frame() if period == "annual" else result.monthly_frame()
    key = "year" if period == "annual" else "month"
    out = df[[key, "total_km", "productive_km", "revenue"]].copy()
    out["avg_realisation_per_pkm"] = out["revenue"] / out["productive_km"]
    return out
