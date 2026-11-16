"""Operating expenditure forecast view (spec §Module 5B)."""
from __future__ import annotations

import pandas as pd

from forecasting.engine import ModelResult


def opex_forecast(result: ModelResult, period: str = "annual") -> pd.DataFrame:
    df = result.annual_frame() if period == "annual" else result.monthly_frame()
    key = "year" if period == "annual" else "month"
    out = df[[key, "energy_cost", "maintenance", "fixed_costs",
              "battery_capex", "depreciation"]].copy()
    out = out.rename(
        columns={
            "energy_cost": "electricity",
            "fixed_costs": "fixed_operating_costs",
            "battery_capex": "battery_replacement_capex",
        }
    )
    out["total_operating_costs"] = (
        out["electricity"] + out["maintenance"] + out["fixed_operating_costs"]
    )
    return out
