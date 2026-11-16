"""Sensitivity analysis (spec §Module 6A/6B).

One-variable: move each driver across a range, all else constant, and
recompute the FULL model (profit impact is calculated, not assumed
proportional to revenue). Two-variable: grids for management heatmaps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.loader import Params
from forecasting import run_model

# driver -> (Params field, label, relative span, kind)
DRIVERS: dict[str, dict] = {
    "daily_km": {"label": "Daily km per vehicle", "span": 0.30},
    "productive_ratio": {"label": "Productive km ratio", "span": 0.15, "clip": (0.0, 1.0)},
    "tariff": {"label": "Electricity tariff (₹/kWh)", "span": 0.40},
    "vehicle_cost": {"label": "Vehicle acquisition cost (₹)", "span": 0.25},
    "maint_per_km": {"label": "Maintenance (₹/km)", "span": 0.40},
    "batt_calendar_fade_py": {"label": "Battery degradation (%/yr)", "span": 1.0},
    "billing_rate": {"label": "Billing rate (₹/PKM)", "span": 0.30},
    "availability": {"label": "Vehicle availability", "span": 0.15, "clip": (0.0, 1.0)},
    "residual_pct": {"label": "Residual value (%)", "span": 0.60, "clip": (0.0, 0.9)},
}

METRICS = {"npv_fcff": "NPV (₹)", "ebitda_total": "5y EBITDA (₹)",
           "irr_fcff": "IRR", "payback_months_fcff": "Payback (months)"}


def one_variable(p: Params, driver: str, values: np.ndarray | list) -> pd.DataFrame:
    """Full-model outputs across a range of one driver."""
    rows = []
    for v in values:
        res = run_model(p.with_(**{driver: float(v)}))
        row = {"value": float(v)}
        row.update({m: res.metric(m) for m in METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


def tornado(p: Params, metric: str = "npv_fcff", span_scale: float = 1.0) -> pd.DataFrame:
    """Low/high impact of each driver on `metric` vs the base run."""
    base = run_model(p).metric(metric)
    rows = []
    for field_name, cfg in DRIVERS.items():
        base_v = float(getattr(p, field_name))
        span = cfg["span"] * span_scale
        lo_v, hi_v = base_v * (1 - span), base_v * (1 + span)
        if "clip" in cfg:
            lo_v = float(np.clip(lo_v, *cfg["clip"]))
            hi_v = float(np.clip(hi_v, *cfg["clip"]))
        lo = run_model(p.with_(**{field_name: lo_v})).metric(metric)
        hi = run_model(p.with_(**{field_name: hi_v})).metric(metric)
        rows.append(
            {
                "driver": cfg["label"], "param": field_name,
                "low_value": lo_v, "high_value": hi_v,
                "metric_low": lo, "metric_high": hi,
                "swing": abs(hi - lo),
                "delta_low": lo - base, "delta_high": hi - base,
            }
        )
    out = pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)
    out.attrs["base_metric"] = base
    return out


def two_variable_grid(
    p: Params, x_field: str, x_values, y_field: str, y_values,
    metric: str = "ebitda_pm_per_vehicle",
) -> pd.DataFrame:
    """Heatmap grid. metric 'ebitda_pm_per_vehicle' = monthly EBITDA per
    owned vehicle (spec's suggested cell value); otherwise any summary
    metric name from the engine."""
    grid = np.zeros((len(y_values), len(x_values)))
    for i, y in enumerate(y_values):
        for j, x in enumerate(x_values):
            res = run_model(p.with_(**{x_field: float(x), y_field: float(y)}))
            if metric == "ebitda_pm_per_vehicle":
                grid[i, j] = res.metric("ebitda_total") / p.horizon_months / p.fleet_size
            else:
                grid[i, j] = res.metric(metric)
    return pd.DataFrame(grid, index=list(y_values), columns=list(x_values))


def standard_heatmaps(p: Params) -> dict[str, pd.DataFrame]:
    """The two heatmaps the spec names: km x billing rate, and
    km x electricity tariff."""
    km = np.round(np.linspace(40, 130, 10), 0)
    rates = np.round(np.linspace(3.0, 10.0, 8), 1)
    tariffs = np.round(np.linspace(6.0, 22.0, 9), 1)
    return {
        "km_vs_rate": two_variable_grid(p, "daily_km", km, "billing_rate", rates),
        "km_vs_tariff": two_variable_grid(p, "daily_km", km, "tariff", tariffs),
    }
