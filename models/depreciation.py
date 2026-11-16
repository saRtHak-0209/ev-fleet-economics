"""Depreciation schedules (spec §Module 3D). Straight-line."""
from __future__ import annotations

import numpy as np


def annual_straight_line(purchase_cost, residual_value, useful_life_years):
    """(Cost - residual) / life. Base case: (120000-18000)/5 = 20400."""
    return (np.asarray(purchase_cost, dtype=float) - np.asarray(residual_value, dtype=float)) / np.asarray(
        useful_life_years, dtype=float
    )


def monthly_straight_line(purchase_cost, residual_value, useful_life_months):
    return (np.asarray(purchase_cost, dtype=float) - np.asarray(residual_value, dtype=float)) / np.asarray(
        useful_life_months, dtype=float
    )


def schedule(cost: float, residual: float, life_months: int, horizon_months: int,
             start_month: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Monthly depreciation and end-of-month net book value for one asset
    purchased at the start of `start_month` (0-indexed).

    Returns (dep[t], nbv[t]) arrays of length horizon_months."""
    dep = np.zeros(horizon_months)
    live = np.arange(horizon_months)
    active = (live >= start_month) & (live < start_month + life_months)
    dep[active] = (cost - residual) / life_months
    accdep = np.cumsum(dep)
    nbv = np.where(live >= start_month, cost - accdep, 0.0)
    return dep, nbv
