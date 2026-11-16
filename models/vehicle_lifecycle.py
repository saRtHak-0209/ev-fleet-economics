"""Vehicle lifecycle helpers: age, residual value, salvage."""
from __future__ import annotations

import numpy as np


def age_years_mid_month(horizon_months: int) -> np.ndarray:
    """Vehicle age in years at the middle of each month, for a vehicle
    purchased at t0. Shape (T,)."""
    return (np.arange(horizon_months, dtype=float) + 0.5) / 12.0


def residual_value(purchase_cost, residual_pct):
    return np.asarray(purchase_cost, dtype=float) * np.asarray(residual_pct, dtype=float)


def salvage_at_horizon(fleet_size, purchase_cost, residual_pct):
    """Sale of the fleet at residual (book) value at end of horizon —
    booked at NBV, so no P&L gain or loss under base assumptions."""
    return float(fleet_size) * residual_value(purchase_cost, residual_pct)
