"""Battery degradation & replacement model (spec §Module 3C).

SOH_t = SOH_0 - cumulative degradation (calendar fade + optional cycle
fade per km). When SOH crosses the replacement threshold the pack is
replaced (a capital expenditure, never buried in maintenance), SOH
resets to 1.0 and continues fading.

All functions are vectorised over a leading simulation axis: inputs of
shape (n_sims,) or scalars produce (n_sims, T) paths.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _col(x, n):
    """Shape scalar or (n,) input to an (n,1) column."""
    a = np.asarray(x, dtype=float)
    if a.ndim == 0:
        a = np.full(n, float(a))
    return a.reshape(-1, 1)


@dataclass
class BatteryPath:
    soh: np.ndarray                 # (n, T) state of health AFTER any replacement
    replacement_month: np.ndarray   # (n,) int, -1 when no replacement in horizon
    replacement_capex_per_vehicle: np.ndarray  # (n, T) INR per vehicle


def soh_path(
    months: int,
    calendar_fade_py,
    cycle_fade_per_10k_km=0.0,
    monthly_km_per_vehicle=0.0,
    n_sims: int = 1,
) -> np.ndarray:
    """Pre-replacement SOH path, shape (n_sims, months). Month t value is
    the SOH at the END of month t (0-indexed)."""
    t = np.arange(1, months + 1, dtype=float)[None, :]            # (1, T)
    cal = _col(calendar_fade_py, n_sims) * t / 12.0
    km = _col(monthly_km_per_vehicle, n_sims)
    if np.ndim(monthly_km_per_vehicle) == 2:                      # (n, T) km path
        cum_km = np.cumsum(np.asarray(monthly_km_per_vehicle, dtype=float), axis=1)
    else:
        cum_km = km * t
    cyc = _col(cycle_fade_per_10k_km, n_sims) * cum_km / 10_000.0
    return 1.0 - cal - cyc


def battery_lifecycle(
    months: int,
    calendar_fade_py,
    cycle_fade_per_10k_km,
    monthly_km_per_vehicle,
    replace_threshold: float,
    battery_kwh: float,
    cost_per_kwh: float,
    min_remaining_months: int = 6,
    n_sims: int = 1,
) -> BatteryPath:
    """SOH path with (at most one) replacement inside the horizon."""
    base = soh_path(months, calendar_fade_py, cycle_fade_per_10k_km,
                    monthly_km_per_vehicle, n_sims)
    n = base.shape[0]
    below = base < float(replace_threshold)
    any_below = below.any(axis=1)
    first = np.where(any_below, below.argmax(axis=1), months)      # month index
    # replace only if enough horizon remains to justify the pack
    eligible = any_below & (first <= months - 1 - int(min_remaining_months))
    rep_month = np.where(eligible, first, -1)

    # after replacement at month r, SOH jumps back by the fade accrued to r
    t_idx = np.arange(months)[None, :]
    r = rep_month[:, None]
    fade_at_r = np.where(eligible[:, None], 1.0 - np.take_along_axis(
        base, np.maximum(r, 0), axis=1), 0.0)
    soh = base + np.where((r >= 0) & (t_idx >= r), fade_at_r, 0.0)
    soh = np.clip(soh, 0.0, 1.0)

    capex = np.zeros((n, months))
    rows = np.nonzero(eligible)[0]
    capex[rows, rep_month[rows]] = float(battery_kwh) * float(cost_per_kwh)
    return BatteryPath(soh=soh, replacement_month=rep_month,
                       replacement_capex_per_vehicle=capex)


def availability_penalty(soh: np.ndarray, threshold: float, k: float) -> np.ndarray:
    """Multiplier (<=1) on availability when SOH drops below `threshold`:
    1 - k x (threshold - SOH)+ . Returns same shape as soh."""
    shortfall = np.clip(float(threshold) - soh, 0.0, None)
    return np.clip(1.0 - float(k) * shortfall, 0.0, 1.0)
