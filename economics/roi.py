"""Return metrics: NPV, IRR, payback, ROI, ROIC (spec §Module 4F/4G).

IRR is a vectorised bisection on the monthly rate — robust for
conventional cash-flow profiles (outflows first, inflows later) and
returns NaN where no sign change exists.
"""
from __future__ import annotations

import numpy as np


def npv(rate_per_period: float, cashflows: np.ndarray) -> np.ndarray:
    """NPV of cashflows along the last axis. cashflows[..., 0] occurs at
    the END of period 1 (i.e. everything is discounted at least once);
    pass the initial outflow inside the series at t=0 undiscounted by
    using `npv_t0`."""
    cf = np.asarray(cashflows, dtype=float)
    t = np.arange(1, cf.shape[-1] + 1, dtype=float)
    disc = (1.0 + rate_per_period) ** (-t)
    return (cf * disc).sum(axis=-1)


def npv_t0(rate_per_period: float, cf0: float | np.ndarray,
           cashflows: np.ndarray) -> np.ndarray:
    """NPV = cf0 + sum cf_t/(1+r)^t  with cf0 at t=0 (spec formula)."""
    return np.asarray(cf0, dtype=float) + npv(rate_per_period, cashflows)


def _npv_full(rate, cf):
    """NPV treating cf[..., 0] as t=0. `rate` may be (n,1) for vectorised
    bisection."""
    t = np.arange(cf.shape[-1], dtype=float)
    return (cf * (1.0 + rate) ** (-t)).sum(axis=-1)


def irr(cashflows: np.ndarray, lo: float = -0.9999, hi: float = 10.0,
        iters: int = 80) -> np.ndarray:
    """Per-period IRR of cashflows (t=0 first element), vectorised over
    leading axes via bisection. NaN where IRR is undefined in [lo, hi]."""
    cf = np.atleast_2d(np.asarray(cashflows, dtype=float))
    n = cf.shape[0]
    lo_v = np.full((n, 1), lo)
    hi_v = np.full((n, 1), hi)
    f_lo = _npv_full(lo_v, cf)
    f_hi = _npv_full(hi_v, cf)
    valid = (np.sign(f_lo) != np.sign(f_hi)) & np.isfinite(f_lo) & np.isfinite(f_hi)
    for _ in range(iters):
        mid = (lo_v + hi_v) / 2.0
        f_mid = _npv_full(mid, cf)
        go_hi = np.sign(f_mid) == np.sign(_npv_full(lo_v, cf))
        lo_v = np.where(go_hi[:, None], mid, lo_v)
        hi_v = np.where(go_hi[:, None], hi_v, mid)
    out = ((lo_v + hi_v) / 2.0).ravel()
    out[~valid] = np.nan
    if np.ndim(cashflows) == 1:
        return out[0]
    return out


def irr_annual_from_monthly(monthly_cashflows: np.ndarray) -> np.ndarray:
    r_m = irr(monthly_cashflows)
    return (1.0 + r_m) ** 12 - 1.0


def payback_months(cashflows: np.ndarray) -> np.ndarray:
    """First month (1-indexed) where cumulative cash flow >= 0.
    cashflows[..., 0] is t=0 (includes the initial outflow). Returns NaN
    when never recovered within the series."""
    cf = np.atleast_2d(np.asarray(cashflows, dtype=float))
    cum = np.cumsum(cf, axis=-1)
    recovered = cum >= 0.0
    ever = recovered.any(axis=-1)
    first = recovered.argmax(axis=-1).astype(float)  # index 0 == t0
    first[~ever] = np.nan
    if np.ndim(cashflows) == 1:
        return first[0]
    return first


def simple_payback_years(initial_investment: float, annual_operating_cash_flow: float) -> float:
    """Undiscounted payback = I0 / annual operating cash flow (spec §4F)."""
    if annual_operating_cash_flow <= 0:
        return float("nan")
    return initial_investment / annual_operating_cash_flow


def roi_pct(annual_net_operating_return: float, initial_investment: float) -> float:
    """ROI = annual net operating return / initial investment x 100."""
    if initial_investment <= 0:
        return float("nan")
    return annual_net_operating_return / initial_investment * 100.0


def roic_pct(nopat_annual: float, invested_capital: float) -> float:
    """Return on invested capital = NOPAT / invested capital x 100."""
    if invested_capital <= 0:
        return float("nan")
    return nopat_annual / invested_capital * 100.0
