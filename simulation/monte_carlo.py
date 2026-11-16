"""Monte Carlo simulation (spec §Module 6C).

Runs the SAME vectorised five-year engine across n parameter draws.
All operational variables use bounded distributions so the simulation
cannot generate impossible values (negative operating days, productive
ratio > 1, etc.).

Default distributions (all centred on the active assumptions):
  daily_km            truncated normal, sd 12%, bounds [30, 160]
  productive_ratio    Beta (concentration 60) -> always in (0, 1)
  availability        Beta (concentration 80) -> always in (0, 1)
  tariff              triangular [0.85x, base, 1.35x]
  maint_per_km        lognormal sd ~20%, clipped [0.10, 2.00]
  batt_calendar_fade  uniform [2%, 7%] per year
  residual_pct        triangular [5%, base, 25%]
plus multiplicative monthly demand noise on km (sd 5%, clipped +-3 sd).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from config.loader import Params
from forecasting import run_model
from forecasting.engine import ModelResult


def _trunc_normal(rng, mean, sd, lo, hi, n):
    a, b = (lo - mean) / sd, (hi - mean) / sd
    return sps.truncnorm.rvs(a, b, loc=mean, scale=sd, size=n, random_state=rng)


def _beta_around(rng, mean, concentration, n, lo=1e-3, hi=1 - 1e-3):
    mean = float(np.clip(mean, lo, hi))
    a, b = mean * concentration, (1 - mean) * concentration
    return rng.beta(a, b, size=n)


def draw_inputs(p: Params, n: int, seed: int = 42) -> tuple[dict, np.ndarray]:
    rng = np.random.default_rng(seed)
    draws = {
        "daily_km": _trunc_normal(rng, p.daily_km, 0.12 * p.daily_km, 30.0, 160.0, n),
        "productive_ratio": _beta_around(rng, p.productive_ratio, 60.0, n),
        "availability": _beta_around(rng, p.availability, 80.0, n),
        "tariff": rng.triangular(0.85 * p.tariff, p.tariff, 1.35 * p.tariff, size=n),
        "maint_per_km": np.clip(
            p.maint_per_km * rng.lognormal(mean=-0.02, sigma=0.20, size=n), 0.10, 2.00
        ),
        "batt_calendar_fade_py": rng.uniform(0.02, 0.07, size=n),
        "residual_pct": rng.triangular(0.05, p.residual_pct, 0.25, size=n),
    }
    noise = np.exp(np.clip(rng.normal(0.0, 0.05, size=(n, p.horizon_months)),
                           -0.15, 0.15))
    return draws, noise


@dataclass
class MCResult:
    n_sims: int
    draws: dict[str, np.ndarray]
    result: ModelResult
    stats: dict
    driver_importance: pd.DataFrame

    def metric_array(self, name: str) -> np.ndarray:
        return self.result.summary[name]


def run_monte_carlo(p: Params, n_sims: int = 10_000, seed: int = 42) -> MCResult:
    draws, noise = draw_inputs(p, n_sims, seed)
    res = run_model(p, draws=draws, monthly_km_noise=noise, n_sims=n_sims)

    npv = res.summary["npv_fcff"]
    irr = res.summary["irr_fcff"]
    payback = res.summary["payback_months_fcff"]
    ebitda_pm = res.summary["ebitda_total"] / p.horizon_months
    min_cash = res.summary["min_cash"]

    stats_out = {
        "n_sims": n_sims,
        "p_npv_positive": float(np.mean(npv > 0)),
        "p_payback_within_36m": float(np.mean(np.nan_to_num(payback, nan=1e9) <= 36)),
        "p_battery_replaced": float(np.mean(res.summary["battery_replaced"])),
        "p_cash_breach": float(np.mean(min_cash < 0)),
        "npv_mean": float(np.mean(npv)),
        "npv_p05": float(np.percentile(npv, 5)),
        "npv_p50": float(np.percentile(npv, 50)),
        "npv_p95": float(np.percentile(npv, 95)),
        "npv_cvar05": float(npv[npv <= np.percentile(npv, 5)].mean()),
        "irr_mean": float(np.nanmean(irr)),
        "irr_p05": float(np.nanpercentile(irr, 5)),
        "irr_p50": float(np.nanpercentile(irr, 50)),
        "irr_p95": float(np.nanpercentile(irr, 95)),
        "expected_monthly_ebitda": float(np.mean(ebitda_pm)),
        "ebitda_pm_p05": float(np.percentile(ebitda_pm, 5)),
        "payback_median_months": float(np.nanmedian(payback)),
    }

    # drivers of bad outcomes: Spearman rank correlation with NPV, and the
    # shift of each driver's mean inside the worst decile
    worst = npv <= np.percentile(npv, 10)
    rows = []
    for name, arr in draws.items():
        rho = sps.spearmanr(arr, npv).statistic
        shift = (arr[worst].mean() - arr.mean()) / (arr.std() + 1e-12)
        rows.append({"driver": name, "spearman_rho_vs_npv": rho,
                     "worst_decile_shift_sd": shift})
    imp = (
        pd.DataFrame(rows)
        .assign(abs_rho=lambda d: d["spearman_rho_vs_npv"].abs())
        .sort_values("abs_rho", ascending=False)
        .drop(columns="abs_rho")
        .reset_index(drop=True)
    )
    return MCResult(n_sims=n_sims, draws=draws, result=res,
                    stats=stats_out, driver_importance=imp)
