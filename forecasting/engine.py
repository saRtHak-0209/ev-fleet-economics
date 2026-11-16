"""Integrated five-year financial model (spec §Module 5).

One vectorised implementation produces P&L, cash flow and balance sheet
for n parallel parameter draws (n=1 for the deterministic case; n=10,000
for Monte Carlo). All monthly arrays have shape (n, T).

Accounting rules enforced here (spec §12B):
  - principal repayment never touches EBITDA (financing cash flow only);
  - interest is a P&L financing expense;
  - battery replacement is capitalised capex, not maintenance;
  - vehicles depreciate straight-line to residual; fleet sold at book
    value at the end of the horizon (no gain/loss);
  - Assets = Liabilities + Equity in every month;
  - opening cash + net cash flow = closing cash.

Simplifications (documented, spec §13): tax accrues monthly at the
corporate rate with loss carry-forward and no refunds; staff cost scales
continuously with fleet; debtors are the only working-capital item and
are released in the valuation series at the end of the horizon.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.loader import Params
from models.battery import availability_penalty, battery_lifecycle
from models.energy_cost import effective_kwh_per_km
from models.financing import amortisation
from economics import roi as roi_mod

# Params fields that Monte Carlo / sensitivity may draw per simulation
DRAWABLE = {
    "daily_km", "productive_ratio", "availability", "operating_days",
    "billing_rate", "tariff", "kwh_per_km", "charging_efficiency",
    "maint_per_km", "maint_age_escalation_py", "batt_calendar_fade_py",
    "residual_pct", "vehicle_cost", "receivable_days", "battery_cost_per_kwh",
}


def _col(p: Params, name: str, draws: dict | None, n: int) -> np.ndarray:
    """Per-sim column (n,1) for a parameter, honouring draws."""
    if draws and name in draws:
        a = np.asarray(draws[name], dtype=float).reshape(-1)
        if a.size != n:
            raise ValueError(f"draw '{name}' has size {a.size}, expected {n}")
        return a.reshape(n, 1)
    return np.full((n, 1), float(getattr(p, name)))


@dataclass
class ModelResult:
    p: Params
    n_sims: int
    monthly: dict[str, np.ndarray]          # (n, T) arrays
    summary: dict[str, np.ndarray]          # (n,) metrics
    t0: dict[str, np.ndarray]               # (n,) opening flows
    _frames: dict = field(default_factory=dict, repr=False)

    # -------- convenience accessors (deterministic n=1) --------
    def metric(self, name: str) -> float:
        return float(np.asarray(self.summary[name]).ravel()[0])

    def series(self, name: str) -> np.ndarray:
        return self.monthly[name][0]

    def monthly_frame(self) -> pd.DataFrame:
        if "monthly" not in self._frames:
            cols = {k: v[0] for k, v in self.monthly.items()}
            df = pd.DataFrame(cols)
            df.insert(0, "month", np.arange(1, df.shape[0] + 1))
            df.insert(1, "year", (df["month"] - 1) // 12 + 1)
            self._frames["monthly"] = df
        return self._frames["monthly"]

    def annual_frame(self) -> pd.DataFrame:
        df = self.monthly_frame()
        flow_cols = [c for c in df.columns if c not in (
            "month", "year", "cash", "loan_balance", "debtors",
            "nbv_vehicles", "nbv_infra", "nbv_battery", "equity",
            "total_assets", "liabilities_equity", "battery_soh",
        )]
        agg = {c: "sum" for c in flow_cols}
        for c in ("cash", "loan_balance", "debtors", "nbv_vehicles", "nbv_infra",
                  "nbv_battery", "equity", "total_assets", "liabilities_equity",
                  "battery_soh"):
            if c in df.columns:
                agg[c] = "last"
        return df.groupby("year").agg(agg).reset_index()


def run_model(p: Params, draws: dict[str, np.ndarray] | None = None,
              monthly_km_noise: np.ndarray | None = None,
              n_sims: int | None = None) -> ModelResult:
    """Run the integrated model.

    draws: optional {param_name: (n,) array} per-simulation values.
    monthly_km_noise: optional (n, T) multiplicative noise on daily km.
    """
    T = int(p.horizon_months)
    if n_sims is None:
        n_sims = 1
        if draws:
            n_sims = max(np.asarray(v).size for v in draws.values())
        if monthly_km_noise is not None:
            n_sims = max(n_sims, np.asarray(monthly_km_noise).shape[0])
    n = int(n_sims)
    if draws:
        bad = set(draws) - DRAWABLE
        if bad:
            raise KeyError(f"non-drawable parameters: {sorted(bad)}")

    F = float(p.fleet_size)
    months = np.arange(T, dtype=float)[None, :]            # (1, T)
    age_mid_years = (months + 0.5) / 12.0

    # ---- per-sim parameters ----
    daily_km = _col(p, "daily_km", draws, n)
    prod_ratio = np.clip(_col(p, "productive_ratio", draws, n), 0.0, 1.0)
    avail_base = np.clip(_col(p, "availability", draws, n), 0.0, 1.0)
    op_days = np.clip(_col(p, "operating_days", draws, n), 0.0, 31.0)
    rate = _col(p, "billing_rate", draws, n)
    tariff = _col(p, "tariff", draws, n)
    kwh_km = _col(p, "kwh_per_km", draws, n)
    chg_eff = _col(p, "charging_efficiency", draws, n)
    maint_km_rate = _col(p, "maint_per_km", draws, n)
    maint_esc = _col(p, "maint_age_escalation_py", draws, n)
    cal_fade = _col(p, "batt_calendar_fade_py", draws, n)
    residual_pct = np.clip(_col(p, "residual_pct", draws, n), 0.0, 0.9)
    vehicle_cost = _col(p, "vehicle_cost", draws, n)
    recv_days = _col(p, "receivable_days", draws, n)
    batt_cost_kwh = _col(p, "battery_cost_per_kwh", draws, n)

    noise = np.ones((n, T)) if monthly_km_noise is None else np.asarray(monthly_km_noise, dtype=float)

    # ---- battery path (planned km before SOH penalty) ----
    planned_km_pv = daily_km * noise * op_days * avail_base      # (n, T) per vehicle
    batt = battery_lifecycle(
        T, cal_fade.ravel(), p.batt_cycle_fade_per_10k_km, planned_km_pv,
        p.soh_replace_threshold, p.battery_kwh, 1.0,             # unit cost applied below
        p.batt_min_remaining_months, n_sims=n,
    )
    soh = batt.soh
    batt_capex_pv = batt.replacement_capex_per_vehicle * batt_cost_kwh  # kWh * INR/kWh
    avail_eff = avail_base * availability_penalty(
        soh, p.batt_avail_coupling_soh, p.batt_avail_coupling_k)

    # ---- activity ----
    total_km = F * avail_eff * op_days * daily_km * noise        # (n, T)
    pkm = total_km * prod_ratio
    revenue = pkm * rate

    # ---- variable costs ----
    eff_kwh = effective_kwh_per_km(kwh_km, soh, p.consumption_soh_coupling)
    energy_cost = total_km * eff_kwh / chg_eff * tariff
    maintenance = total_km * maint_km_rate * (1.0 + maint_esc * age_mid_years)
    variable_costs = energy_cost + maintenance
    contribution = revenue - variable_costs

    # ---- fixed operating costs ----
    points = F / p.vehicles_per_charge_point
    staff = (F / p.vehicles_per_supervisor) * p.supervisor_salary_pm
    fixed_pm = (
        F * p.insurance_py / 12.0
        + F * p.telematics_pm
        + staff
        + p.hub_rent_pm
        + p.admin_overhead_pm
        + p.city_fixed_pm
        + points * p.charger_maint_pm_per_point
    )
    fixed_costs = np.full((n, T), fixed_pm)
    ebitda = contribution - fixed_costs

    # ---- capex & depreciation ----
    vehicle_capex0 = F * (vehicle_cost.ravel() + p.registration_cost)          # (n,)
    infra_capex0 = points * p.charge_point_capex + p.grid_connection_capex + p.setup_capex
    capex0 = vehicle_capex0 + infra_capex0                                     # (n,)

    life = int(p.vehicle_life_months)
    dep_veh_m = F * (vehicle_cost.ravel() * (1.0 - residual_pct.ravel())
                     + p.registration_cost) / life                             # (n,)
    dep_vehicles = np.where(months < life, dep_veh_m[:, None], 0.0)

    infra_life = int(p.infra_life_months)
    dep_infra_m = infra_capex0 / infra_life
    dep_infra = np.where(months < infra_life, dep_infra_m, 0.0) * np.ones((n, 1))

    batt_capex = F * batt_capex_pv                                             # (n, T)
    rep_m = batt.replacement_month                                             # (n,)
    rem = np.where(rep_m >= 0, T - rep_m, 1)
    batt_total = batt_capex.sum(axis=1)
    dep_batt_m = np.where(rep_m >= 0, batt_total / rem, 0.0)
    dep_battery = np.where(months >= rep_m[:, None], dep_batt_m[:, None], 0.0)
    dep_battery = np.where(rep_m[:, None] >= 0, dep_battery, 0.0)

    depreciation = dep_vehicles + dep_infra + dep_battery
    ebit = ebitda - depreciation

    # ---- financing ----
    if p.financing_mode == "financed":
        loan0_scalar = float(np.mean(vehicle_capex0)) * (1.0 - p.down_payment_pct)
        sched = amortisation(loan0_scalar, p.interest_rate_py, p.tenure_months, T)
        interest = np.broadcast_to(sched.interest, (n, T)).copy()
        principal = np.broadcast_to(sched.principal, (n, T)).copy()
        loan_balance = np.broadcast_to(sched.balance, (n, T)).copy()
        loan0 = np.full(n, loan0_scalar)
    else:
        interest = np.zeros((n, T)); principal = np.zeros((n, T))
        loan_balance = np.zeros((n, T)); loan0 = np.zeros(n)

    pbt = ebit - interest

    # ---- tax (monthly accrual, loss carry-forward, no refunds) ----
    def tax_from(profit: np.ndarray) -> np.ndarray:
        if p.loss_carryforward:
            cum = np.cumsum(profit, axis=1)
            cum_tax = p.tax_rate * np.maximum.accumulate(np.clip(cum, 0.0, None), axis=1)
            return np.diff(cum_tax, axis=1, prepend=0.0)
        return p.tax_rate * np.clip(profit, 0.0, None)

    tax = tax_from(pbt)
    pat = pbt - tax
    tax_unlevered = tax_from(ebit)
    nopat = ebit - tax_unlevered

    # ---- working capital ----
    debtors = revenue * recv_days / 30.0
    d_wc = np.diff(debtors, axis=1, prepend=0.0)

    # ---- salvage at end of horizon (sale at book value) ----
    salvage = np.zeros((n, T))
    nbv_veh_end = F * vehicle_cost.ravel() * residual_pct.ravel()
    nbv_infra_end = infra_capex0 - dep_infra.sum(axis=1)
    salvage[:, -1] = nbv_veh_end + np.clip(nbv_infra_end, 0.0, None)

    capex = batt_capex.copy()                                   # monthly capex
    # ---- cash flow statement ----
    cfo = pat + depreciation - d_wc
    cfi = -capex + salvage
    cff = -principal
    equity0 = capex0 - loan0 + p.opening_cash_buffer
    opening_cash = np.full(n, p.opening_cash_buffer)
    cash = opening_cash[:, None] + np.cumsum(cfo + cfi + cff, axis=1)

    # ---- balance sheet ----
    nbv_vehicles = (F * (vehicle_cost.ravel() + p.registration_cost))[:, None] \
        - np.cumsum(dep_vehicles, axis=1)
    nbv_infra = infra_capex0[..., None] if np.ndim(infra_capex0) else np.full((n, 1), infra_capex0)
    nbv_infra = nbv_infra - np.cumsum(dep_infra, axis=1)
    nbv_battery = np.cumsum(batt_capex, axis=1) - np.cumsum(dep_battery, axis=1)
    # sale removes remaining book value in the final month
    nbv_vehicles[:, -1] -= nbv_veh_end
    nbv_infra[:, -1] -= np.clip(nbv_infra_end, 0.0, None)
    equity = equity0[:, None] + np.cumsum(pat, axis=1)
    total_assets = cash + debtors + nbv_vehicles + nbv_infra + nbv_battery
    liab_equity = loan_balance + equity

    # ---- free cash flows & valuation ----
    fcff = ebitda - tax_unlevered - d_wc - capex + salvage
    fcfe = pat + depreciation - d_wc - capex + salvage - principal
    fcff_val = fcff.copy(); fcfe_val = fcfe.copy()
    fcff_val[:, -1] += debtors[:, -1]          # WC release (valuation only)
    fcfe_val[:, -1] += debtors[:, -1]

    rm = p.monthly_discount_rate
    fcff_series = np.concatenate([-capex0[:, None], fcff_val], axis=1)
    fcfe_series = np.concatenate([-equity0[:, None], fcfe_val], axis=1)
    npv_fcff = roi_mod.npv_t0(rm, -capex0, fcff_val)
    npv_fcfe = roi_mod.npv_t0(rm, -equity0, fcfe_val)
    irr_fcff = roi_mod.irr_annual_from_monthly(fcff_series)
    irr_fcfe = roi_mod.irr_annual_from_monthly(fcfe_series)
    payback_fcff = roi_mod.payback_months(np.concatenate([-capex0[:, None], fcff], axis=1))
    payback_fcfe = roi_mod.payback_months(np.concatenate([-equity0[:, None], fcfe], axis=1))

    year1 = slice(0, 12)
    ebitda_y1 = ebitda[:, year1].sum(axis=1)
    nopat_y1 = nopat[:, year1].sum(axis=1)
    invested = capex0 + p.opening_cash_buffer
    runway = np.where(
        ebitda[:, 0] < 0, cash[:, 0] / np.maximum(-ebitda[:, 0], 1e-9), np.inf
    )

    monthly = {
        "revenue": revenue, "energy_cost": energy_cost, "maintenance": maintenance,
        "variable_costs": variable_costs, "contribution": contribution,
        "fixed_costs": fixed_costs, "ebitda": ebitda, "depreciation": depreciation,
        "ebit": ebit, "interest": interest, "pbt": pbt, "tax": tax, "pat": pat,
        "total_km": total_km, "productive_km": pkm, "battery_soh": soh,
        "capex": capex, "battery_capex": batt_capex, "salvage": salvage,
        "cfo": cfo, "cfi": cfi, "cff": cff, "principal_repaid": principal,
        "d_working_capital": d_wc, "debtors": debtors, "cash": cash,
        "loan_balance": loan_balance, "nbv_vehicles": nbv_vehicles,
        "nbv_infra": nbv_infra, "nbv_battery": nbv_battery, "equity": equity,
        "total_assets": total_assets, "liabilities_equity": liab_equity,
        "fcff": fcff, "fcfe": fcfe,
    }
    summary = {
        "capex0": capex0, "equity0": equity0, "loan0": loan0,
        "npv_fcff": npv_fcff, "npv_fcfe": npv_fcfe,
        "irr_fcff": irr_fcff, "irr_fcfe": irr_fcfe,
        "payback_months_fcff": payback_fcff, "payback_months_fcfe": payback_fcfe,
        "revenue_total": revenue.sum(axis=1), "ebitda_total": ebitda.sum(axis=1),
        "pat_total": pat.sum(axis=1), "ebitda_y1": ebitda_y1,
        "revenue_y1": revenue[:, year1].sum(axis=1),
        "ebitda_margin": np.divide(ebitda.sum(axis=1), revenue.sum(axis=1),
                                   out=np.zeros(n), where=revenue.sum(axis=1) > 0),
        "roi_y1_pct": ebitda_y1 / invested * 100.0,
        "roic_y1_pct": nopat_y1 / invested * 100.0,
        "battery_replaced": (batt.replacement_month >= 0).astype(float),
        "battery_replacement_month": batt.replacement_month.astype(float),
        "min_cash": cash.min(axis=1),
        "cash_runway_months_m1": runway,
        "bs_max_abs_gap": np.abs(total_assets - liab_equity).max(axis=1),
    }
    t0 = {"capex0": capex0, "equity0": equity0, "loan0": loan0,
          "opening_cash": opening_cash}
    return ModelResult(p=p, n_sims=n, monthly=monthly, summary=summary, t0=t0)
