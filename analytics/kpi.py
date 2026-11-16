"""Business KPIs (spec §7 Screen 1 & 2).

Two sources, kept distinct:
  - OBSERVED: aggregations of the (synthetic) vehicle-day table;
  - MODELLED: outputs of the five-year engine under current assumptions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.loader import Params
from economics.break_even import break_even_summary
from economics.unit_economics import (
    capital_charge_per_vehicle_pm,
    compute_unit_economics,
    fixed_cost_per_vehicle_pm,
)
from forecasting import run_model
from models.utilisation import (
    city_utilisation,
    monthly_utilisation_trend,
    vehicle_utilisation_summary,
)


def executive_kpis(ops: pd.DataFrame, vehicles: pd.DataFrame, p: Params) -> dict:
    """Screen-1 KPI tiles. Observed values use the latest full data month."""
    df = ops.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    last_month = df["month"].max()
    cur = df[df["month"] == last_month]

    total_fleet = vehicles["vehicle_id"].nunique()
    active = cur["vehicle_id"].nunique()
    monthly_revenue = cur["revenue"].sum()
    variable = cur["charging_cost"].sum() + cur["maintenance_cost"].sum()
    fixed_pm_pv = fixed_cost_per_vehicle_pm(p.with_(fleet_size=total_fleet))
    ebitda_obs = monthly_revenue - variable - fixed_pm_pv * total_fleet

    ue = compute_unit_economics(p)
    be = break_even_summary(p)
    res = run_model(p)
    ebitda_pm_model = res.metric("ebitda_total") / p.horizon_months
    runway = res.metric("cash_runway_months_m1")

    days_in_month = cur["date"].nunique()
    return {
        "as_of_month": str(last_month),
        "total_fleet": total_fleet,
        "active_fleet": active,
        "monthly_revenue_observed": monthly_revenue,
        "fleet_ebitda_observed": ebitda_obs,
        "ebitda_margin_observed": ebitda_obs / monthly_revenue if monthly_revenue else np.nan,
        "contribution_per_vehicle_pm": (monthly_revenue - variable) / max(active, 1),
        "avg_daily_utilisation_km": cur["total_km"].sum() / max(active, 1) / max(days_in_month, 1),
        "avg_daily_pkm": cur["productive_km"].sum() / max(active, 1) / max(days_in_month, 1),
        "break_even_pkm_per_day": be["break_even_pkm_day"],
        "break_even_pkm_per_day_full": be["break_even_pkm_day_full_cost"],
        "break_even_fleet_utilisation": be["break_even_fleet_utilisation"],
        "payback_months_model": res.metric("payback_months_fcff"),
        "npv_model": res.metric("npv_fcff"),
        "irr_model": res.metric("irr_fcff"),
        "ebitda_pm_model": ebitda_pm_model,
        "cash_runway_months": runway,
        "contribution_per_pkm": ue.contribution_per_pkm,
        "contribution_margin_pct": ue.contribution_margin_pct,
    }


def monthly_financial_trend(ops: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Observed monthly revenue / variable costs / EBITDA vs the model's
    projection for the same fleet (actual vs projected, Screen 1)."""
    df = ops.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    g = df.groupby("month")
    out = pd.DataFrame(
        {
            "revenue": g["revenue"].sum(),
            "charging_cost": g["charging_cost"].sum(),
            "maintenance_cost": g["maintenance_cost"].sum(),
            "active_vehicles": g["vehicle_id"].nunique(),
        }
    ).reset_index()
    fleet = int(ops["vehicle_id"].nunique())
    fixed = fixed_cost_per_vehicle_pm(p.with_(fleet_size=fleet)) * fleet
    out["opex"] = out["charging_cost"] + out["maintenance_cost"] + fixed
    out["ebitda"] = out["revenue"] - out["opex"]
    # model projection for the same fleet size (steady state month 1)
    res = run_model(p.with_(fleet_size=fleet))
    out["revenue_projected"] = res.series("revenue")[0]
    out["ebitda_projected"] = res.series("ebitda")[0]
    return out


def city_profitability(ops: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Observed per-city monthly economics per vehicle."""
    util = city_utilisation(ops)
    df = ops.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    months = df["month"].nunique()
    g = df.groupby("city", observed=True)
    out = pd.DataFrame(
        {
            "vehicles": g["vehicle_id"].nunique(),
            "revenue_pm_per_vehicle": g["revenue"].sum(),
            "variable_pm_per_vehicle": g["charging_cost"].sum() + g["maintenance_cost"].sum(),
        }
    )
    out["revenue_pm_per_vehicle"] /= out["vehicles"] * months
    out["variable_pm_per_vehicle"] /= out["vehicles"] * months
    out["contribution_pm_per_vehicle"] = out["revenue_pm_per_vehicle"] - out["variable_pm_per_vehicle"]
    fixed = fixed_cost_per_vehicle_pm(p)
    out["ebitda_pm_per_vehicle"] = out["contribution_pm_per_vehicle"] - fixed
    return out.reset_index().merge(util[["city", "productive_ratio", "availability"]], on="city")


def vehicle_profitability_table(ops: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Screen-2 vehicle-level table with status classification."""
    s = vehicle_utilisation_summary(ops)
    months = pd.to_datetime(ops["date"]).dt.to_period("M").nunique()
    fixed = fixed_cost_per_vehicle_pm(p)
    s["revenue_pm"] = s["revenue"] / months
    s["variable_pm"] = (s["charging_cost"] + s["maintenance_cost"]) / months
    s["contribution_pm"] = s["revenue_pm"] - s["variable_pm"]
    s["ebitda_pm"] = s["contribution_pm"] - fixed
    ue = compute_unit_economics(p)
    cap = capital_charge_per_vehicle_pm(p)
    full_cost = fixed + cap
    s["status"] = np.select(
        [s["ebitda_pm"] - cap > 0, s["ebitda_pm"] > 0],
        ["Profitable", "Monitor"],
        default="Underperforming",
    )
    s.attrs["thresholds"] = {
        "Profitable": "monthly EBITDA covers the capital recovery charge "
                      f"(₹{cap:,.0f}/month)",
        "Monitor": "positive EBITDA but not covering capital recovery",
        "Underperforming": "negative monthly EBITDA",
        "fixed_cost_pm": fixed,
        "full_cost_pm": full_cost,
        "contribution_per_pkm": ue.contribution_per_pkm,
    }
    return s


def utilisation_trend(ops: pd.DataFrame) -> pd.DataFrame:
    return monthly_utilisation_trend(ops)
