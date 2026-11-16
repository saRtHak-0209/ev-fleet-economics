"""Break-even analysis (spec §Module 4E)."""
from __future__ import annotations

import numpy as np

from config.loader import Params
from economics.unit_economics import (
    capital_charge_per_vehicle_pm,
    compute_unit_economics,
    fixed_cost_per_vehicle_pm,
)


def break_even_pkm_month(monthly_fixed_costs, billing_rate, variable_cost_per_pkm):
    """BE PKM = fixed / (rate - variable cost per PKM)."""
    margin = np.asarray(billing_rate, dtype=float) - np.asarray(variable_cost_per_pkm, dtype=float)
    fixed = np.asarray(monthly_fixed_costs, dtype=float)
    return np.divide(fixed, margin, out=np.full_like(fixed + margin, np.inf), where=margin > 0)


def break_even_fleet_utilisation(p: Params) -> float:
    """Share of the fleet that must be deployed (at assumed per-vehicle
    activity) for fleet EBITDA to reach zero."""
    ue = compute_unit_economics(p)
    fleet_fixed = fixed_cost_per_vehicle_pm(p) * p.fleet_size
    per_deployed_contribution = ue.monthly_contribution
    if per_deployed_contribution <= 0:
        return float("inf")
    needed = fleet_fixed / per_deployed_contribution
    return needed / p.fleet_size


def min_viable_operating_days(p: Params) -> float:
    """Operating days per month needed to break even (EBITDA basis) at the
    assumed daily activity."""
    ue = compute_unit_economics(p)
    daily_contribution = ue.contribution_per_pkm * p.daily_km * p.productive_ratio
    if daily_contribution <= 0:
        return float("inf")
    return ue.monthly_fixed_cost / daily_contribution


def additional_pkm_needed(current_pkm_day: float, p: Params, full_cost: bool = False) -> dict:
    """Spec question: a vehicle doing X productive km/day — how much more
    utilisation does it need to break even?"""
    ue = compute_unit_economics(p)
    target = ue.break_even_pkm_day_full if full_cost else ue.break_even_pkm_day
    gap = max(0.0, target - current_pkm_day)
    return {
        "current_pkm_per_day": current_pkm_day,
        "break_even_pkm_per_day": target,
        "additional_pkm_per_day_needed": gap,
        "uplift_pct": (gap / current_pkm_day * 100.0) if current_pkm_day > 0 else float("inf"),
        "basis": "full economic cost (incl. capital recovery)" if full_cost else "cash fixed costs (EBITDA basis)",
    }


def break_even_summary(p: Params) -> dict:
    ue = compute_unit_economics(p)
    return {
        "break_even_pkm_month": ue.break_even_pkm_month,
        "break_even_pkm_day": ue.break_even_pkm_day,
        "break_even_pkm_day_full_cost": ue.break_even_pkm_day_full,
        "break_even_fleet_utilisation": break_even_fleet_utilisation(p),
        "min_viable_billing_rate": ue.min_viable_rate,
        "min_viable_operating_days": min_viable_operating_days(p),
        "capital_charge_per_vehicle_pm": capital_charge_per_vehicle_pm(p),
    }
