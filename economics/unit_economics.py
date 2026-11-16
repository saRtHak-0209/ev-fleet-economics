"""Unit economics engine (spec §Module 4) — per-vehicle snapshot built
directly from the spec's formulas, used for the explorer screen and as
ground truth for tests. (The 5-year engine in `forecasting` reuses the
same primitives with aging, battery and financing effects layered in.)
"""
from __future__ import annotations

from dataclasses import dataclass

from config.loader import Params
from economics.revenue import monthly_revenue_per_vehicle
from models.energy_cost import electricity_cost_per_km


@dataclass(frozen=True)
class UnitEconomics:
    # per-km / per-PKM
    electricity_cost_per_km: float
    maintenance_cost_per_km: float
    variable_cost_per_total_km: float
    variable_cost_per_pkm: float
    contribution_per_pkm: float
    contribution_margin_pct: float
    # monthly (deployed vehicle, year-1)
    monthly_pkm: float
    monthly_revenue: float
    monthly_variable_cost: float
    monthly_contribution: float
    monthly_fixed_cost: float
    monthly_ebitda: float
    monthly_depreciation: float
    monthly_ebit: float
    # break-even & recovery
    break_even_pkm_month: float
    break_even_pkm_day: float
    break_even_pkm_day_full: float   # incl. capital recovery (annuity)
    min_viable_rate: float
    simple_payback_months: float


def fixed_cost_per_vehicle_pm(p: Params) -> float:
    """Allocated monthly fixed operating cost per owned vehicle:
    insurance + telematics + staff + hub + admin + charger upkeep + city
    overhead, spread over the fleet."""
    fleet = max(p.fleet_size, 1)
    staff = (fleet / p.vehicles_per_supervisor) * p.supervisor_salary_pm
    points = fleet / p.vehicles_per_charge_point
    fleet_fixed = (
        staff
        + p.hub_rent_pm
        + p.admin_overhead_pm
        + p.city_fixed_pm
        + points * p.charger_maint_pm_per_point
    )
    return p.insurance_py / 12.0 + p.telematics_pm + fleet_fixed / fleet


def capital_charge_per_vehicle_pm(p: Params) -> float:
    """Annuity that recovers the vehicle's net capital cost over its life
    at the discount rate — used for the 'full economic cost' break-even."""
    r = p.monthly_discount_rate
    n = p.vehicle_life_months
    net_capex = p.vehicle_cost + p.registration_cost
    residual = p.vehicle_cost * p.residual_pct
    pv_residual = residual / (1.0 + r) ** n
    annuity = r / (1.0 - (1.0 + r) ** (-n)) if r > 0 else 1.0 / n
    return (net_capex - pv_residual) * annuity


def compute_unit_economics(p: Params) -> UnitEconomics:
    elec_km = float(electricity_cost_per_km(p.kwh_per_km, p.charging_efficiency, p.tariff))
    maint_km = p.maint_per_km
    var_km = elec_km + maint_km
    var_pkm = var_km / p.productive_ratio
    contrib_pkm = p.billing_rate - var_pkm
    cm_pct = contrib_pkm / p.billing_rate * 100.0 if p.billing_rate else float("nan")

    monthly_km = p.daily_km * p.operating_days
    monthly_pkm = monthly_km * p.productive_ratio
    revenue = float(
        monthly_revenue_per_vehicle(p.daily_km, p.productive_ratio, p.operating_days, p.billing_rate)
    )
    var_cost = var_km * monthly_km
    contribution = revenue - var_cost
    fixed = fixed_cost_per_vehicle_pm(p)
    ebitda = contribution - fixed
    dep = (p.vehicle_cost * (1.0 - p.residual_pct)) / p.vehicle_life_months
    ebit = ebitda - dep

    be_pkm_month = fixed / contrib_pkm if contrib_pkm > 0 else float("inf")
    be_pkm_day = be_pkm_month / p.operating_days
    cap_charge = capital_charge_per_vehicle_pm(p)
    be_pkm_day_full = (
        (fixed + cap_charge) / contrib_pkm / p.operating_days
        if contrib_pkm > 0
        else float("inf")
    )
    # minimum billing rate that covers variable + fixed + capital at the
    # assumed utilisation
    min_rate = var_pkm + (fixed + cap_charge) / monthly_pkm if monthly_pkm > 0 else float("nan")

    invest = p.vehicle_cost + p.registration_cost
    payback = invest / ebitda if ebitda > 0 else float("nan")

    return UnitEconomics(
        electricity_cost_per_km=elec_km,
        maintenance_cost_per_km=maint_km,
        variable_cost_per_total_km=var_km,
        variable_cost_per_pkm=var_pkm,
        contribution_per_pkm=contrib_pkm,
        contribution_margin_pct=cm_pct,
        monthly_pkm=monthly_pkm,
        monthly_revenue=revenue,
        monthly_variable_cost=var_cost,
        monthly_contribution=contribution,
        monthly_fixed_cost=fixed,
        monthly_ebitda=ebitda,
        monthly_depreciation=dep,
        monthly_ebit=ebit,
        break_even_pkm_month=be_pkm_month,
        break_even_pkm_day=be_pkm_day,
        break_even_pkm_day_full=be_pkm_day_full,
        min_viable_rate=min_rate,
        simple_payback_months=payback,
    )
