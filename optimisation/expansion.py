"""City expansion economics (spec §Module 8).

For each candidate city, the five-year engine is run with that city's
operating overrides to obtain per-vehicle marginal NPV and the city's
fixed (setup + overhead) NPV cost. The expansion decision classification
uses transparent, configurable thresholds — never a subjective verdict.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.loader import Params, load_city_config, load_params
from forecasting import run_model

_REF_SMALL, _REF_LARGE = 100, 200   # fleets used to isolate marginal economics


@dataclass
class CityEconomics:
    city: str
    status: str
    confidence: str
    params: Params
    fleet_ref: int
    npv_per_vehicle: float          # marginal NPV per additional vehicle
    npv_fixed_city: float           # NPV of city fixed block (negative)
    capex_per_vehicle: float
    setup_capex: float
    revenue_per_vehicle_pm: float
    contribution_per_vehicle_pm: float
    ebitda_per_vehicle_pm: float
    break_even_fleet_utilisation: float
    npv_ref: float                  # NPV at reference fleet
    irr_ref: float
    payback_ref_months: float
    max_fleet: int
    min_fleet: int
    city_fixed_pm: float


def _city_params(city_cfg: dict, base_overrides: dict | None = None) -> Params:
    overrides = dict(base_overrides or {})
    overrides.update(city_cfg.get("overrides") or {})
    exp = city_cfg.get("expansion", {})
    p = load_params(overrides)
    return p.with_(
        setup_capex=float(exp.get("setup_capex", 0.0)),
        city_fixed_pm=float(exp.get("city_fixed_pm", 0.0)),
        # expansion deployments are new books of business: no legacy admin
        admin_overhead_pm=0.0,
        hub_rent_pm=0.0,
        opening_cash_buffer=0.0,
    )


def city_economics(name: str, city_cfg: dict,
                   base_overrides: dict | None = None) -> CityEconomics:
    exp = city_cfg.get("expansion", {})
    cap_points = float(exp.get("charger_capacity_points", np.inf))
    p = _city_params(city_cfg, base_overrides)

    max_by_demand = int(exp.get("max_fleet_demand", 10_000))
    max_by_chargers = int(cap_points * p.vehicles_per_charge_point) if np.isfinite(cap_points) else 10_000
    max_fleet = min(max_by_demand, max_by_chargers)
    min_fleet = int(exp.get("min_fleet", 0))

    small = run_model(p.with_(fleet_size=_REF_SMALL))
    large = run_model(p.with_(fleet_size=_REF_LARGE))
    npv_small, npv_large = small.metric("npv_fcff"), large.metric("npv_fcff")
    npv_pv = (npv_large - npv_small) / (_REF_LARGE - _REF_SMALL)
    npv_fixed = npv_small - _REF_SMALL * npv_pv

    ref = large
    T = p.horizon_months
    rev_pv_pm = ref.metric("revenue_total") / T / _REF_LARGE
    contribution = ref.series("contribution").sum() / T / _REF_LARGE
    ebitda_pv_pm = ref.metric("ebitda_total") / T / _REF_LARGE
    fixed_pm = ref.series("fixed_costs")[0]
    contr_pm_per_deployed = ref.series("contribution")[0] / (_REF_LARGE * p.availability)
    be_util = (fixed_pm / contr_pm_per_deployed / _REF_LARGE) if contr_pm_per_deployed > 0 else np.inf

    capex_pv = (
        p.vehicle_cost + p.registration_cost
        + p.charge_point_capex / p.vehicles_per_charge_point
    )
    return CityEconomics(
        city=name,
        status=city_cfg.get("status", "candidate"),
        confidence=city_cfg.get("confidence", "low"),
        params=p,
        fleet_ref=_REF_LARGE,
        npv_per_vehicle=npv_pv,
        npv_fixed_city=npv_fixed,
        capex_per_vehicle=capex_pv,
        setup_capex=float(exp.get("setup_capex", 0.0)),
        revenue_per_vehicle_pm=rev_pv_pm,
        contribution_per_vehicle_pm=contribution,
        ebitda_per_vehicle_pm=ebitda_pv_pm,
        break_even_fleet_utilisation=be_util,
        npv_ref=npv_large,
        irr_ref=ref.metric("irr_fcff"),
        payback_ref_months=ref.metric("payback_months_fcff"),
        max_fleet=max_fleet,
        min_fleet=min_fleet,
        city_fixed_pm=float(exp.get("city_fixed_pm", 0.0)),
    )


def all_city_economics(base_overrides: dict | None = None,
                       include_existing: bool = True) -> dict[str, CityEconomics]:
    cfg = load_city_config()
    out = {}
    for name, c in cfg["cities"].items():
        if not include_existing and c.get("status") == "existing":
            continue
        out[name] = city_economics(name, c, base_overrides)
    return out


def city_comparison_table(econ: dict[str, CityEconomics]) -> pd.DataFrame:
    rows = []
    for e in econ.values():
        rows.append(
            {
                "city": e.city,
                "status": e.status,
                "data_confidence": e.confidence,
                "revenue_per_vehicle_pm": e.revenue_per_vehicle_pm,
                "ebitda_per_vehicle_pm": e.ebitda_per_vehicle_pm,
                "npv_per_vehicle": e.npv_per_vehicle,
                "capex_per_vehicle": e.capex_per_vehicle,
                "setup_capex": e.setup_capex,
                "break_even_fleet_utilisation": e.break_even_fleet_utilisation,
                f"npv_at_{e.fleet_ref}_vehicles": e.npv_ref,
                "irr_ref_fleet": e.irr_ref,
                "payback_ref_months": e.payback_ref_months,
                "max_fleet_supported": e.max_fleet,
            }
        )
    return pd.DataFrame(rows)


def classify_city(e: CityEconomics, p_base: Params) -> dict:
    """Transparent Expand / Pilot / Do-not-expand classification
    (spec §Module 8 decision framework)."""
    hurdle, max_payback = p_base.hurdle_irr, p_base.max_payback_months
    irr_ok = np.isfinite(e.irr_ref) and e.irr_ref >= hurdle
    irr_near = np.isfinite(e.irr_ref) and e.irr_ref >= hurdle - 0.02
    payback_ok = np.isfinite(e.payback_ref_months) and e.payback_ref_months <= max_payback
    npv_ok = e.npv_ref > 0
    confident = e.confidence in ("high", "medium")

    reasons = []
    if not npv_ok:
        reasons.append(f"NPV at reference fleet is negative (₹{e.npv_ref:,.0f})")
    if not irr_ok:
        reasons.append(f"IRR {e.irr_ref:.1%} below hurdle {hurdle:.0%}")
    if not payback_ok:
        reasons.append(f"payback {e.payback_ref_months:.0f}m exceeds {max_payback}m limit")
    if not confident:
        reasons.append("city assumptions are illustrative (low confidence) and need validation")

    if npv_ok and irr_ok and payback_ok and confident:
        decision = "Expand"
        rationale = "Meets NPV, IRR-hurdle and payback thresholds on validated assumptions."
    elif npv_ok and irr_near and payback_ok:
        decision = "Pilot"
        rationale = ("Economics clear thresholds on current inputs, but " + "; ".join(reasons)
                     if reasons else "Marginal on IRR hurdle.")
    else:
        decision = "Do not expand"
        rationale = "; ".join(reasons) or "Fails financial thresholds."
    return {
        "city": e.city, "decision": decision, "rationale": rationale,
        "npv_ref": e.npv_ref, "irr_ref": e.irr_ref,
        "payback_ref_months": e.payback_ref_months, "confidence": e.confidence,
        "thresholds": {"hurdle_irr": hurdle, "max_payback_months": max_payback,
                       "npv": "> 0", "confidence_for_expand": "medium or high"},
    }
