"""Rule-based insight engine (spec §7 Screen 6).

Insights are produced by DETERMINISTIC rules over computed metrics —
never by a language model. Each finding quantifies its financial impact
by re-running the five-year engine with the affected parameter moved
from its observed value to its target, so the impact is calculated, not
asserted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.loader import Params
from forecasting import run_model


@dataclass
class Insight:
    rule_id: str
    severity: str                  # high | medium | low
    problem: str
    metrics: dict
    financial_impact_inr: float    # annual EBITDA impact unless stated
    impact_basis: str
    action: str
    expected_outcome: str
    confidence_note: str
    sort_key: float = field(default=0.0)


def _annual_ebitda(p: Params) -> float:
    return run_model(p).metric("ebitda_y1")


def _delta_ebitda(p: Params, **changes) -> float:
    """Annual EBITDA uplift from moving parameters to target values."""
    return _annual_ebitda(p.with_(**changes)) - _annual_ebitda(p)


def generate_insights(ops: pd.DataFrame, vehicles: pd.DataFrame, p: Params,
                      city_npv: dict[str, float] | None = None) -> list[Insight]:
    insights: list[Insight] = []
    df = ops.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    months = df["month"].nunique()
    fleet = df["vehicle_id"].nunique()

    # observed operating facts
    obs_daily_km = df.groupby(["vehicle_id", "month"], observed=True)["total_km"].sum().mean() / 26.0
    obs_prod_ratio = df["productive_km"].sum() / df["total_km"].sum()
    obs_downtime = df["downtime_hours"].mean()
    obs_energy_per_km = df["charging_cost"].sum() / df["total_km"].sum()
    model_energy_per_km = p.kwh_per_km / p.charging_efficiency * p.tariff
    rev = df["revenue"].sum() / months
    energy_share = df["charging_cost"].sum() / months / rev if rev else 0.0

    # --- Rule 1: productive ratio below plan -> routing/repositioning ---
    if obs_prod_ratio < p.productive_ratio - 0.01:
        impact = _delta_ebitda(p.with_(productive_ratio=obs_prod_ratio),
                               productive_ratio=p.productive_ratio)
        insights.append(Insight(
            rule_id="LOW_PRODUCTIVE_RATIO", severity="high",
            problem=(f"Productive-km ratio is {obs_prod_ratio:.1%} against a plan of "
                     f"{p.productive_ratio:.0%} — the fleet is running unbilled kilometres."),
            metrics={"observed_productive_ratio": round(obs_prod_ratio, 4),
                     "planned_productive_ratio": p.productive_ratio,
                     "unbilled_km_per_month": round((1 - obs_prod_ratio) * df["total_km"].sum() / months)},
            financial_impact_inr=impact,
            impact_basis="annual fleet EBITDA recovered by closing the ratio gap (model-computed)",
            action="Investigate routing and repositioning inefficiencies; tighten order "
                   "clustering and rider allocation before buying more vehicles.",
            expected_outcome=f"≈ ₹{impact:,.0f} additional annual EBITDA at current activity.",
            confidence_note="Observed ratio from operational data (synthetic in this build); "
                            "impact assumes billing rate holds.",
            sort_key=impact))

    # --- Rule 2: daily utilisation below plan -> demand allocation ---
    if obs_daily_km < p.daily_km * 0.95:
        impact = _delta_ebitda(p.with_(daily_km=obs_daily_km), daily_km=p.daily_km)
        insights.append(Insight(
            rule_id="LOW_UTILISATION", severity="high",
            problem=(f"Average deployed-vehicle distance is {obs_daily_km:.0f} km/day against "
                     f"a plan of {p.daily_km:.0f} km/day — assets are under-utilised."),
            metrics={"observed_daily_km": round(obs_daily_km, 1),
                     "planned_daily_km": p.daily_km},
            financial_impact_inr=impact,
            impact_basis="annual fleet EBITDA at planned vs observed utilisation (model-computed)",
            action="Improve customer demand allocation to existing vehicles before any fleet "
                   "expansion; review idle-time by hub and daypart.",
            expected_outcome=f"≈ ₹{impact:,.0f} annual EBITDA if plan utilisation is reached.",
            confidence_note="Battery range-cap and seasonal demand explain part of the gap; "
                            "validate with charger queue data.",
            sort_key=impact))

    # --- Rule 3: high downtime -> maintenance turnaround ---
    if obs_downtime > 0.6:
        avail_gain = min(0.03, obs_downtime / 10.0 * 0.3)
        impact = _delta_ebitda(p, availability=min(1.0, p.availability + avail_gain))
        insights.append(Insight(
            rule_id="HIGH_DOWNTIME", severity="medium",
            problem=f"Average downtime is {obs_downtime:.2f} hours per vehicle-day.",
            metrics={"avg_downtime_hours": round(obs_downtime, 2),
                     "implied_availability_loss": f"{obs_downtime/10:.1%}"},
            financial_impact_inr=impact,
            impact_basis=f"annual EBITDA from +{avail_gain:.1%} availability (model-computed)",
            action="Investigate maintenance turnaround time and spare-part availability; "
                   "pre-position consumables at hubs.",
            expected_outcome=f"≈ ₹{impact:,.0f} annual EBITDA from faster turnarounds.",
            confidence_note="Assumes downtime reduction converts one-for-one into deployable days.",
            sort_key=impact))

    # --- Rule 4: electricity cost vs plan / tariff opportunity ---
    if obs_energy_per_km > model_energy_per_km * 1.05 or energy_share > 0.12:
        better_tariff = p.tariff * 0.85
        impact = _delta_ebitda(p, tariff=better_tariff)
        insights.append(Insight(
            rule_id="HIGH_ENERGY_COST", severity="medium",
            problem=(f"Charging cost is ₹{obs_energy_per_km:.2f}/km vs ₹{model_energy_per_km:.2f}/km "
                     f"planned ({energy_share:.1%} of revenue)."),
            metrics={"observed_energy_cost_per_km": round(obs_energy_per_km, 3),
                     "planned_energy_cost_per_km": round(model_energy_per_km, 3),
                     "energy_share_of_revenue": round(energy_share, 3)},
            financial_impact_inr=impact,
            impact_basis="annual EBITDA at a 15% lower blended tariff (model-computed)",
            action="Evaluate charging tariff and arrangements: shift load to off-peak/home-rate "
                   "charging, renegotiate the commercial tariff category with the DISCOM.",
            expected_outcome=f"≈ ₹{impact:,.0f} annual EBITDA at a 15% better blended tariff.",
            confidence_note="Tariff opportunity is illustrative until the applicable BESCOM "
                            "tariff schedule is confirmed.",
            sort_key=impact))

    # --- Rule 5: battery degradation faster than plan ---
    soh_by_age = ops.groupby(ops["vehicle_age_months"].round(), observed=True)["battery_soh"].mean()
    if len(soh_by_age) > 6:
        x, y = soh_by_age.index.to_numpy(), soh_by_age.to_numpy()
        slope_py = float(np.polyfit(x, y, 1)[0]) * -12.0
        if slope_py > p.batt_calendar_fade_py * 1.25:
            worse = run_model(p.with_(batt_calendar_fade_py=slope_py))
            base = run_model(p)
            impact = base.metric("npv_fcff") - worse.metric("npv_fcff")
            insights.append(Insight(
                rule_id="FAST_BATTERY_FADE", severity="medium",
                problem=(f"Observed battery fade ≈ {slope_py:.1%}/yr vs {p.batt_calendar_fade_py:.1%}/yr "
                         "planned — replacement capex will arrive earlier."),
                metrics={"observed_fade_py": round(slope_py, 4),
                         "planned_fade_py": p.batt_calendar_fade_py},
                financial_impact_inr=impact,
                impact_basis="5-year NPV erosion if the observed fade persists (model-computed)",
                action="Evaluate battery replacement economics; audit charging behaviour "
                        "(DoD, fast-charge share, heat) and warranty terms.",
                expected_outcome="Earlier, budgeted pack replacement instead of unplanned downtime.",
                confidence_note="Fade estimated from fleet SOH-vs-age slope; chemistry-level "
                                "attribution needs telemetry.",
                sort_key=impact))

    # --- Rule 6: vehicles below break-even ---
    from analytics.kpi import vehicle_profitability_table
    vt = vehicle_profitability_table(ops, p)
    under = vt[vt["status"] == "Underperforming"]
    if len(under) > 0:
        monthly_loss = -under["ebitda_pm"].sum()
        insights.append(Insight(
            rule_id="UNDERPERFORMING_VEHICLES", severity="medium",
            problem=f"{len(under)} vehicles have negative monthly EBITDA.",
            metrics={"count": int(len(under)),
                     "combined_monthly_ebitda": -round(monthly_loss),
                     "worst_vehicles": under.nsmallest(5, "ebitda_pm")["vehicle_id"].tolist()},
            financial_impact_inr=monthly_loss * 12,
            impact_basis="annualised negative EBITDA of the flagged vehicles (observed)",
            action="Reassign flagged vehicles to higher-demand routes, or rotate them out at "
                   "resale value if utilisation cannot be lifted.",
            expected_outcome=f"Up to ₹{monthly_loss*12:,.0f} annual EBITDA recovered.",
            confidence_note="Classification thresholds are transparent: EBITDA>0 and capital "
                            "charge coverage (see Screen 2).",
            sort_key=monthly_loss * 12))

    # --- Rule 7: negative-NPV expansion cities ---
    for city, npv in (city_npv or {}).items():
        if npv < 0:
            insights.append(Insight(
                rule_id="NEGATIVE_CITY_NPV", severity="high",
                problem=f"{city}: five-year NPV is negative (₹{npv:,.0f}) on current assumptions.",
                metrics={"city": city, "npv": npv},
                financial_impact_inr=-npv,
                impact_basis="avoided NPV loss by not expanding on current assumptions",
                action="Reassess commercial pricing and operating assumptions before committing "
                       "capital; run a bounded pilot only if the rate card can be improved.",
                expected_outcome="Capital preserved for positive-NPV deployments.",
                confidence_note="City inputs are illustrative placeholders pending research.",
                sort_key=-npv))

    insights.sort(key=lambda i: i.sort_key, reverse=True)
    return insights
