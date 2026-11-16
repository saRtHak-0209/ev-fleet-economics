"""Strategic recommendation assembly (spec §7 Screen 6 and §16).

Combines the model outputs into the final management recommendation
(Invest / Pilot / Optimise / Defer) using transparent thresholds. The
verdict follows from the numbers; prose only explains them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from analytics.insights import Insight
from config.loader import Params
from forecasting.engine import ModelResult
from optimisation.fleet_allocation import AllocationResult


@dataclass
class FinalRecommendation:
    verdict: str                    # Invest | Pilot | Optimise first | Defer
    headline: str
    reasons: list[str]
    conditions: list[str]
    thresholds: dict


def final_recommendation(
    p: Params,
    base_result: ModelResult,
    mc_stats: dict,
    allocation: AllocationResult,
    insights: list[Insight],
) -> FinalRecommendation:
    npv = base_result.metric("npv_fcff")
    irr = base_result.metric("irr_fcff")
    payback = base_result.metric("payback_months_fcff")
    p_pos = mc_stats.get("p_npv_positive", np.nan)
    p_cash = mc_stats.get("p_cash_breach", np.nan)

    base_ok = npv > 0 and np.isfinite(irr) and irr >= p.hurdle_irr
    payback_ok = np.isfinite(payback) and payback <= p.max_payback_months
    risk_ok = p_pos >= 0.80 and p_cash <= 0.05
    ops_gap = sum(i.financial_impact_inr for i in insights
                  if i.rule_id in ("LOW_PRODUCTIVE_RATIO", "LOW_UTILISATION"))
    ebitda_y1 = base_result.metric("ebitda_y1")
    big_ops_gap = ops_gap > 0.15 * max(ebitda_y1, 1)
    candidates_funded = any(
        v > 0 for c, v in allocation.allocation.items() if c != "Bengaluru"
    )
    only_pilots = all(
        cl["decision"] in ("Pilot", "Do not expand")
        for cl in allocation.classifications if cl["city"] != "Bengaluru"
    )

    reasons, conditions = [], []
    reasons.append(f"Base-case NPV ₹{npv:,.0f}, IRR {irr:.0%} vs {p.hurdle_irr:.0%} hurdle, "
                   f"payback {payback:.0f} months vs {p.max_payback_months} limit.")
    if np.isfinite(p_pos):
        reasons.append(f"Monte Carlo: P(NPV>0) = {p_pos:.0%}; P(cash breach) = {p_cash:.0%}; "
                       f"worst-5% NPV ₹{mc_stats.get('npv_cvar05', float('nan')):,.0f}.")
    if allocation.status != "INFEASIBLE":
        alloc_txt = ", ".join(f"{c}: {v}" for c, v in allocation.allocation.items() if v > 0) or "none"
        reasons.append(f"Optimal ₹{allocation.budget/1e7:.1f} cr allocation -> {alloc_txt} "
                       f"(₹{allocation.unused_capital:,.0f} held back).")

    if not base_ok:
        verdict = "Defer"
        headline = "Defer expansion: base-case economics do not clear the return thresholds."
        conditions.append("Re-run after validating billing rate, utilisation and tariff inputs.")
    elif big_ops_gap:
        verdict = "Optimise first"
        headline = ("Optimise the existing fleet before expanding: the model attributes "
                    f"≈ ₹{ops_gap:,.0f}/yr of EBITDA to closing utilisation gaps — capture it "
                    "first, then expand from a proven base.")
        conditions.append("Lift productive-km ratio and daily utilisation to plan.")
        conditions.append("Re-run the allocation once operating data confirms plan economics.")
    elif base_ok and payback_ok and risk_ok and not only_pilots and candidates_funded:
        verdict = "Invest"
        headline = "Invest: thresholds met with validated city data; deploy per the optimiser."
    else:
        verdict = "Pilot"
        headline = ("Run a controlled pilot: base economics clear every threshold, but the "
                    "candidate-city inputs are illustrative and must be validated in the field "
                    "before committing the full ₹5 crore.")
        conditions.append("Pilot 50-vehicle cohorts in the top-ranked candidate cities; "
                          "hold the remaining capital until contract rates and utilisation are observed.")
        conditions.append("Convert to 'Invest' when pilot telemetry supports the assumed "
                          "billing rate, daily km and tariff within ±10%.")

    thresholds = {
        "hurdle_irr": p.hurdle_irr,
        "max_payback_months": p.max_payback_months,
        "min_p_npv_positive": 0.80,
        "max_p_cash_breach": 0.05,
        "ops_gap_trigger": "ops EBITDA gap > 15% of Y1 EBITDA -> Optimise first",
        "expand_needs_confidence": "city data confidence medium/high",
    }
    return FinalRecommendation(verdict=verdict, headline=headline, reasons=reasons,
                               conditions=conditions, thresholds=thresholds)


def insights_frame(insights: list[Insight]) -> pd.DataFrame:
    rows = []
    for i in insights:
        d = asdict(i)
        d.pop("sort_key", None)
        rows.append(d)
    return pd.DataFrame(rows)
