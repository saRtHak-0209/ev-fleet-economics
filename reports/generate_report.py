"""Management investment report (spec §15.5 / §16).

Renders a markdown report that answers the ten management questions from
the project spec, ending with the model-driven recommendation. Also
exports the financial statements and the assumptions register to Excel.
All numbers come from the engines — this module only formats them.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.insights import Insight
from analytics.recommendations import FinalRecommendation
from config.loader import Params, load_registry
from economics.break_even import additional_pkm_needed, break_even_summary
from economics.tco import compare_tco
from economics.unit_economics import compute_unit_economics
from forecasting import balance_sheet, cash_flow_statement, income_statement, t0_flows
from forecasting.engine import ModelResult
from optimisation.fleet_allocation import AllocationResult

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"


def _inr(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    if abs(x) >= 1e7:
        return f"₹{x/1e7:,.2f} cr"
    if abs(x) >= 1e5:
        return f"₹{x/1e5:,.2f} lakh"
    return f"₹{x:,.0f}"


def build_report(
    p: Params,
    base: ModelResult,
    scen_table: pd.DataFrame,
    mc_stats: dict,
    mc_importance: pd.DataFrame,
    tornado: pd.DataFrame,
    allocation: AllocationResult,
    insights: list[Insight],
    final: FinalRecommendation,
) -> str:
    ue = compute_unit_economics(p)
    be = break_even_summary(p)
    tco = compare_tco(p)
    gap40 = additional_pkm_needed(40.0, p, full_cost=True)

    irr = base.metric("irr_fcff")
    L = []
    L.append("# VoltMove Mobility — EV Fleet Investment Report")
    L.append(f"*Generated {date.today().isoformat()} · base-case assumptions register "
             f"v{load_registry().meta.get('as_of', '')} · all operational data is SIMULATED*")
    L.append("")
    L.append("> **Recommendation: " + final.verdict + ".** " + final.headline)
    L.append("")
    L.append("---")
    L.append("## 1. Is operating an EV fleet economically viable under base-case assumptions?")
    L.append(f"Yes under the current register: a {p.fleet_size}-vehicle Bengaluru fleet produces "
             f"five-year NPV of **{_inr(base.metric('npv_fcff'))}** at a {p.discount_rate_py:.0%} "
             f"discount rate, IRR **{irr:.0%}**, and year-1 EBITDA of "
             f"{_inr(base.metric('ebitda_y1'))} on revenue of {_inr(base.metric('revenue_y1'))} "
             f"({base.metric('ebitda_y1')/base.metric('revenue_y1'):.0%} margin). "
             f"Viability is assumption-driven — the billing rate "
             f"(₹{p.billing_rate}/PKM, confidence: low) dominates; see Q5 and §Risks.")
    L.append("")
    L.append("## 2. What is the contribution margin per vehicle?")
    L.append(f"Each productive km bills ₹{p.billing_rate:.2f} and costs "
             f"₹{ue.variable_cost_per_pkm:.3f} in variable cost (electricity "
             f"₹{ue.electricity_cost_per_km:.3f}/km + maintenance ₹{ue.maintenance_cost_per_km:.2f}/km, "
             f"grossed up for the {1-p.productive_ratio:.0%} unbilled distance), leaving "
             f"**₹{ue.contribution_per_pkm:.2f} per PKM ({ue.contribution_margin_pct:.0f}%)**. "
             f"A deployed vehicle contributes **{_inr(ue.monthly_contribution)}/month** "
             f"({_inr(ue.monthly_ebitda)} EBITDA after {_inr(ue.monthly_fixed_cost)} allocated fixed cost).")
    L.append("")
    L.append("## 3. What is the minimum daily utilisation required to break even?")
    L.append(f"- Cash (EBITDA) break-even: **{ue.break_even_pkm_day:.0f} productive km/day** "
             f"({ue.break_even_pkm_month:,.0f} PKM/month).")
    L.append(f"- Full economic cost incl. capital recovery: **{ue.break_even_pkm_day_full:.0f} PKM/day**.")
    L.append(f"- Break-even fleet deployment: {be['break_even_fleet_utilisation']:.1%} of the fleet; "
             f"minimum viable billing rate at plan utilisation: ₹{be['min_viable_billing_rate']:.2f}/PKM.")
    if gap40["additional_pkm_per_day_needed"] > 0:
        L.append(f"- A vehicle running 40 PKM/day needs **+{gap40['additional_pkm_per_day_needed']:.0f} "
                 f"PKM/day ({gap40['uplift_pct']:.0f}% uplift)** to cover full economic cost.")
    else:
        L.append(f"- The spec's worked example — a vehicle at 40 PKM/day — already clears the "
                 f"full-cost break-even of {gap40['break_even_pkm_per_day']:.0f} PKM/day under these "
                 "assumptions; the margin of safety is "
                 f"{40 - gap40['break_even_pkm_per_day']:.0f} PKM/day.")
    L.append("")
    L.append("## 4. How does EV economics compare with petrol?")
    cheaper = "cheaper" if tco["ev_is_cheaper"] else "more expensive"
    L.append(f"On an identical workload, the EV runs at **₹{tco['cost_per_km_ev']:.2f}/km** vs "
             f"**₹{tco['cost_per_km_petrol']:.2f}/km** for petrol (total cost incl. acquisition, "
             f"energy, maintenance, replacements, net of resale). Five-year discounted TCO: "
             f"{_inr(tco['tco_5y_ev_discounted'])} vs {_inr(tco['tco_5y_petrol_discounted'])} — the EV is "
             f"**{cheaper}** by {_inr(abs(tco['ev_saving_vs_petrol_5y_discounted']))} per vehicle. "
             f"The ₹{tco['ev_purchase_premium']:,.0f} purchase premium pays back in "
             f"**{tco['premium_payback_months']:.0f} months** of operating savings.")
    L.append("")
    L.append("## 5. What are the three largest drivers of profitability?")
    L.append("One-variable sensitivity of five-year NPV (full model recomputation):")
    for i, row in tornado.head(3).iterrows():
        L.append(f"{i+1}. **{row['driver']}** — NPV swing {_inr(row['swing'])} across the tested range.")
    L.append(f"Monte Carlo rank-correlation confirms: "
             + ", ".join(f"{r.driver} (ρ={r.spearman_rho_vs_npv:+.2f})"
                         for r in mc_importance.head(3).itertuples()) + ".")
    L.append("")
    L.append("## 6. What is the probability of generating a positive NPV?")
    L.append(f"Across {mc_stats['n_sims']:,} simulations with bounded input distributions: "
             f"**P(NPV>0) = {mc_stats['p_npv_positive']:.0%}**; "
             f"P(payback ≤ 36 months) = {mc_stats['p_payback_within_36m']:.0%}; "
             f"median IRR {mc_stats['irr_p50']:.0%} (5th-95th pct: {mc_stats['irr_p05']:.0%} to "
             f"{mc_stats['irr_p95']:.0%}); worst-5% average NPV {_inr(mc_stats['npv_cvar05'])}; "
             f"battery replacement occurs within the horizon in {mc_stats['p_battery_replaced']:.0%} "
             f"of simulations.")
    L.append("")
    L.append("## 7. Which city offers the strongest investment opportunity?")
    best = allocation.table.sort_values("npv_per_vehicle", ascending=False).iloc[0]
    L.append(f"**{best['city']}** has the highest marginal NPV per vehicle "
             f"({_inr(best['npv_per_vehicle'])}) and the lowest incremental fixed cost. "
             "Candidate-city figures rest on illustrative inputs (confidence: low) and rank, "
             "not prove, the opportunity:")
    for r in allocation.table.itertuples():
        L.append(f"- {r.city}: NPV/vehicle {_inr(r.npv_per_vehicle)}, "
                 f"ref-fleet IRR {r.irr_ref:.0%}, data confidence {r.data_confidence}.")
    L.append("")
    L.append("## 8. How should the company allocate its ₹5 crore expansion capital?")
    alloc_lines = [f"- **{c}: {v} vehicles**" for c, v in allocation.allocation.items() if v > 0]
    L.append(f"The constrained optimiser ({allocation.solver}, status {allocation.status}) "
             f"maximises portfolio NPV subject to the ₹{allocation.budget/1e7:.0f} cr budget, "
             "per-city demand ceilings, minimum viable cohorts and charger capacity:")
    L.extend(alloc_lines or ["- No deployment clears the constraints."])
    L.append(f"- Capital used: {_inr(allocation.total_capex)}; held back: "
             f"{_inr(allocation.unused_capital)}; portfolio NPV {_inr(allocation.total_npv)}.")
    L.append("Decision classification per city (transparent thresholds): "
             + "; ".join(f"{c['city']} → **{c['decision']}**" for c in allocation.classifications) + ".")
    L.append("")
    L.append("## 9. What are the major risks to the recommendation?")
    L.append(f"- **Billing rate** (₹{p.billing_rate}/PKM) is a low-confidence assumption; the minimum "
             f"viable rate is ₹{be['min_viable_billing_rate']:.2f}/PKM at plan utilisation — a thin "
             "contract market erases the margin.")
    top_op_driver = mc_importance.iloc[0]["driver"].replace("_", " ")
    L.append(f"- **Utilisation risk**: {top_op_driver} carries the highest rank-correlation with "
             "NPV in simulation; demand is assumed, not proven (spec §13.3).")
    L.append(f"- **Battery risk**: at the pessimistic 6%/yr fade the pack crosses the "
             f"{p.soh_replace_threshold:.0%} threshold inside the horizon, adding "
             f"{_inr(p.fleet_size * p.battery_kwh * p.battery_cost_per_kwh)} fleet-wide capex.")
    L.append("- **Tariff & policy**: commercial tariffs and incentives can change; the model's "
             "tariff is a blended placeholder pending the applicable DISCOM schedule.")
    L.append("- **Model risk**: the optimiser output is only as good as its inputs; every run "
             "publishes its constraints and assumption register (spec §13.8).")
    L.append("- **Data**: all operational records in this build are synthetic and labelled as such; "
             "no claim of real-world predictive accuracy is made (spec §12D).")
    L.append("")
    L.append("## 10. What should management do in the next 90 days?")
    for i, ins in enumerate(insights[:5], 1):
        L.append(f"{i}. **{ins.action}** — {ins.expected_outcome}")
    L.append(f"{min(5, len(insights)) + 1}. Replace low-confidence register entries (billing rate, "
             "tariff, maintenance, residual value) with dated quotations and contracts, then re-run "
             "this report.")
    L.append("")
    L.append("---")
    L.append(f"## Final recommendation: **{final.verdict}**")
    L.append(final.headline)
    L.append("")
    L.append("**Why:**")
    L.extend(f"- {r}" for r in final.reasons)
    if final.conditions:
        L.append("")
        L.append("**Conditions / next gates:**")
        L.extend(f"- {c}" for c in final.conditions)
    L.append("")
    L.append("**Decision thresholds used:** " +
             "; ".join(f"{k} = {v}" for k, v in final.thresholds.items()) + ".")
    L.append("")
    L.append("## Scenario summary")
    L.append("")
    L.append(_table_md(scen_table))
    L.append("")
    L.append("*Prepared by the VoltMove decision-support system. Recommendations are produced by "
             "deterministic rules over model outputs; they support, and do not replace, management "
             "judgement. Financial figures are modelled estimates on documented assumptions, not "
             "forecasts of any real company.*")
    return "\n".join(L)


def _table_md(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in d.columns:
        d[c] = d[c].map(lambda x: f"{x:,.2f}" if isinstance(x, (int, float, np.floating)) else x)
    header = "| metric | " + " | ".join(d.columns) + " |"
    sep = "|---" * (len(d.columns) + 1) + "|"
    rows = [f"| {idx} | " + " | ".join(map(str, row)) + " |" for idx, row in d.iterrows()]
    return "\n".join([header, sep] + rows)


def export_statements_xlsx(base: ModelResult, path: Path | None = None) -> Path:
    path = path or REPORTS_DIR / "financial_statements_base.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        income_statement(base, "annual").to_excel(xw, sheet_name="P&L (annual)")
        cash_flow_statement(base, "annual").to_excel(xw, sheet_name="Cash flow (annual)")
        balance_sheet(base, "annual").to_excel(xw, sheet_name="Balance sheet (annual)")
        t0_flows(base).to_excel(xw, sheet_name="t0 funding")
        income_statement(base, "monthly").to_excel(xw, sheet_name="P&L (monthly)")
        cash_flow_statement(base, "monthly").to_excel(xw, sheet_name="Cash flow (monthly)")
        balance_sheet(base, "monthly").to_excel(xw, sheet_name="Balance sheet (monthly)")
    return path


def export_register_xlsx(path: Path | None = None) -> Path:
    path = path or REPORTS_DIR / "assumptions_register.xlsx"
    load_registry().to_frame().to_excel(path, index=False)
    return path
