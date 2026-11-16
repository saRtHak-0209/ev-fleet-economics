"""Screen 6 — Strategic recommendations (spec §7): rule-engine findings
with quantified impact, and the final investment recommendation."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from analytics.insights import generate_insights
from analytics.recommendations import final_recommendation
from dashboard.common import base_params, inr, require_data, simulated_badge
from forecasting import run_model
from optimisation.expansion import all_city_economics
from optimisation.fleet_allocation import optimise_allocation
from config.loader import load_city_config
from simulation.monte_carlo import run_monte_carlo

ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource(show_spinner="Assembling the decision picture…")
def _pipeline():
    p = base_params()
    base = run_model(p)
    mc = run_monte_carlo(p, n_sims=5000)
    econ = all_city_economics()
    alloc = optimise_allocation(econ, float(load_city_config()["budget_inr"]), p)
    return p, base, mc, econ, alloc


def render():
    st.title("Strategic recommendations")
    simulated_badge()
    ops, veh = require_data()
    p, base, mc, econ, alloc = _pipeline()
    city_npv = {name: e.npv_ref for name, e in econ.items()}
    insights = generate_insights(ops, veh, p, city_npv=city_npv)
    final = final_recommendation(p, base, mc.stats, alloc, insights)

    colour = {"Invest": "green", "Pilot": "orange",
              "Optimise first": "blue", "Defer": "red"}[final.verdict]
    st.markdown(f"## :{colour}[{final.verdict}]")
    st.markdown(final.headline)
    with st.container(border=True):
        st.markdown("**Why:**")
        for r in final.reasons:
            st.markdown(f"- {r}")
        if final.conditions:
            st.markdown("**Conditions / next gates:**")
            for c in final.conditions:
                st.markdown(f"- {c}")
        st.caption("Thresholds: " + "; ".join(f"{k} = {v}" for k, v in final.thresholds.items()))

    st.divider()
    st.subheader(f"Ranked findings ({len(insights)})")
    st.caption("Produced by deterministic rules over computed metrics; financial impact is "
               "model-computed by re-running the five-year engine (never asserted by an LLM).")
    sev_colour = {"high": "red", "medium": "orange", "low": "gray"}
    for i, ins in enumerate(insights, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. :{sev_colour[ins.severity]}[{ins.severity.upper()}] "
                        f"{ins.problem}**")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Estimated financial impact", inr(ins.financial_impact_inr))
                st.caption(ins.impact_basis)
            with c2:
                st.markdown(f"**Recommended action:** {ins.action}")
                st.markdown(f"**Expected outcome:** {ins.expected_outcome}")
                st.caption(f"⚠ {ins.confidence_note}")
            with st.expander("Supporting metrics"):
                st.json(ins.metrics)

    st.divider()
    report_path = ROOT / "reports" / "management_report.md"
    if report_path.exists():
        st.download_button("⬇ Download the management investment report (Markdown)",
                           data=report_path.read_text(encoding="utf-8"),
                           file_name="voltmove_management_report.md", mime="text/markdown")
        with st.expander("Preview report"):
            st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.info("Run `python bootstrap.py` to generate the management report.")
