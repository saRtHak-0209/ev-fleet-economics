"""Screen 4 — Scenario simulator (spec §7): what-if, scenario comparison,
sensitivity heatmaps and Monte Carlo."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import base_params, inr, pct, simulated_badge
from forecasting import run_model
from simulation.monte_carlo import run_monte_carlo
from simulation.scenarios import run_all_scenarios, scenario_comparison
from simulation.sensitivity import standard_heatmaps, tornado


@st.cache_data(show_spinner="Running scenarios…")
def _scenarios():
    return scenario_comparison(run_all_scenarios())


@st.cache_data(show_spinner="Computing one-variable sensitivity (full model per point)…")
def _tornado():
    return tornado(base_params())


@st.cache_data(show_spinner="Computing heatmaps (full model per cell)…")
def _heatmaps():
    return standard_heatmaps(base_params())


@st.cache_data(show_spinner="Running Monte Carlo…")
def _mc(n_sims: int, seed: int):
    mc = run_monte_carlo(base_params(), n_sims=n_sims, seed=seed)
    return (mc.stats, mc.driver_importance,
            mc.metric_array("npv_fcff"), mc.metric_array("irr_fcff"))


def render():
    st.title("Scenario simulator")
    simulated_badge()
    p = base_params()

    # ---- instant what-if ----
    st.subheader("Instant what-if")
    c = st.columns(3)
    d_tariff = c[0].slider("Electricity tariff change", -30, 50, 0, 5, format="%d%%")
    d_pkm = c[1].slider("Daily productive km (base 85)", 50, 120, 85, 5)
    batt_life = c[2].slider("Battery hits threshold after (years)",
                            1.0, 10.0, p.batt_calendar_fade_py and round((1 - p.soh_replace_threshold) / p.batt_calendar_fade_py, 1) or 8.3, 0.5)
    base_res = run_model(p)
    what_if = run_model(p.with_(
        tariff=p.tariff * (1 + d_tariff / 100),
        daily_km=d_pkm / p.productive_ratio,
        batt_calendar_fade_py=(1 - p.soh_replace_threshold) / batt_life,
    ))
    w = st.columns(4)
    w[0].metric("EBITDA (5y)", inr(what_if.metric("ebitda_total")),
                inr(what_if.metric("ebitda_total") - base_res.metric("ebitda_total")))
    w[1].metric("NPV", inr(what_if.metric("npv_fcff")),
                inr(what_if.metric("npv_fcff") - base_res.metric("npv_fcff")))
    irr_wi = what_if.metric("irr_fcff")
    w[2].metric("IRR", pct(irr_wi) if np.isfinite(irr_wi) else "n/a")
    pb = what_if.metric("payback_months_fcff")
    w[3].metric("Payback", f"{pb:.0f} m" if np.isfinite(pb) else "not recovered",
                f"{pb - base_res.metric('payback_months_fcff'):+.0f} m" if np.isfinite(pb) else None)

    st.divider()
    st.subheader("Base · optimistic · pessimistic")
    comp = _scenarios()
    st.dataframe(comp.style.format("{:,.2f}"), use_container_width=True)
    fig = go.Figure()
    for scen in comp.columns:
        fig.add_bar(name=scen, x=["NPV", "5y EBITDA"],
                    y=[comp.loc["NPV @ discount rate (₹)", scen],
                       comp.loc["EBITDA (5y, ₹)", scen]])
    fig.update_layout(barmode="group", height=340, title="Scenario comparison")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("One-variable sensitivity (tornado)")
    tor = _tornado()
    base_npv = tor.attrs.get("base_metric", run_model(p).metric("npv_fcff"))
    fig = go.Figure()
    fig.add_bar(y=tor["driver"], x=tor["delta_high"], orientation="h", name="High case")
    fig.add_bar(y=tor["driver"], x=tor["delta_low"], orientation="h", name="Low case")
    fig.update_layout(barmode="overlay", height=420,
                      title=f"ΔNPV vs base ({inr(base_npv)}) — full model per point",
                      xaxis_title="Δ NPV (₹)")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Two-variable sensitivity heatmaps")
    hm = _heatmaps()
    c1, c2 = st.columns(2)
    for col, (key, ylab, title) in zip(
        (c1, c2),
        [("km_vs_rate", "billing rate (₹/PKM)",
          "Monthly EBITDA / vehicle — daily km × billing rate"),
         ("km_vs_tariff", "tariff (₹/kWh)",
          "Monthly EBITDA / vehicle — daily km × electricity tariff")],
    ):
        grid = hm[key]
        with col:
            fig = go.Figure(go.Heatmap(
                z=grid.values, x=grid.columns, y=grid.index,
                colorscale="RdYlGn", zmid=0,
                colorbar=dict(title="₹/veh/m"),
            ))
            fig.update_layout(title=title, height=420,
                              xaxis_title="daily km", yaxis_title=ylab)
            st.plotly_chart(fig, use_container_width=True)
    st.caption("Green cells are EBITDA-positive per vehicle-month; the zero contour is the "
               "viability frontier the spec asks for (“what billing rate is required at each "
               "level of utilisation?”).")

    st.divider()
    st.subheader("Monte Carlo simulation")
    cc = st.columns([2, 1, 1])
    n_sims = cc[0].select_slider("Simulations", [1000, 2000, 5000, 10000], value=10000)
    seed = cc[1].number_input("Seed", value=42, step=1)
    stats, importance, npv_arr, irr_arr = _mc(int(n_sims), int(seed))

    s = st.columns(5)
    s[0].metric("P(NPV > 0)", pct(stats["p_npv_positive"], 1))
    s[1].metric("P(payback ≤ 36m)", pct(stats["p_payback_within_36m"], 1))
    s[2].metric("Median IRR", pct(stats["irr_p50"]))
    s[3].metric("Worst-5% NPV (CVaR)", inr(stats["npv_cvar05"]))
    s[4].metric("E[monthly EBITDA]", inr(stats["expected_monthly_ebitda"]))

    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure(go.Histogram(x=npv_arr, nbinsx=60))
        fig.add_vline(x=0, line_dash="dot")
        fig.add_vline(x=stats["npv_p05"], line_dash="dash", annotation_text="P5")
        fig.update_layout(title=f"NPV distribution ({n_sims:,} runs)", height=360,
                          xaxis_title="NPV (₹)")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = go.Figure(go.Histogram(x=irr_arr[np.isfinite(irr_arr)], nbinsx=60))
        fig.update_layout(title="IRR distribution", height=360,
                          xaxis_title="IRR (annual)", xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Variables most responsible for poor outcomes** "
                "(rank correlation with NPV; shift of the driver inside the worst decile):")
    st.dataframe(importance.round(3), use_container_width=True)
    st.caption("Distributions are bounded (Beta/triangular/truncated) so no simulation can "
               "produce impossible operations — productive km never exceeds total km and "
               "operating days are never negative (spec §Module 6C).")
