"""Screen 1 — Executive overview (spec §7)."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from analytics.kpi import (
    city_profitability,
    executive_kpis,
    monthly_financial_trend,
    utilisation_trend,
)
from dashboard.common import base_params, inr, pct, require_data, simulated_badge


def render():
    st.title("Executive overview")
    simulated_badge()
    ops, veh = require_data()
    p = base_params()

    k = executive_kpis(ops, veh, p)
    st.caption(f"Observed month: **{k['as_of_month']}** · modelled figures from the "
               "five-year engine on the current assumptions register")

    r1 = st.columns(5)
    r1[0].metric("Total fleet", f"{k['total_fleet']:,}")
    r1[1].metric("Active fleet", f"{k['active_fleet']:,}")
    r1[2].metric("Monthly revenue", inr(k["monthly_revenue_observed"]))
    r1[3].metric("Fleet EBITDA / month", inr(k["fleet_ebitda_observed"]))
    r1[4].metric("EBITDA margin", pct(k["ebitda_margin_observed"]))
    r2 = st.columns(5)
    r2[0].metric("Contribution / vehicle / m", inr(k["contribution_per_vehicle_pm"]))
    r2[1].metric("Avg daily utilisation", f"{k['avg_daily_utilisation_km']:.0f} km "
                 f"({k['avg_daily_pkm']:.0f} PKM)")
    r2[2].metric("Break-even utilisation", f"{k['break_even_pkm_per_day_full']:.0f} PKM/day",
                 help="Full economic cost incl. capital recovery; cash break-even is "
                      f"{k['break_even_pkm_per_day']:.0f} PKM/day")
    r2[3].metric("Payback period", f"{k['payback_months_model']:.0f} months")
    r2[4].metric("Cash runway", "self-funding" if k["cash_runway_months"] == float("inf")
                 else f"{k['cash_runway_months']:.0f} months",
                 help="Months of opening cash at month-1 burn when EBITDA is negative")

    st.divider()
    trend = monthly_financial_trend(ops, p)
    util = utilisation_trend(ops)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_bar(x=trend["month"], y=trend["revenue"], name="Revenue (observed)")
        fig.add_bar(x=trend["month"], y=trend["ebitda"], name="EBITDA (observed)")
        fig.add_scatter(x=trend["month"], y=trend["revenue_projected"],
                        name="Revenue (modelled)", mode="lines", line=dict(dash="dot"))
        fig.add_scatter(x=trend["month"], y=trend["ebitda_projected"],
                        name="EBITDA (modelled)", mode="lines", line=dict(dash="dot"))
        fig.update_layout(title="Monthly revenue & EBITDA — actual vs modelled", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_bar(x=trend["month"], y=trend["revenue"], name="Revenue")
        fig.add_bar(x=trend["month"], y=trend["opex"], name="Operating costs")
        fig.update_layout(title="Revenue vs operating expenditure", barmode="group", height=380)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure()
        fig.add_scatter(x=util["month"], y=util["avg_daily_km"], name="Total km/day", mode="lines+markers")
        fig.add_scatter(x=util["month"], y=util["avg_daily_pkm"], name="Productive km/day", mode="lines+markers")
        fig.add_hline(y=p.daily_km, line_dash="dot", annotation_text="plan km")
        fig.update_layout(title="Fleet utilisation trend (per deployed vehicle)", height=360)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        cp = city_profitability(ops, p)
        fig = go.Figure()
        fig.add_bar(x=cp["city"], y=cp["ebitda_pm_per_vehicle"], name="EBITDA / vehicle / month")
        fig.add_bar(x=cp["city"], y=cp["contribution_pm_per_vehicle"], name="Contribution / vehicle / month")
        fig.update_layout(title="City-wise profitability (observed, per vehicle)", barmode="group", height=360)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("What the numbers say")
    answer1 = "profitable" if k["fleet_ebitda_observed"] > 0 else "loss-making"
    st.markdown(
        f"1. **Is the business profitable?** The observed fleet is **{answer1}** this month "
        f"({inr(k['fleet_ebitda_observed'])} EBITDA, {pct(k['ebitda_margin_observed'])} margin).\n"
        f"2. **What drives it?** Contribution of ₹{k['contribution_per_pkm']:.2f}/PKM "
        f"({k['contribution_margin_pct']:.0f}% margin) against a full-cost break-even of only "
        f"{k['break_even_pkm_per_day_full']:.0f} PKM/day — utilisation is the profit lever.\n"
        f"3. **What should management do next?** See **Strategic recommendations** "
        f"(Screen 6) for the ranked, quantified actions."
    )
