"""Screen 2 — Unit economics explorer (spec §7)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.kpi import vehicle_profitability_table
from dashboard.common import base_params, inr, require_data, simulated_badge
from economics.break_even import additional_pkm_needed, break_even_summary
from economics.tco import compare_tco
from economics.unit_economics import compute_unit_economics


def render():
    st.title("Unit economics explorer")
    simulated_badge()
    ops, veh = require_data()
    p = base_params()
    ue = compute_unit_economics(p)
    be = break_even_summary(p)

    r1 = st.columns(4)
    r1[0].metric("Revenue / vehicle / month", inr(ue.monthly_revenue),
                 help="Deployed vehicle at plan utilisation (100 km × 85% × 26 d × ₹8)")
    r1[1].metric("Energy cost / km", f"₹{ue.electricity_cost_per_km:.3f}")
    r1[2].metric("Maintenance / km", f"₹{ue.maintenance_cost_per_km:.2f}")
    r1[3].metric("Variable cost / PKM", f"₹{ue.variable_cost_per_pkm:.3f}")
    r2 = st.columns(4)
    r2[0].metric("Contribution / PKM", f"₹{ue.contribution_per_pkm:.2f}",
                 f"{ue.contribution_margin_pct:.0f}% margin")
    r2[1].metric("EBITDA / vehicle / month", inr(ue.monthly_ebitda))
    r2[2].metric("Break-even PKM", f"{ue.break_even_pkm_day:.0f}/day cash · "
                 f"{ue.break_even_pkm_day_full:.0f}/day full cost")
    r2[3].metric("Vehicle payback", f"{ue.simple_payback_months:.0f} months",
                 help="Vehicle capex ÷ monthly vehicle EBITDA")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "total"],
            x=["Billing rate", "Electricity", "Maintenance", "Unbilled-km gross-up",
               "Allocated fixed", "Pre-capital profit / PKM"],
            y=[p.billing_rate,
               -ue.electricity_cost_per_km,
               -ue.maintenance_cost_per_km,
               -(ue.variable_cost_per_pkm - ue.variable_cost_per_total_km),
               -(ue.monthly_fixed_cost / ue.monthly_pkm),
               0],
            connector=dict(line=dict(color="#B9B2A6")),
        ))
        fig.update_layout(title="₹ per productive km — where the rate goes", height=400)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        pkm_day = list(range(10, 131, 5))
        monthly_fixed = ue.monthly_fixed_cost
        profit = [(ue.contribution_per_pkm * k * p.operating_days - monthly_fixed) for k in pkm_day]
        fig = go.Figure()
        fig.add_scatter(x=pkm_day, y=profit, mode="lines", name="Monthly vehicle EBITDA")
        fig.add_hline(y=0, line_dash="dot")
        fig.add_vline(x=ue.break_even_pkm_day, line_dash="dot",
                      annotation_text=f"cash BE {ue.break_even_pkm_day:.0f}")
        fig.add_vline(x=ue.break_even_pkm_day_full, line_dash="dash",
                      annotation_text=f"full-cost BE {ue.break_even_pkm_day_full:.0f}")
        fig.update_layout(title="Vehicle EBITDA vs daily productive km", height=400,
                          xaxis_title="productive km/day", yaxis_title="₹/month")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("“A vehicle is doing 40 PKM/day — what does it need to break even?”"):
        g = additional_pkm_needed(40.0, p, full_cost=True)
        st.markdown(
            f"- Full-cost break-even: **{g['break_even_pkm_per_day']:.0f} PKM/day** → the vehicle "
            f"needs **+{g['additional_pkm_per_day_needed']:.0f} PKM/day ({g['uplift_pct']:.0f}% uplift)**.\n"
            f"- Minimum viable billing rate at plan utilisation: "
            f"**₹{be['min_viable_billing_rate']:.2f}/PKM**; minimum operating days: "
            f"**{be['min_viable_operating_days']:.1f}/month**."
        )

    st.divider()
    st.subheader("EV vs petrol — same workload")
    tco = compare_tco(p)
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Cost / km (EV)", f"₹{tco['cost_per_km_ev']:.2f}")
    t2.metric("Cost / km (petrol)", f"₹{tco['cost_per_km_petrol']:.2f}")
    t3.metric("5-yr TCO gap (discounted)", inr(tco["ev_saving_vs_petrol_5y_discounted"]),
              "EV cheaper" if tco["ev_is_cheaper"] else "petrol cheaper")
    t4.metric("EV premium payback", f"{tco['premium_payback_months']:.0f} months")

    st.divider()
    st.subheader("Vehicle-level profitability")
    vt_raw = vehicle_profitability_table(ops, p)
    th = vt_raw.attrs["thresholds"]
    vt = vt_raw.merge(veh[["vehicle_id", "cohort"]], on="vehicle_id", how="left")
    f1, f2, f3, f4 = st.columns(4)
    city = f1.multiselect("City", sorted(vt["city"].unique()))
    cohort = f2.multiselect("Fleet segment", sorted(vt["cohort"].unique()))
    age_band = f3.selectbox("Vehicle age", ["All", "< 12 months", "12–24 months", "> 24 months"])
    status = f4.multiselect("Status", ["Profitable", "Monitor", "Underperforming"])
    view = vt.copy()
    if city:
        view = view[view["city"].isin(city)]
    if cohort:
        view = view[view["cohort"].isin(cohort)]
    if age_band == "< 12 months":
        view = view[view["vehicle_age_months"] < 12]
    elif age_band == "12–24 months":
        view = view[view["vehicle_age_months"].between(12, 24)]
    elif age_band == "> 24 months":
        view = view[view["vehicle_age_months"] > 24]
    if status:
        view = view[view["status"].isin(status)]

    cols = ["vehicle_id", "city", "cohort", "avg_daily_pkm", "productive_ratio",
            "availability", "revenue_pm", "variable_pm", "ebitda_pm",
            "battery_soh", "vehicle_age_months", "status"]
    st.dataframe(
        view[cols].sort_values("ebitda_pm").round(2),
        use_container_width=True, height=380,
        column_config={
            "revenue_pm": st.column_config.NumberColumn("Revenue/m", format="₹%.0f"),
            "variable_pm": st.column_config.NumberColumn("Variable/m", format="₹%.0f"),
            "ebitda_pm": st.column_config.NumberColumn("EBITDA/m", format="₹%.0f"),
            "avg_daily_pkm": st.column_config.NumberColumn("PKM/day", format="%.0f"),
            "productive_ratio": st.column_config.NumberColumn("Prod. ratio", format="%.2f"),
        },
    )
    st.caption(f"Status thresholds — Profitable: {th['Profitable']}; Monitor: {th['Monitor']}; "
               f"Underperforming: {th['Underperforming']}.")
    st.caption(f"Counts: {vt['status'].value_counts().to_dict()}")
