"""Screen 3 — Financial planning & forecasting (spec §7)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import base_params, inr, pct, simulated_badge
from forecasting import (
    balance_sheet,
    cash_flow_statement,
    income_statement,
    run_model,
    t0_flows,
)
from reports.generate_report import export_statements_xlsx


def _assumption_panel(p):
    st.sidebar.header("Financial assumptions")
    st.sidebar.caption("Every change re-runs the full five-year engine.")
    fleet = st.sidebar.slider("Fleet size", 50, 1500, p.fleet_size, 50)
    daily_km = st.sidebar.slider("Daily km / deployed vehicle", 40, 140, int(p.daily_km), 5)
    rate = st.sidebar.slider("Billing rate (₹/PKM)", 3.0, 12.0, float(p.billing_rate), 0.1)
    prod = st.sidebar.slider("Productive ratio", 0.60, 1.00, float(p.productive_ratio), 0.01)
    avail = st.sidebar.slider("Availability", 0.60, 1.00, float(p.availability), 0.01)
    tariff = st.sidebar.slider("Electricity tariff (₹/kWh)", 5.0, 25.0, float(p.tariff), 0.5)
    maint = st.sidebar.slider("Maintenance (₹/km)", 0.10, 1.50, float(p.maint_per_km), 0.05)
    vcost = st.sidebar.slider("Vehicle cost (₹)", 80_000, 200_000, int(p.vehicle_cost), 5_000)
    fade = st.sidebar.slider("Battery fade (%/yr)", 1.0, 10.0, p.batt_calendar_fade_py * 100, 0.5) / 100
    mode = st.sidebar.selectbox("Financing", ["owned", "financed"],
                                index=0 if p.financing_mode == "owned" else 1)
    out = p.with_(fleet_size=fleet, daily_km=float(daily_km), billing_rate=rate,
                  productive_ratio=prod, availability=avail, tariff=tariff,
                  maint_per_km=maint, vehicle_cost=float(vcost),
                  batt_calendar_fade_py=fade, financing_mode=mode)
    if mode == "financed":
        down = st.sidebar.slider("Down payment", 0.10, 0.90, p.down_payment_pct, 0.05)
        irate = st.sidebar.slider("Interest rate (p.a.)", 0.08, 0.24, p.interest_rate_py, 0.005)
        tenure = st.sidebar.slider("Tenure (months)", 12, 60, p.tenure_months, 6)
        out = out.with_(down_payment_pct=down, interest_rate_py=irate, tenure_months=tenure)
    return out


def render():
    st.title("Financial planning & forecasting")
    simulated_badge()
    p = _assumption_panel(base_params())
    res = run_model(p)

    m = st.columns(5)
    m[0].metric("NPV (FCFF @ 12%)", inr(res.metric("npv_fcff")))
    irr = res.metric("irr_fcff")
    m[1].metric("IRR", pct(irr) if np.isfinite(irr) else "n/a")
    pb = res.metric("payback_months_fcff")
    m[2].metric("Payback", f"{pb:.0f} months" if np.isfinite(pb) else "not recovered")
    m[3].metric("Initial capital", inr(res.metric("capex0")))
    m[4].metric("5-yr EBITDA", inr(res.metric("ebitda_total")))
    if res.metric("min_cash") < 0:
        st.error(f"Cash goes negative (minimum {inr(res.metric('min_cash'))}) — "
                 "increase the opening buffer or financing.")

    ann = res.annual_frame()
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for col, label in [("revenue", "Revenue"), ("ebitda", "EBITDA"), ("pat", "Net profit (PAT)")]:
            fig.add_bar(x=[f"Y{int(y)}" for y in ann["year"]], y=ann[col], name=label)
        fig.update_layout(title="Five-year projection", barmode="group", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        mo = res.monthly_frame()
        cum = np.concatenate([[-res.metric("capex0")],
                              -res.metric("capex0") + np.cumsum(res.series("fcff"))])
        fig = go.Figure()
        fig.add_scatter(x=list(range(0, len(cum))), y=cum, name="Cumulative FCFF", mode="lines")
        fig.add_hline(y=0, line_dash="dot")
        if np.isfinite(pb):
            fig.add_vline(x=int(pb), line_dash="dash",
                          annotation_text=f"payback M{int(pb)}")
        fig.update_layout(title="Cumulative free cash flow & payback", height=380,
                          xaxis_title="month", yaxis_title="₹")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure()
        fig.add_scatter(x=mo["month"], y=mo["cfo"], name="Operating CF", mode="lines")
        fig.add_bar(x=mo["month"], y=-mo["battery_capex"], name="Battery replacement capex")
        fig.add_scatter(x=mo["month"], y=mo["cash"], name="Cash balance", mode="lines",
                        line=dict(dash="dot"))
        fig.update_layout(title="Monthly cash flow, battery capex & cash", height=360)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        if p.financing_mode == "financed":
            fig = go.Figure()
            fig.add_bar(x=mo["month"], y=mo["interest"], name="Interest")
            fig.add_bar(x=mo["month"], y=mo["principal_repaid"], name="Principal")
            fig.add_scatter(x=mo["month"], y=mo["loan_balance"], name="Outstanding", mode="lines")
            fig.update_layout(title="Loan repayment schedule", barmode="stack", height=360)
        else:
            fig = go.Figure()
            fig.add_scatter(x=mo["month"], y=mo["battery_soh"], name="Battery SOH", mode="lines")
            fig.add_hline(y=p.soh_replace_threshold, line_dash="dot",
                          annotation_text="replacement threshold")
            fig.update_layout(title="Battery state of health", height=360,
                              yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    tabs = st.tabs(["P&L", "Cash flow", "Balance sheet", "t0 funding"])
    with tabs[0]:
        st.dataframe(income_statement(res).style.format("{:,.0f}"), use_container_width=True)
    with tabs[1]:
        st.dataframe(cash_flow_statement(res).style.format("{:,.0f}"), use_container_width=True)
    with tabs[2]:
        bs = balance_sheet(res)
        st.dataframe(bs.style.format("{:,.0f}"), use_container_width=True)
        gap = res.metric("bs_max_abs_gap")
        st.caption(f"Reconciliation: max |Assets − (Liabilities + Equity)| across all 60 months "
                   f"= ₹{gap:,.6f}")
    with tabs[3]:
        st.dataframe(t0_flows(res).style.format("{:,.0f}"))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_statements_xlsx(res, path=Path(tmpdir) / "statements.xlsx")
        data = path.read_bytes()
    st.download_button("⬇ Download statements (Excel)", data=data,
                       file_name="voltmove_financials.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
