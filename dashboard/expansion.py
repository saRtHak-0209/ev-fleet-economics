"""Screen 5 — City expansion planner (spec §7): city economics, India map,
constrained capital allocation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.loader import load_city_config
from dashboard.common import base_params, inr, pct, simulated_badge
from optimisation.expansion import all_city_economics, city_comparison_table
from optimisation.fleet_allocation import optimise_allocation


@st.cache_resource(show_spinner="Running the five-year engine per city…")
def _city_econ():
    return all_city_economics()


def render():
    st.title("City expansion planner")
    simulated_badge()
    p = base_params()
    cfg = load_city_config()
    econ = _city_econ()
    table = city_comparison_table(econ)

    st.warning("Candidate-city parameters are **illustrative placeholders** — the spec forbids "
               "manufacturing city statistics. Replace `config/city_parameters.yaml` entries with "
               "researched values (DISCOM tariff orders, contracted rates, pilot telemetry) "
               "before using the allocation for a real decision.", icon="⚠️")

    # ---- map ----
    cities = cfg["cities"]
    lat = [c["lat"] for c in cities.values()]
    lon = [c["lon"] for c in cities.values()]
    names = list(cities.keys())
    npv_pv = [econ[n].npv_per_vehicle for n in names]
    fig = go.Figure(go.Scattergeo(
        lat=lat, lon=lon, text=[
            f"{n}<br>NPV/vehicle: {inr(v)}<br>status: {cities[n].get('status')}"
            for n, v in zip(names, npv_pv)],
        mode="markers+text", textposition="top center",
        marker=dict(size=[max(10, v / 25_000) for v in npv_pv],
                    color=npv_pv, colorscale="RdYlGn", cmid=0, line_width=1),
        textfont=dict(size=12),
        hoverinfo="text",
        name="",
    ))
    fig.update_geos(
        scope="asia", center=dict(lat=16.5, lon=77.5),
        lataxis_range=[8, 25], lonaxis_range=[68, 90],
        showcountries=True, countrycolor="#B9B2A6",
        showland=True, landcolor="#F4F1EA", showocean=True, oceancolor="#E6EEF0",
    )
    fig.update_layout(title="Markets under evaluation (marker size & colour = NPV per vehicle)",
                      height=460, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("City economics (five-year engine per city)")
    st.dataframe(
        table.style.format({
            "revenue_per_vehicle_pm": "₹{:,.0f}", "ebitda_per_vehicle_pm": "₹{:,.0f}",
            "npv_per_vehicle": "₹{:,.0f}", "capex_per_vehicle": "₹{:,.0f}",
            "setup_capex": "₹{:,.0f}", "break_even_fleet_utilisation": "{:.1%}",
            f"npv_at_{econ[names[0]].fleet_ref}_vehicles": "₹{:,.0f}",
            "irr_ref_fleet": "{:.0%}", "payback_ref_months": "{:.0f}",
        }),
        use_container_width=True,
    )
    with st.expander("City operating assumptions in force"):
        rows = []
        for n, c in cities.items():
            r = {"city": n, "confidence": c.get("confidence")}
            r.update({k.split(".")[-1]: v for k, v in (c.get("overrides") or {}).items()})
            r.update({f"exp_{k}": v for k, v in (c.get("expansion") or {}).items()})
            rows.append(r)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.divider()
    st.subheader("Capital allocation optimiser")
    cc = st.columns(3)
    budget = cc[0].number_input("Expansion budget (₹)", 10_000_000, 500_000_000,
                                int(cfg.get("budget_inr", 50_000_000)), 5_000_000,
                                format="%d")
    max_share = cc[1].slider("Max capital share per city", 0.25, 1.0, 1.0, 0.05,
                             help="Force diversification by capping any one city's share")
    irr_filter = cc[2].toggle(f"Only cities above {p.hurdle_irr:.0%} IRR hurdle", value=False)

    alloc = optimise_allocation(econ, float(budget), p,
                                max_share_per_city=float(max_share),
                                min_irr_filter=bool(irr_filter))
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Solver", alloc.solver.split("(")[0], alloc.status)
    a2.metric("Portfolio NPV", inr(alloc.total_npv))
    a3.metric("Capital deployed", inr(alloc.total_capex))
    a4.metric("Held back", inr(alloc.unused_capital))

    fig = go.Figure()
    tbl = alloc.table
    fig.add_bar(x=tbl["city"], y=tbl["vehicles"], name="Vehicles allocated",
                text=tbl["vehicles"], textposition="outside")
    fig.update_layout(title="Optimal vehicle allocation", height=340)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        tbl.style.format({
            "capex": "₹{:,.0f}", "npv": "₹{:,.0f}", "npv_per_vehicle": "₹{:,.0f}",
            "capex_per_vehicle": "₹{:,.0f}", "setup_capex": "₹{:,.0f}",
            "irr_ref": "{:.0%}",
        }),
        use_container_width=True,
    )

    st.subheader("Expansion decision per city")
    for cl in alloc.classifications:
        colour = {"Expand": "green", "Pilot": "orange", "Do not expand": "red"}[cl["decision"]]
        with st.container(border=True):
            st.markdown(f"**{cl['city']}** — :{colour}[**{cl['decision']}**]  ·  "
                        f"NPV {inr(cl['npv_ref'])} · IRR {pct(cl['irr_ref'], 0)} · "
                        f"payback {cl['payback_ref_months']:.0f}m · "
                        f"data confidence: {cl['confidence']}")
            st.caption(cl["rationale"])
    th = alloc.classifications[0]["thresholds"]
    st.caption("Transparent thresholds: IRR ≥ "
               f"{th['hurdle_irr']:.0%}, payback ≤ {th['max_payback_months']}m, NPV {th['npv']}, "
               f"confidence {th['confidence_for_expand']} — configurable in "
               "`config/assumptions.yaml` (valuation block).")
    st.caption("Constraints applied: budget; per-city demand ceiling and charger capacity; "
               "minimum viable cohort; optional per-city capital cap and IRR pre-filter. "
               "The recommendation is an allocation, not a ranking (spec §Module 8).")
