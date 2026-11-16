"""VoltMove Mobility — EV Fleet Economics & Strategic Expansion Intelligence.

Streamlit entry point. All financial logic lives in the engine packages
(models/, economics/, forecasting/, simulation/, optimisation/,
analytics/) — the dashboard only renders their outputs (spec §9).

Run:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="VoltMove — EV Fleet Economics",
    page_icon="⚡",
    layout="wide",
)

from dashboard import (  # noqa: E402
    expansion,
    forecasting,
    overview,
    recommendations,
    scenarios,
    unit_economics,
)

pages = [
    st.Page(overview.render, title="Executive overview", icon="📊", url_path="overview",
            default=True),
    st.Page(unit_economics.render, title="Unit economics explorer", icon="🛵",
            url_path="unit-economics"),
    st.Page(forecasting.render, title="Financial planning", icon="📈",
            url_path="forecasting"),
    st.Page(scenarios.render, title="Scenario simulator", icon="🎲", url_path="scenarios"),
    st.Page(expansion.render, title="City expansion planner", icon="🗺️",
            url_path="expansion"),
    st.Page(recommendations.render, title="Strategic recommendations", icon="🧭",
            url_path="recommendations"),
]

nav = st.navigation(pages)
with st.sidebar:
    st.markdown("### ⚡ VoltMove Mobility")
    st.caption("EV fleet investment & operating strategy decision support · "
               "Bengaluru · 500 e-2W · per-km B2B billing")
nav.run()
