"""Shared dashboard helpers: cached data access, formatting, chart theme.
The dashboard NEVER computes finance itself (spec §9 architecture rule) —
it calls the engine modules and renders their outputs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from config.loader import Params, load_params

ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "data" / "synthetic"

ACCENT = "#0F7B6C"
ACCENT2 = "#C76E00"
NEGATIVE = "#B3261E"
GRID = "#E8E4DC"

_template = go.layout.Template()
_template.layout = go.Layout(
    font=dict(family="Georgia, 'Times New Roman', serif", size=13, color="#2A2722"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    colorway=[ACCENT, ACCENT2, "#5B5EA6", NEGATIVE, "#7A7265", "#3E7CB1"],
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
pio.templates["voltmove"] = _template
pio.templates.default = "voltmove"


def inr(x: float, decimals: int = 1) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if x != x:
        return "n/a"
    if abs(x) >= 1e7:
        return f"₹{x/1e7:,.{decimals}f} cr"
    if abs(x) >= 1e5:
        return f"₹{x/1e5:,.{decimals}f} L"
    return f"₹{x:,.0f}"


def pct(x: float, decimals: int = 1) -> str:
    try:
        return f"{float(x)*100:,.{decimals}f}%"
    except (TypeError, ValueError):
        return "n/a"


@st.cache_data(show_spinner=False)
def load_ops() -> pd.DataFrame | None:
    path = SYNTH / "daily_operations.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_vehicles() -> pd.DataFrame | None:
    path = SYNTH / "vehicles.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def base_params() -> Params:
    return load_params()


def require_data():
    ops, veh = load_ops(), load_vehicles()
    if ops is None or veh is None:
        st.error(
            "Operational dataset not found. Generate it first:\n\n"
            "```\npython bootstrap.py\n```"
        )
        st.stop()
    return ops, veh


def simulated_badge():
    st.caption(":orange-badge[⚠ SIMULATED DATA] All operational records are "
               "synthetic (spec §13.1); assumptions carry source & confidence "
               "tags in `config/assumptions.yaml`.")
