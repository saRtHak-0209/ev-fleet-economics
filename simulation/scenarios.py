"""Predefined scenario engine (spec §Module 6): optimistic / base /
pessimistic, defined declaratively in config/scenarios.yaml."""
from __future__ import annotations

import pandas as pd

from config.loader import Params, load_params, load_scenarios
from forecasting import run_model
from forecasting.engine import ModelResult

SUMMARY_METRICS = [
    ("Revenue (5y, ₹)", "revenue_total"),
    ("EBITDA (5y, ₹)", "ebitda_total"),
    ("EBITDA margin", "ebitda_margin"),
    ("PAT (5y, ₹)", "pat_total"),
    ("NPV @ discount rate (₹)", "npv_fcff"),
    ("IRR (annual)", "irr_fcff"),
    ("Payback (months)", "payback_months_fcff"),
    ("Battery replaced in horizon", "battery_replaced"),
    ("Minimum cash (₹)", "min_cash"),
]


def scenario_params(name: str, base_overrides: dict | None = None) -> Params:
    scen = load_scenarios()[name]
    overrides = dict(base_overrides or {})
    overrides.update(scen.get("overrides") or {})
    return load_params(overrides)


def run_scenario(name: str, base_overrides: dict | None = None) -> ModelResult:
    return run_model(scenario_params(name, base_overrides))


def run_all_scenarios(base_overrides: dict | None = None) -> dict[str, ModelResult]:
    return {name: run_scenario(name, base_overrides) for name in load_scenarios()}


def scenario_comparison(results: dict[str, ModelResult]) -> pd.DataFrame:
    scen_cfg = load_scenarios()
    data = {}
    for name, res in results.items():
        label = scen_cfg.get(name, {}).get("label", name)
        data[label] = {metric_label: res.metric(key) for metric_label, key in SUMMARY_METRICS}
    return pd.DataFrame(data)
