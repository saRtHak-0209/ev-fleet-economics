"""Behavioural validation (spec §12C) + Monte Carlo guardrails."""
import numpy as np
import pytest

from forecasting import run_model
from simulation.monte_carlo import draw_inputs, run_monte_carlo
from simulation.scenarios import run_all_scenarios
from simulation.sensitivity import one_variable, tornado


def _m(p, metric="npv_fcff", **chg):
    return run_model(p.with_(**chg)).metric(metric)


def test_behaviour_table(p):
    """Spec §12C: the model must respond logically to input changes."""
    base_energy = run_model(p).series("energy_cost").sum()
    assert run_model(p.with_(tariff=p.tariff * 1.2)).series("energy_cost").sum() > base_energy

    base_rev = run_model(p).metric("revenue_total")
    assert _m(p, "revenue_total", productive_ratio=0.75) < base_rev
    assert _m(p, "revenue_total", availability=0.80) < base_rev
    assert _m(p, "revenue_total", daily_km=p.daily_km * 1.2) > base_rev

    ue_base = run_model(p).series("contribution").sum()
    assert run_model(p.with_(billing_rate=9.0)).series("contribution").sum() > ue_base

    assert _m(p, "capex0", vehicle_cost=150_000) > _m(p, "capex0")
    assert _m(p, "ebitda_total", maint_per_km=0.8) < _m(p, "ebitda_total")

    from economics.tco import compare_tco
    assert (compare_tco(p.with_(residual_pct=0.05))["tco_5y_ev_discounted"]
            > compare_tco(p)["tco_5y_ev_discounted"])


def test_scenarios_are_ordered(p):
    res = run_all_scenarios()
    npv = {k: v.metric("npv_fcff") for k, v in res.items()}
    assert npv["optimistic"] > npv["base"] > npv["pessimistic"]
    assert res["pessimistic"].metric("battery_replaced") == 1.0
    assert res["base"].metric("battery_replaced") == 0.0


def test_one_variable_sensitivity_uses_full_model(p):
    df = one_variable(p, "daily_km", [40, 100, 120])
    npv40, npv100, npv120 = df["npv_fcff"].tolist()
    assert npv40 < npv100 < npv120
    # profit impact is NOT proportional to revenue (fixed costs & capital):
    rev_ratio = 40 / 100
    npv_ratio = npv40 / npv100
    assert abs(npv_ratio - rev_ratio) > 0.05


def test_tornado_ranks_rate_and_utilisation_top(p):
    tor = tornado(p)
    top3 = set(tor.head(3)["param"])
    assert "billing_rate" in top3
    assert "daily_km" in top3 or "productive_ratio" in top3


def test_mc_draws_are_bounded(p):
    draws, noise = draw_inputs(p, 5000, seed=7)
    assert (draws["productive_ratio"] > 0).all() and (draws["productive_ratio"] < 1).all()
    assert (draws["availability"] > 0).all() and (draws["availability"] <= 1).all()
    assert (draws["daily_km"] >= 30).all() and (draws["daily_km"] <= 160).all()
    assert (draws["maint_per_km"] >= 0.10).all()
    assert (draws["batt_calendar_fade_py"] >= 0.02).all()
    assert (noise > 0).all()


def test_mc_outputs_internally_impossible_values_never_occur(p):
    mc = run_monte_carlo(p, n_sims=2000, seed=11)
    res = mc.result
    assert (res.monthly["productive_km"] <= res.monthly["total_km"] + 1e-9).all()
    assert (res.monthly["total_km"] >= 0).all()
    assert (res.monthly["battery_soh"] <= 1.0 + 1e-12).all()
    assert np.isfinite(mc.stats["npv_mean"])
    assert 0.0 <= mc.stats["p_npv_positive"] <= 1.0


def test_mc_centres_near_deterministic(p):
    mc = run_monte_carlo(p, n_sims=4000, seed=3)
    det = run_model(p).metric("npv_fcff")
    # median within 15% of the deterministic base (skewed inputs allowed)
    assert abs(mc.stats["npv_p50"] - det) / det < 0.15
