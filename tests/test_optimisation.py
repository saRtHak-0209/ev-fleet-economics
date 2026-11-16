"""Capital allocation optimiser tests (spec §Module 8)."""
import numpy as np
import pytest

from config.loader import load_city_config
from optimisation.expansion import CityEconomics, all_city_economics, classify_city
from optimisation.fleet_allocation import optimise_allocation


def _toy_city(name, npv_pv, capex_pv=128_250.0, setup=2_000_000.0,
              min_fleet=50, max_fleet=300, irr=0.5, conf="low"):
    return CityEconomics(
        city=name, status="candidate", confidence=conf, params=None, fleet_ref=200,
        npv_per_vehicle=npv_pv, npv_fixed_city=-setup * 0.9,
        capex_per_vehicle=capex_pv, setup_capex=setup,
        revenue_per_vehicle_pm=15_000, contribution_per_vehicle_pm=12_000,
        ebitda_per_vehicle_pm=10_000, break_even_fleet_utilisation=0.15,
        npv_ref=npv_pv * 200, irr_ref=irr, payback_ref_months=20,
        max_fleet=max_fleet, min_fleet=min_fleet, city_fixed_pm=200_000,
    )


@pytest.fixture(scope="module")
def econ():
    return all_city_economics()


def test_budget_constraint_respected(econ, p):
    budget = float(load_city_config()["budget_inr"])
    r = optimise_allocation(econ, budget, p)
    assert r.status in ("OPTIMAL", "FEASIBLE")
    assert r.total_capex <= budget + 1.0
    assert r.total_capex + r.unused_capital == pytest.approx(budget)


def test_min_max_fleet_bounds(econ, p):
    r = optimise_allocation(econ, float(load_city_config()["budget_inr"]), p)
    for city, x in r.allocation.items():
        e = econ[city]
        assert x == 0 or (e.min_fleet <= x <= e.max_fleet)


def test_allocation_is_an_allocation_not_a_ranking(econ, p):
    r = optimise_allocation(econ, 5e7, p)
    assert sum(r.allocation.values()) > 0
    assert isinstance(next(iter(r.allocation.values())), int)


def test_optimiser_beats_single_city_on_crafted_inputs(p):
    # City A: great per-vehicle NPV but tiny demand cap; City B: moderate.
    # Optimal must split, beating either single-city plan.
    econ = {
        "A": _toy_city("A", npv_pv=400_000, max_fleet=100),
        "B": _toy_city("B", npv_pv=250_000, max_fleet=400),
    }
    budget = 3.5e7
    r = optimise_allocation(econ, budget, p)
    assert r.allocation["A"] == 100          # cap binds
    assert r.allocation["B"] >= 50           # remaining budget deployed
    npv_single_a = 100 * 400_000 - 2_000_000 * 0.9
    assert r.total_npv > npv_single_a


def test_max_share_constraint(p):
    econ = {
        "A": _toy_city("A", npv_pv=400_000, max_fleet=1000),
        "B": _toy_city("B", npv_pv=250_000, max_fleet=1000),
    }
    budget = 5e7
    r = optimise_allocation(econ, budget, p, max_share_per_city=0.5)
    for city in econ:
        e = econ[city]
        spend = r.allocation[city] * e.capex_per_vehicle + (
            e.setup_capex if r.allocation[city] > 0 else 0)
        assert spend <= budget * 0.5 + 1.0


def test_solver_parity_cpsat_vs_milp(econ, p):
    budget = float(load_city_config()["budget_inr"])
    a = optimise_allocation(econ, budget, p, force_solver="cpsat")
    b = optimise_allocation(econ, budget, p, force_solver="milp")
    assert a.total_npv == pytest.approx(b.total_npv, rel=1e-3)


def test_negative_npv_city_gets_nothing(p):
    econ = {
        "Good": _toy_city("Good", npv_pv=300_000),
        "Bad": _toy_city("Bad", npv_pv=-50_000),
    }
    r = optimise_allocation(econ, 5e7, p)
    assert r.allocation["Bad"] == 0
    assert r.allocation["Good"] > 0


def test_classification_thresholds(p):
    strong = _toy_city("S", npv_pv=300_000, irr=0.40, conf="high")
    assert classify_city(strong, p)["decision"] == "Expand"
    unvalidated = _toy_city("U", npv_pv=300_000, irr=0.40, conf="low")
    assert classify_city(unvalidated, p)["decision"] == "Pilot"
    weak = _toy_city("W", npv_pv=-10_000, irr=0.05, conf="high")
    weak.npv_ref = -2_000_000
    assert classify_city(weak, p)["decision"] == "Do not expand"
