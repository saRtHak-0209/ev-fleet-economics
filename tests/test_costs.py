"""Mathematical validation of cost models against the spec's own
hand-calculated examples (spec §12A)."""
import numpy as np
import pytest

from models import depreciation, energy_cost, financing
from models.battery import availability_penalty, battery_lifecycle


def test_electricity_cost_per_km_matches_spec(p):
    # spec §Module 3A: 0.04 / 0.90 * 12 = 0.533 ₹/km
    val = energy_cost.electricity_cost_per_km(p.kwh_per_km, p.charging_efficiency, p.tariff)
    assert val == pytest.approx(0.5333, abs=1e-3)


def test_daily_electricity_spend_matches_spec(p):
    # ~₹53.33 per vehicle per day at 100 km
    cost = energy_cost.electricity_cost(100, p.kwh_per_km, p.charging_efficiency, p.tariff)
    assert cost == pytest.approx(53.33, abs=0.01)


def test_grid_energy_grossup():
    # 100 km * 0.04 / 0.9 = 4.444 kWh from the grid
    assert energy_cost.grid_energy_kwh(100, 0.04, 0.9) == pytest.approx(4.4444, abs=1e-3)


def test_annual_depreciation_matches_spec(p):
    # spec §Module 3D: (120000 - 18000) / 5 = 20,400 / year
    dep = depreciation.annual_straight_line(
        p.vehicle_cost, p.vehicle_cost * p.residual_pct, 5)
    assert dep == pytest.approx(20_400.0)


def test_depreciation_schedule_reaches_residual(p):
    dep, nbv = depreciation.schedule(120_000, 18_000, 60, 60)
    assert dep.sum() == pytest.approx(102_000.0)
    assert nbv[-1] == pytest.approx(18_000.0)


def test_emi_against_closed_form():
    # ₹100,000 at 12% for 12 months -> 8,884.88 (standard EMI table value)
    assert financing.emi(100_000, 0.12, 12) == pytest.approx(8_884.88, abs=0.01)


def test_amortisation_identities():
    sched = financing.amortisation(84_000, 0.14, 36, 60)
    # principal repaid sums to the loan; balance ends at zero
    assert sched.principal.sum() == pytest.approx(84_000, abs=1e-6)
    assert sched.balance[36:].max() == pytest.approx(0.0, abs=1e-9)
    # total payments = EMI * n = principal + interest
    assert sched.emi * 36 == pytest.approx(84_000 + sched.interest.sum(), abs=1e-6)


def test_battery_replacement_triggers_on_threshold():
    # 6%/yr fade crosses 75% SOH at ~50 months -> replacement inside 60m
    path = battery_lifecycle(60, 0.06, 0.0, 2340.0, 0.75, 3.0, 9000.0, 6)
    assert path.replacement_month[0] > 0
    assert path.replacement_capex_per_vehicle.sum() == pytest.approx(27_000.0)
    # SOH resets at replacement and never exceeds 1
    assert path.soh.max() <= 1.0
    r = path.replacement_month[0]
    assert path.soh[0, r] > path.soh[0, r - 1]


def test_battery_no_replacement_in_base_case(p):
    # 3%/yr -> 85% SOH at month 60, above the 75% threshold
    path = battery_lifecycle(60, p.batt_calendar_fade_py, 0.0, 2340.0,
                             p.soh_replace_threshold, 3.0, 9000.0, 6)
    assert path.replacement_month[0] == -1
    assert path.soh[0, -1] == pytest.approx(0.85, abs=1e-6)


def test_availability_penalty_only_below_threshold():
    soh = np.array([[0.95, 0.80, 0.70]])
    pen = availability_penalty(soh, threshold=0.80, k=0.5)
    assert pen[0, 0] == 1.0 and pen[0, 1] == 1.0
    assert pen[0, 2] == pytest.approx(0.95)
