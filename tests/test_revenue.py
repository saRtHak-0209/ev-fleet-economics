"""Revenue & unit economics vs the spec's hand-calculated examples."""
import pytest

from economics.break_even import break_even_pkm_month, break_even_summary
from economics.revenue import fleet_monthly_revenue, monthly_revenue_per_vehicle
from economics.tco import compare_tco
from economics.unit_economics import compute_unit_economics
from economics import roi


def test_monthly_revenue_matches_spec(p):
    # spec §Module 4A: 100 x 0.85 x 26 x 8 = ₹17,680 / month
    rev = monthly_revenue_per_vehicle(p.daily_km, p.productive_ratio,
                                      p.operating_days, p.billing_rate)
    assert rev == pytest.approx(17_680.0)


def test_fleet_revenue_formula(p):
    # spec §Module 5A: FleetSize x DeployedRatio x PKM x Days x Rate
    rev = fleet_monthly_revenue(500, 0.9, 100, 0.85, 26, 8)
    assert rev == pytest.approx(500 * 0.9 * 17_680.0)


def test_variable_cost_per_pkm_matches_spec(p):
    # spec §Module 4B: (0.533 + 0.40) / 0.85 = ₹1.098 / PKM
    ue = compute_unit_economics(p)
    assert ue.variable_cost_per_total_km == pytest.approx(0.9333, abs=1e-3)
    assert ue.variable_cost_per_pkm == pytest.approx(1.098, abs=1e-3)


def test_contribution_matches_spec(p):
    # spec §Module 4C: 8 - 1.098 = ₹6.902 / PKM
    ue = compute_unit_economics(p)
    assert ue.contribution_per_pkm == pytest.approx(6.902, abs=1e-3)
    assert ue.contribution_margin_pct == pytest.approx(86.27, abs=0.05)


def test_break_even_formula(p):
    # BE PKM = fixed / (rate - varcost/PKM); hand example: 2000/(8-1.098)
    assert break_even_pkm_month(2000.0, 8.0, 1.098) == pytest.approx(2000 / 6.902, rel=1e-6)
    ue = compute_unit_economics(p)
    assert ue.break_even_pkm_month == pytest.approx(
        ue.monthly_fixed_cost / ue.contribution_per_pkm, rel=1e-9)


def test_break_even_summary_is_consistent(p):
    be = break_even_summary(p)
    ue = compute_unit_economics(p)
    # at the minimum viable rate, contribution covers fixed + capital charge
    margin = be["min_viable_billing_rate"] - ue.variable_cost_per_pkm
    covers = margin * ue.monthly_pkm
    assert covers == pytest.approx(
        ue.monthly_fixed_cost + be["capital_charge_per_vehicle_pm"], rel=1e-6)


def test_npv_and_irr_known_case():
    # classic: -1000 then 5 x 300 at 10% -> NPV 137.24; IRR 15.24%
    cfs = [300.0] * 5
    npv = roi.npv_t0(0.10, -1000.0, cfs)
    assert npv == pytest.approx(137.24, abs=0.01)
    irr = roi.irr([-1000.0] + cfs)
    assert irr == pytest.approx(0.15238, abs=1e-4)


def test_payback_months_first_nonnegative():
    pb = roi.payback_months([-100.0, 30, 30, 30, 30])
    assert pb == 4  # cumulative turns >= 0 at the 4th flow (t0 + 4)
    assert roi.payback_months([-100.0, 10, 10]) != roi.payback_months([-100.0, 60, 60])


def test_tco_same_workload_and_double_count_guard(p):
    cmp = compare_tco(p)
    ev, ice = cmp["ev"], cmp["petrol"]
    assert ev.total_km == pytest.approx(ice.total_km)
    assert ev.productive_km == pytest.approx(ice.productive_km)
    # financing-neutral: owned mode carries zero interest in both columns
    assert ev.monthly_interest.sum() == 0.0 and ice.monthly_interest.sum() == 0.0
    # verdict is computed, not assumed
    assert cmp["ev_is_cheaper"] == (cmp["tco_5y_ev_discounted"] < cmp["tco_5y_petrol_discounted"])


def test_tco_ev_not_automatically_cheaper(p):
    # push tariff very high & petrol very cheap -> petrol must win
    cmp = compare_tco(p.with_(tariff=60.0, petrol_price=60.0, petrol_maint_per_km=0.40))
    assert not cmp["ev_is_cheaper"]
