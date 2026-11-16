"""Financial consistency checks on the integrated model (spec §12B)."""
import numpy as np
import pytest

from forecasting import run_model


def _modes(p):
    return [p, p.with_(financing_mode="financed"),
            p.with_(batt_calendar_fade_py=0.06),           # forces battery capex
            p.with_(billing_rate=3.0),                      # loss-making case
            p.with_(financing_mode="financed", batt_calendar_fade_py=0.07,
                    billing_rate=4.0, receivable_days=45)]


def test_balance_sheet_balances_every_month(p):
    for params in _modes(p):
        res = run_model(params)
        gap = res.metric("bs_max_abs_gap")
        scale = max(1.0, abs(res.series("total_assets")).max())
        assert gap / scale < 1e-9, f"A != L+E for {params.financing_mode}"


def test_cash_ties_to_cash_flow_statement(p):
    for params in _modes(p):
        res = run_model(params)
        cash = res.series("cash")
        net = res.series("cfo") + res.series("cfi") + res.series("cff")
        opening = float(np.asarray(res.t0["opening_cash"]).ravel()[0])
        rebuilt = opening + np.cumsum(net)
        assert np.allclose(cash, rebuilt, atol=1e-6)


def test_principal_repayment_never_touches_ebitda(p):
    owned = run_model(p)
    financed = run_model(p.with_(financing_mode="financed"))
    # EBITDA identical across financing modes...
    assert np.allclose(owned.series("ebitda"), financed.series("ebitda"))
    # ...while interest reduces PBT only in the financed case
    assert financed.series("interest").sum() > 0
    assert owned.series("interest").sum() == 0
    assert financed.metric("pat_total") < owned.metric("pat_total")


def test_vehicle_purchase_and_depreciation_mechanics(p):
    res = run_model(p)
    nbv = res.series("nbv_vehicles")
    gross = p.fleet_size * (p.vehicle_cost + p.registration_cost)
    # depreciation reduces book value monotonically
    assert nbv[0] < gross
    assert (np.diff(nbv) < 0).all()
    # fleet is sold at residual book value in the final month (NBV -> 0)
    assert nbv[-1] == pytest.approx(0.0, abs=1e-6)
    assert res.series("salvage")[-1] >= p.fleet_size * p.vehicle_cost * p.residual_pct


def test_loan_principal_reduces_outstanding_debt(p):
    res = run_model(p.with_(financing_mode="financed"))
    bal = res.series("loan_balance")
    loan0 = float(np.asarray(res.t0["loan0"]).ravel()[0])
    principal = res.series("principal_repaid")
    assert loan0 > 0
    assert bal[0] == pytest.approx(loan0 - principal[0], rel=1e-9)
    assert bal[p.tenure_months - 1] == pytest.approx(0.0, abs=1e-6)
    assert principal.sum() == pytest.approx(loan0, rel=1e-9)


def test_battery_replacement_not_double_counted(p):
    fast = p.with_(batt_calendar_fade_py=0.08)
    res = run_model(fast)
    batt_capex = res.series("battery_capex").sum()
    assert batt_capex > 0
    # replacement shows in capex/investing, never in maintenance opex:
    base_maint = run_model(p).series("maintenance")
    fast_maint = res.series("maintenance")
    # maintenance may differ only via km path (availability coupling), never
    # by the pack cost
    assert abs(fast_maint.sum() - base_maint.sum()) < batt_capex * 0.5
    # the pack is fully depreciated by the end of the horizon
    assert res.series("nbv_battery")[-1] == pytest.approx(0.0, abs=1e-6)
    assert res.series("depreciation").sum() >= run_model(p).series("depreciation").sum()


def test_tax_loss_carryforward_no_refunds(p):
    # loss-making early months must not create negative tax
    res = run_model(p.with_(billing_rate=3.2))
    tax = res.series("tax")
    cum_tax = np.cumsum(tax)
    assert (cum_tax >= -1e-9).all()
    # cumulative tax equals rate x running max of positive cumulative PBT
    cum_pbt = np.cumsum(res.series("pbt"))
    expected = p.tax_rate * np.maximum.accumulate(np.clip(cum_pbt, 0, None))
    assert np.allclose(cum_tax, expected, atol=1e-6)


def test_mc_single_run_consistency(p):
    """A 1-sim run with draws pinned to base values reproduces the
    deterministic engine exactly (single source of truth)."""
    det = run_model(p)
    pinned = {k: np.array([getattr(p, k)]) for k in
              ("daily_km", "productive_ratio", "availability", "tariff",
               "maint_per_km", "batt_calendar_fade_py", "residual_pct")}
    sim = run_model(p, draws=pinned, n_sims=1)
    assert sim.metric("npv_fcff") == pytest.approx(det.metric("npv_fcff"), rel=1e-12)
    assert np.allclose(sim.monthly["ebitda"], det.monthly["ebitda"])
