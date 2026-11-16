"""One-command project bootstrap.

    python bootstrap.py [--sims 10000] [--skip-data]

1. Generates the synthetic operational dataset (labelled simulated).
2. Validates it and loads SQLite (or PostgreSQL via DATABASE_URL).
3. Runs the full analytical pipeline: base model, scenarios, sensitivity,
   Monte Carlo, EV-vs-petrol TCO, city economics, capital allocation,
   insights.
4. Writes reports/: management_report.md, financial_statements_base.xlsx,
   assumptions_register.xlsx, monte_carlo_summary.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=10_000, help="Monte Carlo runs")
    ap.add_argument("--skip-data", action="store_true",
                    help="reuse existing synthetic data & database")
    args = ap.parse_args()
    t0 = time.time()

    import pandas as pd

    from analytics.insights import generate_insights
    from analytics.recommendations import final_recommendation
    from config.loader import load_city_config, load_params
    from database.connection import get_engine, init_schema
    from forecasting import run_model
    from ingestion.generate import SYNTH_DIR, generate, write_outputs
    from ingestion.load_financials import load_financials
    from ingestion.load_operations import load_operations
    from ingestion.load_vehicles import load_vehicles
    from optimisation.expansion import all_city_economics
    from optimisation.fleet_allocation import optimise_allocation
    from reports.generate_report import (
        REPORTS_DIR,
        build_report,
        export_register_xlsx,
        export_statements_xlsx,
    )
    from simulation.monte_carlo import run_monte_carlo
    from simulation.scenarios import run_all_scenarios, scenario_comparison
    from simulation.sensitivity import tornado

    print("── VoltMove bootstrap ─────────────────────────────────────────")
    if args.skip_data and (SYNTH_DIR / "daily_operations.parquet").exists():
        print("1-2. Reusing existing dataset & database")
        vehicles = pd.read_csv(SYNTH_DIR / "vehicles.csv")
        ops = pd.read_parquet(SYNTH_DIR / "daily_operations.parquet")
    else:
        print("1. Generating synthetic operational dataset (SIMULATED)…")
        vehicles, ops = generate()
        write_outputs(vehicles, ops)
        print(f"   {len(vehicles):,} vehicles · {len(ops):,} vehicle-day rows")
        print("2. Validating and loading the database…")
        eng = get_engine()
        init_schema(eng)
        load_vehicles(engine=eng)
        load_operations(vehicles=vehicles, engine=eng)
        load_financials(engine=eng)
        print(f"   loaded -> {eng.url}")

    p = load_params()
    print("3. Base five-year model…")
    base = run_model(p)
    print(f"   NPV {base.metric('npv_fcff'):,.0f} · IRR {base.metric('irr_fcff'):.1%} · "
          f"payback {base.metric('payback_months_fcff'):.0f}m · "
          f"BS gap {base.metric('bs_max_abs_gap'):.2e}")

    print("4. Scenarios…")
    scen = scenario_comparison(run_all_scenarios())

    print("5. Sensitivity (tornado, full model per point)…")
    tor = tornado(p)

    print(f"6. Monte Carlo ({args.sims:,} runs)…")
    mc = run_monte_carlo(p, n_sims=args.sims)
    print(f"   P(NPV>0) {mc.stats['p_npv_positive']:.1%} · "
          f"median IRR {mc.stats['irr_p50']:.1%} · "
          f"CVaR5 {mc.stats['npv_cvar05']:,.0f}")

    print("7. City economics & ₹5 cr capital allocation…")
    econ = all_city_economics()
    alloc = optimise_allocation(econ, float(load_city_config()["budget_inr"]), p)
    print(f"   {alloc.solver} {alloc.status} -> "
          + ", ".join(f"{c}:{v}" for c, v in alloc.allocation.items()))

    print("8. Insight rules…")
    insights = generate_insights(ops, vehicles, p,
                                 city_npv={n: e.npv_ref for n, e in econ.items()})
    final = final_recommendation(p, base, mc.stats, alloc, insights)
    print(f"   {len(insights)} findings · final verdict: {final.verdict}")

    print("9. Writing reports/…")
    REPORTS_DIR.mkdir(exist_ok=True)
    report_md = build_report(p, base, scen, mc.stats, mc.driver_importance,
                             tor, alloc, insights, final)
    (REPORTS_DIR / "management_report.md").write_text(report_md, encoding="utf-8")
    export_statements_xlsx(base)
    export_register_xlsx()
    (REPORTS_DIR / "monte_carlo_summary.json").write_text(
        json.dumps(mc.stats, indent=2), encoding="utf-8")
    tor.to_csv(REPORTS_DIR / "sensitivity_tornado.csv", index=False)
    alloc.table.to_csv(REPORTS_DIR / "capital_allocation.csv", index=False)

    print(f"── done in {time.time()-t0:.1f}s ──")
    print("Launch the dashboard:  streamlit run app.py")


if __name__ == "__main__":
    main()
