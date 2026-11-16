# EV Fleet Economics & Strategic Expansion Intelligence

*A data-driven financial modelling and decision-support platform for evaluating the
unit economics, profitability, operational efficiency and expansion strategy of
electric two-wheeler fleets in India.*

The system simulates **VoltMove Mobility**, a hypothetical operator of 500 electric
scooters in Bengaluru billed per productive kilometre to logistics companies, and
answers the core management question:

> Given a fixed amount of investment capital, what is the most economically viable
> way to operate and expand an electric two-wheeler fleet while maintaining
> profitability and acceptable investment returns?

**Core capabilities**

- Vehicle-level and fleet-level unit economics (contribution, break-even, payback)
- Integrated five-year financial model — P&L, cash flow and balance sheet that
  reconcile to the rupee every month (`Assets = Liabilities + Equity` is unit-tested)
- EV vs petrol total cost of ownership on an identical workload
- Battery degradation, replacement capex and availability coupling
- Debt vs owned financing (EMI schedule; principal never touches EBITDA)
- Scenario engine (base / optimistic / pessimistic), one-variable tornado,
  two-variable heatmaps — every point is a full model re-run
- 10,000-run vectorised Monte Carlo with bounded distributions (~1 s)
- Multi-city expansion benchmarking + **constrained capital allocation** via
  OR-Tools CP-SAT (scipy/HiGHS fallback) under a ₹5 crore budget
- Rule-based strategic recommendations with **model-computed** financial impact
- Six-screen Streamlit management dashboard + auto-generated investment report

---

## Quick start

```bash
pip install -r requirements.txt
python bootstrap.py          # generate data -> load DB -> run full pipeline -> write reports/
streamlit run app.py         # open the six-screen dashboard
pytest                       # 53 tests: spec hand-calcs, statement reconciliation, behaviour
```

`bootstrap.py` takes ~5 s and produces:

| Output | Path |
|---|---|
| Synthetic vehicle-day dataset (labelled **simulated**) | `data/synthetic/` |
| SQLite database (PostgreSQL via `DATABASE_URL`) | `data/voltmove.db` |
| Management investment report (spec §16's ten questions) | `reports/management_report.md` |
| Five-year statements (annual + monthly) | `reports/financial_statements_base.xlsx` |
| Assumptions register (value·unit·source·date·confidence) | `reports/assumptions_register.xlsx` |
| Monte Carlo summary / tornado / allocation | `reports/*.json`, `reports/*.csv` |

To use PostgreSQL instead of SQLite:

```bash
pip install psycopg2-binary
export DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/voltmove
python bootstrap.py
```

## Architecture

```
config/        assumptions.yaml (register with source/unit/date/confidence),
               scenarios.yaml, city_parameters.yaml, loader.py (pydantic)
ingestion/     synthetic data generator (relationship-driven), validation, DB loaders
database/      schema.sql (PostgreSQL-compatible), connection.py (SQLite default)
models/        utilisation, energy cost, maintenance, battery, depreciation, financing
economics/     revenue, unit economics, break-even, EV-vs-petrol TCO, NPV/IRR/payback
forecasting/   integrated 5-year engine (vectorised over simulations) + statement views
simulation/    scenarios, sensitivity (tornado & heatmaps), Monte Carlo
optimisation/  per-city economics, CP-SAT capital allocation, Expand/Pilot/No framework
analytics/     KPIs, rule-based insights (impact computed by re-running the engine)
dashboard/     six Streamlit screens — rendering only, no financial logic
reports/       management report generator + Excel exports
tests/         53 pytest cases incl. the spec's hand-calculated examples
```

Architectural rule (spec §9): **the dashboard never computes finance** — every number
on screen comes from the engine packages, which are importable and unit-tested on
their own.

The five-year engine is written once, vectorised over a leading simulation axis:
the deterministic forecast is `n=1`, Monte Carlo is `n=10,000` of the *same* code
(a pinned-draw equality test enforces this), so the risk analysis can never drift
from the plan model.

## Verified against the spec's hand calculations

| Quantity | Spec value | Test |
|---|---|---|
| Electricity cost/km (0.04/0.90 × ₹12) | ₹0.533 | `test_electricity_cost_per_km_matches_spec` |
| Annual depreciation (₹1.2L, 15%, 5y) | ₹20,400 | `test_annual_depreciation_matches_spec` |
| Monthly revenue (100 × 0.85 × 26 × ₹8) | ₹17,680 | `test_monthly_revenue_matches_spec` |
| Variable cost per PKM | ₹1.098 | `test_variable_cost_per_pkm_matches_spec` |
| Contribution per PKM | ₹6.902 | `test_contribution_matches_spec` |

Financial consistency (every month, every scenario): balance sheet balances; opening
cash + net cash flow = closing cash; EBITDA is identical across financing modes;
loan principal reduces debt, not profit; battery replacement is capex, counted once.

## Data & honesty

All operational records are **synthetic** and labelled as such throughout the UI and
report. Fields are generated through relationships (demand → km → energy → cost;
age → breakdowns → downtime → km; age + km → SOH → range cap), then validated
(productive km ≤ total km, SOH ∈ (0,1], monotone fade, referential integrity).

Every assumption in `config/assumptions.yaml` carries **value, unit, source, as-of
date and a confidence tag**; low-confidence entries (billing rate, tariff,
maintenance, residual value, all candidate-city parameters) are flagged in the UI
and must be replaced with dated quotations, DISCOM tariff schedules and signed
contract rates before any real investment decision. Calibration sources: VAHAN,
NITI Aayog, PM E-DRIVE, BESCOM, Indian Oil, OEM quotations (spec §10C).

## Limitations (spec §13)

Synthetic data proves the machinery, not market performance; battery fade is a
simplified calendar+cycle model; demand is assumed, not forecast; charging is a
capacity ratio, not a grid simulation; tax/GST treatment is simplified and
documented; the assumed interest rate is not a financing offer; the optimiser's
allocation is only as good as its inputs — every run publishes its constraints and
register. Decision thresholds (18% IRR hurdle, 36-month payback, confidence gates)
are transparent and configurable, and the final verdict (Invest / Pilot / Optimise
first / Defer) follows mechanically from them.
