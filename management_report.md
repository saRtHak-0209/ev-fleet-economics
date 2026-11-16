# VoltMove Mobility — EV Fleet Investment Report
*Generated 2026-11-16 · base-case assumptions register v2026-11-17 · all operational data is SIMULATED*

> **Recommendation: Pilot.** Run a controlled pilot: base economics clear every threshold, but the candidate-city inputs are illustrative and must be validated in the field before committing the full ₹5 crore.

---
## 1. Is operating an EV fleet economically viable under base-case assumptions?
Yes under the current register: a 500-vehicle Bengaluru fleet produces five-year NPV of **₹14.44 cr** at a 12% discount rate, IRR **105%**, and year-1 EBITDA of ₹7.00 cr on revenue of ₹9.55 cr (73% margin). Viability is assumption-driven — the billing rate (₹8.0/PKM, confidence: low) dominates; see Q5 and §Risks.

## 2. What is the contribution margin per vehicle?
Each productive km bills ₹8.00 and costs ₹1.098 in variable cost (electricity ₹0.533/km + maintenance ₹0.40/km, grossed up for the 15% unbilled distance), leaving **₹6.90 per PKM (86%)**. A deployed vehicle contributes **₹15,253/month** (₹13,223 EBITDA after ₹2,030 allocated fixed cost).

## 3. What is the minimum daily utilisation required to break even?
- Cash (EBITDA) break-even: **11 productive km/day** (294 PKM/month).
- Full economic cost incl. capital recovery: **25 PKM/day**.
- Break-even fleet deployment: 13.3% of the fleet; minimum viable billing rate at plan utilisation: ₹3.13/PKM.
- The spec's worked example — a vehicle at 40 PKM/day — already clears the full-cost break-even of 25 PKM/day under these assumptions; the margin of safety is 15 PKM/day.

## 4. How does EV economics compare with petrol?
On an identical workload, the EV runs at **₹1.97/km** vs **₹3.84/km** for petrol (total cost incl. acquisition, energy, maintenance, replacements, net of resale). Five-year discounted TCO: ₹2.42 lakh vs ₹4.32 lakh — the EV is **cheaper** by ₹1.90 lakh per vehicle. The ₹35,000 purchase premium pays back in **8 months** of operating savings.

## 5. What are the three largest drivers of profitability?
One-variable sensitivity of five-year NPV (full model recomputation):
1. **Billing rate (₹/PKM)** — NPV swing ₹16.08 cr across the tested range.
2. **Daily km per vehicle** — NPV swing ₹13.67 cr across the tested range.
3. **Productive km ratio** — NPV swing ₹8.04 cr across the tested range.
Monte Carlo rank-correlation confirms: daily_km (ρ=+0.82), productive_ratio (ρ=+0.43), availability (ρ=+0.25).

## 6. What is the probability of generating a positive NPV?
Across 10,000 simulations with bounded input distributions: **P(NPV>0) = 100%**; P(payback ≤ 36 months) = 100%; median IRR 103% (5th-95th pct: 69% to 143%); worst-5% average NPV ₹7.78 cr; battery replacement occurs within the horizon in 29% of simulations.

## 7. Which city offers the strongest investment opportunity?
**Bengaluru** has the highest marginal NPV per vehicle (₹3.19 lakh) and the lowest incremental fixed cost. Candidate-city figures rest on illustrative inputs (confidence: low) and rank, not prove, the opportunity:
- Bengaluru: NPV/vehicle ₹3.19 lakh, ref-fleet IRR 101%, data confidence medium.
- Hyderabad: NPV/vehicle ₹2.75 lakh, ref-fleet IRR 75%, data confidence low.
- Chennai: NPV/vehicle ₹2.21 lakh, ref-fleet IRR 60%, data confidence low.
- Pune: NPV/vehicle ₹2.97 lakh, ref-fleet IRR 84%, data confidence low.

## 8. How should the company allocate its ₹5 crore expansion capital?
The constrained optimiser (ortools-cpsat, status OPTIMAL) maximises portfolio NPV subject to the ₹5 cr budget, per-city demand ceilings, minimum viable cohorts and charger capacity:
- **Bengaluru: 378 vehicles**
- Capital used: ₹5.00 cr; held back: ₹21,500; portfolio NPV ₹11.67 cr.
Decision classification per city (transparent thresholds): Bengaluru → **Expand**; Hyderabad → **Pilot**; Chennai → **Pilot**; Pune → **Pilot**.

## 9. What are the major risks to the recommendation?
- **Billing rate** (₹8.0/PKM) is a low-confidence assumption; the minimum viable rate is ₹3.13/PKM at plan utilisation — a thin contract market erases the margin.
- **Utilisation risk**: daily km carries the highest rank-correlation with NPV in simulation; demand is assumed, not proven (spec §13.3).
- **Battery risk**: at the pessimistic 6%/yr fade the pack crosses the 75% threshold inside the horizon, adding ₹1.35 cr fleet-wide capex.
- **Tariff & policy**: commercial tariffs and incentives can change; the model's tariff is a blended placeholder pending the applicable DISCOM schedule.
- **Model risk**: the optimiser output is only as good as its inputs; every run publishes its constraints and assumption register (spec §13.8).
- **Data**: all operational records in this build are synthetic and labelled as such; no claim of real-world predictive accuracy is made (spec §12D).

## 10. What should management do in the next 90 days?
1. **Improve customer demand allocation to existing vehicles before any fleet expansion; review idle-time by hub and daypart.** — ≈ ₹6,182,645 annual EBITDA if plan utilisation is reached.
2. **Evaluate battery replacement economics; audit charging behaviour (DoD, fast-charge share, heat) and warranty terms.** — Earlier, budgeted pack replacement instead of unplanned downtime.
3. Replace low-confidence register entries (billing rate, tariff, maintenance, residual value) with dated quotations and contracts, then re-run this report.

---
## Final recommendation: **Pilot**
Run a controlled pilot: base economics clear every threshold, but the candidate-city inputs are illustrative and must be validated in the field before committing the full ₹5 crore.

**Why:**
- Base-case NPV ₹144,427,645, IRR 105% vs 18% hurdle, payback 16 months vs 36 limit.
- Monte Carlo: P(NPV>0) = 100%; P(cash breach) = 0%; worst-5% NPV ₹77,752,628.
- Optimal ₹5.0 cr allocation -> Bengaluru: 378 (₹21,500 held back).

**Conditions / next gates:**
- Pilot 50-vehicle cohorts in the top-ranked candidate cities; hold the remaining capital until contract rates and utilisation are observed.
- Convert to 'Invest' when pilot telemetry supports the assumed billing rate, daily km and tariff within ±10%.

**Decision thresholds used:** hurdle_irr = 0.18; max_payback_months = 36; min_p_npv_positive = 0.8; max_p_cash_breach = 0.05; ops_gap_trigger = ops EBITDA gap > 15% of Y1 EBITDA -> Optimise first; expand_needs_confidence = city data confidence medium/high.

## Scenario summary

| metric | Base | Optimistic | Pessimistic |
|---|---|---|---|
| Revenue (5y, ₹) | 477,360,000.00 | 586,872,000.00 | 318,556,617.60 |
| EBITDA (5y, ₹) | 345,324,000.00 | 451,888,466.67 | 187,132,153.18 |
| EBITDA margin | 0.72 | 0.77 | 0.59 |
| PAT (5y, ₹) | 217,289,536.70 | 298,378,667.11 | 85,669,667.72 |
| NPV @ discount rate (₹) | 144,427,645.38 | 204,925,036.86 | 46,775,319.53 |
| IRR (annual) | 1.05 | 1.49 | 0.42 |
| Payback (months) | 16.00 | 13.00 | 28.00 |
| Battery replaced in horizon | 0.00 | 0.00 | 1.00 |
| Minimum cash (₹) | 11,650,139.62 | 11,147,525.19 | 9,710,273.23 |

*Prepared by the VoltMove decision-support system. Recommendations are produced by deterministic rules over model outputs; they support, and do not replace, management judgement. Financial figures are modelled estimates on documented assumptions, not forecasts of any real company.*