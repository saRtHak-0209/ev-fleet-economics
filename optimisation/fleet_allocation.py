"""Constrained capital allocation (spec §Module 8).

    max  sum_c [ x_c * npv_per_vehicle_c + y_c * npv_fixed_c ]
    s.t. sum_c [ x_c * capex_per_vehicle_c + y_c * setup_capex_c ] <= budget
         y_c * min_fleet_c <= x_c <= y_c * max_fleet_c        (semi-continuous)
         x_c integer, y_c binary
         optional: max capital share per city, IRR pre-filter

Solved with OR-Tools CP-SAT; falls back to scipy.optimize.milp (HiGHS)
when OR-Tools is unavailable. Money is scaled to ₹'000 integers for the
CP-SAT model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.loader import Params
from optimisation.expansion import CityEconomics, classify_city

try:
    from ortools.sat.python import cp_model
    _HAS_ORTOOLS = True
except Exception:                                    # pragma: no cover
    _HAS_ORTOOLS = False

_K = 1_000  # money scale for integer solver


@dataclass
class AllocationResult:
    solver: str
    status: str
    allocation: dict[str, int]               # vehicles per city
    opened: dict[str, bool]
    total_npv: float
    total_capex: float
    budget: float
    unused_capital: float
    table: pd.DataFrame
    constraints: dict
    classifications: list[dict]


def _build_table(econ: dict[str, CityEconomics], alloc: dict[str, int]) -> pd.DataFrame:
    rows = []
    for name, e in econ.items():
        x = alloc.get(name, 0)
        capex = x * e.capex_per_vehicle + (e.setup_capex if x > 0 else 0.0)
        npv = x * e.npv_per_vehicle + (e.npv_fixed_city if x > 0 else 0.0)
        rows.append(
            {
                "city": name, "vehicles": x, "capex": capex, "npv": npv,
                "npv_per_vehicle": e.npv_per_vehicle,
                "capex_per_vehicle": e.capex_per_vehicle,
                "setup_capex": e.setup_capex,
                "min_fleet": e.min_fleet, "max_fleet": e.max_fleet,
                "irr_ref": e.irr_ref, "data_confidence": e.confidence,
            }
        )
    return pd.DataFrame(rows).sort_values("npv", ascending=False).reset_index(drop=True)


def optimise_allocation(
    econ: dict[str, CityEconomics],
    budget: float,
    p_base: Params,
    max_share_per_city: float = 1.0,
    min_irr_filter: bool = False,
    force_solver: str | None = None,
) -> AllocationResult:
    cities = list(econ.keys())
    if min_irr_filter:
        cities = [c for c in cities
                  if np.isfinite(econ[c].irr_ref) and econ[c].irr_ref >= p_base.hurdle_irr]
    constraints = {
        "budget": budget,
        "max_share_per_city": max_share_per_city,
        "min_irr_filter": p_base.hurdle_irr if min_irr_filter else None,
        "per_city_bounds": {c: (econ[c].min_fleet, econ[c].max_fleet) for c in cities},
    }

    solver_choice = force_solver or ("cpsat" if _HAS_ORTOOLS else "milp")
    if solver_choice == "cpsat" and _HAS_ORTOOLS:
        alloc, status = _solve_cpsat(econ, cities, budget, max_share_per_city)
        solver = "ortools-cpsat"
    else:
        alloc, status = _solve_milp(econ, cities, budget, max_share_per_city)
        solver = "scipy-milp(HiGHS)"

    table = _build_table(econ, alloc)
    total_capex = float(table["capex"].sum())
    total_npv = float(table["npv"].sum())
    return AllocationResult(
        solver=solver, status=status, allocation=alloc,
        opened={c: alloc.get(c, 0) > 0 for c in econ},
        total_npv=total_npv, total_capex=total_capex, budget=budget,
        unused_capital=budget - total_capex, table=table,
        constraints=constraints,
        classifications=[classify_city(econ[c], p_base) for c in econ],
    )


def _solve_cpsat(econ, cities, budget, max_share):
    m = cp_model.CpModel()
    x, y = {}, {}
    for c in cities:
        e = econ[c]
        x[c] = m.NewIntVar(0, e.max_fleet, f"x_{c}")
        y[c] = m.NewBoolVar(f"y_{c}")
        m.Add(x[c] >= e.min_fleet).OnlyEnforceIf(y[c])
        m.Add(x[c] <= e.max_fleet).OnlyEnforceIf(y[c])
        m.Add(x[c] == 0).OnlyEnforceIf(y[c].Not())
        if max_share < 1.0:
            m.Add(
                x[c] * int(round(e.capex_per_vehicle / _K))
                + y[c] * int(round(e.setup_capex / _K))
                <= int(round(budget * max_share / _K))
            )
    m.Add(
        sum(
            x[c] * int(round(econ[c].capex_per_vehicle / _K))
            + y[c] * int(round(econ[c].setup_capex / _K))
            for c in cities
        )
        <= int(round(budget / _K))
    )
    m.Maximize(
        sum(
            x[c] * int(round(econ[c].npv_per_vehicle / _K))
            + y[c] * int(round(econ[c].npv_fixed_city / _K))
            for c in cities
        )
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    status = solver.Solve(m)
    name = {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE"}.get(status, "INFEASIBLE")
    alloc = {c: int(solver.Value(x[c])) for c in cities} if name != "INFEASIBLE" else {c: 0 for c in cities}
    return alloc, name


def _solve_milp(econ, cities, budget, max_share):
    from scipy.optimize import LinearConstraint, milp, Bounds
    nc = len(cities)
    # variables: [x_0..x_{n-1}, y_0..y_{n-1}]
    obj = np.concatenate(
        [[-econ[c].npv_per_vehicle for c in cities],
         [-econ[c].npv_fixed_city for c in cities]]
    )
    A, lb, ub = [], [], []
    row = np.concatenate(
        [[econ[c].capex_per_vehicle for c in cities],
         [econ[c].setup_capex for c in cities]]
    )
    A.append(row); lb.append(-np.inf); ub.append(budget)
    for i, c in enumerate(cities):
        e = econ[c]
        r1 = np.zeros(2 * nc); r1[i] = 1; r1[nc + i] = -e.max_fleet   # x <= max*y
        A.append(r1); lb.append(-np.inf); ub.append(0)
        r2 = np.zeros(2 * nc); r2[i] = 1; r2[nc + i] = -e.min_fleet   # x >= min*y
        A.append(r2); lb.append(0); ub.append(np.inf)
        if max_share < 1.0:
            r3 = np.zeros(2 * nc)
            r3[i] = e.capex_per_vehicle; r3[nc + i] = e.setup_capex
            A.append(r3); lb.append(-np.inf); ub.append(budget * max_share)
    res = milp(
        c=obj,
        constraints=LinearConstraint(np.array(A), np.array(lb), np.array(ub)),
        integrality=np.ones(2 * nc),
        bounds=Bounds(np.zeros(2 * nc),
                      np.concatenate([[econ[c].max_fleet for c in cities], np.ones(nc)])),
    )
    if not res.success:
        return {c: 0 for c in cities}, "INFEASIBLE"
    xs = np.round(res.x[:nc]).astype(int)
    return {c: int(v) for c, v in zip(cities, xs)}, "OPTIMAL"
