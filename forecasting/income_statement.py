"""P&L statement view (spec §Module 5C)."""
from __future__ import annotations

import pandas as pd

from forecasting.engine import ModelResult

PNL_ROWS = [
    ("Revenue", "revenue"),
    ("Variable costs", "variable_costs"),
    ("  Electricity", "energy_cost"),
    ("  Maintenance", "maintenance"),
    ("Contribution", "contribution"),
    ("Fixed operating costs", "fixed_costs"),
    ("EBITDA", "ebitda"),
    ("Depreciation", "depreciation"),
    ("EBIT", "ebit"),
    ("Interest", "interest"),
    ("PBT", "pbt"),
    ("Tax", "tax"),
    ("PAT", "pat"),
]


def income_statement(result: ModelResult, period: str = "annual") -> pd.DataFrame:
    """P&L with line items as rows and periods as columns (INR)."""
    df = result.annual_frame() if period == "annual" else result.monthly_frame()
    key = "year" if period == "annual" else "month"
    out = {}
    for label, col in PNL_ROWS:
        out[label] = df.set_index(key)[col]
    stmt = pd.DataFrame(out).T
    stmt.columns = [f"{'Y' if period == 'annual' else 'M'}{int(c)}" for c in stmt.columns]
    return stmt
