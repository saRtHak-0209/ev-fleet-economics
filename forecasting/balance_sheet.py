"""Projected balance sheet view (spec §Module 5E). The three statements
reconcile: Assets = Liabilities + Equity is asserted by tests for every
month of every scenario."""
from __future__ import annotations

import pandas as pd

from forecasting.engine import ModelResult


def balance_sheet(result: ModelResult, period: str = "annual") -> pd.DataFrame:
    df = result.annual_frame() if period == "annual" else result.monthly_frame()
    key = "year" if period == "annual" else "month"
    idx = df[key].astype(int)
    rows = {
        "Net vehicle assets": df["nbv_vehicles"],
        "Charging infrastructure (net)": df["nbv_infra"],
        "Battery assets (net)": df["nbv_battery"],
        "Trade receivables": df["debtors"],
        "Cash": df["cash"],
        "TOTAL ASSETS": df["total_assets"],
        "Loan outstanding": df["loan_balance"],
        "Shareholders' equity": df["equity"],
        "TOTAL LIABILITIES + EQUITY": df["liabilities_equity"],
        "Check (A - L - E)": df["total_assets"] - df["liabilities_equity"],
    }
    stmt = pd.DataFrame(rows).set_index(idx).T
    stmt.columns = [f"{'Y' if period == 'annual' else 'M'}{c}" for c in stmt.columns]
    return stmt
