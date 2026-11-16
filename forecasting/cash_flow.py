"""Cash flow statement view (spec §Module 5D)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.engine import ModelResult


def cash_flow_statement(result: ModelResult, period: str = "annual") -> pd.DataFrame:
    df = result.annual_frame() if period == "annual" else result.monthly_frame()
    key = "year" if period == "annual" else "month"
    idx = df[key].astype(int)
    rows = {
        "Operating cash flow (PAT + dep - dWC)": df["cfo"],
        "  PAT": df["pat"],
        "  Depreciation add-back": df["depreciation"],
        "  Change in working capital": -df["d_working_capital"],
        "Investing cash flow": df["cfi"],
        "  Battery replacement capex": -df["battery_capex"],
        "  Vehicle & infra salvage": df["salvage"],
        "Financing cash flow": df["cff"],
        "  Loan principal repayment": -df["principal_repaid"],
        "Net cash flow": df["cfo"] + df["cfi"] + df["cff"],
        "Closing cash": df["cash"],
        "FCFF": df["fcff"],
        "FCFE": df["fcfe"],
    }
    stmt = pd.DataFrame(rows).set_index(idx).T
    stmt.columns = [f"{'Y' if period == 'annual' else 'M'}{c}" for c in stmt.columns]
    return stmt


def t0_flows(result: ModelResult) -> pd.DataFrame:
    """Opening (month 0) investment and funding flows."""
    g = lambda k: float(np.asarray(result.t0[k]).ravel()[0])
    return pd.DataFrame(
        {
            "INR": {
                "Vehicle + infrastructure capex": -g("capex0"),
                "Loan drawdown": g("loan0"),
                "Equity contribution": g("equity0"),
                "Opening cash": g("opening_cash"),
            }
        }
    )
