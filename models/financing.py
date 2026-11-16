"""Debt financing model (spec §Module 3E).

Reducing-balance EMI loan. Interest is a P&L expense; principal
repayment is a financing cash flow and must never touch EBITDA.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def emi(principal: float, annual_rate: float, tenure_months: int) -> float:
    """Standard EMI with monthly rests."""
    if tenure_months <= 0 or principal <= 0:
        return 0.0
    i = annual_rate / 12.0
    if i == 0:
        return principal / tenure_months
    return principal * i / (1.0 - (1.0 + i) ** (-tenure_months))


@dataclass
class LoanSchedule:
    emi: float
    interest: np.ndarray      # (T,)
    principal: np.ndarray     # (T,)
    balance: np.ndarray       # (T,) end-of-month outstanding
    drawdown: float           # amount borrowed at t0


def amortisation(principal: float, annual_rate: float, tenure_months: int,
                 horizon_months: int) -> LoanSchedule:
    """Full amortisation schedule, padded with zeros to the horizon."""
    T = horizon_months
    interest = np.zeros(T)
    principal_paid = np.zeros(T)
    balance = np.zeros(T)
    if principal <= 0 or tenure_months <= 0:
        return LoanSchedule(0.0, interest, principal_paid, balance, 0.0)
    pay = emi(principal, annual_rate, tenure_months)
    i = annual_rate / 12.0
    bal = principal
    for t in range(min(tenure_months, T)):
        intr = bal * i
        prin = min(pay - intr, bal)
        bal -= prin
        interest[t] = intr
        principal_paid[t] = prin
        balance[t] = bal
    # if tenure exceeds horizon, remaining balance stays outstanding
    if tenure_months > T:
        balance[T - 1] = bal
    return LoanSchedule(pay, interest, principal_paid, balance, principal)
