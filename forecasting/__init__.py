"""Five-year integrated financial model (spec §Module 5)."""
from forecasting.engine import ModelResult, run_model
from forecasting.income_statement import income_statement
from forecasting.cash_flow import cash_flow_statement, t0_flows
from forecasting.balance_sheet import balance_sheet
from forecasting.revenue_forecast import revenue_forecast
from forecasting.opex_forecast import opex_forecast

__all__ = [
    "ModelResult", "run_model", "income_statement", "cash_flow_statement",
    "t0_flows", "balance_sheet", "revenue_forecast", "opex_forecast",
]
