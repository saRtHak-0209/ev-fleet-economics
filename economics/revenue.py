"""Revenue model (spec §Module 4A). Per-productive-km billing."""
from __future__ import annotations

import numpy as np


def daily_revenue(daily_total_km, productive_ratio, billing_rate):
    return (
        np.asarray(daily_total_km, dtype=float)
        * np.asarray(productive_ratio, dtype=float)
        * np.asarray(billing_rate, dtype=float)
    )


def monthly_revenue_per_vehicle(daily_total_km, productive_ratio, operating_days, billing_rate):
    """Base case: 100 x 0.85 x 26 x 8 = 17,680 INR/month (deployed vehicle)."""
    return daily_revenue(daily_total_km, productive_ratio, billing_rate) * np.asarray(
        operating_days, dtype=float
    )


def fleet_monthly_revenue(fleet_size, deployed_ratio, daily_total_km,
                          productive_ratio, operating_days, billing_rate):
    """FleetRevenue = FleetSize x DeployedRatio x PKM_daily x Days x Rate
    (spec §Module 5A)."""
    return (
        np.asarray(fleet_size, dtype=float)
        * np.asarray(deployed_ratio, dtype=float)
        * monthly_revenue_per_vehicle(daily_total_km, productive_ratio,
                                      operating_days, billing_rate)
    )
