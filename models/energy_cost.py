"""Electricity / charging cost model (spec §Module 3A).

Grid energy = (km x kWh/km) / charging efficiency
Cost        = grid energy x tariff
"""
from __future__ import annotations

import numpy as np


def energy_per_km(kwh_consumed, total_km):
    """Observed energy intensity kWh/km."""
    km = np.asarray(total_km, dtype=float)
    return np.divide(kwh_consumed, km, out=np.zeros_like(km), where=km > 0)


def grid_energy_kwh(total_km, kwh_per_km, charging_efficiency):
    """Electricity drawn from the grid for a given distance."""
    return np.asarray(total_km, dtype=float) * np.asarray(kwh_per_km, dtype=float) / np.asarray(
        charging_efficiency, dtype=float
    )


def electricity_cost(total_km, kwh_per_km, charging_efficiency, tariff):
    """Charging cost for a given distance."""
    return grid_energy_kwh(total_km, kwh_per_km, charging_efficiency) * np.asarray(
        tariff, dtype=float
    )


def electricity_cost_per_km(kwh_per_km, charging_efficiency, tariff):
    """INR per total km. Base case: 0.04/0.90*12 = 0.5333."""
    return electricity_cost(1.0, kwh_per_km, charging_efficiency, tariff)


def effective_kwh_per_km(kwh_per_km, soh, coupling: float = 0.0):
    """Optionally degrade energy efficiency as the battery ages:
    consumption x (1 + coupling x (1 - SOH)). coupling = 0 disables."""
    return np.asarray(kwh_per_km, dtype=float) * (
        1.0 + float(coupling) * (1.0 - np.asarray(soh, dtype=float))
    )
