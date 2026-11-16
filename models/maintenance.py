"""Maintenance cost model (spec §Module 3B).

Variable cost per km with an age-dependent escalation:
    cost = km x base_rate x (1 + escalation_py x age_years)
"""
from __future__ import annotations

import numpy as np


def age_multiplier(age_years, escalation_py: float):
    return 1.0 + np.asarray(escalation_py, dtype=float) * np.asarray(age_years, dtype=float)


def maintenance_cost(total_km, rate_per_km, age_years=0.0, escalation_py: float = 0.0):
    return (
        np.asarray(total_km, dtype=float)
        * np.asarray(rate_per_km, dtype=float)
        * age_multiplier(age_years, escalation_py)
    )
