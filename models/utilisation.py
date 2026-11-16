"""Vehicle utilisation engine (spec §Module 2).

Pure functions + a data-driven analyser. All numeric functions accept
scalars or numpy arrays and broadcast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def availability(available_hours, scheduled_hours):
    """Availability = available hours / scheduled hours."""
    scheduled = np.asarray(scheduled_hours, dtype=float)
    return np.divide(available_hours, scheduled,
                     out=np.zeros_like(scheduled), where=scheduled > 0)


def productive_ratio(productive_km, total_km):
    """Productive ratio = productive km / total km."""
    total = np.asarray(total_km, dtype=float)
    return np.divide(productive_km, total,
                     out=np.zeros_like(total), where=total > 0)


def daily_productive_km(total_km, prod_ratio):
    """PKM = total km x productive ratio."""
    return np.asarray(total_km, dtype=float) * np.asarray(prod_ratio, dtype=float)


def fleet_utilisation(vehicles_deployed, total_fleet):
    """Fleet utilisation = vehicles deployed / total fleet."""
    fleet = np.asarray(total_fleet, dtype=float)
    return np.divide(vehicles_deployed, fleet,
                     out=np.zeros_like(fleet), where=fleet > 0)


# ---------------------------------------------------------------------
# Data-driven utilisation analytics over the vehicle-day table
# ---------------------------------------------------------------------
SCHEDULED_HOURS_PER_DAY = 10.0  # scheduled commercial operating window


def vehicle_utilisation_summary(ops: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the daily operational table to one row per vehicle."""
    g = ops.groupby("vehicle_id", observed=True)
    out = pd.DataFrame(
        {
            "city": g["city"].first(),
            "days_observed": g["date"].nunique(),
            "total_km": g["total_km"].sum(),
            "productive_km": g["productive_km"].sum(),
            "avg_daily_km": g["total_km"].mean(),
            "avg_daily_pkm": g["productive_km"].mean(),
            "downtime_hours": g["downtime_hours"].sum(),
            "avg_downtime_hours": g["downtime_hours"].mean(),
            "revenue": g["revenue"].sum(),
            "energy_kwh": g["energy_consumed_kwh"].sum(),
            "charging_cost": g["charging_cost"].sum(),
            "maintenance_cost": g["maintenance_cost"].sum(),
            "battery_soh": g["battery_soh"].last(),
            "vehicle_age_months": g["vehicle_age_months"].last(),
        }
    )
    out["productive_ratio"] = productive_ratio(out["productive_km"], out["total_km"])
    out["availability"] = 1.0 - out["avg_downtime_hours"] / SCHEDULED_HOURS_PER_DAY
    return out.reset_index()


def flag_underperformers(
    summary: pd.DataFrame,
    low_util_km: float = 60.0,
    low_prod_ratio: float = 0.78,
    high_downtime_hours: float = 1.5,
) -> pd.DataFrame:
    """Identify vehicles with low utilisation / low productive ratio /
    excessive downtime (spec §Module 2 identification tasks)."""
    s = summary.copy()
    s["flag_low_utilisation"] = s["avg_daily_km"] < low_util_km
    s["flag_low_productive_ratio"] = s["productive_ratio"] < low_prod_ratio
    s["flag_high_downtime"] = s["avg_downtime_hours"] > high_downtime_hours
    s["n_flags"] = (
        s[["flag_low_utilisation", "flag_low_productive_ratio", "flag_high_downtime"]]
        .sum(axis=1)
        .astype(int)
    )
    return s


def city_utilisation(ops: pd.DataFrame) -> pd.DataFrame:
    """City-level utilisation differences."""
    g = ops.groupby("city", observed=True)
    out = pd.DataFrame(
        {
            "vehicles": g["vehicle_id"].nunique(),
            "avg_daily_km": g["total_km"].mean(),
            "avg_daily_pkm": g["productive_km"].mean(),
            "productive_ratio": g["productive_km"].sum() / g["total_km"].sum(),
            "avg_downtime_hours": g["downtime_hours"].mean(),
            "revenue_per_vehicle_day": g["revenue"].mean(),
        }
    )
    out["availability"] = 1.0 - out["avg_downtime_hours"] / SCHEDULED_HOURS_PER_DAY
    return out.reset_index()


def monthly_utilisation_trend(ops: pd.DataFrame) -> pd.DataFrame:
    """Fleet productivity trend by calendar month."""
    df = ops.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    g = df.groupby("month")
    out = pd.DataFrame(
        {
            "avg_daily_km": g["total_km"].mean(),
            "avg_daily_pkm": g["productive_km"].mean(),
            "productive_ratio": g["productive_km"].sum() / g["total_km"].sum(),
            "avg_downtime_hours": g["downtime_hours"].mean(),
            "active_vehicles": g["vehicle_id"].nunique(),
        }
    )
    out["availability"] = 1.0 - out["avg_downtime_hours"] / SCHEDULED_HOURS_PER_DAY
    return out.reset_index()
