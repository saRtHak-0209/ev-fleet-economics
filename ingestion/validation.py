"""Data validation (spec §Module 1 / §10): the model must never ingest
impossible operational records."""
from __future__ import annotations

import pandas as pd

REQUIRED_OPS_COLUMNS = [
    "date", "vehicle_id", "city", "vehicle_age_months", "total_km",
    "productive_km", "availability", "downtime_hours", "energy_consumed_kwh",
    "charging_cost", "maintenance_cost", "battery_soh", "revenue",
]
REQUIRED_VEHICLE_COLUMNS = [
    "vehicle_id", "city", "cohort", "purchase_cost", "battery_capacity",
    "rated_range", "actual_range", "purchase_date", "useful_life",
    "residual_value",
]


class ValidationError(ValueError):
    pass


def _check(cond: pd.Series | bool, message: str, errors: list[str]) -> None:
    if isinstance(cond, bool):
        ok = cond
        n_bad = 0 if ok else 1
    else:
        n_bad = int((~cond).sum())
        ok = n_bad == 0
    if not ok:
        errors.append(f"{message} ({n_bad:,} rows)")


def validate_vehicles(vehicles: pd.DataFrame) -> dict:
    errors: list[str] = []
    missing = [c for c in REQUIRED_VEHICLE_COLUMNS if c not in vehicles.columns]
    if missing:
        raise ValidationError(f"vehicle master missing columns: {missing}")
    _check(~vehicles["vehicle_id"].duplicated(), "duplicate vehicle_id", errors)
    _check(vehicles["purchase_cost"] > 0, "non-positive purchase cost", errors)
    _check(vehicles["battery_capacity"] > 0, "non-positive battery capacity", errors)
    _check(vehicles["residual_value"] >= 0, "negative residual value", errors)
    _check(vehicles["actual_range"] <= vehicles["rated_range"] * 1.1,
           "actual range implausibly above rated range", errors)
    if errors:
        raise ValidationError("; ".join(errors))
    return {"vehicles": len(vehicles), "status": "ok"}


def validate_operations(ops: pd.DataFrame, vehicles: pd.DataFrame | None = None) -> dict:
    errors: list[str] = []
    missing = [c for c in REQUIRED_OPS_COLUMNS if c not in ops.columns]
    if missing:
        raise ValidationError(f"operations table missing columns: {missing}")
    _check(ops["total_km"] >= 0, "negative total_km", errors)
    _check(ops["productive_km"] >= 0, "negative productive_km", errors)
    _check(ops["productive_km"] <= ops["total_km"] + 1e-6,
           "productive_km exceeds total_km", errors)
    _check(ops["availability"].between(0, 1), "availability outside [0,1]", errors)
    _check(ops["battery_soh"].between(0, 1), "battery SOH outside [0,1]", errors)
    _check(ops["downtime_hours"] >= 0, "negative downtime", errors)
    for col in ("energy_consumed_kwh", "charging_cost", "maintenance_cost", "revenue"):
        _check(ops[col] >= 0, f"negative {col}", errors)
    _check(~ops.duplicated(["date", "vehicle_id"]), "duplicate vehicle-day rows", errors)
    # SOH must never increase for a vehicle (no replacements in year 1 data)
    soh_increase = (
        ops.sort_values(["vehicle_id", "date"])
        .groupby("vehicle_id", observed=True)["battery_soh"]
        .diff()
        .fillna(0)
    )
    _check(soh_increase <= 1e-6, "battery SOH increased without replacement", errors)
    if vehicles is not None:
        unknown = ~ops["vehicle_id"].isin(vehicles["vehicle_id"])
        _check(~unknown, "operations reference unknown vehicle_id", errors)
    if errors:
        raise ValidationError("; ".join(errors))
    return {"rows": len(ops), "vehicles": ops["vehicle_id"].nunique(), "status": "ok"}
