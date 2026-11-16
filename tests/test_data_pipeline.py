"""Synthetic data generator & validation tests (spec §10/§Module 1)."""
import numpy as np
import pandas as pd
import pytest

from ingestion.generate import generate
from ingestion.validation import ValidationError, validate_operations, validate_vehicles


@pytest.fixture(scope="module")
def dataset():
    return generate(seed=7, sim_days=120, main_fleet=60)


def test_generator_passes_validation(dataset):
    vehicles, ops = dataset
    assert validate_vehicles(vehicles)["status"] == "ok"
    assert validate_operations(ops, vehicles)["status"] == "ok"


def test_relationships_not_independent(dataset):
    """Spec §10B: fields must be generated through relationships."""
    _, ops = dataset
    # revenue is exactly productive_km x rate (per-city constant)
    per_km = ops["revenue"] / ops["productive_km"].replace(0, np.nan)
    assert per_km.groupby(ops["city"]).std().max() < 0.02
    # charging cost scales with energy consumed (within a city: one tariff)
    corr = (
        ops.groupby("city")[["energy_consumed_kwh", "charging_cost"]]
        .corr().unstack().iloc[:, 1]
    )
    assert corr.min() > 0.98
    # downtime days produce fewer km
    high_dt = ops[ops["downtime_hours"] > 4]["total_km"].mean()
    low_dt = ops[ops["downtime_hours"] < 1]["total_km"].mean()
    assert high_dt < low_dt
    # maintenance rises with vehicle age (controls: same city)
    blr = ops[ops["city"] == "Bengaluru"]
    old = blr[blr["vehicle_age_months"] > 15]
    young = blr[blr["vehicle_age_months"] < 8]
    assert (old["maintenance_cost"] / old["total_km"]).median() > \
           (young["maintenance_cost"] / young["total_km"]).median()


def test_soh_declines_with_age(dataset):
    _, ops = dataset
    s = ops.sort_values(["vehicle_id", "date"]).groupby("vehicle_id")["battery_soh"]
    assert (s.last() <= s.first()).all()


def test_validation_catches_impossible_rows(dataset):
    vehicles, ops = dataset
    bad = ops.copy()
    bad.loc[bad.index[0], "productive_km"] = bad.loc[bad.index[0], "total_km"] + 10
    with pytest.raises(ValidationError, match="productive_km exceeds"):
        validate_operations(bad, vehicles)
    bad2 = ops.copy()
    bad2.loc[bad2.index[0], "battery_soh"] = 1.4
    with pytest.raises(ValidationError, match="SOH"):
        validate_operations(bad2, vehicles)
