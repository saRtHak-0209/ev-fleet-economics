"""Synthetic operational dataset generator (spec §10).

The data is SIMULATED and labelled as such. Fields are generated through
relationships, never as independent random draws (spec §10B):

  demand/seasonality -> daily km -> energy -> charging cost
  vehicle age        -> breakdown probability -> downtime -> available km
  age + cumulative km -> battery SOH (monotone) -> range cap on daily km
  productive km = fraction of total km;  revenue = productive km x rate
  maintenance = per-km routine (age-escalated) + repair events

Default fleet: 500 main vehicles in Bengaluru + 25-vehicle pilot cohorts
in each candidate city (so city-level utilisation differences are
observable), one year of operating days (Sundays off ~ 26 days/month).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config.loader import load_city_config, load_params

ROOT = Path(__file__).resolve().parents[1]
SYNTH_DIR = ROOT / "data" / "synthetic"

SIM_START = "2025-11-01"
SIM_DAYS = 365
PILOT_FLEET = 25
SCHEDULED_HOURS = 10.0


def _city_param(city_cfg: dict, key: str, default: float) -> float:
    return float((city_cfg.get("overrides") or {}).get(key, default))


def generate(seed: int = 42, sim_days: int = SIM_DAYS,
             main_fleet: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (vehicles, daily_operations)."""
    rng = np.random.default_rng(seed)
    p = load_params()
    cities_cfg = load_city_config()["cities"]
    main_fleet = int(main_fleet or p.fleet_size)

    # ----- vehicle master -----
    rows = []
    counter = 1
    for city, cfg in cities_cfg.items():
        n = main_fleet if cfg.get("status") == "existing" else PILOT_FLEET
        cohort = "main" if cfg.get("status") == "existing" else "pilot"
        for _ in range(n):
            age_months_at_start = rng.uniform(1.0, 18.0)
            purchase = pd.Timestamp(SIM_START) - pd.to_timedelta(
                int(age_months_at_start * 30.44), unit="D")
            quality = rng.normal(1.0, 0.04)
            rows.append(
                {
                    "vehicle_id": f"EV{counter:04d}",
                    "city": city,
                    "cohort": cohort,
                    "purchase_cost": float(np.round(rng.normal(p.vehicle_cost, 3000), -2)),
                    "battery_capacity": p.battery_kwh,
                    "rated_range": 85.0,
                    "actual_range": float(np.round(75.0 * quality, 1)),
                    "purchase_date": purchase.date().isoformat(),
                    "useful_life": p.vehicle_life_months,
                    "residual_value": p.vehicle_cost * p.residual_pct,
                    "_quality": quality,
                    "_age_months_start": age_months_at_start,
                }
            )
            counter += 1
    vehicles = pd.DataFrame(rows)

    # ----- per-city operating parameters -----
    city_names = vehicles["city"].to_numpy()
    base_km_city = np.array([_city_param(cities_cfg[c], "operations.daily_km", p.daily_km)
                             for c in city_names])
    prod_city = np.array([_city_param(cities_cfg[c], "operations.productive_ratio", p.productive_ratio)
                          for c in city_names])
    rate_city = np.array([_city_param(cities_cfg[c], "operations.billing_rate", p.billing_rate)
                          for c in city_names])
    tariff_city = np.array([_city_param(cities_cfg[c], "energy.tariff", p.tariff)
                            for c in city_names])
    maint_city = np.array([_city_param(cities_cfg[c], "maintenance.cost_per_km", p.maint_per_km)
                           for c in city_names])

    V = len(vehicles)
    D = sim_days
    dates = pd.date_range(SIM_START, periods=D, freq="D")
    quality = vehicles["_quality"].to_numpy()[:, None]                     # (V,1)
    age_start_m = vehicles["_age_months_start"].to_numpy()[:, None]
    age_months = age_start_m + np.arange(D)[None, :] / 30.44               # (V,D)
    age_years = age_months / 12.0

    # vehicle-level persistent heterogeneity
    km_vehicle_factor = rng.normal(1.0, 0.08, size=(V, 1))

    # seasonality: festive uplift Oct-Nov, monsoon drag Jun-Sep
    month_of = dates.month.to_numpy()[None, :]
    season = np.where(np.isin(month_of, [10, 11]), 1.08,
                      np.where(np.isin(month_of, [6, 7, 8, 9]), 0.93, 1.0))
    demand_noise = rng.normal(1.0, 0.07, size=(V, D))

    # downtime: breakdown probability rises with age
    p_breakdown = np.clip(0.015 + 0.012 * age_years, 0.0, 0.25)
    breakdown = rng.random((V, D)) < p_breakdown
    downtime = np.where(breakdown, rng.uniform(2.0, 10.0, size=(V, D)),
                        rng.uniform(0.0, 0.5, size=(V, D)))
    availability_day = np.clip(1.0 - downtime / SCHEDULED_HOURS, 0.0, 1.0)

    # battery SOH: calendar + cycle fade, with starting age reflected
    # (computed from planned km so the relationship stays acyclic)
    planned_km = base_km_city[:, None] * quality * km_vehicle_factor * season
    cal_fade_daily = p.batt_calendar_fade_py / 365.0
    cycle_fade_per_km = 0.01 / 10_000.0
    fade_noise = rng.normal(1.0, 0.1, size=(V, 1))
    start_fade = (age_start_m / 12.0) * p.batt_calendar_fade_py * fade_noise
    cum_fade = (
        start_fade
        + cal_fade_daily * fade_noise * np.arange(1, D + 1)[None, :]
        + cycle_fade_per_km * np.cumsum(planned_km, axis=1)
    )
    soh = np.clip(1.0 - cum_fade, 0.55, 1.0)

    # realised distance: demand x availability, capped by charge-cycle range
    range_cap = 2.0 * p.battery_kwh * soh / p.kwh_per_km / 1.3   # ~2 charges/day, derated
    total_km = np.clip(planned_km * demand_noise * availability_day, 0.0, range_cap)

    # productive share
    prod_ratio_day = rng.beta(prod_city[:, None] * 50, (1 - prod_city[:, None]) * 50,
                              size=(V, D))
    productive_km = total_km * prod_ratio_day

    # energy & charging (consumption worsens slightly as the pack ages)
    kwh = total_km * p.kwh_per_km * (1.0 + 0.05 * (1.0 - soh))
    charging_cost = kwh / p.charging_efficiency * tariff_city[:, None] \
        * rng.normal(1.0, 0.02, size=(V, D)).clip(0.9, 1.1)

    # maintenance: routine per-km (age-escalated) + repair events
    routine = total_km * maint_city[:, None] * (1.0 + p.maint_age_escalation_py * age_years)
    repair = np.where(breakdown, rng.lognormal(mean=6.2, sigma=0.5, size=(V, D)), 0.0)
    maintenance = routine + repair

    revenue = productive_km * rate_city[:, None]

    # operating-day mask: Sundays off (~26 operating days/month)
    operating = (dates.dayofweek.to_numpy() != 6)[None, :] & np.ones((V, 1), dtype=bool)

    vid = np.repeat(vehicles["vehicle_id"].to_numpy(), D)
    city_rep = np.repeat(city_names, D)
    date_rep = np.tile(dates.values, V)
    mask = operating.ravel()

    ops = pd.DataFrame(
        {
            "date": pd.to_datetime(date_rep[mask]).date,
            "vehicle_id": vid[mask],
            "city": city_rep[mask],
            "vehicle_age_months": np.round(age_months.ravel()[mask], 2),
            "total_km": np.round(total_km.ravel()[mask], 2),
            "productive_km": np.round(np.minimum(productive_km, total_km).ravel()[mask], 2),
            "availability": np.round(availability_day.ravel()[mask], 3),
            "downtime_hours": np.round(downtime.ravel()[mask], 2),
            "energy_consumed_kwh": np.round(kwh.ravel()[mask], 3),
            "charging_cost": np.round(charging_cost.ravel()[mask], 2),
            "maintenance_cost": np.round(maintenance.ravel()[mask], 2),
            "battery_soh": np.round(soh.ravel()[mask], 3),
            "revenue": np.round(revenue.ravel()[mask], 2),
        }
    )
    vehicles = vehicles.drop(columns=["_quality", "_age_months_start"])
    return vehicles, ops


def write_outputs(vehicles: pd.DataFrame, ops: pd.DataFrame,
                  out_dir: Path | None = None) -> dict[str, Path]:
    out = out_dir or SYNTH_DIR
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "vehicles_csv": out / "vehicles.csv",
        "ops_parquet": out / "daily_operations.parquet",
        "ops_sample_csv": out / "daily_operations_sample.csv",
    }
    vehicles.to_csv(paths["vehicles_csv"], index=False)
    ops.to_parquet(paths["ops_parquet"], index=False)
    ops.head(2000).to_csv(paths["ops_sample_csv"], index=False)
    return paths


if __name__ == "__main__":
    v, o = generate()
    paths = write_outputs(v, o)
    print(f"vehicles: {len(v):,}  vehicle-day rows: {len(o):,}")
    for k, pth in paths.items():
        print(f"  {k}: {pth}")
