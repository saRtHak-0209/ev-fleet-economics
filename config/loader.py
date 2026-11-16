"""Assumption registry loader.

Reads config/assumptions.yaml into a validated registry where every
assumption carries value / unit / source / as_of / confidence
(spec §Module 1: "Every assumption must have a source, date, unit and
confidence classification"), and flattens it into a plain `Params`
object consumed by the engines.

Overrides are dotted paths ("energy.tariff") mapped to raw values, used
by scenarios, city parameters, sensitivity runs and dashboard inputs.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, field_validator

CONFIG_DIR = Path(__file__).resolve().parent


class Entry(BaseModel):
    """One registered assumption."""

    value: Any
    unit: str = ""
    source: str = "assumption"
    as_of: str = ""
    confidence: Literal["high", "medium", "low", "assumption"] = "assumption"

    @field_validator("as_of", mode="before")
    @classmethod
    def _coerce_date(cls, v: Any) -> str:
        return str(v) if v is not None else ""


class Registry:
    """Nested {group: {name: Entry}} with dotted-path access."""

    def __init__(self, raw: dict):
        self.meta = raw.get("meta", {})
        self.groups: dict[str, dict[str, Entry]] = {}
        for group, items in raw.items():
            if group == "meta":
                continue
            self.groups[group] = {k: Entry(**v) for k, v in items.items()}

    def get(self, path: str) -> Any:
        group, name = path.split(".", 1)
        return self.groups[group][name].value

    def set(self, path: str, value: Any) -> None:
        group, name = path.split(".", 1)
        if group not in self.groups or name not in self.groups[group]:
            raise KeyError(f"Unknown assumption path: {path}")
        entry = self.groups[group][name]
        self.groups[group][name] = entry.model_copy(update={"value": value})

    def with_overrides(self, overrides: dict[str, Any] | None) -> "Registry":
        reg = copy.deepcopy(self)
        for path, value in (overrides or {}).items():
            reg.set(path, value)
        return reg

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for group, items in self.groups.items():
            for name, e in items.items():
                rows.append(
                    {
                        "group": group,
                        "assumption": name,
                        "value": e.value,
                        "unit": e.unit,
                        "source": e.source,
                        "as_of": e.as_of,
                        "confidence": e.confidence,
                    }
                )
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class Params:
    """Flat, typed engine inputs (one source of truth for defaults is the
    YAML register; this class only mirrors it)."""

    # fleet / horizon
    fleet_size: int = 500
    horizon_months: int = 60
    # vehicle
    vehicle_cost: float = 120_000.0
    vehicle_life_months: int = 60
    residual_pct: float = 0.15
    registration_cost: float = 2_000.0
    # operations
    daily_km: float = 100.0
    productive_ratio: float = 0.85
    availability: float = 0.90
    operating_days: float = 26.0
    billing_rate: float = 8.0
    # energy
    kwh_per_km: float = 0.04
    charging_efficiency: float = 0.90
    tariff: float = 12.0
    consumption_soh_coupling: float = 0.0
    # maintenance
    maint_per_km: float = 0.40
    maint_age_escalation_py: float = 0.08
    # battery
    battery_kwh: float = 3.0
    battery_cost_per_kwh: float = 9_000.0
    batt_calendar_fade_py: float = 0.03
    batt_cycle_fade_per_10k_km: float = 0.0
    soh_replace_threshold: float = 0.75
    batt_min_remaining_months: int = 6
    batt_avail_coupling_k: float = 0.5
    batt_avail_coupling_soh: float = 0.80
    # fixed costs
    insurance_py: float = 6_000.0
    telematics_pm: float = 80.0
    vehicles_per_supervisor: float = 50.0
    supervisor_salary_pm: float = 28_000.0
    hub_rent_pm: float = 120_000.0
    admin_overhead_pm: float = 300_000.0
    city_fixed_pm: float = 0.0           # incremental city overhead (expansion)
    # infrastructure
    vehicles_per_charge_point: float = 4.0
    charge_point_capex: float = 25_000.0
    grid_connection_capex: float = 1_000_000.0
    infra_life_months: int = 84
    charger_maint_pm_per_point: float = 200.0
    setup_capex: float = 0.0             # one-time city setup (expansion)
    # financing
    financing_mode: str = "owned"        # "owned" | "financed"
    down_payment_pct: float = 0.30
    interest_rate_py: float = 0.14
    tenure_months: int = 36
    # tax
    tax_rate: float = 0.2517
    loss_carryforward: bool = True
    # working capital
    receivable_days: float = 30.0
    opening_cash_buffer: float = 15_000_000.0
    # valuation
    discount_rate_py: float = 0.12
    hurdle_irr: float = 0.18
    max_payback_months: int = 36
    # petrol comparator
    petrol_cost: float = 85_000.0
    petrol_mileage: float = 48.0
    petrol_price: float = 103.0
    petrol_maint_per_km: float = 0.75
    petrol_maint_age_escalation_py: float = 0.10
    petrol_insurance_py: float = 6_500.0
    petrol_residual_pct: float = 0.25
    petrol_overhaul_cost: float = 8_000.0
    petrol_life_months: int = 60

    def with_(self, **kwargs) -> "Params":
        return replace(self, **kwargs)

    @property
    def monthly_discount_rate(self) -> float:
        return (1.0 + self.discount_rate_py) ** (1.0 / 12.0) - 1.0


# dotted assumption path -> Params field
_PATH_MAP: dict[str, str] = {
    "fleet.initial_fleet": "fleet_size",
    "fleet.horizon_months": "horizon_months",
    "vehicle.purchase_cost": "vehicle_cost",
    "vehicle.useful_life_months": "vehicle_life_months",
    "vehicle.residual_value_pct": "residual_pct",
    "vehicle.registration_cost": "registration_cost",
    "operations.daily_km": "daily_km",
    "operations.productive_ratio": "productive_ratio",
    "operations.availability": "availability",
    "operations.operating_days": "operating_days",
    "operations.billing_rate": "billing_rate",
    "energy.kwh_per_km": "kwh_per_km",
    "energy.charging_efficiency": "charging_efficiency",
    "energy.tariff": "tariff",
    "energy.consumption_soh_coupling": "consumption_soh_coupling",
    "maintenance.cost_per_km": "maint_per_km",
    "maintenance.age_escalation_py": "maint_age_escalation_py",
    "battery.capacity_kwh": "battery_kwh",
    "battery.replacement_cost_per_kwh": "battery_cost_per_kwh",
    "battery.calendar_fade_py": "batt_calendar_fade_py",
    "battery.cycle_fade_per_10k_km": "batt_cycle_fade_per_10k_km",
    "battery.replacement_threshold_soh": "soh_replace_threshold",
    "battery.min_remaining_months_for_replacement": "batt_min_remaining_months",
    "battery.availability_coupling_k": "batt_avail_coupling_k",
    "battery.availability_coupling_soh_threshold": "batt_avail_coupling_soh",
    "fixed_costs.insurance_per_vehicle_py": "insurance_py",
    "fixed_costs.telematics_per_vehicle_pm": "telematics_pm",
    "fixed_costs.staff_vehicles_per_supervisor": "vehicles_per_supervisor",
    "fixed_costs.supervisor_salary_pm": "supervisor_salary_pm",
    "fixed_costs.hub_rent_pm": "hub_rent_pm",
    "fixed_costs.admin_overhead_pm": "admin_overhead_pm",
    "infrastructure.vehicles_per_charge_point": "vehicles_per_charge_point",
    "infrastructure.charge_point_capex": "charge_point_capex",
    "infrastructure.grid_connection_capex": "grid_connection_capex",
    "infrastructure.infra_life_months": "infra_life_months",
    "infrastructure.charger_maintenance_pm_per_point": "charger_maint_pm_per_point",
    "financing.mode": "financing_mode",
    "financing.down_payment_pct": "down_payment_pct",
    "financing.interest_rate_py": "interest_rate_py",
    "financing.tenure_months": "tenure_months",
    "tax.corporate_rate": "tax_rate",
    "tax.loss_carryforward": "loss_carryforward",
    "working_capital.receivable_days": "receivable_days",
    "working_capital.opening_cash_buffer": "opening_cash_buffer",
    "valuation.discount_rate_py": "discount_rate_py",
    "valuation.hurdle_irr": "hurdle_irr",
    "valuation.max_payback_months": "max_payback_months",
    "petrol.purchase_cost": "petrol_cost",
    "petrol.mileage_km_per_litre": "petrol_mileage",
    "petrol.petrol_price_per_litre": "petrol_price",
    "petrol.maintenance_per_km": "petrol_maint_per_km",
    "petrol.maintenance_age_escalation_py": "petrol_maint_age_escalation_py",
    "petrol.insurance_py": "petrol_insurance_py",
    "petrol.residual_value_pct": "petrol_residual_pct",
    "petrol.fuel_system_overhaul_cost": "petrol_overhaul_cost",
    "petrol.useful_life_months": "petrol_life_months",
}

_INT_FIELDS = {f.name for f in fields(Params) if f.type in ("int",)}


def load_registry(path: str | Path | None = None) -> Registry:
    path = Path(path) if path else CONFIG_DIR / "assumptions.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return Registry(yaml.safe_load(fh))


def params_from_registry(reg: Registry) -> Params:
    kwargs: dict[str, Any] = {}
    for path, field_name in _PATH_MAP.items():
        v = reg.get(path)
        if field_name in _INT_FIELDS:
            v = int(v)
        kwargs[field_name] = v
    return Params(**kwargs)


def load_params(overrides: dict[str, Any] | None = None,
                path: str | Path | None = None) -> Params:
    """Load base params, optionally with dotted-path overrides.

    Overrides may use assumption paths (``energy.tariff``) or direct
    Params field names (``tariff``)."""
    reg = load_registry(path)
    p = params_from_registry(reg)
    direct: dict[str, Any] = {}
    for k, v in (overrides or {}).items():
        if k in _PATH_MAP:
            direct[_PATH_MAP[k]] = v
        else:
            direct[k] = v
    for k in list(direct):
        if k in _INT_FIELDS:
            direct[k] = int(direct[k])
    return p.with_(**direct) if direct else p


def load_scenarios(path: str | Path | None = None) -> dict[str, dict]:
    path = Path(path) if path else CONFIG_DIR / "scenarios.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_city_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else CONFIG_DIR / "city_parameters.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
