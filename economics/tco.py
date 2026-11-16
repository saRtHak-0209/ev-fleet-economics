"""EV versus petrol total cost of ownership (spec §Module 7).

Both powertrains are evaluated on the SAME workload (total km, operating
days). The comparison is financing-neutral: ownership costs are
discounted at the project discount rate and financing is reported
separately, never double-counted.

The model does NOT assume the EV is cheaper — the result follows from
the assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config.loader import Params
from models.battery import battery_lifecycle
from models.energy_cost import electricity_cost_per_km
from models.financing import amortisation


@dataclass
class VehicleTCO:
    label: str
    acquisition: float
    monthly_operating: np.ndarray        # (T,) energy+maintenance+insurance
    replacement_capex: np.ndarray        # (T,) battery pack / engine overhaul
    residual_value: float                # inflow at horizon end
    horizon_months: int
    discount_rate_py: float
    monthly_interest: np.ndarray = field(default=None)   # informational only
    total_km: float = 0.0
    productive_km: float = 0.0

    @property
    def _disc(self) -> np.ndarray:
        r = (1.0 + self.discount_rate_py) ** (1.0 / 12.0) - 1.0
        t = np.arange(1, self.horizon_months + 1, dtype=float)
        return (1.0 + r) ** (-t)

    @property
    def tco_undiscounted(self) -> float:
        return float(
            self.acquisition
            + self.monthly_operating.sum()
            + self.replacement_capex.sum()
            - self.residual_value
        )

    @property
    def tco_discounted(self) -> float:
        d = self._disc
        return float(
            self.acquisition
            + (self.monthly_operating * d).sum()
            + (self.replacement_capex * d).sum()
            - self.residual_value * d[-1]
        )

    @property
    def cost_per_km(self) -> float:
        return self.tco_undiscounted / self.total_km if self.total_km else float("nan")

    @property
    def cost_per_pkm(self) -> float:
        return self.tco_undiscounted / self.productive_km if self.productive_km else float("nan")

    @property
    def avg_monthly_operating(self) -> float:
        return float(self.monthly_operating.mean())


def _workload(p: Params) -> tuple[float, np.ndarray, np.ndarray]:
    """Common monthly workload: km per month and age (years) per month."""
    T = p.horizon_months
    monthly_km = p.daily_km * p.operating_days * p.availability
    age_years = (np.arange(T, dtype=float) + 0.5) / 12.0
    return monthly_km, np.full(T, monthly_km), age_years


def ev_tco(p: Params) -> VehicleTCO:
    T = p.horizon_months
    monthly_km, km_path, age = _workload(p)
    elec_km = float(electricity_cost_per_km(p.kwh_per_km, p.charging_efficiency, p.tariff))
    energy = km_path * elec_km
    maint = km_path * p.maint_per_km * (1.0 + p.maint_age_escalation_py * age)
    insurance = np.full(T, p.insurance_py / 12.0)
    batt = battery_lifecycle(
        T, p.batt_calendar_fade_py, p.batt_cycle_fade_per_10k_km, monthly_km,
        p.soh_replace_threshold, p.battery_kwh, p.battery_cost_per_kwh,
        p.batt_min_remaining_months, n_sims=1,
    )
    loan = amortisation(
        (p.vehicle_cost) * (1.0 - p.down_payment_pct) if p.financing_mode == "financed" else 0.0,
        p.interest_rate_py, p.tenure_months, T,
    )
    return VehicleTCO(
        label="EV",
        acquisition=p.vehicle_cost + p.registration_cost,
        monthly_operating=energy + maint + insurance,
        replacement_capex=batt.replacement_capex_per_vehicle[0],
        residual_value=p.vehicle_cost * p.residual_pct,
        horizon_months=T,
        discount_rate_py=p.discount_rate_py,
        monthly_interest=loan.interest,
        total_km=float(km_path.sum()),
        productive_km=float(km_path.sum() * p.productive_ratio),
    )


def petrol_tco(p: Params) -> VehicleTCO:
    T = p.petrol_life_months
    monthly_km, km_path, age = _workload(p)
    fuel = km_path * (p.petrol_price / p.petrol_mileage)
    maint = km_path * p.petrol_maint_per_km * (1.0 + p.petrol_maint_age_escalation_py * age)
    insurance = np.full(T, p.petrol_insurance_py / 12.0)
    overhaul = np.zeros(T)
    if T > 36:
        overhaul[35] = p.petrol_overhaul_cost
    loan = amortisation(
        p.petrol_cost * (1.0 - p.down_payment_pct) if p.financing_mode == "financed" else 0.0,
        p.interest_rate_py, p.tenure_months, T,
    )
    return VehicleTCO(
        label="Petrol",
        acquisition=p.petrol_cost + p.registration_cost,
        monthly_operating=fuel + maint + insurance,
        replacement_capex=overhaul,
        residual_value=p.petrol_cost * p.petrol_residual_pct,
        horizon_months=T,
        discount_rate_py=p.discount_rate_py,
        monthly_interest=loan.interest,
        total_km=float(km_path.sum()),
        productive_km=float(km_path.sum() * p.productive_ratio),
    )


def compare_tco(p: Params) -> dict:
    ev = ev_tco(p)
    ice = petrol_tco(p)
    premium = ev.acquisition - ice.acquisition
    monthly_saving = ice.monthly_operating - ev.monthly_operating
    # payback of the EV purchase premium from cumulative operating savings
    net = np.concatenate([[-premium], monthly_saving - (ev.replacement_capex - ice.replacement_capex)])
    cum = np.cumsum(net)
    recovered = np.nonzero(cum >= 0)[0]
    premium_payback = float(recovered[0]) if recovered.size else float("nan")
    return {
        "ev": ev,
        "petrol": ice,
        "tco_5y_ev": ev.tco_undiscounted,
        "tco_5y_petrol": ice.tco_undiscounted,
        "tco_5y_ev_discounted": ev.tco_discounted,
        "tco_5y_petrol_discounted": ice.tco_discounted,
        "cost_per_km_ev": ev.cost_per_km,
        "cost_per_km_petrol": ice.cost_per_km,
        "cost_per_pkm_ev": ev.cost_per_pkm,
        "cost_per_pkm_petrol": ice.cost_per_pkm,
        "monthly_operating_ev": ev.avg_monthly_operating,
        "monthly_operating_petrol": ice.avg_monthly_operating,
        "ev_saving_vs_petrol_5y": ice.tco_undiscounted - ev.tco_undiscounted,
        "ev_saving_vs_petrol_5y_discounted": ice.tco_discounted - ev.tco_discounted,
        "ev_purchase_premium": premium,
        "premium_payback_months": premium_payback,
        "ev_is_cheaper": ev.tco_discounted < ice.tco_discounted,
    }
