"""Snapshot the assumption register and city market reference into the
database so every model run is reproducible (spec §Module 1C/1D)."""
from __future__ import annotations

from datetime import date

import pandas as pd

from config.loader import load_city_config, load_registry
from database.connection import get_engine, write_frame


def load_financials(engine=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    engine = engine or get_engine()
    reg = load_registry().to_frame().rename(columns={"group": "grp"})
    reg.insert(0, "snapshot_date", date.today().isoformat())
    reg["value"] = reg["value"].astype(str)
    write_frame(reg, "assumptions_register", engine)

    cities = load_city_config()["cities"]
    rows = []
    for city, cfg in cities.items():
        for path, value in (cfg.get("overrides") or {}).items():
            rows.append({"city": city, "parameter": path, "value": str(value),
                         "source": "config/city_parameters.yaml — illustrative placeholder",
                         "confidence": cfg.get("confidence", "low")})
        for k, v in (cfg.get("expansion") or {}).items():
            rows.append({"city": city, "parameter": f"expansion.{k}", "value": str(v),
                         "source": "config/city_parameters.yaml — illustrative placeholder",
                         "confidence": cfg.get("confidence", "low")})
    market = pd.DataFrame(rows)
    write_frame(market, "market_reference", engine)
    return reg, market
