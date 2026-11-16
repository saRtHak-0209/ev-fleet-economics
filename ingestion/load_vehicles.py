"""Load vehicle master data into the database (spec §Module 1A)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from database.connection import get_engine, write_frame
from ingestion.generate import SYNTH_DIR
from ingestion.validation import validate_vehicles


def load_vehicles(path: str | Path | None = None, engine=None) -> pd.DataFrame:
    path = Path(path) if path else SYNTH_DIR / "vehicles.csv"
    vehicles = pd.read_csv(path)
    validate_vehicles(vehicles)
    write_frame(vehicles, "vehicles", engine or get_engine())
    return vehicles
