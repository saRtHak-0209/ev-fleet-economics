"""Load daily operational data into the database (spec §Module 1B)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from database.connection import get_engine, write_frame
from ingestion.generate import SYNTH_DIR
from ingestion.validation import validate_operations


def load_operations(path: str | Path | None = None,
                    vehicles: pd.DataFrame | None = None,
                    engine=None) -> pd.DataFrame:
    path = Path(path) if path else SYNTH_DIR / "daily_operations.parquet"
    ops = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    validate_operations(ops, vehicles)
    ops["date"] = pd.to_datetime(ops["date"]).dt.date.astype(str)
    write_frame(ops, "daily_operations", engine or get_engine())
    return ops
