"""Database connection (spec §Module 1 storage).

Default: zero-setup SQLite file at data/voltmove.db.
PostgreSQL: set DATABASE_URL, e.g.
    export DATABASE_URL=postgresql+psycopg2://user:pass@localhost/voltmove
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SQLITE = f"sqlite:///{ROOT / 'data' / 'voltmove.db'}"


def get_engine(url: str | None = None):
    return create_engine(url or os.environ.get("DATABASE_URL", DEFAULT_SQLITE))


def init_schema(engine=None) -> None:
    engine = engine or get_engine()
    raw = (ROOT / "database" / "schema.sql").read_text()
    # strip SQL comments before splitting into statements
    lines = []
    for line in raw.splitlines():
        if line.strip().startswith("--"):
            continue
        lines.append(line.split("--")[0])
    for stmt in "\n".join(lines).split(";"):
        if stmt.strip():
            with engine.begin() as conn:
                conn.execute(text(stmt))


def write_frame(df: pd.DataFrame, table: str, engine=None, replace: bool = True) -> None:
    engine = engine or get_engine()
    df.to_sql(table, engine, if_exists="replace" if replace else "append",
              index=False, chunksize=20_000)


def read_table(table: str, engine=None) -> pd.DataFrame:
    engine = engine or get_engine()
    return pd.read_sql_table(table, engine)


def query(sql: str, engine=None) -> pd.DataFrame:
    engine = engine or get_engine()
    return pd.read_sql_query(text(sql), engine)
