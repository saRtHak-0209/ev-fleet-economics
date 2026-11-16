import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.loader import load_params  # noqa: E402


@pytest.fixture(scope="session")
def p():
    """Base-case parameters from the assumptions register."""
    return load_params()
