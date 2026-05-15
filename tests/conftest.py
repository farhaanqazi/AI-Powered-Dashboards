"""Shared pytest fixtures for the AI-Powered Dashboards test suite."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main as main_module

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient that auto-attaches guest-mode auth headers."""
    c = TestClient(main_module.app)
    c.headers.update({
        "X-Guest-Mode": "1",
        "X-Guest-Session-Id": "pytest-session",
    })
    yield c


@pytest.fixture(autouse=True)
def _reset_storage():
    """Clear the in-process dashboard store between tests."""
    main_module.dashboard_storage.clear()
    yield
    main_module.dashboard_storage.clear()


@pytest.fixture
def sample_csv_bytes() -> bytes:
    return (FIXTURES / "sample_data.csv").read_bytes()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "sample_data.csv")


@pytest.fixture
def upload_files(sample_csv_bytes):
    return {"dataset": ("sample_data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
