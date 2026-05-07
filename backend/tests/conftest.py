"""
Shared pytest fixtures for CardioSense backend tests.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


def pytest_configure(config):
    """Use a fresh SQLite file so tests never touch dev cardiosense.db."""
    p = Path(__file__).resolve().parent / "_cardiosense_test.sqlite"
    if p.exists():
        try:
            p.unlink()
        except OSError:
            pass
    os.environ["CARDIOSENSE_DATABASE_URL"] = f"sqlite:///{p}"


# ── ECG signal fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def sample_ecg_array():
    """12-lead ECG as (12, 1000) float32 numpy array with realistic amplitudes."""
    rng = np.random.default_rng(42)
    ecg = rng.normal(0, 0.1, (12, 1000)).astype(np.float32)
    # Inject a synthetic QRS-like bump in each lead
    t = np.linspace(0, 2 * np.pi, 1000)
    for lead in range(12):
        ecg[lead] += 0.5 * np.sin(t * 5) * np.exp(-((t - np.pi) ** 2) / 0.5)
    return ecg


@pytest.fixture
def flat_ecg_list(sample_ecg_array):
    """ECG as a flat Python list (12000 elements) — matches API input format."""
    return sample_ecg_array.flatten().tolist()


@pytest.fixture
def api_client():
    """FastAPI TestClient — runs app lifespan (SQLite init + demo user)."""
    from main import app
    with TestClient(app) as client:
        yield client


@pytest.fixture
def auth_headers(api_client):
    """JWT for demo user (seeded on app startup)."""
    r = api_client.post(
        "/auth/login",
        json={"email": "demo@example.com", "password": "demo123"},
    )
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
