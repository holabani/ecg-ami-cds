"""
Shared pytest fixtures for CardioSense backend tests.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient


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
    """FastAPI TestClient — starts the app without a real server."""
    from main import app
    return TestClient(app)
