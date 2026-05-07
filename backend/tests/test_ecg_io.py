"""Tests for ecg_io CSV / WFDB parsing."""

import numpy as np
import pytest

from ecg_io import parse_ecg_csv, coerce_twelve_lead, MIN_SAMPLES


def test_csv_sample_major():
    rng = np.random.default_rng(0)
    sig = rng.normal(size=(MIN_SAMPLES + 50, 12)).astype(float)
    lines = '\n'.join(','.join(f'{x:.6f}' for x in row) for row in sig)
    arr = parse_ecg_csv(lines.encode('utf-8'))
    assert arr.shape == (12, sig.shape[0])
    np.testing.assert_allclose(arr[0][:5], sig.T[0][:5], rtol=1e-4)


def test_csv_lead_major():
    rng = np.random.default_rng(1)
    sig = rng.normal(size=(12, MIN_SAMPLES + 20)).astype(float)
    lines = '\n'.join(','.join(f'{x:.6f}' for x in row) for row in sig)
    arr = parse_ecg_csv(lines.encode('utf-8'))
    assert arr.shape == (12, MIN_SAMPLES + 20)


def test_coerce_rejects_odd_shape():
    a = np.zeros((10, 10), dtype=np.float32)

    with pytest.raises(ValueError):
        coerce_twelve_lead(a)


def test_csv_skips_non_numeric_header():
    rng = np.random.default_rng(2)
    sig = rng.normal(size=(MIN_SAMPLES + 10, 12))
    hdr = ','.join([f'l{i}' for i in range(12)])
    body = '\n'.join(','.join(f'{x:.6f}' for x in row) for row in sig)
    arr = parse_ecg_csv((hdr + '\n' + body).encode())
    assert arr.shape == (12, MIN_SAMPLES + 10)
