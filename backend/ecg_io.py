"""
Parse uploaded 12-lead ECG data from CSV/TSV or WFDB (.hea + .dat) pairs.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import wfdb as _wfdb
except ImportError:
    _wfdb = None

MIN_SAMPLES = 200


def coerce_twelve_lead(arr: np.ndarray) -> np.ndarray:
    """Normalize to shape (12, T) with T >= MIN_SAMPLES."""

    if arr.ndim != 2:
        raise ValueError(f'Expected 2D numeric table; got shape {arr.shape}')
    n0, n1 = arr.shape
    if n0 == 12 and n1 >= MIN_SAMPLES:
        out = arr.astype(np.float32, copy=False)
    elif n1 == 12 and n0 >= MIN_SAMPLES:
        out = arr.T.astype(np.float32, copy=False)
    elif n0 == 12 and n1 < MIN_SAMPLES:
        raise ValueError(
            f'12 leads × {n1} samples is too short — need at least {MIN_SAMPLES} samples per lead.'
        )
    elif n1 == 12 and n0 < MIN_SAMPLES:
        raise ValueError(
            f'T samples × 12 leads layout has only {n0} samples — need at least {MIN_SAMPLES}.'
        )
    else:
        raise ValueError(
            f'Shape {arr.shape} is not 12-lead: one axis must be exactly 12 and '
            f'the other at least {MIN_SAMPLES}.'
        )
    return out


def parse_ecg_csv(content: bytes) -> np.ndarray:
    """
    Parse CSV / TSV where each **row** is one time step and **columns** are 12 leads
    (PhysioNet / spreadsheet export style).

    **Also accepts** 12 rows × T columns (lead-major) if T >= MIN_SAMPLES.

    Header lines (non-numeric) are skipped automatically.
    """

    text = content.decode('utf-8-sig', errors='replace')
    rows_as_time: list[list[float]] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if ',' in line:
            parts = [p.strip() for p in line.split(',') if p.strip() != '']
        elif '\t' in line:
            parts = [p.strip() for p in line.split('\t') if p.strip() != '']
        elif ';' in line:
            parts = [p.strip() for p in line.split(';') if p.strip() != '']
        else:
            parts = line.split()
        vals: list[float] = []
        for seg in parts:
            s = seg.strip().replace(',', '.')
            try:
                vals.append(float(s))
            except ValueError:
                vals = []
                break
        if len(vals) >= 12:
            rows_as_time.append(vals)

    if not rows_as_time:
        raise ValueError('No numeric rows with at least 12 values found in CSV.')

    arr = np.asarray(rows_as_time, dtype=np.float32)
    try:
        return coerce_twelve_lead(arr)
    except ValueError:
        # Lead-major: flip
        if arr.shape[0] == 12 and arr.shape[1] >= MIN_SAMPLES:
            return arr.astype(np.float32, copy=False)
        raise


def load_wfdb_pair(hea_bytes: bytes, dat_bytes: bytes, hea_filename: str) -> np.ndarray:
    """
    Write ``.hea`` / ``.dat`` to a temp directory and load with ``wfdb``.
    Returns ``(12, T)`` float32.
    """

    if _wfdb is None:
        raise RuntimeError('The wfdb package is required for WFDB uploads (pip install wfdb).')

    stem = Path(hea_filename).stem
    if not stem:
        raise ValueError('WFDB header filename must look like record.hea.')

    with tempfile.TemporaryDirectory(prefix='wfdb_ul_') as tmp:
        base = os.path.join(tmp, stem)
        with open(base + '.hea', 'wb') as fh:
            fh.write(hea_bytes)
        with open(base + '.dat', 'wb') as fh:
            fh.write(dat_bytes)
        try:
            sig, _ = _wfdb.rdsamp(base)
        except Exception as exc:
            raise ValueError(f'Could not read WFDB record {stem!r}: {exc}') from exc

    if sig.ndim != 2 or sig.shape[1] != 12:
        raise ValueError(
            f'Expected 12 channels in WFDB record; got signals shape {sig.shape}.'
        )
    out = sig.T.astype(np.float32)
    if out.shape[1] < MIN_SAMPLES:
        raise ValueError(f'Record too short: {out.shape[1]} samples (min {MIN_SAMPLES}).')
    logger.info('WFDB %s → shape=%s', stem, out.shape)
    return out
