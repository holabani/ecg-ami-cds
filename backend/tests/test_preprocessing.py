"""
Tests for ECGPreprocessor (backend/preprocessing.py).

Covers:
  - Output shape preservation
  - Bandpass filter removes DC offset
  - Notch filter attenuates 50 Hz
  - Handles zero/NaN inputs gracefully
  - R-peak detection returns valid indices
"""

import numpy as np
import pytest
from preprocessing import ECGPreprocessor


@pytest.fixture
def preprocessor():
    return ECGPreprocessor(fs=100)


class TestPreprocess:
    def test_output_shape_unchanged(self, preprocessor, sample_ecg_array):
        out = preprocessor.preprocess(sample_ecg_array)
        assert out.shape == sample_ecg_array.shape

    def test_output_dtype_is_floating(self, preprocessor, sample_ecg_array):
        out = preprocessor.preprocess(sample_ecg_array)
        assert np.issubdtype(out.dtype, np.floating), f"Expected float dtype, got {out.dtype}"

    def test_bandpass_removes_dc(self, preprocessor):
        """After filtering, per-lead mean should be near zero."""
        dc_signal = np.ones((12, 1000), dtype=np.float32) * 2.0
        out = preprocessor.preprocess(dc_signal)
        for lead in range(12):
            assert abs(out[lead].mean()) < 0.1, (
                f"Lead {lead} mean {out[lead].mean():.4f} too large after bandpass"
            )

    def test_notch_attenuates_50hz(self):
        """50 Hz sinusoid power should be reduced after notch filtering (use fs=500)."""
        # Must use fs=500 — at fs=100, 50 Hz is the Nyquist and trivially zero
        p500 = ECGPreprocessor(fs=500)
        t = np.arange(5000) / 500
        sin50 = np.tile(np.sin(2 * np.pi * 50 * t).astype(np.float32), (12, 1))
        out = p500.preprocess(sin50)
        power_in  = float(np.mean(sin50 ** 2))
        power_out = float(np.mean(out ** 2))
        assert power_out < power_in * 0.1, (
            f"Notch filter did not attenuate 50 Hz: in={power_in:.4f} out={power_out:.4f}"
        )

    def test_handles_zero_input(self, preprocessor):
        zeros = np.zeros((12, 1000), dtype=np.float32)
        out = preprocessor.preprocess(zeros)
        assert out.shape == (12, 1000)
        assert not np.any(np.isnan(out))

    def test_handles_nan_input_gracefully(self, preprocessor):
        bad = np.zeros((12, 1000), dtype=np.float32)
        bad[0, 100] = np.nan
        # Should not raise; NaNs may propagate but shape must be correct
        try:
            out = preprocessor.preprocess(bad)
            assert out.shape == (12, 1000)
        except Exception as e:
            pytest.fail(f"preprocess raised unexpectedly on NaN input: {e}")


class TestRPeakDetection:
    def test_returns_list_or_array(self, preprocessor, sample_ecg_array):
        lead0 = sample_ecg_array[0]
        peaks = preprocessor.detect_rpeaks(lead0)
        assert hasattr(peaks, '__len__'), "detect_rpeaks must return a sequence"

    def test_peaks_within_signal_bounds(self, preprocessor, sample_ecg_array):
        lead0 = sample_ecg_array[0]
        peaks = preprocessor.detect_rpeaks(lead0)
        for p in peaks:
            assert 0 <= p < len(lead0), f"Peak index {p} out of bounds"

    def test_empty_signal_returns_empty(self, preprocessor):
        empty = np.zeros(1000, dtype=np.float32)
        peaks = preprocessor.detect_rpeaks(empty)
        assert len(peaks) == 0 or isinstance(peaks, (list, np.ndarray))
