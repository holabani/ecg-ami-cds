"""
Tests for ECGFeatureExtractor (backend/features.py).

Covers:
  - extract_all returns a non-empty dict
  - All feature values are finite floats
  - Per-lead features are computed for all 12 leads
  - Inter-lead features (reciprocal change, ST range) are present
  - get_top_features returns correct count and sorts by magnitude
"""

import numpy as np
import pytest
from features import ECGFeatureExtractor


@pytest.fixture
def extractor():
    return ECGFeatureExtractor()


class TestExtractAll:
    def test_returns_dict(self, extractor, sample_ecg_array):
        feats = extractor.extract_all(sample_ecg_array)
        assert isinstance(feats, dict)
        assert len(feats) > 0

    def test_all_values_are_finite(self, extractor, sample_ecg_array):
        feats = extractor.extract_all(sample_ecg_array)
        for key, val in feats.items():
            assert np.isfinite(val), f"Feature '{key}' is not finite: {val}"

    def test_all_values_are_float(self, extractor, sample_ecg_array):
        feats = extractor.extract_all(sample_ecg_array)
        for key, val in feats.items():
            assert isinstance(val, (int, float, np.floating)), (
                f"Feature '{key}' has unexpected type {type(val)}"
            )

    def test_per_lead_features_all_12_leads(self, extractor, sample_ecg_array):
        feats = extractor.extract_all(sample_ecg_array)
        # PTB-XL lead names: I, II, III, aVR, aVL, aVF, V1-V6
        expected_leads = {'I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                          'V1', 'V2', 'V3', 'V4', 'V5', 'V6'}
        leads_present = {k.split('_')[0] for k in feats}
        missing = expected_leads - leads_present
        assert not missing, f"Missing leads in features: {missing}"

    def test_interlead_features_present(self, extractor, sample_ecg_array):
        feats = extractor.extract_all(sample_ecg_array)
        interlead_keys = [k for k in feats if 'reciprocal' in k or 'st_range' in k
                          or 'st_max' in k or 'st_min' in k]
        assert len(interlead_keys) > 0, "No inter-lead features found"

    def test_zero_ecg_does_not_raise(self, extractor):
        zeros = np.zeros((12, 1000), dtype=np.float32)
        try:
            feats = extractor.extract_all(zeros)
            assert isinstance(feats, dict)
        except Exception as e:
            pytest.fail(f"extract_all raised on zero ECG: {e}")


class TestGetTopFeatures:
    def test_returns_n_features(self, extractor, sample_ecg_array):
        feats = extractor.extract_all(sample_ecg_array)
        top5 = extractor.get_top_features(feats, n=5)
        assert len(top5) == 5

    def test_sorted_by_absolute_magnitude(self, extractor, sample_ecg_array):
        feats = extractor.extract_all(sample_ecg_array)
        top = extractor.get_top_features(feats, n=10)
        # get_top_features returns a dict — check values are sorted descending by abs
        values = [abs(v) for v in top.values()]
        assert values == sorted(values, reverse=True), (
            "get_top_features not sorted by absolute magnitude"
        )

    def test_n_larger_than_features_returns_all(self, extractor, sample_ecg_array):
        feats = extractor.extract_all(sample_ecg_array)
        top = extractor.get_top_features(feats, n=9999)
        assert len(top) == len(feats)
