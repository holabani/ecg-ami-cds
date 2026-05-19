"""
Tests for CardioSense model inference (backend/model.py).

Covers:
  - predict() returns two floats in [0, 1]
  - predict() accepts both flat list and 2-D array inputs
  - predict() handles short/long signals by padding/cropping
  - predict_with_internals() returns expected keys
  - Model output is deterministic for same input
"""

import numpy as np
import pytest
from model import predict, predict_with_internals, SEQ_LEN, N_LEADS


class TestPredict:
    def test_returns_two_floats(self, flat_ecg_list):
        ami, revasc = predict(flat_ecg_list)
        assert isinstance(ami,   float)
        assert isinstance(revasc, float)

    def test_outputs_in_unit_interval(self, flat_ecg_list):
        ami, revasc = predict(flat_ecg_list)
        assert 0.0 <= ami   <= 1.0, f"AMI prob out of range: {ami}"
        assert 0.0 <= revasc <= 1.0, f"Revasc prob out of range: {revasc}"

    def test_accepts_2d_array(self, sample_ecg_array):
        ami, revasc = predict(sample_ecg_array)
        assert 0.0 <= ami   <= 1.0
        assert 0.0 <= revasc <= 1.0

    def test_accepts_flat_list(self, flat_ecg_list):
        ami, revasc = predict(flat_ecg_list)
        assert 0.0 <= ami   <= 1.0
        assert 0.0 <= revasc <= 1.0

    def test_handles_short_signal(self):
        """Signal shorter than SEQ_LEN should be padded, not raise."""
        short = np.zeros((N_LEADS, SEQ_LEN // 2), dtype=np.float32)
        ami, revasc = predict(short)
        assert 0.0 <= ami   <= 1.0
        assert 0.0 <= revasc <= 1.0

    def test_handles_long_signal(self):
        """Signal longer than SEQ_LEN should be cropped, not raise."""
        long = np.zeros((N_LEADS, SEQ_LEN * 3), dtype=np.float32)
        ami, revasc = predict(long)
        assert 0.0 <= ami   <= 1.0
        assert 0.0 <= revasc <= 1.0

    def test_deterministic_output(self, flat_ecg_list):
        """Same input must produce identical outputs (model in eval mode)."""
        ami1, revasc1 = predict(flat_ecg_list)
        ami2, revasc2 = predict(flat_ecg_list)
        assert ami1   == ami2,   "AMI prediction is non-deterministic"
        assert revasc1 == revasc2, "Revasc prediction is non-deterministic"

    def test_different_inputs_may_differ(self):
        """Two different random ECGs should not always produce identical outputs."""
        rng = np.random.default_rng(0)
        ecg_a = rng.normal(0, 0.5, (N_LEADS, SEQ_LEN)).astype(np.float32)
        ecg_b = rng.normal(0, 0.5, (N_LEADS, SEQ_LEN)).astype(np.float32)
        ami_a, _ = predict(ecg_a)
        ami_b, _ = predict(ecg_b)
        # With random weights the outputs are unlikely to be identical
        # (not a hard requirement — just a sanity check)
        assert not (ami_a == ami_b), "Different inputs produced identical output (suspicious)"


class TestPredictWithInternals:
    def test_returns_dict_with_required_keys(self, sample_ecg_array):
        result = predict_with_internals(sample_ecg_array)
        assert isinstance(result, dict)
        for key in ('ami_prob', 'revasc_prob'):
            assert key in result, f"Missing key '{key}' in predict_with_internals output"

    def test_probs_in_unit_interval(self, sample_ecg_array):
        result = predict_with_internals(sample_ecg_array)
        assert 0.0 <= result['ami_prob']   <= 1.0
        assert 0.0 <= result['revasc_prob'] <= 1.0
