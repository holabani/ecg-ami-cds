"""
Ground-truth confusion metrics — unit + API smoke tests.

``ami_ground_truth`` on POST /predict drives Prometheus TP/TN/FP/FN counters
(reference label vs binary prediction at ``AMI_BINARY_THRESHOLD``).
"""

import pytest

from metrics import (
    AMI_BINARY_THRESHOLD,
    evaluate_ami_vs_ground_truth,
)


class TestEvaluateAmiVsGroundTruth:
    def test_cell_logic(self):
        assert evaluate_ami_vs_ground_truth(0.9, True) == "tp"
        assert evaluate_ami_vs_ground_truth(0.1, False) == "tn"
        assert evaluate_ami_vs_ground_truth(0.9, False) == "fp"
        assert evaluate_ami_vs_ground_truth(0.1, True) == "fn"

    def test_threshold_inclusive_predicted_positive(self):
        assert evaluate_ami_vs_ground_truth(AMI_BINARY_THRESHOLD, True) == "tp"
        assert evaluate_ami_vs_ground_truth(AMI_BINARY_THRESHOLD, False) == "fp"
        assert evaluate_ami_vs_ground_truth(AMI_BINARY_THRESHOLD - 1e-6, True) == "fn"


def test_predict_with_ground_truth_returns_evaluation_cell(api_client, auth_headers):
    payload = {
        "patient_id": "gt-test-1",
        "ecg_data": [[0.0] * 1000 for _ in range(12)],
        "ami_ground_truth": True,
    }
    r = api_client.post("/predict", json=payload, headers=auth_headers)
    assert r.status_code == 200
    data = r.json()
    assert data["ami_evaluation_vs_ground_truth"] in ("tp", "tn", "fp", "fn")


def test_predict_without_ground_truth_null_evaluation(api_client, auth_headers):
    r = api_client.post(
        "/predict",
        json={"patient_id": "gt-test-2", "ecg_data": [[0.0] * 1000 for _ in range(12)]},
        headers=auth_headers,
    )
    assert r.status_code == 200
    assert r.json().get("ami_evaluation_vs_ground_truth") is None


def test_metrics_exposes_confusion_counter(api_client, auth_headers):
    api_client.post(
        "/predict",
        json={
            "patient_id": "gt-metrics",
            "ecg_data": [[0.01] * 1000 for _ in range(12)],
            "ami_ground_truth": False,
        },
        headers=auth_headers,
    )
    txt = api_client.get("/metrics").text
    assert "ecg_ami_confusion_vs_ground_truth_total" in txt
    assert "ecg_predict_with_ami_ground_truth_total" in txt
