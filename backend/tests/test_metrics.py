"""
Prometheus instrumentation smoke tests — HTTP + AMI labelled counters.
"""

import pytest


def test_metrics_contains_declared_histograms(api_client):
    txt = api_client.get("/metrics").text
    assert "ecg_http_request_duration_seconds" in txt


def test_http_middleware_records_request(api_client):
    api_client.get("/health")
    txt = api_client.get("/metrics").text
    assert "ecg_http_requests_total{" in txt
    assert "/health" in txt or 'path="' in txt


def test_predict_increments_ami_counters(api_client, auth_headers):
    api_client.post(
        "/predict",
        json={
            "patient_id": "metrics_pt",
            "ecg_data": [[0.0] * 1000 for _ in range(12)],
        },
        headers=auth_headers,
    )
    txt = api_client.get("/metrics").text
    assert "ecg_ami_binary_predictions_total" in txt
    assert "ecg_ami_risk_prediction_total" in txt
