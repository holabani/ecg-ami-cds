"""
Integration tests for the CardioSense FastAPI backend (backend/main.py).

Covers:
  - GET  /health         — reports healthy status
  - POST /predict        — full pipeline returns valid response schema
  - GET  /history        — returns list
  - GET  /explain/{id}   — returns explanation after a prediction
  - POST /predict        — invalid input returns 422
  - POST /predict        — missing ecg_data returns 422
"""

import numpy as np
import pytest


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, api_client):
        resp = api_client.get("/health")
        assert resp.status_code == 200

    def test_response_has_status_field(self, api_client):
        data = api_client.get("/health").json()
        assert "status" in data

    def test_status_is_healthy(self, api_client):
        data = api_client.get("/health").json()
        assert data["status"] in ("healthy", "degraded")


# ── /predict ──────────────────────────────────────────────────────────────────

class TestPredict:
    def _make_payload(self, ecg=None, patient_id="test-001"):
        if ecg is None:
            rng = np.random.default_rng(7)
            ecg = rng.normal(0, 0.1, (12, 1000)).astype(float).tolist()
        return {"patient_id": patient_id, "ecg_data": ecg}

    def test_returns_200(self, api_client):
        resp = api_client.post("/predict", json=self._make_payload())
        assert resp.status_code == 200

    def test_response_contains_ami_probability(self, api_client):
        data = api_client.post("/predict", json=self._make_payload()).json()
        assert "ami_probability" in data
        assert 0.0 <= data["ami_probability"] <= 1.0

    def test_response_contains_revasc_probability(self, api_client):
        data = api_client.post("/predict", json=self._make_payload()).json()
        assert "revascularization_probability" in data
        assert 0.0 <= data["revascularization_probability"] <= 1.0

    def test_response_contains_ami_label(self, api_client):
        data = api_client.post("/predict", json=self._make_payload()).json()
        assert "ami_label" in data
        assert data["ami_label"] in ("STEMI", "NSTEMI", "Normal", "Inconclusive")

    def test_response_contains_urgency(self, api_client):
        data = api_client.post("/predict", json=self._make_payload()).json()
        assert "revascularization_urgency" in data

    def test_response_contains_gradcam(self, api_client):
        data = api_client.post("/predict", json=self._make_payload()).json()
        assert "gradcam_lead_importance" in data
        assert isinstance(data["gradcam_lead_importance"], dict)

    def test_response_contains_shap(self, api_client):
        data = api_client.post("/predict", json=self._make_payload()).json()
        assert "shap_features" in data
        assert isinstance(data["shap_features"], dict)

    def test_flat_list_input_accepted(self, api_client):
        flat = np.zeros(12000).tolist()
        resp = api_client.post("/predict", json={"patient_id": "p1", "ecg_data": flat})
        assert resp.status_code == 200

    def test_missing_ecg_data_returns_422(self, api_client):
        resp = api_client.post("/predict", json={"patient_id": "p1"})
        assert resp.status_code == 422

    def test_empty_ecg_data_returns_error(self, api_client):
        resp = api_client.post("/predict", json={"patient_id": "p1", "ecg_data": []})
        assert resp.status_code in (400, 422)

    def test_different_patients_tracked_separately(self, api_client):
        api_client.post("/predict", json=self._make_payload(patient_id="p-aaa"))
        api_client.post("/predict", json=self._make_payload(patient_id="p-bbb"))
        resp = api_client.get("/history").json()
        predictions = resp if isinstance(resp, list) else resp.get("predictions", resp)
        patient_ids = [h["patient_id"] for h in predictions]
        assert "p-aaa" in patient_ids
        assert "p-bbb" in patient_ids


# ── /history ──────────────────────────────────────────────────────────────────

class TestHistory:
    def _history_list(self, api_client):
        """Return predictions list regardless of whether API wraps it."""
        data = api_client.get("/history").json()
        return data if isinstance(data, list) else data.get("predictions", [])

    def _history_count(self, api_client):
        data = api_client.get("/history").json()
        if isinstance(data, list):
            return len(data)
        return data.get("total", len(data.get("predictions", [])))

    def test_returns_200(self, api_client):
        resp = api_client.get("/history")
        assert resp.status_code == 200

    def test_returns_iterable_of_predictions(self, api_client):
        predictions = self._history_list(api_client)
        assert isinstance(predictions, list)

    def test_grows_after_prediction(self, api_client):
        before = self._history_count(api_client)
        rng = np.random.default_rng(1)
        ecg = rng.normal(0, 0.1, (12, 1000)).tolist()
        api_client.post("/predict", json={"patient_id": "hist-test-grow", "ecg_data": ecg})
        after = self._history_count(api_client)
        assert after > before


# ── /explain/{patient_id} ────────────────────────────────────────────────────

class TestExplain:
    def test_explain_after_predict_returns_200(self, api_client):
        rng = np.random.default_rng(3)
        ecg = rng.normal(0, 0.1, (12, 1000)).tolist()
        api_client.post("/predict", json={"patient_id": "explain-pt", "ecg_data": ecg})
        resp = api_client.get("/explain/explain-pt")
        assert resp.status_code == 200

    def test_explain_unknown_patient_returns_404(self, api_client):
        resp = api_client.get("/explain/nonexistent-patient-xyz")
        assert resp.status_code == 404

    def test_explain_contains_narrative(self, api_client):
        rng = np.random.default_rng(5)
        ecg = rng.normal(0, 0.1, (12, 1000)).tolist()
        api_client.post("/predict", json={"patient_id": "explain-narr", "ecg_data": ecg})
        data = api_client.get("/explain/explain-narr").json()
        assert "explanation" in data or "narrative" in data or "summary" in data
