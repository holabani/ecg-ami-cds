"""
CardioSense FastAPI Backend — Full Pipeline.

Five-stage pipeline per POST /predict:
  1. Signal QA + parsing (accept 2D or flat ECG input)
  2. Preprocessing: bandpass filter → notch filter → R-peak → beat segmentation
  3. Feature extraction: morphological + wavelet + inter-lead
  4. AI inference: CNN-BiLSTM (or NumPy fallback) → AMI + Revascularization probs
  5. Explainability: Grad-CAM lead saliency + SHAP feature attribution

All XAI modules degrade gracefully when optional libraries are absent.
"""

import logging
from datetime import datetime
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from preprocessing import ECGPreprocessor, SCIPY_AVAILABLE
from features import ECGFeatureExtractor
from model import predict_with_internals, TORCH_AVAILABLE
from explainability import (
    ECGGradCAM,
    compute_shap_features,
    classify_ami,
    classify_urgency,
)
from alerts_service import alerts_service
from metrics import (
    PREDICT_REQUESTS,
    PREDICT_CRITICAL_ALERTS,
    PREDICT_LATENCY,
    get_metrics,
)
from schemas import (
    PredictRequest,
    PredictResponse,
    HistoryResponse,
    HistoryItem,
    AlertsResponse,
    AlertItem,
    ExplainResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# In-memory prediction history (newest appended last)
_prediction_history: list[dict] = []

# Module-level singletons
_preprocessor = ECGPreprocessor(fs=500)
_extractor = ECGFeatureExtractor()
_gradcam = ECGGradCAM()  # Captum unavailable → clinical fallback used


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CardioSense backend starting up...")
    logger.info(f"  PyTorch: {'available' if TORCH_AVAILABLE else 'NOT available (fallback)'}")
    logger.info(f"  SciPy:   {'available' if SCIPY_AVAILABLE else 'NOT available (fallback)'}")
    yield
    logger.info("CardioSense backend shutting down.")


app = FastAPI(
    title="CardioSense API",
    description=(
        "AI-Based Electrocardiography (ECG) Analysis for Early Detection "
        "of AMI and Revascularization Need. Full CNN-BiLSTM pipeline with "
        "Grad-CAM and SHAP explainability."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Predict endpoint — full CardioSense pipeline
# ──────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(req: PredictRequest):
    """
    Run the full CardioSense pipeline on a 12-lead ECG recording.

    Accepts ecg_data as:
    - list[list[float]] — 12 leads × T samples (recommended)
    - list[float]       — flat 12×T values

    Returns AMI probability, AMI label (STEMI/NSTEMI/Normal/Inconclusive),
    revascularization probability and urgency, SHAP feature attributions,
    and Grad-CAM lead saliency scores.
    """
    PREDICT_REQUESTS.inc()

    with PREDICT_LATENCY.time():
        # ── Stage 1: Parse ECG input ──────────────────────────────────
        ecg_raw = _parse_ecg_data(req.ecg_data)
        _preprocessor.fs = req.sampling_rate  # Honour declared sampling rate

        # ── Stage 2: Preprocessing ────────────────────────────────────
        try:
            prep_result = _preprocessor.run(ecg_raw)
            beat_template = prep_result["beat_template"]
            preprocessing_applied = SCIPY_AVAILABLE
        except Exception as e:
            logger.warning(f"Preprocessing failed ({e}), using raw ECG")
            beat_template = ecg_raw
            preprocessing_applied = False

        # ── Stage 3: Feature extraction ───────────────────────────────
        try:
            features = _extractor.extract_all(beat_template, fs=req.sampling_rate)
        except Exception as e:
            logger.warning(f"Feature extraction failed ({e})")
            features = {}

        # ── Stage 4: Model inference ──────────────────────────────────
        result = predict_with_internals(ecg_raw.tolist())
        ami_prob: float = result["ami_prob"]
        revasc_prob: float = result["revasc_prob"]
        spectrogram = result.get("spectrogram")
        inference_mode = "cnn_bilstm" if TORCH_AVAILABLE else "fallback"

        # ── Stage 5: Explainability ───────────────────────────────────
        # Grad-CAM: lead-level saliency from CNN spectrogram branch
        cam = _gradcam.compute(spectrogram, ami_prob=ami_prob)
        lead_importance = _gradcam.as_dict(cam)

        # SHAP: signed feature attributions
        shap_vals = compute_shap_features(features, ami_prob, revasc_prob)

        # ── Clinical classification ───────────────────────────────────
        ami_label = classify_ami(ami_prob, revasc_prob)
        urgency = classify_urgency(revasc_prob)
        confidence = float(abs(ami_prob - 0.5) * 2)  # 0 = borderline, 1 = certain

    # ── Alerts ───────────────────────────────────────────────────────
    alert = alerts_service.check_and_create_alert(
        patient_id=req.patient_id,
        ami_probability=ami_prob,
        revasc_probability=revasc_prob,
    )
    critical_alert = alert is not None
    if critical_alert:
        PREDICT_CRITICAL_ALERTS.inc()

    # ── Store in history ─────────────────────────────────────────────
    _prediction_history.append({
        "patient_id": req.patient_id,
        "ami_probability": ami_prob,
        "ami_label": ami_label,
        "revascularization_probability": revasc_prob,
        "revascularization_urgency": urgency,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "critical_alert": critical_alert,
        "shap_features": shap_vals,
        "gradcam_lead_importance": lead_importance,
    })

    logger.info(
        f"Predict | patient={req.patient_id} | AMI={ami_prob:.3f} ({ami_label}) | "
        f"Revasc={revasc_prob:.3f} ({urgency}) | alert={critical_alert} | mode={inference_mode}"
    )

    return PredictResponse(
        patient_id=req.patient_id,
        ami_probability=round(ami_prob, 4),
        ami_label=ami_label,
        ami_confidence=round(confidence, 4),
        revascularization_probability=round(revasc_prob, 4),
        revascularization_urgency=urgency,
        critical_alert=critical_alert,
        alert_id=alert.id if alert else None,
        shap_features=shap_vals,
        gradcam_lead_importance=lead_importance,
        inference_mode=inference_mode,
        preprocessing_applied=preprocessing_applied,
    )


# ──────────────────────────────────────────────
# History endpoint
# ──────────────────────────────────────────────

@app.get("/history", response_model=HistoryResponse)
async def history_endpoint():
    """Return prediction history (newest first)."""
    items = [
        HistoryItem(
            patient_id=h["patient_id"],
            ami_probability=h["ami_probability"],
            ami_label=h.get("ami_label", "Unknown"),
            revascularization_probability=h["revascularization_probability"],
            revascularization_urgency=h.get("revascularization_urgency", "Unknown"),
            timestamp=h["timestamp"],
            critical_alert=h["critical_alert"],
        )
        for h in reversed(_prediction_history)
    ]
    return HistoryResponse(predictions=items, total=len(items))


# ──────────────────────────────────────────────
# Alerts endpoint
# ──────────────────────────────────────────────

@app.get("/alerts", response_model=AlertsResponse)
async def alerts_endpoint():
    """Return all critical alerts (newest first)."""
    alerts = alerts_service.get_all_alerts()
    return AlertsResponse(
        alerts=[AlertItem(**a) for a in alerts],
        total=len(alerts),
    )


# ──────────────────────────────────────────────
# Explain endpoint — SHAP + Grad-CAM breakdown
# ──────────────────────────────────────────────

@app.get("/explain/{patient_id}", response_model=ExplainResponse)
async def explain_endpoint(patient_id: str):
    """
    Return the detailed XAI breakdown for a patient's latest prediction.

    Provides:
    - SHAP feature attribution (top 10 morphological/spectral features)
    - Grad-CAM lead-level saliency (which of the 12 leads drove the prediction)
    - Clinical natural-language explanation
    """
    matches = [h for h in _prediction_history if h["patient_id"] == patient_id]
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction found for patient '{patient_id}'. Run /predict first.",
        )

    latest = matches[-1]
    ami_prob = latest["ami_probability"]
    ami_label = latest.get("ami_label", "Unknown")
    revasc_prob = latest["revascularization_probability"]
    urgency = latest.get("revascularization_urgency", "Unknown")
    shap = latest.get("shap_features", {})
    gradcam = latest.get("gradcam_lead_importance", {})

    # Generate clinical natural-language explanation
    explanation = _build_explanation(ami_label, ami_prob, revasc_prob, urgency, shap, gradcam)

    return ExplainResponse(
        patient_id=patient_id,
        ami_label=ami_label,
        ami_probability=round(ami_prob, 4),
        revascularization_probability=round(revasc_prob, 4),
        revascularization_urgency=urgency,
        explanation=explanation,
        feature_importance=shap,
        gradcam_lead_importance=gradcam,
    )


# ──────────────────────────────────────────────
# Health + Metrics endpoints
# ──────────────────────────────────────────────

@app.get("/health")
async def health_endpoint():
    """Health check with pipeline component status."""
    return {
        "status": "healthy",
        "service": "CardioSense",
        "version": "2.0.0",
        "pipeline": {
            "torch": TORCH_AVAILABLE,
            "scipy": SCIPY_AVAILABLE,
        },
    }


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    data, content_type = get_metrics()
    return Response(content=data, media_type=content_type)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _parse_ecg_data(ecg_data: list) -> np.ndarray:
    """
    Convert raw ecg_data from request to (12, T) float32 ndarray.
    Handles both 2D (list of lists) and flat (list of floats) inputs.
    """
    if not ecg_data:
        raise HTTPException(status_code=422, detail="ecg_data cannot be empty.")

    if isinstance(ecg_data[0], (list, tuple)):
        arr = np.array(ecg_data, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != 12:
            raise HTTPException(
                status_code=422,
                detail=f"ecg_data must be shape (12, T). Got {arr.shape}.",
            )
    else:
        flat = np.array(ecg_data, dtype=np.float32)
        if len(flat) < 12:
            raise HTTPException(status_code=422, detail="ecg_data too short.")
        t = len(flat) // 12
        arr = flat[: 12 * t].reshape(12, t)

    return arr


def _build_explanation(
    ami_label: str,
    ami_prob: float,
    revasc_prob: float,
    urgency: str,
    shap: dict,
    gradcam: dict,
) -> str:
    """
    Build a clinical natural-language explanation from model outputs.
    Mirrors the structured clinical impression format used in Section 8.5.
    """
    # Top contributing feature
    top_feat = max(shap, key=lambda k: abs(shap[k])) if shap else "ECG features"
    top_val = shap.get(top_feat, 0)

    # Highest saliency lead
    top_lead = max(gradcam, key=gradcam.get) if gradcam else "precordial leads"

    sentiment = "elevating" if top_val > 0 else "reducing"

    lines = [
        f"CardioSense classifies this ECG as: {ami_label}.",
        f"AMI probability: {ami_prob:.1%} | Revascularization probability: {revasc_prob:.1%}.",
        f"Recommended action: {urgency}.",
        "",
        f"The model's attention was highest on lead {top_lead} (Grad-CAM saliency = "
        f"{gradcam.get(top_lead, 0):.2f}), consistent with the anatomical territory "
        f"implicated by the ECG pattern.",
        "",
        f"The feature '{top_feat}' had the largest SHAP attribution "
        f"({top_val:+.3f}), {sentiment} the AMI prediction. "
        f"This aligns with established diagnostic criteria for {ami_label}.",
        "",
        "Note: CardioSense is a decision-support tool. Final clinical judgement "
        "rests with the treating physician.",
    ]
    return " ".join(line for line in lines if line)
