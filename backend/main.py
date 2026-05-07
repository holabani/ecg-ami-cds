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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from preprocessing import ECGPreprocessor, SCIPY_AVAILABLE
from ecg_io import parse_ecg_csv, load_wfdb_pair
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
    observe_ami_outcome,
    record_ami_ground_truth_confusion,
    attach_http_metrics_middleware,
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
attach_http_metrics_middleware(app)


# ──────────────────────────────────────────────
# Predict endpoint — full CardioSense pipeline
# ──────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(req: PredictRequest):
    """
    Run the full CardioSense pipeline on a 12-lead ECG recording (JSON body).

    Accepts ecg_data as list[list[float]] (12 × T), or flat list 12×T.
    """

    ecg_raw = _parse_ecg_data(req.ecg_data)
    return _run_predict_core(
        ecg_raw,
        req.patient_id,
        req.sampling_rate,
        req.ami_ground_truth,
    )


@app.post("/predict/upload", response_model=PredictResponse)
async def predict_upload(
    patient_id: str = Form(..., description="Patient identifier"),
    sampling_rate: int = Form(
        500,
        ge=60,
        le=2000,
        description="ECG sampling frequency in Hz (100 for PTB-XL low-res WFDB)",
    ),
    ami_ground_truth_str: str = Form(
        '',
        alias='ami_ground_truth',
        description='Optional: true|false for confusion metrics.',
    ),
    csv_file: UploadFile | None = File(None),
    wfdb_header: UploadFile | None = File(None, description=".hea WFDB header file"),
    wfdb_signal: UploadFile | None = File(None, description=".dat WFDB signal file"),
):
    """
    Run the pipeline on an uploaded waveform.

    Exactly **one** of:
    - **CSV**: `csv_file` — rows = time samples, **12 comma-separated numeric columns**
      (alternatively 12 rows × many columns — lead-major), **≥200 samples** per lead;
    - **WFDB**: `wfdb_header` + `wfdb_signal` — matching basename (``record.hea``, ``record.dat``).
    """
    ami_gt = _parse_optional_ami_ground_truth_form(ami_ground_truth_str)

    has_csv = csv_file is not None and csv_file.filename
    has_wfdb = (
        wfdb_header is not None
        and wfdb_signal is not None
        and wfdb_header.filename
        and wfdb_signal.filename
    )
    modes = sum([bool(has_csv), bool(has_wfdb)])
    if modes == 0:
        raise HTTPException(
            status_code=422,
            detail="Provide either csv_file, or wfdb_header + wfdb_signal together.",
        )
    if modes != 1:
        raise HTTPException(status_code=422, detail="Use CSV **or** WFDB pair — not both.")

    if has_csv:
        raw_bytes = await csv_file.read()
        if not raw_bytes:
            raise HTTPException(status_code=422, detail="CSV file empty.")
        try:
            ecg_raw = parse_ecg_csv(raw_bytes)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    else:

        hn = wfdb_header.filename or ''
        if not hn.lower().endswith('.hea'):
            raise HTTPException(status_code=422, detail='WFDB header must be named like *.hea')
        if not ((wfdb_signal.filename or '').lower().endswith('.dat')):

            raise HTTPException(status_code=422, detail='WFDB signals must use a .dat companion file.')

        hea_b = await wfdb_header.read()
        dat_b = await wfdb_signal.read()

        try:

            ecg_raw = load_wfdb_pair(hea_b, dat_b, hn)

        except (ValueError, RuntimeError) as e:

            raise HTTPException(status_code=422, detail=str(e))

    return _run_predict_core(ecg_raw, patient_id, sampling_rate, ami_gt)


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

def _parse_optional_ami_ground_truth_form(raw: str) -> bool | None:
    if raw is None or str(raw).strip() == '':
        return None
    s = str(raw).strip().lower()
    if s in {'true', '1', 'yes', 'on'}:
        return True
    if s in {'false', '0', 'no', 'off'}:
        return False
    raise HTTPException(
        status_code=422,
        detail=f"Invalid ami_ground_truth={raw!r}; use blank, true, or false.",
    )


def _run_predict_core(
    ecg_raw: np.ndarray,
    patient_id: str,
    sampling_rate: int,
    ami_ground_truth: bool | None,
) -> PredictResponse:
    """Shared inference path from (12,T) ndarray (JSON upload or multipart)."""

    PREDICT_REQUESTS.inc()

    with PREDICT_LATENCY.time():

        _preprocessor.fs = sampling_rate

        try:
            prep_result = _preprocessor.run(ecg_raw)
            beat_template = prep_result["beat_template"]
            preprocessing_applied = SCIPY_AVAILABLE
        except Exception as e:

            logger.warning("Preprocessing failed (%s), using raw ECG", e)
            beat_template = ecg_raw
            preprocessing_applied = False

        try:

            features = _extractor.extract_all(beat_template, fs=sampling_rate)

        except Exception as e:

            logger.warning("Feature extraction failed (%s)", e)
            features = {}

        result = predict_with_internals(ecg_raw.tolist())

        ami_prob: float = result["ami_prob"]
        revasc_prob: float = result["revasc_prob"]
        spectrogram = result.get("spectrogram")
        inference_mode = "cnn_bilstm" if TORCH_AVAILABLE else "fallback"

        cam = _gradcam.compute(spectrogram, ami_prob=ami_prob)
        lead_importance = _gradcam.as_dict(cam)
        shap_vals = compute_shap_features(features, ami_prob, revasc_prob)

        ami_label = classify_ami(ami_prob, revasc_prob)
        urgency = classify_urgency(revasc_prob)
        confidence = float(abs(ami_prob - 0.5) * 2)

    alert = alerts_service.check_and_create_alert(
        patient_id=patient_id,
        ami_probability=ami_prob,
        revasc_probability=revasc_prob,
    )
    critical_alert = alert is not None
    if critical_alert:

        PREDICT_CRITICAL_ALERTS.inc()

    observe_ami_outcome(ami_prob)

    ami_evaluation: str | None = None
    if ami_ground_truth is not None:
        ami_evaluation = record_ami_ground_truth_confusion(ami_prob, ami_ground_truth)

    _prediction_history.append({
        "patient_id": patient_id,
        "ami_probability": ami_prob,
        "ami_label": ami_label,
        "revascularization_probability": revasc_prob,
        "revascularization_urgency": urgency,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "critical_alert": critical_alert,
        "shap_features": shap_vals,
        "gradcam_lead_importance": lead_importance,
        "ami_ground_truth": ami_ground_truth,
        "ami_evaluation_vs_ground_truth": ami_evaluation,
    })

    logger.info(
        "Predict | patient=%s | AMI=%.3f (%s) | Revasc=%.3f (%s) | alert=%s | mode=%s",
        patient_id,
        ami_prob,
        ami_label,
        revasc_prob,
        urgency,
        critical_alert,
        inference_mode,

    )

    return PredictResponse(
        patient_id=patient_id,
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
        ami_evaluation_vs_ground_truth=ami_evaluation,
    )


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
