"""Pydantic schemas for CardioSense API request and response models."""

import re

from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Literal, Optional


def validate_password_policy(password: str) -> str:
    """Registration: ≥8 chars, lowercase, uppercase, and one special character."""
    if len(password) > 128:
        raise ValueError('Password must be at most 128 characters.')
    if len(password) < 8:
        raise ValueError('Password must be at least 8 characters.')
    if not re.search(r'[a-z]', password):
        raise ValueError('Password must include a lowercase letter.')
    if not re.search(r'[A-Z]', password):
        raise ValueError('Password must include an uppercase letter.')
    if not re.search(r'[^A-Za-z0-9]', password):
        raise ValueError('Password must include at least one special character.')
    return password


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator('password')
    @classmethod
    def password_strength(cls, v: str) -> str:
        return validate_password_policy(v)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PredictRequest(BaseModel):
    """
    POST /predict request body.

    ecg_data accepts two formats:
    - list[list[float]] — 12 leads × T samples (recommended, matches report format)
    - list[float]       — flat list of 12 × T values (backward compatible)

    Sampling rate should be declared so preprocessing can apply correct filter
    cutoffs and R-peak detection.
    """
    patient_id: str = Field(..., description="Unique patient identifier")
    ecg_data: list = Field(
        ...,
        description=(
            "12-lead ECG data. Preferred: list of 12 lists (one per lead), each with T samples. "
            "Also accepted: flat list of 12×T values (lead-major order)."
        ),
    )
    sampling_rate: int = Field(
        500,
        description="ECG sampling frequency in Hz. 500 Hz for clinical recordings, 100 Hz for PTB-XL.",
    )
    ami_ground_truth: Optional[bool] = Field(
        None,
        description=(
            "Optional reference label for retrospective monitoring only. "
            "true = AMI present (e.g. angiographer-confirmed MI), false = no AMI. "
            "When set, Prometheus updates confusion counters (tp/tn/fp/fn) using the "
            "same threshold as ami_probability ≥ 0.5 for predicted positive."
        ),
    )


class PredictResponse(BaseModel):
    """POST /predict response — full CardioSense clinical decision support output."""

    patient_id: str

    # --- AMI Detection ---
    ami_probability: float = Field(..., ge=0, le=1, description="Probability of AMI (0–1)")
    ami_label: str = Field(
        ...,
        description="Classification label: STEMI | NSTEMI | Normal | Inconclusive",
    )
    ami_confidence: float = Field(
        ..., ge=0, le=1,
        description="Model confidence (distance from 0.5 decision boundary)",
    )

    # --- Revascularization Prediction ---
    revascularization_probability: float = Field(
        ..., ge=0, le=1, description="Probability patient will require revascularization (0–1)"
    )
    revascularization_urgency: str = Field(
        ...,
        description=(
            "Recommended urgency: Immediate PCI | Early Invasive (24h) | "
            "Deferred Invasive | Conservative Management"
        ),
    )

    # --- Alert ---
    critical_alert: bool = Field(
        ..., description="True when AMI > 80% OR Revascularization > 75%"
    )
    alert_id: Optional[str] = Field(None, description="Alert ID if critical_alert is True")

    # --- Explainability ---
    shap_features: dict[str, float] = Field(
        default_factory=dict,
        description="Top 10 SHAP feature attributions (positive = pushes toward AMI)",
    )
    gradcam_lead_importance: dict[str, float] = Field(
        default_factory=dict,
        description="Grad-CAM per-lead saliency (0–1, higher = more important for prediction)",
    )

    # --- Pipeline metadata ---
    inference_mode: str = Field(
        ...,
        description="cnn_bilstm | fallback — indicates whether full model or NumPy fallback was used",
    )
    preprocessing_applied: bool = Field(
        ..., description="True if SciPy bandpass/notch filtering was applied"
    )

    # --- Ground-truth evaluation (only when ami_ground_truth was sent on request)
    ami_evaluation_vs_ground_truth: Optional[
        Literal["tp", "tn", "fp", "fn"]
    ] = Field(
        None,
        description=(
            "If request included ami_ground_truth: confusion cell vs AMI_BINARY_THRESHOLD — "
            "tp=true positive, tn=true negative, fp=false positive, fn=false negative"
        ),
    )


class HistoryItem(BaseModel):
    """Single prediction history entry."""
    patient_id: str
    ami_probability: float
    ami_label: str
    revascularization_probability: float
    revascularization_urgency: str
    timestamp: str
    critical_alert: bool


class HistoryResponse(BaseModel):
    """GET /history response."""
    predictions: list[HistoryItem]
    total: int


class AlertItem(BaseModel):
    """Single critical alert entry."""
    id: str
    patient_id: str
    alert_type: str
    ami_probability: float
    revasc_probability: float
    message: str
    timestamp: str
    severity: str


class AlertsResponse(BaseModel):
    """GET /alerts response."""
    alerts: list[AlertItem]
    total: int


class ExplainResponse(BaseModel):
    """GET /explain/{patient_id} response — detailed XAI breakdown."""
    patient_id: str

    # Clinical summary
    ami_label: str
    ami_probability: float
    revascularization_probability: float
    revascularization_urgency: str

    # Explanation text
    explanation: str
    note: str = "CardioSense XAI — Grad-CAM + SHAP feature attribution"

    # SHAP feature attribution (bar chart data)
    feature_importance: dict[str, float]

    # Grad-CAM lead-level saliency (heatmap data)
    gradcam_lead_importance: dict[str, float]
