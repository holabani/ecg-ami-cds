"""DB helpers for users and predictions."""

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from models import Prediction, User


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.execute(select(User).where(User.email == email.lower().strip())).scalar_one_or_none()


def create_user(db: Session, email: str, password_hash: str) -> User:
    u = User(email=email.lower().strip(), hashed_password=password_hash)
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.get(User, user_id)


def save_prediction(
    db: Session,
    *,
    user_id: int,
    patient_id: str,
    ami_probability: float,
    ami_label: str,
    revascularization_probability: float,
    revascularization_urgency: str,
    timestamp: str,
    critical_alert: bool,
    shap_features: dict[str, float],
    gradcam_lead_importance: dict[str, float],
    ami_ground_truth: bool | None,
    ami_evaluation_vs_ground_truth: str | None,
) -> Prediction:
    row = Prediction(
        user_id=user_id,
        patient_id=patient_id,
        ami_probability=ami_probability,
        ami_label=ami_label,
        revascularization_probability=revascularization_probability,
        revascularization_urgency=revascularization_urgency,
        timestamp=timestamp,
        critical_alert=critical_alert,
        shap_json=json.dumps(shap_features, default=float),
        gradcam_json=json.dumps(gradcam_lead_importance, default=float),
        ami_ground_truth=ami_ground_truth,
        ami_evaluation=ami_evaluation_vs_ground_truth,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def list_predictions_for_user(db: Session, user_id: int) -> list[Prediction]:
    q = (
        select(Prediction)
        .where(Prediction.user_id == user_id)
        .order_by(Prediction.id.asc())
    )
    return list(db.execute(q).scalars().all())


def latest_prediction_for_patient(db: Session, user_id: int, patient_id: str) -> Prediction | None:
    q = (
        select(Prediction)
        .where(Prediction.user_id == user_id, Prediction.patient_id == patient_id)
        .order_by(Prediction.id.desc())
        .limit(1)
    )
    return db.execute(q).scalar_one_or_none()


def prediction_row_to_history_dict(row: Prediction) -> dict[str, Any]:
    return {
        "patient_id": row.patient_id,
        "ami_probability": row.ami_probability,
        "ami_label": row.ami_label,
        "revascularization_probability": row.revascularization_probability,
        "revascularization_urgency": row.revascularization_urgency,
        "timestamp": row.timestamp,
        "critical_alert": row.critical_alert,
        "shap_features": row.shap_dict(),
        "gradcam_lead_importance": row.gradcam_dict(),
        "ami_ground_truth": row.ami_ground_truth,
        "ami_evaluation_vs_ground_truth": row.ami_evaluation,
    }
