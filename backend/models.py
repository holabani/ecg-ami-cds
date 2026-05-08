"""SQLAlchemy models."""

import json
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)

    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction", back_populates="user", cascade="all, delete-orphan"
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    patient_id: Mapped[str] = mapped_column(String(512), index=True, nullable=False)
    ami_probability: Mapped[float] = mapped_column(Float, nullable=False)
    ami_label: Mapped[str] = mapped_column(String(64), nullable=False)
    revascularization_probability: Mapped[float] = mapped_column(Float, nullable=False)
    revascularization_urgency: Mapped[str] = mapped_column(String(128), nullable=False)
    timestamp: Mapped[str] = mapped_column(String(64), nullable=False)
    critical_alert: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    shap_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    gradcam_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    ami_ground_truth: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    ami_evaluation: Mapped[str | None] = mapped_column(String(8), nullable=True)

    user: Mapped["User"] = relationship("User", back_populates="predictions")

    def shap_dict(self) -> dict:
        try:
            return json.loads(self.shap_json or "{}")
        except json.JSONDecodeError:
            return {}

    def gradcam_dict(self) -> dict:
        try:
            return json.loads(self.gradcam_json or "{}")
        except json.JSONDecodeError:
            return {}


class PendingRegistration(Base):
    """Stores registration until the user verifies email with a one-time code."""

    __tablename__ = "pending_registrations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    otp_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
