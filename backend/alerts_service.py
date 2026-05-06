"""
Local mock alerts service for AMI/Revascularization critical alerts.
No AWS SNS - uses in-memory storage and console logging.
"""

import logging
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Represents a critical clinical alert."""
    id: str
    patient_id: str
    alert_type: str
    ami_probability: float
    revasc_probability: float
    message: str
    timestamp: str
    severity: str = "Critical"


class AlertsService:
    """In-memory alerts service - no AWS dependencies."""
    
    AMI_THRESHOLD = 0.80
    REVASC_THRESHOLD = 0.75
    
    def __init__(self):
        self._alerts: list[Alert] = []
        self._alert_counter = 0
    
    def _generate_alert_id(self) -> str:
        self._alert_counter += 1
        return f"ALT-{datetime.now().strftime('%Y%m%d')}-{self._alert_counter:04d}"
    
    def check_and_create_alert(
        self,
        patient_id: str,
        ami_probability: float,
        revasc_probability: float,
    ) -> Optional[Alert]:
        """
        Check if AMI or Revascularization probability exceeds threshold.
        If so, create Critical Alert, log to console, store in memory.
        Returns the Alert if created, None otherwise.
        """
        is_critical = (
            ami_probability > self.AMI_THRESHOLD or
            revasc_probability > self.REVASC_THRESHOLD
        )
        
        if not is_critical:
            return None
        
        # Determine alert reason
        reasons = []
        if ami_probability > self.AMI_THRESHOLD:
            reasons.append(f"AMI probability {ami_probability:.2%} > {self.AMI_THRESHOLD:.0%}")
        if revasc_probability > self.REVASC_THRESHOLD:
            reasons.append(f"Revascularization probability {revasc_probability:.2%} > {self.REVASC_THRESHOLD:.0%}")
        
        message = "Critical Alert: " + "; ".join(reasons)
        
        alert = Alert(
            id=self._generate_alert_id(),
            patient_id=patient_id,
            alert_type="Critical",
            ami_probability=ami_probability,
            revasc_probability=revasc_probability,
            message=message,
            timestamp=datetime.utcnow().isoformat() + "Z",
            severity="Critical",
        )
        
        self._alerts.append(alert)
        
        # Log to console
        logger.critical(
            f"[CRITICAL ALERT] {alert.id} | Patient: {patient_id} | "
            f"AMI: {ami_probability:.2%} | Revasc: {revasc_probability:.2%} | "
            f"{message}"
        )
        print(
            f"\n{'='*60}\n"
            f"🚨 CRITICAL ALERT - {alert.id}\n"
            f"Patient ID: {patient_id}\n"
            f"AMI Probability: {ami_probability:.2%}\n"
            f"Revascularization Probability: {revasc_probability:.2%}\n"
            f"Message: {message}\n"
            f"Timestamp: {alert.timestamp}\n"
            f"{'='*60}\n"
        )
        
        return alert
    
    def get_all_alerts(self) -> list[dict]:
        """Return all stored alerts as list of dicts (newest first)."""
        return [
            {
                "id": a.id,
                "patient_id": a.patient_id,
                "alert_type": a.alert_type,
                "ami_probability": a.ami_probability,
                "revasc_probability": a.revasc_probability,
                "message": a.message,
                "timestamp": a.timestamp,
                "severity": a.severity,
            }
            for a in reversed(self._alerts)
        ]


# Singleton instance
alerts_service = AlertsService()
