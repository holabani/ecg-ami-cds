"""Prometheus metrics for the ECG CDS backend."""

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Request counters
PREDICT_REQUESTS = Counter(
    "ecg_predict_requests_total",
    "Total number of predict requests",
)
PREDICT_CRITICAL_ALERTS = Counter(
    "ecg_critical_alerts_total",
    "Total number of critical alerts generated",
)

# Latency histogram
PREDICT_LATENCY = Histogram(
    "ecg_predict_latency_seconds",
    "Latency of predict endpoint in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)


def get_metrics() -> tuple[bytes, str]:
    """Return Prometheus metrics in text format."""
    return generate_latest(), CONTENT_TYPE_LATEST
