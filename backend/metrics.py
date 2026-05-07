"""
Prometheus metrics for the ECG CDS backend.

Includes:
  - Predict-path counters (backward-compatible) + AMI outcome / risk tiers
  - HTTP layer: request counts by method/path/status, 5xx counter, latency histogram

AMI binary labels use ``AMI_BINARY_THRESHOLD``: predicted positive if ami_probability >= threshold.
Risk tiers mirror common CDS cutoffs for dashboards (modifiable for your study).
"""

from __future__ import annotations

import time

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


# ─── Clinical thresholds (document in thesis / ops runbooks) ────────────────

AMI_BINARY_THRESHOLD = 0.5
AMI_HIGH_RISK_THRESHOLD = 0.65   # aligns with CDS “elevated” band in UI thresholds
AMI_CRITICAL_THRESHOLD = 0.80    # aligns with alert / STEMI gate in explainability

# ─── Predict endpoint (fine-grained, kept for Grafana panels on /predict only) ─

PREDICT_REQUESTS = Counter(
    "ecg_predict_requests_total",
    "Total number of POST /predict calls",
)

PREDICT_CRITICAL_ALERTS = Counter(
    "ecg_critical_alerts_total",
    "Total number of critical alerts generated on predict",
)

PREDICT_LATENCY = Histogram(
    "ecg_predict_latency_seconds",
    "Wall time of POST /predict handler (pipeline + alerting + history)",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 15.0],
)

# ─── AMI outcome & risk (updated each successful prediction) ──────────────

AMI_BINARY_PREDICTIONS = Counter(
    "ecg_ami_binary_predictions_total",
    'AMI predictions labelled "positive" or "negative" vs AMI_BINARY_THRESHOLD',
    ("binary",),
)

AMI_RISK_TIER = Counter(
    "ecg_ami_risk_prediction_total",
    "AMI predictions bucketed into risk tiers for monitoring drift",
    ("tier",),
)

# ─── All HTTP routes (Starlette middleware) ─────────────────────────────────

HTTP_REQUESTS = Counter(
    "ecg_http_requests_total",
    "HTTP requests processed",
    ("method", "path", "status_code"),
)

HTTP_5XX_RESPONSES = Counter(
    "ecg_http_server_errors_total",
    "Responses with HTTP 5xx status (application or unhandled errors)",
)

HTTP_REQUEST_DURATION = Histogram(
    "ecg_http_request_duration_seconds",
    "End-to-end HTTP request latency (seconds)",
    ("method", "path"),
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 15.0, 60.0],
)


class PrometheusMetricsMiddleware(BaseHTTPMiddleware):
    """
    Record latency / status for every HTTP request.

    Uses the mounted route template (e.g. ``/explain/{patient_id}``) as the ``path`` label
    to avoid exploding Prometheus cardinality when patient IDs vary.
    """

    async def dispatch(self, request: Request, call_next):
        # Avoid counting Prometheus scrapes as application traffic / latency spikes
        if request.url.path == "/metrics":
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        method = request.method
        path = _http_path_label(request)
        status_code = str(response.status_code)

        HTTP_REQUEST_DURATION.labels(method=method, path=path).observe(elapsed)
        HTTP_REQUESTS.labels(
            method=method, path=path, status_code=status_code
        ).inc()
        if response.status_code >= 500:
            HTTP_5XX_RESPONSES.inc()
        return response


def _http_path_label(request: Request) -> str:
    route = request.scope.get("route")
    if route is not None:
        p = getattr(route, "path", None)
        if p:
            return p
    return "__unhandled_route__"


def attach_http_metrics_middleware(app) -> None:
    """
    Append Starlette middleware (outermost layer) after CORS/other inner layers.

    Call **after** ``add_middleware(CORSMiddleware)`` so HTTP timings include full stack.
    """

    app.add_middleware(PrometheusMetricsMiddleware)


def observe_ami_outcome(ami_probability: float) -> None:
    """
    Record AMI+/AMI− and coarse risk tiers from the model probability.

    * binary: ``positive`` if prob >= AMI_BINARY_THRESHOLD else ``negative``
    * tier: ``critical`` | ``high`` | ``moderate`` | ``low`` (exclusive buckets, low inclusive)
    """
    p = float(ami_probability)
    if p >= AMI_BINARY_THRESHOLD:
        AMI_BINARY_PREDICTIONS.labels(binary="positive").inc()
    else:
        AMI_BINARY_PREDICTIONS.labels(binary="negative").inc()

    if p >= AMI_CRITICAL_THRESHOLD:
        AMI_RISK_TIER.labels(tier="critical").inc()
    elif p >= AMI_HIGH_RISK_THRESHOLD:
        AMI_RISK_TIER.labels(tier="high").inc()
    elif p >= AMI_BINARY_THRESHOLD:
        AMI_RISK_TIER.labels(tier="moderate").inc()
    else:
        AMI_RISK_TIER.labels(tier="low").inc()


def get_metrics() -> tuple[bytes, str]:
    """Return Prometheus exposition format for scrape."""
    return generate_latest(), CONTENT_TYPE_LATEST
