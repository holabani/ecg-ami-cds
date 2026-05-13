#!/usr/bin/env python3
"""
Premium IEEE-style Major Project viva deck for CardioSense (ecg-ami-cds).

Requires: pip install python-pptx matplotlib numpy
Output:   presentations/CardioSense_Major_Project_Viva.pptx

Each slide is built around high-DPI matplotlib figures (infographics, architecture,
dashboards) plus Times New Roman titles in PowerPoint for editability.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_mpl_cache = ROOT / ".matplotlib-cache"
_mpl_cache.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

OUT_DIR = ROOT / "presentations"
OUT_PPTX = OUT_DIR / "CardioSense_Major_Project_Viva.pptx"

# ── Brand (PPTX) ─────────────────────────────────────────────────────────────
NAVY = RGBColor(13, 27, 42)
NAVY_CARD = RGBColor(27, 38, 59)
WHITE = RGBColor(255, 255, 255)
SILVER = RGBColor(198, 208, 224)
TEAL_PPT = RGBColor(42, 157, 143)  # #2a9d8f

# ── Matplotlib hex palette (navy + teal + clinical) ─────────────────────
C_BG = "#0d1b2a"
C_CARD = "#1b263b"
C_TEAL = "#2a9d8f"
C_TEAL_L = "#4ecdc4"
C_WHITE = "#ffffff"
C_MUTED = "#8d99ae"
C_GRID = "#2b3a4f"
C_WARN = "#e76f51"
C_OK = "#2a9d8f"


def synthetic_ecg_like(n: int = 2400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 6, n)
    ph = (t * 1.15 * 2 * np.pi) % (2 * np.pi)
    qrs = np.exp(-((ph - 0.35) ** 2) / 0.0018) * 1.5
    qrs += np.exp(-((ph - 6.6) ** 2) / 0.0018) * 1.45
    y = qrs + 0.06 * np.sin(2 * np.pi * 0.65 * t) + 0.04 * rng.standard_normal(n)
    return t, y


def fig_to_png_bytes(fig, dpi: int = 170) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _rounded_box(ax, x, y, w, h, text, fc, ec=None, fontsize=9, tc="white", lw=1.4):
    if ec is None:
        ec = C_TEAL
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.06",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=tc,
        fontfamily="DejaVu Sans",
        weight="medium",
        zorder=3,
    )
    return box


def _arrow(ax, x1, y1, x2, y2, color=C_TEAL_L, lw=1.8):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=2, shrinkB=2),
        zorder=1,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Slide figure generators
# ═══════════════════════════════════════════════════════════════════════════


def fig_cover_ecg_background() -> bytes:
    """Layered ECG traces + grid for cover (PPT adds title on top)."""
    np.random.seed(44)
    fig = plt.figure(figsize=(14, 8), facecolor=C_BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=C_BG)
    # subtle grid
    for g in np.linspace(0, 1, 24):
        ax.axhline(g * 2 - 1, color=C_GRID, lw=0.35, alpha=0.45)
        ax.axvline(g * 14, color=C_GRID, lw=0.35, alpha=0.45)
    for i, off in enumerate([0.0, 0.35, -0.25]):
        t, y = synthetic_ecg_like(2800, seed=40 + i)
        y = (y - y.mean()) * (0.55 - i * 0.1) + off
        col = [C_TEAL_L, "#6c9bcf", "#5fa8d3"][i]
        ax.plot(t, y, color=col, lw=1.1 - i * 0.15, alpha=0.5 - i * 0.08)
    ax.set_xlim(0, 6)
    ax.set_ylim(-1.4, 1.4)
    ax.axis("off")
    return fig_to_png_bytes(fig, dpi=180)


def fig_problem_infographic() -> bytes:
    """Patient → ECG → delay → bottleneck → risk (infographic)."""
    fig, ax = plt.subplots(figsize=(13.2, 5.8), facecolor=C_BG)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.set_facecolor(C_BG)
    ax.axis("off")

    stages = [
        (0.4, 2.2, 1.55, 1.9, "Patient\npresentation", C_CARD),
        (2.35, 2.2, 1.45, 1.9, "12-lead\nECG capture", C_CARD),
        (4.15, 2.2, 1.55, 1.9, "Manual read\nlatency", C_WARN),
        (6.1, 2.2, 1.65, 1.9, "Clinician\nbottleneck", "#7f5539"),
        (8.2, 2.2, 1.75, 1.9, "Monitoring\n& comms gaps", "#6c757d"),
        (10.35, 2.2, 1.95, 1.9, "AMI risk /\nmissed STEMI", "#9d0208"),
    ]
    for x, y, w, h, txt, c in stages:
        _rounded_box(ax, x, y, w, h, txt, c, ec=C_TEAL, fontsize=10)

    for i in range(len(stages) - 1):
        xa = stages[i][0] + stages[i][2]
        ya = stages[i][1] + stages[i][3] / 2
        xb = stages[i + 1][0]
        yb = stages[i + 1][1] + stages[i + 1][3] / 2
        _arrow(ax, xa + 0.02, ya, xb - 0.02, yb)

    # mini ECG strip
    t, y = synthetic_ecg_like(900, 99)
    ax.plot(np.linspace(1.0, 3.8, len(t)), y * 0.25 + 5.1, color=C_TEAL_L, lw=1.6)
    ax.text(0.45, 5.55, "Clinical pathway · where delay amplifies harm", fontsize=12, color=C_WHITE, fontfamily="DejaVu Serif", style="italic")

    ax.text(
        0.45,
        0.55,
        "CardioSense motivation — manual interpretation does not scale; CDS + observability required.",
        fontsize=10,
        color=C_MUTED,
        fontfamily="DejaVu Sans",
    )
    return fig_to_png_bytes(fig)


def fig_objective_cards() -> bytes:
    objs = [
        ("AMI detection", "CNN–BiLSTM \n+ torch inference"),
        ("PCI / CABG signal", "Revascularization \nprobability head"),
        ("Explainable AI", "Grad-CAM + SHAP-style\nfeatures"),
        ("AWS deployment", "EC2 · Compose · nginx"),
        ("Real-time monitoring", "Prometheus + Grafana\n/metrics"),
        ("Clinical CDS UX", "Next.js · alerts · history"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.9), facecolor=C_BG)
    fig.patch.set_facecolor(C_BG)
    for ax, (title, sub) in zip(axes.flat, objs):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor(C_BG)
        fancy = FancyBboxPatch(
            (0.06, 0.1),
            0.88,
            0.78,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=C_CARD,
            edgecolor=C_TEAL,
            linewidth=1.8,
        )
        ax.add_patch(fancy)
        bar = Rectangle((0.06, 0.1), 0.04, 0.78, facecolor=C_TEAL, zorder=3)
        ax.add_patch(bar)
        ax.text(0.52, 0.72, title, ha="center", va="center", fontsize=12.5, color=C_WHITE, fontfamily="DejaVu Serif", weight="bold")
        ax.text(0.52, 0.38, sub, ha="center", va="center", fontsize=9.5, color=C_MUTED, fontfamily="DejaVu Sans")

    fig.suptitle(
        "Project objectives — aligned with ecg-ami-cds modules",
        fontsize=14,
        color=C_WHITE,
        fontfamily="DejaVu Serif",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    return fig_to_png_bytes(fig)


def fig_comparison_split() -> bytes:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.8), facecolor=C_BG, gridspec_kw={"wspace": 0.12})
    for ax, title, rows, accent in [
        (
            axL,
            "Traditional ECG workflow",
            [
                "Interpretation: human reader bottleneck",
                "Turnaround: variable across shifts",
                "Transparency: implicit pattern recognition",
                "Scale: queue-bound during surge",
                "Ops visibility: fragmented logs",
            ],
            C_WARN,
        ),
        (
            axR,
            "CardioSense · AI-assisted workflow",
            [
                "Interpretation: model + clinician oversight",
                "Turnaround: FastAPI inference path (ms–s)",
                "Transparency: Grad-CAM + SHAP-style bars",
                "Scale: containerized on EC2 / Compose",
                "Observability: /metrics → Prometheus → Grafana",
            ],
            C_TEAL,
        ),
    ]:
        ax.set_facecolor(C_CARD)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        hdr = FancyBboxPatch((0.04, 0.86), 0.92, 0.11, boxstyle="round,pad=0.008", facecolor=accent, edgecolor="none")
        ax.add_patch(hdr)
        ax.text(0.5, 0.915, title, ha="center", va="center", fontsize=11.5, color="white", fontfamily="DejaVu Serif", weight="bold")
        y = 0.78
        for r in rows:
            ax.text(0.08, y, "▸ " + r, ha="left", va="top", fontsize=10, color=C_WHITE, fontfamily="DejaVu Sans")
            y -= 0.125
    fig.text(0.5, 0.02, "Industry-style comparison · no fictional timing gains — architecture & capabilities focus.", ha="center", fontsize=9, color=C_MUTED)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.1)
    return fig_to_png_bytes(fig)


def fig_system_architecture_full() -> bytes:
    """Primary architecture slide — SaaS-style layers."""
    fig, ax = plt.subplots(figsize=(13.5, 6.9), facecolor=C_BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.set_facecolor(C_BG)
    ax.axis("off")

    # Cloud banner
    cloud = FancyBboxPatch(
        (0.35, 6.55),
        13.3,
        1.2,
        boxstyle="round,pad=0.02",
        facecolor="#14213d",
        edgecolor=C_TEAL,
        linewidth=1.5,
    )
    ax.add_patch(cloud)
    ax.text(7, 7.15, "AWS EC2  ·  Docker Compose  ·  nginx TLS reverse proxy", ha="center", va="center", fontsize=11, color=C_WHITE, fontfamily="DejaVu Serif")
    ax.text(7, 6.75, "deploy/nginx/cardiosense-ec2.conf · loopback bind · sqlite + prometheus volumes", ha="center", fontsize=8.5, color=C_MUTED)

    # Main traffic column
    col_x = 1.0
    y0 = 5.45
    stack = [
        "User\n(clinician browser)",
        "Next.js 14 · TS\nTailwind",
        "Axios client\nNEXT_PUBLIC_API_URL",
        "nginx\n443 → services",
        "FastAPI · Uvicorn\nmain.py",
        "ECGPreprocessor +\nFeatureExtractor",
        "PyTorch CNN–BiLSTM\n(STFT + BiLSTM fuse)",
        "Explainability\nGrad-CAM · SHAP feats",
        "SQLAlchemy\n→ SQLite",
    ]
    h = 0.52
    gap = 0.05
    y = y0
    for i, s in enumerate(stack):
        _rounded_box(ax, col_x, y, 3.25, h, s, C_CARD, fontsize=8.8)
        if i < len(stack) - 1:
            _arrow(ax, col_x + 1.62, y - gap, col_x + 1.62, y + h - 0.02)
        y -= (h + gap + 0.08)

    # Parallel: metrics bus
    ax.text(6.15, 5.35, "Observability path", fontsize=11, color=C_TEAL_L, fontfamily="DejaVu Serif", weight="bold")
    obs = [
        (6.1, 4.75, 3.4, 0.55, "GET /metrics\nprometheus_client"),
        (6.1, 3.95, 3.4, 0.55, "Prometheus :9090\nscrape"),
        (6.1, 3.15, 3.4, 0.55, "Grafana 11.x\nCardioSense Overview"),
    ]
    oy = 4.75
    for x, yy, w, ht, lab in obs:
        _rounded_box(ax, x, yy, w, ht, lab, "#14213d", fontsize=8.5)
    _arrow(ax, 4.35, 3.5, 6.05, 4.05)
    ax.text(5.0, 3.85, "/metrics", fontsize=7, color=C_TEAL_L, rotation=28)

    # JWT / OTP sidecar
    _rounded_box(ax, 10.0, 4.9, 3.35, 1.15, "Auth plane\nJWT Bearer · POST /auth/*\nOTP email verify", "#1d3557", fontsize=9)

    # Docker strip
    dock = Rectangle((1.0, 0.45), 12.2, 0.85, facecolor="#102a43", edgecolor=C_TEAL, linewidth=1.2)
    ax.add_patch(dock)
    ax.text(
        7.1,
        0.88,
        "Containers: backend · frontend · prometheus:v2.54.1 · grafana:11.4.0  —  docker-compose.production.yml",
        ha="center",
        va="center",
        fontsize=9,
        color=C_WHITE,
        fontfamily="DejaVu Sans",
    )
    return fig_to_png_bytes(fig, dpi=175)


def fig_ecg_processing_premium() -> bytes:
    np.random.seed(7)
    t, y = synthetic_ecg_like(2000, 11)
    t = t / t.max() * 8

    fig, (axw, axp) = plt.subplots(2, 1, figsize=(13, 6.1), facecolor=C_BG, height_ratios=[1.15, 1])

    # Waveform + QRS markers (Lead II style)
    axw.set_facecolor(C_BG)
    axw.plot(t, y, color=C_TEAL_L, lw=1.35)
    # detect approximate peaks for annotation
    thr = np.percentile(y, 92)
    peaks = []
    for i in range(2, len(y) - 2):
        if y[i] > thr and y[i] > y[i - 1] and y[i] > y[i + 1]:
            if not peaks or i - peaks[-1] > 80:
                peaks.append(i)
    for p in peaks[:8]:
        axw.axvline(t[p], color=C_WARN, lw=0.9, alpha=0.35)
        circ = Circle((t[p], y[p]), 0.06, fill=False, edgecolor=C_TEAL, lw=2)
        axw.add_patch(circ)
    axw.set_title("Filtered ECG (exemplar) · R-peak emphasis for Pan–Tompkins context", color=C_WHITE, fontsize=12, fontfamily="DejaVu Serif", pad=12)
    axw.tick_params(colors=C_MUTED)
    axw.set_xlabel("time (normalized)", color=C_MUTED, fontsize=9)
    axw.set_ylabel("mV (norm)", color=C_MUTED, fontsize=9)
    for spine in axw.spines.values():
        spine.set_color(C_GRID)

    steps = [
        "Raw upload\nparse_ecg_csv / WFDB",
        "Butterworth\n0.5–40 Hz",
        "50 Hz notch",
        "Pan–Tompkins\nLead II (NK2)",
        "Median beat +\nfeatures.py",
        "STFT + tensor\n→ model.forward",
    ]
    xs = np.linspace(0.45, 12.55, len(steps))
    for i, (x, lab) in enumerate(zip(xs, steps)):
        _rounded_box(axp, x - 0.95, 2.1, 1.85, 1.25, lab, C_CARD, fontsize=8.4)
        if i < len(steps) - 1:
            _arrow(axp, x + 0.92, 2.72, xs[i + 1] - 0.95, 2.72)

    axp.set_xlim(0, 14)
    axp.set_ylim(0, 4.2)
    axp.set_facecolor(C_BG)
    axp.axis("off")

    axp.text(
        7,
        0.6,
        "preprocessing.py · fs=500 · zero-phase sosfiltfilt · implementation-true parameters",
        ha="center",
        fontsize=9,
        color=C_MUTED,
    )
    plt.tight_layout()
    return fig_to_png_bytes(fig)


def fig_model_research() -> bytes:
    fig, ax = plt.subplots(figsize=(13, 6.2), facecolor="white")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_facecolor("white")

    # branch labels
    ax.text(3.5, 6.35, "Frequency branch (spectrogram)", ha="center", fontsize=11, color=C_TEAL, weight="bold", fontfamily="DejaVu Serif")
    ax.text(3.5, 2.1, "Temporal branch (sequence)", ha="center", fontsize=11, color=C_TEAL, weight="bold", fontfamily="DejaVu Serif")

    # CNN stack
    ys = [5.45, 4.65, 3.85]
    for yi in ys:
        _rounded_box(ax, 1.8, yi, 3.4, 0.62, "Conv2D block +\nBatchNorm · ReLU", "#264653", ec=C_TEAL, tc="white", fontsize=8.5)
    _rounded_box(ax, 1.8, 2.85, 3.4, 0.55, "GAP / spatial pool", "#415a77", fontsize=8.5)

    # mini feature maps
    for j in range(3):
        xi = 6.0 + j * 0.55
        rng = np.random.RandomState(j)
        im = rng.rand(6, 8)
        ax.imshow(im, extent=[xi, xi + 0.45, 4.0 + j * 0.2, 4.15 + j * 0.2], cmap="viridis", alpha=0.85)

    # BiLSTM
    _rounded_box(ax, 1.85, 1.35, 3.3, 0.7, "BiLSTM ×2 · hidden 128\n(batch, T, 12)", C_CARD, fontsize=8.5)
    _arrow(ax, 3.5, 2.8, 3.5, 2.15)

    # Fusion + heads
    _rounded_box(ax, 6.8, 2.85, 2.8, 1.3, "Attention\nfusion gate", "#2a6f97", fontsize=10)
    _arrow(ax, 5.25, 5.0, 6.75, 3.55)
    _arrow(ax, 5.25, 1.7, 6.75, 3.05)

    _rounded_box(ax, 10.35, 3.55, 2.8, 0.75, "AMI · sigmoid\nP(AMI)", "#1b4332", fontsize=9)
    _rounded_box(ax, 10.35, 2.65, 2.8, 0.75, "Revasc · sigmoid\nPCI/CABG need", "#1b4332", fontsize=9)

    ax.text(
        7,
        0.45,
        "model.py · CardioSenseModel · predict_with_internals · STFT n_fft=128 hop=32 · PyTorch CPU on EC2",
        ha="center",
        fontsize=9,
        color="#457b9d",
        fontfamily="DejaVu Sans",
    )
    return fig_to_png_bytes(fig)


def fig_xai_premium() -> bytes:
    rng = np.random.RandomState(42)
    fig = plt.figure(figsize=(13, 6.2), facecolor=C_BG)

    # wave + heat overlay strip
    ax0 = fig.add_axes([0.06, 0.72, 0.88, 0.2])
    t, y = synthetic_ecg_like(1600, 5)
    ax0.plot(t, y, color=C_TEAL_L, lw=1.3)
    heat = np.outer(np.linspace(0.3, 1, len(t)), np.ones(3))
    ax0.imshow(heat.T, extent=[t.min(), t.max(), y.min() - 0.2, y.max() + 0.2], aspect="auto", cmap="RdYlGn_r", alpha=0.25)
    ax0.set_facecolor(C_BG)
    ax0.set_title("Temporal explanation overlay (illustrative)", color=C_WHITE, fontsize=11, fontfamily="DejaVu Serif", loc="left")
    ax0.tick_params(colors=C_MUTED)
    ax0.set_xlabel("samples", color=C_MUTED)

    ax1 = fig.add_axes([0.06, 0.38, 0.4, 0.28])
    spec = rng.standard_normal((40, 56)) * 0.07 + np.outer(np.linspace(1, 0.2, 40), np.linspace(0.3, 1, 56))
    im = ax1.imshow(spec, cmap="RdBu_r", aspect="auto")
    ax1.set_title("Grad-CAM · spectrogram branch", color=C_WHITE, fontsize=10, fontfamily="DejaVu Serif")
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_axes([0.54, 0.42, 0.4, 0.24])
    leads = ["V4", "V5", "II", "aVL", "III"][::-1]
    v = np.array([0.9, 0.72, 0.55, 0.41, 0.28])
    ax2.barh(leads, v, color=C_TEAL, edgecolor="white", linewidth=0.6)
    ax2.set_facecolor(C_CARD)
    ax2.tick_params(colors=C_MUTED)
    ax2.set_title("Lead saliency · gradcam_lead_importance", color=C_WHITE, fontsize=10, fontfamily="DejaVu Serif")

    ax3 = fig.add_axes([0.54, 0.08, 0.4, 0.22])
    feats = ["ST V4", "recip aVL", "DWT E", "QRS width"]
    s = np.array([0.35, -0.22, 0.18, -0.11])
    cols = [C_WARN if x >= 0 else C_TEAL_L for x in s]
    ax3.barh(feats, s, color=cols, edgecolor="white", linewidth=0.5)
    ax3.axvline(0, color=C_WHITE, lw=0.7)
    ax3.set_facecolor(C_CARD)
    ax3.set_title("SHAP-style features · compute_shap_features()", color=C_WHITE, fontsize=10, fontfamily="DejaVu Serif")
    ax3.tick_params(colors=C_MUTED)

    return fig_to_png_bytes(fig)


def fig_api_auth_flow() -> bytes:
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=C_BG)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_facecolor(C_BG)

    _rounded_box(ax, 0.5, 4.8, 2.3, 1.1, "Next.js 14\nfrontend", C_CARD)
    _rounded_box(ax, 3.5, 5.05, 2.1, 0.6, "Axios\nHTTP client", "#14213d")
    _rounded_box(ax, 6.2, 4.7, 3.3, 1.3, "FastAPI + Uvicorn\nJWT Depends() routes", "#1d3557")
    _rounded_box(ax, 10.2, 4.85, 2.2, 0.9, "SQLite\nSQLAlchemy ORM", C_CARD)

    _arrow(ax, 2.85, 5.35, 3.45, 5.35)
    _arrow(ax, 5.65, 5.35, 6.15, 5.35)
    _arrow(ax, 9.55, 5.35, 10.15, 5.35)

    _rounded_box(ax, 2.8, 2.6, 2.8, 1.25, "Auth\nregister → OTP email\nverify → JWT", "#6c584c")
    _rounded_box(ax, 6.5, 2.45, 3.8, 1.55, "Core endpoints\nPOST /predict · /predict/upload\nGET /history /alerts\n/explain/{id}\n/health /metrics", "#264653", fontsize=8.8)

    _arrow(ax, 4.2, 4.75, 4.8, 3.9)
    _arrow(ax, 7.85, 4.7, 7.85, 3.95)

    ax.text(0.6, 1.15, "Bearer token on protected routes · CORS in main.py · metrics middleware excludes /metrics churn", fontsize=9.5, color=C_MUTED)
    return fig_to_png_bytes(fig)


def fig_monitoring_premium() -> bytes:
    fig = plt.figure(figsize=(13, 6.3), facecolor="#0e0e0e")

    # Prometheus pipeline banner
    axb = fig.add_axes([0.04, 0.86, 0.92, 0.1])
    axb.set_facecolor("#151515")
    axb.set_xlim(0, 1)
    axb.set_ylim(0, 1)
    axb.axis("off")
    steps = ["FastAPI", "/metrics", "Prometheus", "Grafana DS", "Dashboards"]
    xs = np.linspace(0.08, 0.92, len(steps))
    for i, (x, s) in enumerate(zip(xs, steps)):
        rect = FancyBboxPatch(
            (x - 0.065, 0.25),
            0.13,
            0.5,
            boxstyle="round,pad=0.01",
            facecolor="#1e3a5f",
            edgecolor=C_TEAL,
        )
        axb.add_patch(rect)
        axb.text(x, 0.52, s, ha="center", va="center", fontsize=9, color="white", fontfamily="DejaVu Sans", weight="bold")
        if i < len(steps) - 1:
            axb.annotate("", xy=(xs[i + 1] - 0.07, 0.5), xytext=(x + 0.07, 0.5), arrowprops=dict(arrowstyle="-|>", color=C_TEAL_L, lw=1.5))

    titles = [
        "ecg_http_request_duration_seconds",
        "ecg_predict_latency_seconds",
        "ecg_ami_risk_prediction_total",
        "ecg_predictions (exemplar rate)",
    ]
    for i, title in enumerate(titles):
        ax = fig.add_axes([0.04 + (i % 2) * 0.48, 0.42 - (i // 2) * 0.38, 0.44, 0.32])
        ax.set_facecolor("#161616")
        rng = np.random.RandomState(i * 17)
        x = np.arange(24)
        y = np.cumsum(rng.randn(24) * 0.03 + 0.04)
        ax.fill_between(x, y, alpha=0.35, color="#73BF69")
        ax.plot(x, y, color="#73BF69", lw=1.8)
        ax.set_title(title, color="#ddd", fontsize=8, loc="left", fontfamily="DejaVu Sans Mono")
        ax.tick_params(colors="#777", labelsize=6)
        for spine in ax.spines.values():
            spine.set_color("#333")

    fig.text(0.5, 0.02, "backend/metrics.py · PREDICT_LATENCY · AMI tier counters · HTTP middleware path templates", ha="center", color="#888", fontsize=9)
    return fig_to_png_bytes(fig)


def fig_aws_premium() -> bytes:
    fig, ax = plt.subplots(figsize=(13.5, 6.4), facecolor="#f8f9fa")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # EC2
    ec2 = FancyBboxPatch(
        (0.4, 0.6),
        13.2,
        6.8,
        boxstyle="round,pad=0.03",
        facecolor="white",
        edgecolor="#ff9900",
        linewidth=2.5,
    )
    ax.add_patch(ec2)
    ax.text(7, 7.15, "Amazon EC2 instance", ha="center", fontsize=13, weight="bold", color="#232f3e", fontfamily="DejaVu Serif")
    ax.text(7, 6.75, "Security groups · elastic IP optional · volumes: backend SQLite + Prometheus TSDB", ha="center", fontsize=9, color="#555")

    # Compose box
    cb = FancyBboxPatch((0.9, 1.2), 12.2, 5.0, boxstyle="round,pad=0.02", facecolor="#e9ecef", edgecolor=C_TEAL, linewidth=1.5)
    ax.add_patch(cb)
    ax.text(7, 5.85, "Docker Compose (docker-compose.production.yml)", ha="center", fontsize=11, weight="bold", color="#212529")

    svcs = [
        (1.4, 3.8, "nginx\nhost proxy"),
        (4.0, 3.8, "frontend\nNext.js :3000"),
        (6.6, 3.8, "backend\nFastAPI :8000"),
        (9.2, 3.8, "prometheus\n:9090"),
        (11.35, 3.8, "grafana\n:3010 map"),
    ]
    for x, y, name in svcs:
        _rounded_box(ax, x, y, 2.0, 1.0, name, "#415a77", ec="#212529", fontsize=9)

    _arrow(ax, 3.45, 4.3, 3.95, 4.3)
    _arrow(ax, 6.05, 4.3, 6.55, 4.3)
    _arrow(ax, 8.65, 4.3, 9.15, 4.3)
    _arrow(ax, 11.05, 4.3, 11.35, 4.3)

    ax.text(7, 2.35, "HTTPS → nginx → 127.0.0.1 services · Prometheus scrapes backend:8000/metrics on Docker network", ha="center", fontsize=9.5, color="#495057")
    return fig_to_png_bytes(fig)


def fig_results_premium() -> bytes:
    fig, ax = plt.subplots(figsize=(13, 6.2), facecolor=C_BG)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_facecolor(C_BG)

    metrics = [
        ("AUROC (AMI)", "—", "From thesis validation cohort"),
        ("Sensitivity", "—", "Lock to clinical endpoint definition"),
        ("Specificity", "—", "Lock to clinical endpoint definition"),
        ("Accuracy", "—", "Report alongside CI if available"),
    ]
    x0 = 0.5
    for i, (title, val, sub) in enumerate(metrics):
        xx = x0 + i * 3.05
        fancy = FancyBboxPatch(
            (xx, 5.1),
            2.75,
            1.35,
            boxstyle="round,pad=0.02",
            facecolor=C_CARD,
            edgecolor=C_TEAL,
            linewidth=1.6,
        )
        ax.add_patch(fancy)
        ax.text(xx + 1.37, 6.1, title, ha="center", fontsize=11, color=C_WHITE, weight="bold", fontfamily="DejaVu Serif")
        ax.text(xx + 1.37, 5.55, val, ha="center", fontsize=22, color=C_TEAL_L, fontfamily="DejaVu Sans", weight="bold")
        ax.text(xx + 1.37, 5.15, sub, ha="center", fontsize=7.5, color=C_MUTED)

    ax.text(6.5, 4.35, "Replace em dashes with your measured KPIs — repo does not hard-code study statistics.", ha="center", fontsize=10, color=C_MUTED)

    # Wireframe placeholders "screenshots"
    for i, lab in enumerate(["Analyse UI", "Explain / XAI", "Grafana panel"]):
        xx = 0.7 + i * 4.0
        rect = FancyBboxPatch((xx, 1.05), 3.5, 2.35, boxstyle="round,pad=0.015", facecolor="#111923", edgecolor=C_MUTED, linestyle="--", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(xx + 1.75, 2.0, lab + "\n[screenshot]", ha="center", va="center", fontsize=10, color=C_MUTED)

    ax.text(6.5, 0.45, "Capture live screenshots from deployed CardioSense for final viva PDF appendix.", ha="center", fontsize=9, color=C_TEAL)
    return fig_to_png_bytes(fig)


def fig_challenges_cards() -> bytes:
    chals = [
        ("Image footprint", "Torch + scipy stack\n→ Docker build RAM"),
        ("Compose networking", "Loopback bind + nginx\ntunnel testing"),
        ("Frontend API URL", "Axios base vs TLS host"),
        ("Prometheus labels", "Route templates for\ncardinality control"),
        ("Preprocessing deps", "NK2 / PyWT fallbacks"),
        ("XAI libraries", "Captum / SHAP optional\nheuristic paths"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.9), facecolor=C_BG)
    for ax, (t, b) in zip(axes.flat, chals):
        ax.set_facecolor(C_BG)
        ax.axis("off")
        fancy = FancyBboxPatch(
            (0.05, 0.12),
            0.9,
            0.76,
            boxstyle="round,pad=0.02",
            facecolor=C_CARD,
            edgecolor=C_WARN,
            linewidth=1.4,
        )
        ax.add_patch(fancy)
        ax.text(0.5, 0.72, t, ha="center", fontsize=12, color=C_WHITE, weight="bold", fontfamily="DejaVu Serif")
        ax.text(0.5, 0.4, b, ha="center", fontsize=9.5, color=C_MUTED, fontfamily="DejaVu Sans")

    fig.suptitle("Deployment & integration challenges (honest engineering)", fontsize=14, color=C_WHITE, fontfamily="DejaVu Serif", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    return fig_to_png_bytes(fig)


def fig_conclusion_split() -> bytes:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.8), facecolor=C_BG, gridspec_kw={"wspace": 0.15})
    for ax in (axL, axR):
        ax.set_facecolor(C_BG)
        ax.axis("off")

    fancyL = FancyBboxPatch((0.04, 0.08), 0.92, 0.85, boxstyle="round,pad=0.02", facecolor=C_CARD, edgecolor=C_TEAL, transform=axL.transAxes, linewidth=1.8)
    axL.add_patch(fancyL)
    axL.text(0.5, 0.9, "Conclusion", transform=axL.transAxes, ha="center", fontsize=14, color=C_WHITE, weight="bold", fontfamily="DejaVu Serif")
    concl = [
        "End-to-end CDS: preprocessing → dual-head CNN–BiLSTM → SQLite persistence.",
        "Clinician trust via XAI payloads consumed by Next.js dashboards.",
        "Production-grade ops: /metrics, Grafana, Compose on EC2 behind nginx.",
    ]
    y = 0.72
    for c in concl:
        axL.text(0.08, y, "▹ " + c, transform=axL.transAxes, fontsize=10.5, color=C_WHITE, fontfamily="DejaVu Sans")
        y -= 0.16

    fancyR = FancyBboxPatch((0.04, 0.08), 0.92, 0.85, boxstyle="round,pad=0.02", facecolor="#14213d", edgecolor=C_TEAL_L, transform=axR.transAxes, linewidth=1.8)
    axR.add_patch(fancyR)
    axR.text(0.5, 0.9, "Future scope", transform=axR.transAxes, ha="center", fontsize=14, color=C_WHITE, weight="bold", fontfamily="DejaVu Serif")
    fut = [
        "Wearable / patch ECG streaming",
        "Mobile client + secured APIs",
        "Multi-hospital FHIR-aware rollout",
        "Transformer encoders for 12-lead",
        "PostgreSQL HA + key management",
    ]
    y = 0.74
    for c in fut:
        axR.text(0.08, y, "◇ " + c, transform=axR.transAxes, fontsize=10.5, color=C_MUTED, fontfamily="DejaVu Sans")
        y -= 0.13

    fig.text(0.5, 0.02, "Thank you.", ha="center", fontsize=18, color=C_TEAL_L, fontfamily="DejaVu Serif", style="italic")
    fig.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.12)
    return fig_to_png_bytes(fig)


# ═══════════════════════════════════════════════════════════════════════════
# PowerPoint helpers
# ═══════════════════════════════════════════════════════════════════════════


def apply_background(slide, color: RGBColor) -> None:
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def set_run_title(run, size_pt: int, bold: bool = True, color: RGBColor = WHITE) -> None:
    run.font.name = "Times New Roman"
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    run.font.color.rgb = color


def set_run_caption(run, size_pt: int = 10, color: RGBColor = SILVER) -> None:
    run.font.name = "Calibri"
    run.font.size = Pt(size_pt)
    run.font.bold = False
    run.font.color.rgb = color


def add_teal_accent_bar(slide, prs_w) -> None:
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0.38), prs_w, Inches(0.06))
    bar.fill.solid()
    bar.fill.fore_color.rgb = TEAL_PPT
    bar.line.fill.background()


def add_visual_slide(
    prs: Presentation,
    title: str,
    png_bytes: bytes,
    subtitle: str | None = None,
) -> None:
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)
    apply_background(slide, NAVY)
    prs_w = prs.slide_width
    add_teal_accent_bar(slide, prs_w)

    hdr = slide.shapes.add_textbox(Inches(0.55), Inches(0.52), Inches(12.2), Inches(0.9))
    hp = hdr.text_frame.paragraphs[0]
    hp.text = title
    set_run_title(hp.runs[0], 28)

    y_img = 1.38
    if subtitle:
        sub = slide.shapes.add_textbox(Inches(0.6), Inches(1.08), Inches(12), Inches(0.45))
        sp = sub.text_frame.paragraphs[0]
        sp.text = subtitle
        set_run_caption(sp.runs[0], 12)
        y_img = 1.52

    h_img = 7.5 - y_img - 0.35
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(png_bytes)
        tmp.flush()
        slide.shapes.add_picture(tmp.name, Inches(0.45), Inches(y_img), width=Inches(12.45))


def add_cover_slide(prs: Presentation) -> None:
    blank = prs.slide_layouts[6]
    s1 = prs.slides.add_slide(blank)
    apply_background(s1, NAVY)
    bg_png = fig_cover_ecg_background()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(bg_png)
        tmp.flush()
        s1.shapes.add_picture(tmp.name, 0, 0, width=prs.slide_width)

    overlay = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), prs.slide_width, prs.slide_height)
    overlay.fill.solid()
    overlay.fill.fore_color.rgb = NAVY
    overlay.fill.transparency = 0.45
    overlay.line.fill.background()

    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(1.12), Inches(11.5), Inches(0.09))
    bar.fill.solid()
    bar.fill.fore_color.rgb = TEAL_PPT
    bar.line.fill.background()

    tb = s1.shapes.add_textbox(Inches(0.85), Inches(1.35), Inches(11.6), Inches(2.55))
    tfp = tb.text_frame
    p0 = tfp.paragraphs[0]
    p0.text = (
        "AI-Based ECG Analysis for Early Detection of\n"
        "Acute Myocardial Infarction (AMI)\nand Revascularization Need"
    )
    p0.alignment = PP_ALIGN.CENTER
    set_run_title(p0.runs[0], 28)
    p0.runs[0].font.color.rgb = WHITE

    badge = s1.shapes.add_textbox(Inches(1), Inches(3.98), Inches(11.2), Inches(0.4))
    bp = badge.text_frame.paragraphs[0]
    bp.text = "CardioSense · Clinical decision support platform (production-style stack)"
    bp.alignment = PP_ALIGN.CENTER
    set_run_caption(bp.runs[0], 12, TEAL_PPT)
    bp.runs[0].font.name = "Calibri"
    bp.runs[0].font.color.rgb = RGBColor(180, 220, 215)

    sub = s1.shapes.add_textbox(Inches(1), Inches(4.45), Inches(11), Inches(2.85))
    sf = sub.text_frame
    meta = [
        "Major Project — ecg-ami-cds",
        "Bhavya Saxena · Jiya Bajaj · Harmeet Kaur",
        "Faculty Guide: Dr. Renuka Nagpal",
        "Amity School of Engineering and Technology",
        "Amity University",
    ]
    for i, line in enumerate(meta):
        para = sf.paragraphs[0] if i == 0 else sf.add_paragraph()
        para.text = line
        para.alignment = PP_ALIGN.CENTER
        para.space_before = Pt(8) if i == 1 else Pt(0)
        run = para.runs[0]
        run.font.name = "Times New Roman"
        run.font.size = Pt(14 if i >= 1 else 13)
        run.font.bold = i == 0
        run.font.color.rgb = SILVER if i > 0 else WHITE


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prs = Presentation()
    prs.slide_width = Inches(13.333333)
    prs.slide_height = Inches(7.5)

    add_cover_slide(prs)

    add_visual_slide(prs, "Problem statement — clinical workflow friction", fig_problem_infographic(), "Real-world delays and gaps motivating instrumented CDS + monitoring.")

    add_visual_slide(
        prs,
        "Project objectives",
        fig_objective_cards(),
        "Six capability pillars implemented across FastAPI, Next.js, PyTorch, and the compose stack.",
    )

    add_visual_slide(
        prs,
        "Traditional vs CardioSense workflow",
        fig_comparison_split(),
        "Capability comparison grounded in repository features — not speculative performance claims.",
    )

    add_visual_slide(
        prs,
        "System architecture — deployed CDS topology",
        fig_system_architecture_full(),
        "Single coherent diagram: user path, inference stack, auth, observability, and container footprint.",
    )

    add_visual_slide(
        prs,
        "ECG signal processing & model inputs",
        fig_ecg_processing_premium(),
        "Pan–Tompkins · bandpass/notch · median beat · Daubechies features — matches backend/preprocessing.py and features.py.",
    )

    add_visual_slide(
        prs,
        "CNN–BiLSTM inference graph",
        fig_model_research(),
        "research-grade view of CardioSense dual-branch fusion with concrete STFT hyper-parameters.",
    )

    add_visual_slide(
        prs,
        "Explainable AI — Grad-CAM & SHAP-style outputs",
        fig_xai_premium(),
        "Illustrative only — API keys: gradcam_lead_importance, shap_features · explain.py + Next.js visualisation.",
    )

    add_visual_slide(
        prs,
        "FastAPI backend, Axios client & auth plane",
        fig_api_auth_flow(),
        "JWT + OTP registration + protected predict/history/alerts/explain routes as implemented in main.py.",
    )

    add_visual_slide(
        prs,
        "Monitoring & observability",
        fig_monitoring_premium(),
        "Grafana-style time-series + Prometheus scrape chain reflecting backend/metrics.py instrumentation.",
    )

    add_visual_slide(
        prs,
        "AWS EC2 deployment — Docker & nginx ingress",
        fig_aws_premium(),
        "Light theme cloud diagram · maps to docker-compose.production.yml + EC2 host nginx TLS.",
    )

    add_visual_slide(
        prs,
        "Results, KPIs & evidence slots",
        fig_results_premium(),
        "Placeholders for thesis metrics + replace wireframes with your capture of the live UI and Grafana.",
    )

    add_visual_slide(
        prs,
        "Engineering challenges",
        fig_challenges_cards(),
        "Practical constraints encountered integrating ML + observability + full-stack deployment.",
    )

    add_visual_slide(
        prs,
        "Conclusion & future scope",
        fig_conclusion_split(),
        None,
    )

    prs.save(str(OUT_PPTX))
    print(f"Wrote {OUT_PPTX}")


if __name__ == "__main__":
    main()
