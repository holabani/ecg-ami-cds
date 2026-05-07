# AI-Based ECG Analysis for Early Detection of AMI and Revascularization Need

Full-stack AI-powered Clinical Decision Support System with **no cloud dependencies**. Runs entirely locally via Docker.

## Architecture

- **Frontend**: Next.js (App Router), TypeScript, Tailwind CSS, Recharts, Axios
- **Backend**: FastAPI, PyTorch (checkpoint inference), Prometheus metrics endpoint
- **Alerts**: Local mock implementation (in-memory + console logging) — no AWS SNS
- **Observability**: Prometheus (`/metrics`) + Grafana dashboards (included in Compose)

## Quick Start

```bash
docker-compose up --build
```

### Pretrained model (no training required for clones)

Cloners get trained inference when **`backend/checkpoints/best_model_fold1.pt`** or **`best_model.pt`** is committed (see `backend/checkpoints/README.md`), or when **`CARDIOSENSE_CHECKPOINT_URL`** points at a downloadable `.pt` (e.g. a GitHub Release asset). Without that, the API still runs but uses **uninitialized weights** (demo mode).

PyTorch is required for neural inference; it is listed in `backend/requirements.txt`.

| Service | URL | Notes |
|--------|-----|-------|
| Frontend | http://localhost:3000 | App UI (use `npm run dev -- -p 3001` if port 3000 is in use, e.g. Homebrew Grafana) |
| Backend API | http://localhost:8000 | Swagger: `/docs` |
| Prometheus | http://localhost:9090 | Scrapes **`backend:8000/metrics`** on the Compose network |
| Grafana | http://localhost:3010 | Login **`admin` / `admin`** — Prometheus data source & **CardioSense — Overview** dashboard are auto-provisioned |

For demos: open Grafana, confirm Prometheus **Status → Targets** is up, generate traffic from the Predict page; panels update after scraping.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Run ECG prediction (AMI + Revascularization) |
| GET | `/history` | Prediction history |
| GET | `/alerts` | Critical alerts (in-memory store) |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |
| GET | `/explain/{patient_id}` | SHAP placeholder explanation |

## Alert Logic

- **Critical Alert** when:
  - AMI probability > 0.80 **OR**
  - Revascularization probability > 0.75
- Alerts are logged to console and stored in memory
- `GET /alerts` returns all stored alerts

## Project Structure

```
ecg-ami-cds/
├── backend/
│   ├── alerts_service.py   # Local mock alerts (no AWS)
│   ├── main.py             # FastAPI app
│   ├── model.py            # Dummy PyTorch model
│   ├── schemas.py          # Pydantic models
│   ├── metrics.py          # Prometheus
│   └── Dockerfile
├── frontend/
│   ├── src/
│   └── Dockerfile
├── deploy/
│   ├── prometheus/prometheus.yml   # targets `backend:8000` under Compose
│   └── grafana/                    # datasources + CardioSense overview dashboard
├── docker-compose.yml
└── README.md
```
