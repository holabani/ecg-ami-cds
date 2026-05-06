# AI-Based ECG Analysis for Early Detection of AMI and Revascularization Need

Full-stack AI-powered Clinical Decision Support System with **no cloud dependencies**. Runs entirely locally via Docker.

## Architecture

- **Frontend**: Next.js (App Router), TypeScript, Tailwind CSS, Recharts, Axios
- **Backend**: FastAPI, PyTorch (dummy model), SHAP placeholder, Prometheus metrics
- **Alerts**: Local mock implementation (in-memory + console logging) — no AWS SNS

## Quick Start

```bash
docker-compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

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
│   │   ├── app/            # Next.js App Router pages
│   │   └── lib/api.ts      # Axios API client
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```
