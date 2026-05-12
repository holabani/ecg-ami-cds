## Security group inbound (starter)

| Rule | TCP port | Source | Notes |
|------|----------|--------|--------|
| SSH | 22 | **Your IP/32 only** — avoid `0.0.0.0/0`** | Admin access |
| HTTP | 80 | `0.0.0.0/0` (optional if using nginx) | Nginx frontend |
| HTTPS | 443 | `0.0.0.0/0` (after certbot) | Same |
| Backend API | **8000** | `0.0.0.0/0` **or** narrow to your campus IP | FastAPI — **required** when `NEXT_PUBLIC_API_URL` includes `:8000` on this host |
| Prometheus | 9090 | **omit** from public SG | `docker-compose.production.yml` binds `127.0.0.1:9090` — use SSH tunnel |
| Grafana | 3010 | **omit** from public SG | binds `127.0.0.1:3010` |

**Do not** expose Grafana’s default admin to the internet on a budget instance.
