## Security group inbound (recommended phases)

### Phase A — Temporary (raw ports, no nginx yet)

| Rule | TCP | Source | Notes |
|------|-----|--------|------|
| Custom TCP | 3000, 8000 | As needed | Direct Next + FastAPI (close after Phase B). |
| Session Manager | (none inbound for SSM) | — | **SSM outbound** uses **HTTPS** — no inbound rule required. |

### Phase B — Production (nginx TLS on host)

| Rule | TCP | Source | Notes |
|------|-----|--------|------|
| HTTP | 80 | `0.0.0.0/0` (or narrow) | **Let’s Encrypt** HTTP-01 + redirect to HTTPS. |
| HTTPS | 443 | Your users | Browser traffic to **Next / FastAPI / Grafana** hostnames only. |
| SSH | **22** | **`Your.IP/32` or omit** | **Remove `0.0.0.0/0`**. Prefer **SSM-only** admin → delete port 22. |
| Prometheus | **9090** | **omit public** | Compose uses **`127.0.0.1:9090`**; scrape stays on Docker network. Optional gated `prometheus.*` subdomain. |
| Grafana (direct) | **3010** | **omit** | Loopback; use **`https://grafana.&lt;domain&gt;`** via nginx. |

**After Phase B:** delete inbound **3000** and **8000** once **`https://` + nginx`** work everywhere.

Use a **strong Grafana admin password**. Do **not** expose Grafana on the open internet without TLS and good credentials.
