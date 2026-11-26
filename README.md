# LabAI: Production Multi-Tier LLM Platform on Bare-Metal CPU Nodes

A hardware-aware, self-hosted ChatGPT-class AI platform built to run **24/7 on two commodity CPU-only university lab desktops** (`2× Intel Core i7-12700`, `14.88 GB` single-channel DDR5-4800 each, `100 Mbps` Fast Ethernet).

Instead of wrapping a single model in a basic UI, **LabAI** implements a custom **AI Control Plane** featuring:
- **3-Tier Model Serving (`Fast` / `Reasoning` / `Pro`)** kept hot in RAM across both nodes (`Qwen3-4B`, `Qwen3-8B`, `Qwen3-14B` + `0.6B` speculative draft).
- **3-Stage Auto Router** (`labai-auto`): Heuristics + Local ONNX Semantic Classifier (`BAAI/bge-small-en-v1.5`) + Live Queue-Aware Fallback, alongside manual tier overrides (`labai-fast`, `labai-reasoning`, `labai-pro`).
- **Dynamic Context-Aware Response Budgeting**: Automatically scales response depth (`768` / `2048` / `4096` tokens) and Qwen3 `/think` vs `/no_think` mode based on prompt complexity instead of a fixed token cutoff.
- **Exact + Semantic Redis Cache** to absorb concurrent classroom/department bursts without CPU thrashing.
- **Asymmetric CPU Core Isolation**: Verified physical P-cores (`0,2,4,6,8,10,12,14`) pinned to `llama-server`; E-cores (`16-19`) dedicated to Docker infra (Open WebUI, FastAPI Gateway, Redis, SearXNG, Caddy, Prometheus, Grafana).
- **Live Web Search, Image Lookup & Interactive Diagrams** via self-hosted SearXNG and native Mermaid/SVG rendering.

---

## Repository Structure

```text
├── docs/ARCHITECTURE.md         # Full system & infra design + Mermaid diagrams
├── gateway/                     # Custom FastAPI AI Control Plane
│   ├── app/
│   │   ├── main.py              # OpenAI-compatible SSE API, /metrics, image-search tool
│   │   ├── router.py            # 3-Stage Auto Router + Dynamic Budgeting + /think steering
│   │   ├── cache.py             # Redis Exact & Semantic Cache + Per-user sliding quotas
│   │   ├── backends.py          # Node 01 inference semaphore + Node 02 circuit breaker
│   │   └── config.py            # Environment settings
│   ├── tests/test_router.py     # Unit tests for routing, budgeting & failover
│   ├── Dockerfile
│   └── requirements.txt
├── systemd/                     # P-core pinned bare-metal llama-server units
│   ├── llama-tier1.service      # node01: Qwen3-4B (port 8081)
│   ├── llama-tier2.service      # node01: Qwen3-8B (port 8082)
│   └── llama-tier3.service      # node02: Qwen3-14B + 0.6B Draft (port 8083)
├── deploy/                      # E-core pinned Docker Compose stack (node01)
│   ├── docker-compose.yml       # Open WebUI, Gateway, Redis, SearXNG, Caddy, Prom, Grafana
│   ├── Caddyfile                # Unbuffered SSE reverse proxy
│   ├── prometheus.yml           # Scrapes gateway, all 3 llama tiers, and both nodes
│   └── searxng/settings.yml     # Enables JSON API for web & image search
└── scripts/                     # Master-driven deployment automation
    ├── setup_worker_ssh.sh      # One-time passwordless SSH from node01 -> node02
    ├── node_provision.sh        # Installs deps, builds llama.cpp natively, pulls Qwen3 GGUFs
    └── deploy_cluster.sh        # Run on node01 ONLY to deploy both node01 & node02
```

---

## Quick Start (Master-Driven from `node01`)

You only need to run commands on **`node01`** (`192.168.9.136`) over Tailscale SSH—`node01` automatically provisions and starts **`node02`** (`192.168.9.135`):

```bash
# 1. Clone this repository on node01
git clone <your-repo-url> ~/labai-cluster
cd ~/labai-cluster

# 2. One-time passwordless SSH setup from node01 -> node02
bash scripts/setup_worker_ssh.sh 192.168.9.135 cluster

# 3. Deploy the entire 2-node cluster from node01 (builds llama.cpp, downloads models, starts services)
bash scripts/deploy_cluster.sh
```

Once deployed:
* **Chat UI (Caddy / Open WebUI):** `http://<node01-ip-or-tailscale>:80` (or `:3000`)
* **Grafana Telemetry:** `http://<node01-ip-or-tailscale>:3001`
* **Gateway API / Metrics:** `http://<node01-ip-or-tailscale>:8000/v1/models` and `/metrics`

---

## Documentation

* **[Full System & Infrastructure Architecture](docs/ARCHITECTURE.md)** — Hardware bottleneck analysis, model/runtime trade-offs, Mermaid diagrams, memory budgets, and phased rollout details.
