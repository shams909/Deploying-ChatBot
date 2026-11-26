# LabAI — Complete A-to-Z Project Documentation & Operations Guide

This is the master guide covering **where every file lives**, **what we built**, **why every technical choice was made**, **the exact build sequence ("which one after which one")**, **how requests flow end-to-end**, and **how to operate/maintain the cluster** from home over Tailscale SSH.

---

## 1. Where Everything Lives (Repository & Server Map)

### 1.1 Git Repository Layout (`Deploying-ChatBot/`)

| Path | Role | Runs On |
| :--- | :--- | :--- |
| `README.md` | Project overview, quick-start commands, doc links | GitHub / Docs |
| `docs/ARCHITECTURE.md` | System & infra architecture, Mermaid diagrams, memory budget | GitHub / Docs |
| `docs/DEEP_DIVE_CONCEPTS.md` | Technical theory (bandwidth roofline, P/E cores, KV cache, speculative decoding) | GitHub / Docs |
| `docs/COMPLETE_GUIDE.md` | Full A-to-Z map, build sequence, rationale, and operations runbook | GitHub / Docs |
| `scripts/setup_worker_ssh.sh` | One-time Ed25519 SSH key setup (`node01 -> node02`) | `node01` |
| `scripts/node_provision.sh` | Installs deps, builds `llama.cpp` (`-DGGML_NATIVE=ON`), downloads GGUFs, enables systemd units | `node01` & `node02` |
| `scripts/deploy_cluster.sh` | **Master orchestrator** — syncs files to `node02`, provisions both nodes, starts Docker stack, health-checks all endpoints | `node01` (controls both) |
| `systemd/llama-tier1.service` | `Qwen3-4B-Q4_K_M` (`:8081`, `-np 2`, P-cores `0,2,4,6,8,10,12,14`) | `node01` |
| `systemd/llama-tier2.service` | `Qwen3-8B-Q4_K_M` (`:8082`, `-np 1`, P-cores `0,2,4,6,8,10,12,14`) | `node01` |
| `systemd/llama-tier3.service` | `Qwen3-14B-Q4_K_M` + `Qwen3-0.6B-Q8_0` draft (`:8083`, `-np 1`, P-cores) | `node02` |
| `gateway/app/config.py` | Environment settings (`LABAI_*`), cache thresholds, token budgets, quotas | `node01` (Docker) |
| `gateway/app/router.py` | Internal-task fast-path, 3-stage Auto Router, dynamic `max_tokens`, `/think` & Mermaid steering | `node01` (Docker) |
| `gateway/app/cache.py` | Redis exact + ONNX semantic cache (`bge-small-en-v1.5`), active-slot `INCR`/`DECR`, sliding token quota | `node01` (Docker) |
| `gateway/app/backends.py` | Background health poller, Node 01 inference `Semaphore(1)`, Node 02 circuit breaker | `node01` (Docker) |
| `gateway/app/main.py` | FastAPI OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`, `/metrics`, `/v1/tools/image-search`) | `node01` (Docker) |
| `gateway/tests/test_router.py` | Pytest unit tests for routing, dynamic budgets, diagram hints, and badge stripping | CI / Local |
| `deploy/docker-compose.yml` | E-core pinned (`16-19`) containers: Caddy, Open WebUI, Gateway, Redis, SearXNG, Node Exporter, Prometheus, Grafana | `node01` (Docker) |
| `deploy/Caddyfile` | Reverse proxy on `:80` with `flush_interval -1` for zero-buffering SSE streaming | `node01` (Docker) |
| `deploy/prometheus.yml` | Scrapes Gateway, Tier 1/2/3 `llama-server` `/metrics`, and both `node_exporter`s every `2s` | `node01` (Docker) |
| `deploy/searxng/settings.yml` | Enables `json` output format for Open WebUI & Gateway web/image search | `node01` (Docker) |
| `deploy/grafana/provisioning/` | Auto-provisions Prometheus datasource (`uid: prometheus`) & 11-panel `LabAI Cluster` dashboard | `node01` (Docker) |
| `deploy/.env.example` | Safe template for secrets (real `deploy/.env` is gitignored and lives only on `node01`) | Template |

### 1.2 Server Filesystem & Ports (`node01` & `node02`)

| Location | Contents | Node |
| :--- | :--- | :--- |
| `/home/cluster/Deploying-ChatBot/` | Master git repository & Docker Compose stack | `node01` |
| `/home/cluster/labai-cluster/` | Synced copy pushed from `node01` via `rsync` | `node02` |
| `/opt/llama.cpp/build/bin/llama-server` | Natively compiled (`-DGGML_NATIVE=ON`) inference binary | Both |
| `/opt/labai/models/` | `Qwen3-4B` + `Qwen3-8B` (`node01`) / `Qwen3-14B` + `Qwen3-0.6B` (`node02`) | Both |
| `/etc/systemd/system/llama-tier*.service` | Bare-metal inference `systemd` services | Both |
| `/etc/sysctl.d/99-labai.conf` | `vm.swappiness=1` kernel tuning | Both |
| Docker volume `deploy_openwebui_data` | Open WebUI SQLite DB (`webui.db`: users, logins, chat histories, settings) | `node01` |
| Docker volume `deploy_prometheus_data` | Time-series metrics storage | `node01` |
| Docker volume `deploy_grafana_data` | Grafana DB & user state | `node01` |

**Active Ports:**
* `node01`: `:80` (Caddy → Open WebUI), `:3000` (Open WebUI direct), `:3001` (Grafana), `:8000` (FastAPI Gateway), `:8081` (`Qwen3-4B`), `:8082` (`Qwen3-8B`).
* `node02`: `:8083` (`Qwen3-14B` + `0.6B` draft), `:9100` (`prometheus-node-exporter`).

---

## 2. What We Built & Why Every Decision Was Made

1. **Hardware Discovery First (Before Writing Code):**
   * `nvidia-smi` / PyTorch introspection showed **no GPU** (`cuda_available: false`), so GPU engines (`vLLM`, `SGLang`) were ruled out immediately.
   * `dmidecode` revealed **`1× 16 GB DDR5-4800` (Single-Channel)** per node (`38.4 GB/s` theoretical bandwidth), which caps CPU token decode speed to `Bandwidth / Model_Size`.
   * `ethtool` + `iperf3` revealed **`2.5GbE` NICs capped at `100 Mbps`** by the lab switch (`94 Mbps`, `0.44 ms` RTT). Splitting a single model across nodes (`llama.cpp` RPC) would stall prefill over `100 Mbps`, so we kept each model self-contained on one node and used `100 Mbps` only for tiny HTTP/SSE traffic.
   * `lscpu -e` revealed Alder Lake's exact CPU numbering: **8 P-cores (`0–15` paired HT)** + **4 E-cores (`16–19`)**. Pinning `CPUAffinity=0 2 4 6 8 10 12 14` gives `llama-server` 1 thread per physical P-core (avoiding both E-core barrier stalls and HyperThreading contention), while Docker infra runs isolated on E-cores (`16-19`).
2. **Why `Qwen3` (`4B` / `8B` / `14B`) + `0.6B` Draft:**
   * Exact fit for `2× 14.88 GB` RAM (`13.14 GB` available with Ubuntu Desktop running), strong coding/math/Mermaid generation, native `/think` vs `/no_think` switching, and speculative decoding (`0.6B` draft) boosting `14B` to `~3.93 tok/s`.
   * **Important:** Model weights are **100% official untouched Qwen3 GGUFs**—no fine-tuning or weight modification was performed. Runtime behavior is steered purely via system directives in `gateway/app/router.py` (tone/English anchor, dynamic conciseness/depth, `/think`, and Mermaid double-quoting).
3. **Why Native `llama.cpp` + `systemd` + `Docker Compose` (Instead of Ollama or `k8s`/`k3s`):**
   * Ollama wraps `llama.cpp` without exposing fine-grained P-core affinity, speculative draft flags, or Prometheus metrics as cleanly.
   * `k8s` (`~1.5–2.5 GB/node`) or `k3s` (`~500 MB/node`) wastes precious RAM on a static 2-node cluster. `systemd` uses `< 50 MB` for inference supervision and `Docker Compose` isolates the Control Plane cleanly on E-cores.
4. **Why Custom FastAPI AI Gateway + Open WebUI + SearXNG:**
   * **Open WebUI** gives ChatGPT-grade UX (accounts, history, code highlighting, native Mermaid SVG rendering, RAG, SearXNG web search) out of the box.
   * **FastAPI Gateway** adds real AI infra intelligence: Open WebUI `### Task:` detection, Redis exact + ONNX semantic cache, 3-stage Auto Router, dynamic response length (`768`/`2048`/`4096`), Node 01 inference semaphore, Node 02 circuit breaker, and Prometheus `/metrics`.

---

## 3. Exact Build Order ("Which One After Which One")

1. **Step 1 — Hardware & OS Introspection (`node01` & `node02` via Tailscale SSH):**
   Verified GPU absence, RAM channels (`dmidecode`), NIC/switch speed (`ethtool`), CPU P/E-core IDs (`lscpu -e`), and confirmed `13.14 GB` available RAM with Ubuntu Desktop sleep disabled.
2. **Step 2 — Master-to-Worker Passwordless SSH (`scripts/setup_worker_ssh.sh`):**
   Generated Ed25519 key on `node01` and copied it to `cluster@192.168.9.135` (`node02`) so all deployment happens from `node01` only.
3. **Step 3 — Worker Provisioning (`node02` via `scripts/deploy_cluster.sh`):**
   Installed `cmake` + build tools + `prometheus-node-exporter` (pinned to E-cores `16-19`), tuned `vm.swappiness=1`, compiled `llama.cpp` with `-DGGML_NATIVE=ON`, downloaded `Qwen3-14B-Q4_K_M.gguf` + `Qwen3-0.6B-Q8_0.gguf`, and started `llama-tier3.service` (`:8083`).
4. **Step 4 — Master Inference Provisioning (`node01`):**
   Installed Docker + Compose, compiled `llama.cpp` natively, downloaded `Qwen3-4B-Q4_K_M.gguf` + `Qwen3-8B-Q4_K_M.gguf`, and started `llama-tier1.service` (`:8081`) + `llama-tier2.service` (`:8082`).
5. **Step 5 — Control Plane, UI, Search & Monitoring (`deploy/docker-compose.yml`):**
   Built and started `Caddy`, `Open WebUI`, `FastAPI Gateway`, `Redis`, `SearXNG`, `node-exporter`, `Prometheus`, and `Grafana` all pinned to `cpuset: "16-19"`.
6. **Step 6 — Live Verification & Production Bug Fixes:**
   * Updated `llama-tier3.service` speculative flags (`--spec-draft-model`, `--spec-draft-n-max`, `--spec-draft-p-min`).
   * Enabled `cache_max_temperature = 0.85` so Open WebUI default requests hit Redis semantic cache (`HIT_SEMANTIC(0.96)`).
   * Enabled `ENABLE_FORWARD_USER_INFO_HEADERS=true` and added `is_internal_webui_task()` so Open WebUI's parallel title/tag requests never trigger `429` locks or pollute cache/titles.
   * Added badge-stripping in `apply_dynamic_steering()` and strengthened `MERMAID_SYSTEM_HINT` so parenthesis labels like `["Load Balancer (Nginx)"]` render cleanly.
   * Added `.gitignore` (protecting `deploy/.env` secrets) and `GF_DASHBOARDS_MIN_REFRESH_INTERVAL=1s` + `2s` Prometheus scraping for real-time Grafana telemetry.

---

## 4. End-to-End Request Lifecycle

1. User sends a prompt in **Open WebUI** (`:80` via Caddy, `flush_interval -1`).
2. Open WebUI forwards the request (plus `X-OpenWebUI-User-Id`) to **`gateway:8000/v1/chat/completions`**.
3. **Internal Task Check:** If it's an Open WebUI background `### Task:` (title/tags), it bypasses user locks & cache and goes straight to `Fast (4B)`.
4. **Quota & Concurrency Guard:** Redis checks per-user sliding token quota (`10,000` / 15 min) and active stream limit (`max 2`).
5. **Redis Exact + Semantic Cache:** On first-turn queries (`temperature <= 0.85`), checks SHA-256 exact match, then `BAAI/bge-small-en-v1.5` cosine similarity (`>= 0.88`). Hits return in `<50 ms`.
6. **3-Stage Auto Router (`labai-auto`):** Blends structural heuristics + ONNX complexity score (`0.0–1.0`) + `node02` live health/queue depth to pick `Fast (4B)`, `Reasoning (8B)`, or `Pro (14B)` (with automatic fallback from `node02` → `node01` if `node02` is down/busy), sets dynamic `max_tokens` (`768` / `2048` / `4096`), `/think` vs `/no_think`, and Mermaid rules.
7. **Execution & Telemetry:** `node01` requests acquire `asyncio.Semaphore(1)`; `node02` requests track queue depth & circuit breaker state. Gateway streams SSE tokens back and records `TTFT`, `TPOT`, tokens, and cache outcomes to Prometheus (`/metrics`), graphed live in Grafana (`:3001`).

---

## 5. Day-2 Operations Cheat Sheet (Run on `node01`)

* **Update gateway/config after editing on Windows & pushing to GitHub:**
  ```bash
  cd ~/Deploying-ChatBot && git pull && cd deploy && \
  sudo docker compose up -d --build gateway
  ```
* **Check all 3 model backends & cluster status:**
  ```bash
  curl -s http://127.0.0.1:8000/healthz | python3 -m json.tool
  ```
* **Check `llama-server` logs (`node01` or `node02`):**
  ```bash
  journalctl -u llama-tier1 -u llama-tier2 -n 30 --no-pager
  ssh cluster@192.168.9.135 "journalctl -u llama-tier3 -n 30 --no-pager"
  ```
* **Flush Redis semantic cache & active locks:**
  ```bash
  sudo docker exec labai-redis redis-cli FLUSHDB
  ```
* **Reset Grafana admin password:**
  ```bash
  sudo docker exec -it labai-grafana grafana cli admin reset-admin-password 'NewStrongPassword'
  ```
* **Backup Open WebUI database (users + all chat history):**
  ```bash
  sudo docker cp labai-openwebui:/app/backend/data/webui.db ~/webui-backup-$(date +%F).db
  ```
