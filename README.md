# LabAI: Production Multi-Tier LLM Platform on Bare-Metal CPU Nodes

A hardware-aware, self-hosted ChatGPT-class AI platform built to run **24/7 on two commodity CPU-only university lab desktops** (`2× Intel Core i7-12700`, `14.88 GB` single-channel DDR5-4800 each, `100 Mbps` Fast Ethernet).

Instead of wrapping a single model in a basic UI, **LabAI** implements a custom **AI Control Plane** featuring:
- **3-Tier Model Serving (`Fast` / `Reasoning` / `Pro`)** kept hot in RAM across both nodes.
- **3-Stage Auto Router** (Heuristics + Local ONNX Semantic Classifier + Live Queue-Aware Fallback) alongside manual tier selection.
- **Dynamic Context-Aware Response Budgeting** (short replies for quick questions, deep un-truncated code/analysis when needed).
- **Exact + Semantic Redis Cache** to absorb concurrent classroom/department bursts without CPU thrashing.
- **Asymmetric CPU Core Isolation** (`i7-12700` P-cores pinned to LLM inference; E-cores dedicated to web UI, gateway, search, and monitoring).
- **Live Web Search, Image Lookup & Interactive Diagrams** via self-hosted SearXNG and native Mermaid/SVG rendering.

---

## Documentation

- **[Full System & Infrastructure Architecture](docs/ARCHITECTURE.md)** — Hardware bottleneck analysis, model & runtime trade-offs, Mermaid system/sequence/infra diagrams, memory budgets, and the phased implementation roadmap.

---

## Cluster Hardware Snapshot

| Attribute | Master Node (`node01`) | Worker Node (`node02`) |
| :--- | :--- | :--- |
| **IP Address** | `192.168.9.136` | `192.168.9.135` |
| **CPU** | Intel Core i7-12700 (8 P-cores + 4 E-cores / 20T) | Intel Core i7-12700 (8 P-cores + 4 E-cores / 20T) |
| **GPU / CUDA** | None (`cuda_available: false`) | None (`cuda_available: false`) |
| **Memory** | `14.88 GB` DDR5-4800 (Single-Channel, 1×16 GB) | `14.88 GB` DDR5-4800 (Single-Channel, 1×16 GB) |
| **Storage** | `456.3 GB` SSD | `456.3 GB` SSD |
| **OS** | Ubuntu Desktop (`Linux 7.0.0-44-generic`, `x86_64`) | Ubuntu Desktop (`Linux 7.0.0-44-generic`, `x86_64`) |
| **NIC / Switch** | `2.5GbE` NIC negotiated at `100Mb/s` (`94.7 Mbps`) | `2.5GbE` NIC negotiated at `100Mb/s` (`94.0 Mbps`) |

---

## Implementation Phases

1. **Phase 0 — OS & Bare-Metal Tuning:** Switch Ubuntu Desktop to headless server target (`multi-user.target` to reclaim ~1.5 GB RAM), configure SSH keys, disable swap thrashing, and verify P-core vs. E-core CPU IDs.
2. **Phase 1 — Inference Engine (`llama.cpp`):** Build `llama-server` natively (`-DGGML_NATIVE=ON`), pull `Qwen3-4B`, `Qwen3-8B`, and `Qwen3-14B` (`Q4_K_M`), and run as `systemd` services pinned to P-cores.
3. **Phase 2 — Custom FastAPI AI Gateway:** Implement OpenAI-compatible SSE streaming, 3-stage Auto Router, dynamic token budgeting, Node 01 inference semaphore, Redis semantic cache, and Node 02 circuit breaker.
4. **Phase 3 — Chat UI, Diagrams & Web/Image Search:** Deploy Open WebUI + SearXNG + Redis + Caddy reverse proxy (SSE unbuffered) pinned to E-cores.
5. **Phase 4 — Observability & Exposure:** Wire Prometheus + Grafana + Node Exporter dashboards and expose securely via Cloudflare Tunnel (or campus VPN).
6. **Phase 5 — Load Testing & Hardening:** Stress-test with simulated department traffic, tune queue caps and cache thresholds, and automate backups/restarts.
