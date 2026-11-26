# A-to-Z Engineering & MLOps Concepts Deep Dive

This guide explains every systems, networking, hardware, and machine learning inference concept implemented in **LabAI**—why each choice exists, how it works under the hood, and how it maps to real code in this repository.

---

## 1. Inference Physics & Hardware Mastery

### 1.1 Memory-Bandwidth Roofline (Why CPU Token Speed Has a Hard Ceiling)
LLM inference has two completely different phases:
1. **Prefill (Prompt Evaluation):** Processes all input tokens in parallel (`matrix × matrix`). This is **compute-bound**—your `i7-12700` P-cores spike to 100% and chew through prompts at ~25–120 tok/s depending on model size.
2. **Decode (Token Generation):** Generates one token at a time (`matrix × vector`). Every generated token requires streaming the *entire* quantized model weights from DDR5 RAM into CPU registers. This is **memory-bandwidth bound**:

$$\text{Max Tokens/sec} \approx \frac{\text{Sustained RAM Bandwidth (GB/s)}}{\text{Active Model Footprint (GB)}}$$

Because `node01` and `node02` each run **Single-Channel DDR5-4800** (`1× 16 GB` DIMM, `38.4 GB/s` theoretical peak, ~28–32 GB/s practical), our measured speeds match physics almost exactly:
* `Qwen3-4B (Q4_K_M, ~2.5 GB)` → **~13.45 tok/s**
* `Qwen3-14B (Q4_K_M, ~9.0 GB + 0.6B draft)` → **~3.93 tok/s**

### 1.2 Asymmetric CPU Pinning (Alder Lake P-Cores vs. E-Cores + SMT Avoidance)
The Intel `i7-12700` is a **hybrid CPU** (`lscpu -e` verified):
* **CPUs `0–15` (Cores `0–7`, 4.8–4.9 GHz):** 8 Performance cores with HyperThreading (`(0,1)`, `(2,3)`, ..., `(14,15)`).
* **CPUs `16–19` (Cores `8–11`, 3.6 GHz):** 4 Efficiency cores.

Two critical pitfalls avoided in `systemd/*.service` and `deploy/docker-compose.yml`:
1. **Barrier Synchronization Stalls:** Running `llama-server` across all 20 threads causes fast P-cores to wait at OpenMP/GGML thread barriers for slower E-cores.
2. **SMT Contention:** Pinning 8 threads to `0-7` would pack 8 threads onto only 4 physical cores! Instead, `CPUAffinity=0 2 4 6 8 10 12 14` places **1 thread per physical P-core**, while all Docker containers (`Open WebUI`, `Gateway`, `Redis`, `SearXNG`, `Caddy`, `Prometheus`, `Grafana`) run isolated on E-cores (`cpuset: "16-19"`).

### 1.3 `Q4_K_M` Quantization, FlashAttention & `q8_0` KV-Cache
* **`Q4_K_M` (4-bit K-quant medium):** Compresses weights from 16-bit (~28 GB for 14B) down to ~4.8 bits/weight (~9.0 GB) with minimal perplexity loss.
* **KV-Cache Quantization (`-ctk q8_0 -ctv q8_0` + `--flash-attn on`):** As chat history grows (`--ctx-size 8192`), Key-Value attention tensors consume gigabytes of RAM. Quantizing K and V caches to 8-bit (`q8_0`) cuts KV memory nearly in half—requiring FlashAttention (`--flash-attn on`) in `llama.cpp`.
* **Prefix Cache Reuse (`--cache-reuse 256`):** Avoids recomputing shared system prompts and earlier turns from scratch when continuing a chat.

### 1.4 Speculative Decoding (`node02` Tier 3)
In `systemd/llama-tier3.service`, `Qwen3-14B` is paired with `Qwen3-0.6B-Q8_0` (`--spec-draft-model`, `--spec-draft-n-max 8`, `--spec-draft-p-min 0.5`).
The tiny `0.6B` model rapidly guesses up to 8 candidate tokens, and `14B` verifies them in a **single parallel batch pass**—reusing the 9 GB weight read from RAM across multiple tokens and pushing `14B` up to **~3.93 tok/s**.

### 1.5 Native Compilation & Swap Discipline
* `cmake -DGGML_NATIVE=ON` compiles `ggml-cpu` with `-march=native` (`AVX2` + `AVX-VNNI` on Alder Lake).
* `vm.swappiness=1` prevents the Linux kernel from paging hot GGUF weights onto NVMe SSD.

---

## 2. Distributed Systems & AI Control Plane (`gateway/`)

### 2.1 Why Replication + Routing Beats Model Splitting on 100 Mbps LAN
`ethtool` & `iperf3` showed `2.5GbE` NICs capped at **`100 Mbps` (`~11.8 MB/s`)** by the lab switch. Splitting one model across nodes (`llama.cpp` RPC) stalls prompt prefill by ~1.5–2 seconds per stage over 100 Mbps. Keeping `4B` + `8B` self-contained on `node01` and `14B` on `node02` means the 100 Mbps link only carries tiny JSON/SSE tokens (~0.8 Mbps even at 200 users).

### 2.2 3-Stage Auto Router (`gateway/app/router.py`)
Requests to `labai-auto` pass through three stages:
1. **Stage 0 — Internal Task Fast-Path:** Detects Open WebUI background `### Task:` calls (chat title, tags, follow-ups), bypasses user locks/caches, routes to `Fast (4B)` with `/no_think`, and omits UI badges so sidebar titles stay clean.
2. **Stage 1 + 2 — Heuristic + ONNX Semantic Complexity Scoring:** Combines regex/structural signals (code blocks, stack traces, `"architect"`, `"debug"`, prompt/history length, diagram intent) with local CPU ONNX embeddings (`BAAI/bge-small-en-v1.5` via `fastembed`, 60/40 blend) to classify into `Fast (<0.35)`, `Reasoning (0.35–0.71)`, or `Pro (>=0.72)`.
3. **Stage 3 — Load-Aware Circuit Breaker:** Before sending to `node02`, `BackendManager` checks live `/health`, failure cooldown, and `node2_queue_depth`. If `node02` is busy (`> 2` queued) or offline, requests degrade gracefully to `Reasoning (8B)` on `node01`.

### 2.3 Dynamic Context-Aware Steering & Token Budgeting
* **Dynamic `max_tokens`:** Scales between `768` (`quick`), `2048` (`standard`), and `4096` (`deep`) so simple questions stay snappy while long code refactors never truncate.
* **Qwen3 Hybrid Thinking (`/think` vs `/no_think`):** Automatically enables chain-of-thought reasoning only on `deep` queries routed to `8B`/`14B`.
* **Adaptive Persona & History Hygiene:** Anchors natural conversational tone + English default on ambiguous slang (`"yo"`), injects explicit Mermaid double-quote rules when diagrams are requested, and strips previous routing badges from assistant history so small models never echo them.

### 2.4 Redis Exact + Semantic Cache & Sliding Quotas (`gateway/app/cache.py`)
* **Two-Tier Cache:** Single-turn queries check SHA-256 exact match first, then cosine similarity against a ring buffer of `bge-small` embeddings in Redis (`threshold >= 0.88`), returning hits in `<50 ms` with 0% LLM CPU load.
* **Fairness Controls:** Atomic Redis `INCR`/`DECR` limits active streams per user (`max_active = 2`) alongside a sliding-window token bucket (`10,000` output tokens / 15 min), keyed by `X-OpenWebUI-User-Id`.
* **Node 01 Inference Semaphore:** Serializes `Tier 1 (4B)` and `Tier 2 (8B)` via `asyncio.Semaphore(1)` so both models never fight over single-channel DDR5 bandwidth simultaneously.

---

## 3. Networking, Streaming & Observability (`deploy/`)

* **Unbuffered SSE Streaming (`Caddyfile`):** `flush_interval -1` disables reverse-proxy response buffering so tokens appear word-by-word immediately instead of stalling until completion.
* **Self-Hosted Web Search (`SearXNG`):** Provides live web and image search (`formats: [html, json]`) with zero third-party API costs.
* **Telemetry (`Prometheus` + `Grafana`):** Scrapes `gateway:8000/metrics` (`TTFT`, `TPOT`, cache events, router choices), all 3 `llama-server` `/metrics`, and `node_exporter` on both hosts (`2s` scrape interval), displayed in the auto-provisioned `LabAI Cluster` dashboard.
