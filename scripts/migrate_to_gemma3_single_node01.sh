#!/usr/bin/env bash
set -euo pipefail

# Clean migration to solo Gemma-3-4B-IT-Abliterated on node01 + Qwen3.5-9B on node02,
# removing all old node01 models and resetting Redis + OpenWebUI for a completely fresh start.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="/opt/labai/models"

echo "============================================================"
echo " Migrating node01 -> Solo Gemma-3-4B-IT-Abliterated (+Vision)"
echo " + Deleting old node01 models & nuclear OpenWebUI/Redis reset"
echo "============================================================"

echo "==> [1/5] Stopping & disabling llama-tier1 (eliminating 2-model RAM pressure)..."
sudo systemctl disable --now llama-tier1 || true
sudo systemctl stop llama-tier2 || true
sudo rm -f /etc/systemd/system/llama-tier1.service

echo "==> [2/5] Deleting ALL old models on node01 (node02 Qwen3.5-9B remains untouched)..."
mkdir -p "$MODEL_DIR"
rm -f "${MODEL_DIR}"/tier1-fast.gguf "${MODEL_DIR}"/tier2-reasoning.gguf "${MODEL_DIR}"/tier2-mmproj.gguf "${MODEL_DIR}"/Qwen3-*.gguf

echo "==> [3/5] Downloading Gemma-3-4B-IT-Abliterated (Q4_K_M + mmproj) to node01..."
python3 - <<'PY'
import json, os, subprocess, urllib.request

repos = [
    "bartowski/mlabonne_gemma-3-4b-it-abliterated-GGUF",
    "mradermacher/gemma-3-4b-it-abliterated-GGUF",
]

model_url = mmproj_url = model_size = mmproj_size = None
for repo in repos:
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/main", timeout=20) as r:
            entries = {item["path"]: item.get("size", 0) for item in json.load(r) if item.get("type") == "file"}
        files = list(entries.keys())
        m = next((f for f in files if f.endswith(".Q4_K_M.gguf") and "mmproj" not in f.lower()), None)
        mm = next(
            (f for f in files if "mmproj" in f.lower() and f.lower().endswith(("q8_0.gguf", "f16.gguf", "bf16.gguf"))),
            None,
        )
        if m and mm:
            model_url = f"https://huggingface.co/{repo}/resolve/main/{m}"
            mmproj_url = f"https://huggingface.co/{repo}/resolve/main/{mm}"
            model_size = entries.get(m, 0)
            mmproj_size = entries.get(mm, 0)
            print(f"==> Selected {repo}: {m} + {mm}")
            break
    except Exception as exc:
        print(f"==> Repo check skipped ({repo}): {exc}")

if not model_url or not mmproj_url:
    raise SystemExit("Could not resolve Gemma-3-4B-IT-Abliterated Q4_K_M + mmproj on Hugging Face.")

for url, expected, dest in [
    (model_url, model_size, "/opt/labai/models/tier2-reasoning.gguf"),
    (mmproj_url, mmproj_size, "/opt/labai/models/tier2-mmproj.gguf"),
]:
    print(f"==> Downloading {url} -> {dest}")
    subprocess.run(["curl", "-fL", "--retry", "10", "--retry-all-errors", "--retry-delay", "5", "-o", dest, url], check=True)
    if expected and os.path.getsize(dest) != expected:
        raise SystemExit(f"Size mismatch for {dest}: got {os.path.getsize(dest)}, expected {expected}")
PY

echo "==> [4/5] Installing updated llama-tier2.service (--parallel 3 --ctx-size 12288)..."
sudo cp "${REPO_DIR}/systemd/llama-tier2.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-tier2

ENV_FILE="${REPO_DIR}/deploy/.env"
if [ -f "$ENV_FILE" ]; then
  sed -i '/^LABAI_TIER1_URL=/d;/^LABAI_TIER2_URL=/d;/^LABAI_TIER1_VISION_ENABLED=/d;/^LABAI_VISION_ENABLED=/d;/^LABAI_NODE1_MAX_CONCURRENT=/d' "$ENV_FILE"
  {
    echo "LABAI_TIER1_URL=http://host.docker.internal:8082"
    echo "LABAI_TIER2_URL=http://host.docker.internal:8082"
    echo "LABAI_VISION_ENABLED=true"
    echo "LABAI_TIER1_VISION_ENABLED=false"
    echo "LABAI_NODE1_MAX_CONCURRENT=3"
  } >> "$ENV_FILE"
fi

echo "==> [5/5] Nuclear Reset (Redis FLUSHDB + wipe openwebui_data + rebuild & start)..."
cd "${REPO_DIR}/deploy"
(sudo docker exec labai-redis redis-cli FLUSHDB || true)
sudo docker compose down
sudo docker volume ls -q --filter name=openwebui_data | xargs -r sudo docker volume rm
sudo docker compose up -d --build

echo "==> Waiting for Gemma-3-4B & OpenWebUI to come up..."
sleep 15
free -h
curl -s http://127.0.0.1:8000/healthz
echo ""
echo "Done! Open http://100.64.166.2/ (Incognito / Ctrl+Shift+R) to create your fresh admin account."
