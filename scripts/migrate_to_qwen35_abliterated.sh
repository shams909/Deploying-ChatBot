#!/usr/bin/env bash
set -euo pipefail

# Master migration script — run ONLY on node01 (Master).
# 1. Syncs updated repo to node02
# 2. Downloads Huihui-Qwen3.5-2B/4B/9B-abliterated (+ mmproj) cleanly via HF API
# 3. Deletes ALL old Qwen3-*.gguf models from BOTH node01 and node02
# 4. Updates systemd services on both nodes, enables Vision, and rebuilds gateway
#    (Preserves Open WebUI Admin/users/chats, Grafana, and long-term user memories!)

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER_IP="${WORKER_IP:-192.168.9.135}"
WORKER_USER="${WORKER_USER:-cluster}"
REMOTE_DIR="/home/${WORKER_USER}/labai-cluster"
MODEL_DIR="/opt/labai/models"

if [[ "${1:-}" == "--worker-exec" ]]; then
  # Executed remotely on node02
  mkdir -p "$MODEL_DIR"
  python3 - <<'PY'
import json, os, subprocess, urllib.request

repo = "mradermacher/Huihui-Qwen3.5-9B-abliterated-GGUF"
with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/main", timeout=20) as r:
    entries = {item["path"]: item.get("size", 0) for item in json.load(r) if item.get("type") == "file"}

files = list(entries.keys())
model_file = next((f for f in files if f.endswith(".Q4_K_M.gguf") and "mmproj" not in f.lower()), None)
mmproj_file = next(
    (f for f in files if "mmproj" in f.lower() and f.lower().endswith("q8_0.gguf")),
    next((f for f in files if "mmproj" in f.lower() and f.lower().endswith(("f16.gguf", "bf16.gguf"))), None),
)
if not model_file or not mmproj_file:
    raise SystemExit(f"Could not find Q4_K_M + mmproj in {repo}: {files}")

for fname, dest in [(model_file, "/opt/labai/models/tier3-pro.gguf"), (mmproj_file, "/opt/labai/models/tier3-mmproj.gguf")]:
    expected = entries.get(fname, 0)
    if os.path.exists(dest) and expected and os.path.getsize(dest) == expected:
        print(f"==> [node02] Already complete, skipping: {fname}")
        continue
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    print(f"==> [node02] Downloading/resuming {fname} -> {dest}")
    subprocess.run(["curl", "-fL", "--retry", "10", "--retry-all-errors", "--retry-delay", "5", "-C", "-", "-o", dest, url], check=True)
PY
  echo "==> [node02] Stopping llama-tier3 and deleting old Qwen3-*.gguf models..."
  sudo systemctl stop llama-tier3 || true
  rm -f "${MODEL_DIR}"/Qwen3-*.gguf
  sudo cp "${REPO_DIR}/systemd/llama-tier3.service" /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable --now llama-tier3
  echo "==> [node02] Disk & models:"
  ls -lh "$MODEL_DIR"
  exit 0
fi

echo "============================================================"
echo " Migrating Cluster to Huihui-Qwen3.5 Abliterated (2B/4B/9B)"
echo "============================================================"

echo "==> [1/4] Syncing repo to worker (${WORKER_IP})..."
ssh "${WORKER_USER}@${WORKER_IP}" "mkdir -p ${REMOTE_DIR}"
rsync -az --delete \
  --exclude '.git' --exclude '__pycache__' --exclude '.venv' \
  "${REPO_DIR}/" "${WORKER_USER}@${WORKER_IP}:${REMOTE_DIR}/"

if [[ "${1:-}" != "--master-only" ]]; then
  echo "==> [2/4] Updating node02 (Qwen3.5-9B-abliterated + mmproj, deleting old Qwen3)..."
  ssh -t "${WORKER_USER}@${WORKER_IP}" "bash ${REMOTE_DIR}/scripts/migrate_to_qwen35_abliterated.sh --worker-exec"
else
  echo "==> [2/4] Skipping node02 (--master-only specified)."
fi

echo "==> [3/4] Downloading Qwen3.5-2B & 4B-abliterated (+ mmproj) on node01..."
mkdir -p "$MODEL_DIR"
python3 - <<'PY'
import json, os, subprocess, urllib.request

def download_if_needed(repo, fname, expected, dest):
    if os.path.exists(dest) and expected and os.path.getsize(dest) == expected:
        print(f"==> [node01] Already complete, skipping: {fname}")
        return
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    print(f"==> [node01] Downloading/resuming {fname} -> {dest}")
    subprocess.run(["curl", "-fL", "--retry", "10", "--retry-all-errors", "--retry-delay", "5", "-C", "-", "-o", dest, url], check=True)

def resolve_and_pull(repo, dest_model, dest_mmproj=None):
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/main", timeout=20) as r:
        entries = {item["path"]: item.get("size", 0) for item in json.load(r) if item.get("type") == "file"}
    files = list(entries.keys())
    model_file = next((f for f in files if f.endswith(".Q4_K_M.gguf") and "mmproj" not in f.lower()), None)
    if not model_file:
        raise SystemExit(f"Could not find Q4_K_M in {repo}: {files}")
    download_if_needed(repo, model_file, entries.get(model_file, 0), dest_model)
    if dest_mmproj:
        mmproj_file = next(
            (f for f in files if "mmproj" in f.lower() and f.lower().endswith("q8_0.gguf")),
            next((f for f in files if "mmproj" in f.lower() and f.lower().endswith(("f16.gguf", "bf16.gguf"))), None),
        )
        if not mmproj_file:
            raise SystemExit(f"Could not find mmproj in {repo}: {files}")
        download_if_needed(repo, mmproj_file, entries.get(mmproj_file, 0), dest_mmproj)

resolve_and_pull("mradermacher/Huihui-Qwen3.5-2B-abliterated-GGUF", "/opt/labai/models/tier1-fast.gguf")
resolve_and_pull("mradermacher/Huihui-Qwen3.5-4B-abliterated-GGUF", "/opt/labai/models/tier2-reasoning.gguf", "/opt/labai/models/tier2-mmproj.gguf")
PY

echo "==> [4/4] Removing old Qwen3-*.gguf models, enabling Vision, and restarting services..."
sudo systemctl stop llama-tier1 llama-tier2 || true
rm -f "${MODEL_DIR}"/Qwen3-*.gguf
sudo cp "${REPO_DIR}/systemd/llama-tier1.service" /etc/systemd/system/
sudo cp "${REPO_DIR}/systemd/llama-tier2.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-tier1 llama-tier2

ENV_FILE="${REPO_DIR}/deploy/.env"
if [ -f "$ENV_FILE" ]; then
  if grep -q "^LABAI_VISION_ENABLED=" "$ENV_FILE"; then
    sed -i 's/^LABAI_VISION_ENABLED=.*/LABAI_VISION_ENABLED=true/' "$ENV_FILE"
  else
    echo "LABAI_VISION_ENABLED=true" >> "$ENV_FILE"
  fi
fi

# Clear only stale response caches; preserve user long-term memories (labai:memory:*) & Open WebUI DB
sudo docker exec labai-redis sh -c 'redis-cli --scan --pattern "labai:cache:*" | xargs -r redis-cli DEL' || true
cd "${REPO_DIR}/deploy"
sudo docker compose up -d --build gateway

echo "==> Waiting for models to load into RAM..."
sleep 12
curl -s http://127.0.0.1:8000/healthz | python3 -m json.tool
echo "=== node01 models ==="
ls -lh "$MODEL_DIR"
df -h /opt/labai/models
