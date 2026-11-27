#!/usr/bin/env bash
set -euo pipefail

# Reversible 1-command plug/unplug swap for node02 (:8083) between:
#   - qwen   : Huihui-Qwen3.5-9B-abliterated (+mmproj vision, coding/Pro IQ)
#   - stheno : L3-8B-Stheno-v3.2 Q4_K_M (purpose-built RP / AI-GF model)
# Both GGUFs stay on disk (/opt/labai/models) so swapping back takes ~5 seconds.

TARGET="${1:-}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="/opt/labai/models"
SERVICE_PATH="/etc/systemd/system/llama-tier3.service"
STHENO_FILE="${MODEL_DIR}/stheno-8b.gguf"
STHENO_URL="https://huggingface.co/bartowski/L3-8B-Stheno-v3.2-GGUF/resolve/main/L3-8B-Stheno-v3.2-Q4_K_M.gguf"

if [[ "$TARGET" != "stheno" && "$TARGET" != "qwen" ]]; then
  echo "Usage: bash scripts/swap_node02_model.sh [stheno|qwen]"
  exit 1
fi

if [[ "$TARGET" == "qwen" ]]; then
  if [[ ! -f "${MODEL_DIR}/tier3-pro.gguf" ]]; then
    echo "Error: ${MODEL_DIR}/tier3-pro.gguf not found on node02."
    exit 1
  fi
  echo "==> Plugging Qwen3.5-9B-Abliterated (+Vision) into node02 (:8083)..."
  sudo cp "${REPO_DIR}/systemd/llama-tier3.service" "$SERVICE_PATH"
else
  mkdir -p "$MODEL_DIR"
  if [[ ! -f "$STHENO_FILE" || $(stat -c%s "$STHENO_FILE") -lt 4000000000 ]]; then
    echo "==> Downloading bartowski/L3-8B-Stheno-v3.2-Q4_K_M.gguf (~4.9 GB)..."
    curl -fL --retry 3 -o "${STHENO_FILE}.part" "$STHENO_URL"
    mv "${STHENO_FILE}.part" "$STHENO_FILE"
  else
    echo "==> Using cached ${STHENO_FILE}"
  fi
  echo "==> Plugging L3-8B-Stheno-v3.2 into node02 (:8083)..."
  sudo tee "$SERVICE_PATH" > /dev/null <<'EOF'
[Unit]
Description=LabAI Tier 3 RP Mode (L3-8B-Stheno-v3.2)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=cluster
Group=cluster
CPUAffinity=0 2 4 6 8 10 12 14
Nice=-5
ExecStart=/opt/llama.cpp/build/bin/llama-server \
  --model /opt/labai/models/stheno-8b.gguf \
  --alias labai-pro \
  --host 0.0.0.0 \
  --port 8083 \
  --threads 8 \
  --threads-batch 8 \
  --ctx-size 8192 \
  --parallel 1 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --cache-reuse 256 \
  --jinja \
  --metrics
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
fi

sudo systemctl daemon-reload
sudo systemctl restart llama-tier3
echo "==> Waiting for node02 (:8083) health..."
for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8083/health >/dev/null 2>&1; then
    echo "==> Ready! node02 (:8083) is now running: ${TARGET}"
    curl -s http://127.0.0.1:8083/health && echo ""
    exit 0
  fi
  sleep 2
done
echo "Warning: node02 health check timed out; check 'sudo journalctl -u llama-tier3 -n 30'"
exit 1
