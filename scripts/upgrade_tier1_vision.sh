#!/usr/bin/env bash
set -euo pipefail

# Run on node01 (Master) to upgrade Tier 1 from text-only Qwen3-4B
# to multimodal Qwen3.5-4B + mmproj-F16 so uploaded screenshots work.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="/opt/labai/models"

mkdir -p "$MODEL_DIR"
echo "==> Downloading Qwen3.5-4B-Q4_K_M.gguf (~2.7 GB)..."
curl -fL --retry 5 --retry-delay 5 -C - \
  -o "$MODEL_DIR/Qwen3.5-4B-Q4_K_M.gguf" \
  "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf"

echo "==> Downloading mmproj-F16.gguf (~676 MB)..."
curl -fL --retry 5 --retry-delay 5 -C - \
  -o "$MODEL_DIR/mmproj-F16.gguf" \
  "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/mmproj-F16.gguf"

echo "==> Updating llama-tier1.service to Qwen3.5-4B Vision..."
sudo cp "$REPO_DIR/systemd/llama-tier1-vision.service" /etc/systemd/system/llama-tier1.service
sudo systemctl daemon-reload
sudo systemctl restart llama-tier1

ENV_FILE="$REPO_DIR/deploy/.env"
if [ -f "$ENV_FILE" ]; then
  if grep -q "^LABAI_TIER1_VISION_ENABLED=" "$ENV_FILE"; then
    sed -i 's/^LABAI_TIER1_VISION_ENABLED=.*/LABAI_TIER1_VISION_ENABLED=true/' "$ENV_FILE"
  else
    echo "LABAI_TIER1_VISION_ENABLED=true" >> "$ENV_FILE"
  fi
fi

echo "==> Rebuilding gateway with Vision enabled..."
cd "$REPO_DIR/deploy"
sudo docker compose up -d --build gateway
sleep 5
curl -s http://127.0.0.1:8000/healthz | python3 -m json.tool
echo "Done! Tier 1 is now running Qwen3.5-4B Vision."
