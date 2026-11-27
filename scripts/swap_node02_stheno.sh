#!/usr/bin/env bash
set -euo pipefail

# Plug/unplug L3-8B-Stheno-v3.2 on node02 (:8083) while keeping Qwen3.5-9B on disk.
# Usage:
#   bash scripts/swap_node02_stheno.sh          # plug in L3-8B-Stheno-v3.2
#   bash scripts/swap_node02_stheno.sh qwen     # restore Qwen3.5-9B

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="/opt/labai/models"
STHENO_FILE="${MODEL_DIR}/stheno-8b.gguf"
STHENO_URL="https://huggingface.co/bartowski/L3-8B-Stheno-v3.2-GGUF/resolve/main/L3-8B-Stheno-v3.2-Q4_K_M.gguf"
MODE="${1:-stheno}"

echo "============================================================"
echo " Node02 Tier-3 Hot-Swap -> ${MODE}"
echo "============================================================"

if [[ "${MODE}" == "qwen" || "${MODE}" == "restore" ]]; then
  echo "==> [1/2] Restoring Qwen3.5-9B systemd unit..."
  sudo cp "${REPO_DIR}/systemd/llama-tier3.service" /etc/systemd/system/llama-tier3.service
  sudo systemctl daemon-reload
  sudo systemctl restart llama-tier3
else
  echo "==> [1/3] Checking / downloading L3-8B-Stheno-v3.2 (Q4_K_M ~4.9GB)..."
  sudo mkdir -p "${MODEL_DIR}"
  sudo chown -R "${USER}:${USER}" "${MODEL_DIR}"
  if [[ -f "${STHENO_FILE}" && $(stat -c%s "${STHENO_FILE}") -gt 1000000000 ]]; then
    echo "    Already present (${STHENO_FILE}), skipping download."
  else
    curl -fL --progress-bar -o "${STHENO_FILE}.part" "${STHENO_URL}"
    mv "${STHENO_FILE}.part" "${STHENO_FILE}"
  fi

  echo "==> [2/3] Installing llama-tier3-stheno.service (tier3-pro.gguf stays untouched on disk)..."
  sudo cp "${REPO_DIR}/systemd/llama-tier3-stheno.service" /etc/systemd/system/llama-tier3.service
  sudo systemctl daemon-reload
  sudo systemctl restart llama-tier3
fi

echo "==> Waiting for node02 :8083 health..."
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:8083/health >/dev/null 2>&1; then
    echo "OK: $(curl -s http://127.0.0.1:8083/health)"
    free -h
    exit 0
  fi
  sleep 2
done

echo "ERROR: node02 :8083 did not become healthy in time. Check: sudo journalctl -u llama-tier3 -n 40"
exit 1
