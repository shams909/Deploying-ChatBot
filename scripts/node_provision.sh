#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:-}"
if [[ "$ROLE" != "master" && "$ROLE" != "worker" ]]; then
  echo "Usage: $0 <master|worker>"
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="/opt/labai/models"
LLAMA_DIR="/opt/llama.cpp"

echo "==> [$(hostname)] Provisioning role: $ROLE"

sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
  git cmake build-essential pkg-config libcurl4-openssl-dev curl ca-certificates rsync

if [[ "$ROLE" == "master" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "==> Installing Docker + Compose plugin on master..."
    sudo apt-get install -y docker.io docker-compose-v2
    sudo systemctl enable --now docker
    sudo usermod -aG docker "$USER" || true
  fi
else
  echo "==> Installing Prometheus Node Exporter on worker..."
  sudo apt-get install -y prometheus-node-exporter
  # Pin node exporter to E-cores (16-19)
  sudo mkdir -p /etc/systemd/system/prometheus-node-exporter.service.d
  cat <<'EOF' | sudo tee /etc/systemd/system/prometheus-node-exporter.service.d/override.conf >/dev/null
[Service]
CPUAffinity=16 17 18 19
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable --now prometheus-node-exporter
fi

echo "==> Tuning vm.swappiness=1 for model residency"
echo "vm.swappiness=1" | sudo tee /etc/sysctl.d/99-labai.conf >/dev/null
sudo sysctl --system >/dev/null

echo "==> Building llama.cpp natively (-DGGML_NATIVE=ON)..."
if [ ! -d "$LLAMA_DIR/.git" ]; then
  sudo mkdir -p "$LLAMA_DIR"
  sudo chown -R "$USER:$USER" "$LLAMA_DIR"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
else
  git -C "$LLAMA_DIR" pull --ff-only || true
fi

cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_CURL=ON
cmake --build "$LLAMA_DIR/build" --config Release --target llama-server -j "$(nproc)"

echo "==> Preparing model directory: $MODEL_DIR"
sudo mkdir -p "$MODEL_DIR"
sudo chown -R "$USER:$USER" /opt/labai

download_model() {
  local repo="$1"
  local file="$2"
  local dest="$MODEL_DIR/$file"
  local url="https://huggingface.co/${repo}/resolve/main/${file}"
  echo "==> Ensuring model: $file"
  curl -fL --retry 5 --retry-delay 5 -C - -o "$dest" "$url"
}

if [[ "$ROLE" == "master" ]]; then
  download_model "Qwen/Qwen3-4B-GGUF" "Qwen3-4B-Q4_K_M.gguf"
  download_model "Qwen/Qwen3-8B-GGUF" "Qwen3-8B-Q4_K_M.gguf"

  sudo cp "$REPO_DIR/systemd/llama-tier1.service" /etc/systemd/system/
  sudo cp "$REPO_DIR/systemd/llama-tier2.service" /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable --now llama-tier1 llama-tier2
else
  download_model "Qwen/Qwen3-14B-GGUF" "Qwen3-14B-Q4_K_M.gguf"
  download_model "Qwen/Qwen3-0.6B-GGUF" "Qwen3-0.6B-Q8_0.gguf"

  sudo cp "$REPO_DIR/systemd/llama-tier3.service" /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable --now llama-tier3
fi

echo "==> [$(hostname)] Provisioning complete for role: $ROLE"
