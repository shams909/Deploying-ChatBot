#!/usr/bin/env bash
set -euo pipefail

# Run ONLY on node01 (Master). Provisions node02 remotely over SSH,
# provisions node01 locally, starts the Docker Compose control plane,
# and verifies every service in the cluster.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER_IP="${WORKER_IP:-192.168.9.135}"
WORKER_USER="${WORKER_USER:-cluster}"
REMOTE_DIR="/home/${WORKER_USER}/labai-cluster"

echo "============================================================"
echo " LabAI Master-Driven Cluster Deployment"
echo " Master: $(hostname) (local) | Worker: ${WORKER_USER}@${WORKER_IP}"
echo "============================================================"

if ! ssh -o BatchMode=yes "${WORKER_USER}@${WORKER_IP}" "true" 2>/dev/null; then
  echo "ERROR: Passwordless SSH to ${WORKER_USER}@${WORKER_IP} is not configured."
  echo "Run once: bash ${REPO_DIR}/scripts/setup_worker_ssh.sh ${WORKER_IP} ${WORKER_USER}"
  exit 1
fi

echo "==> [1/4] Syncing deployment files to worker (${WORKER_IP}:${REMOTE_DIR})..."
ssh "${WORKER_USER}@${WORKER_IP}" "mkdir -p ${REMOTE_DIR}"
rsync -az --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.venv' \
  "${REPO_DIR}/" "${WORKER_USER}@${WORKER_IP}:${REMOTE_DIR}/"

echo "==> [2/4] Provisioning worker node02 remotely (Tier 3 + Node Exporter)..."
ssh -t "${WORKER_USER}@${WORKER_IP}" "bash ${REMOTE_DIR}/scripts/node_provision.sh worker"

echo "==> [3/4] Provisioning master node01 locally (Tier 1 + Tier 2 + Docker)..."
bash "${REPO_DIR}/scripts/node_provision.sh" master

echo "==> [4/4] Starting Control Plane stack (Gateway, Open WebUI, SearXNG, Redis, Caddy, Monitoring)..."
cd "${REPO_DIR}/deploy"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "NOTE: Created deploy/.env from .env.example — customize secrets before exposing publicly."
fi

# Use sg docker if current shell session hasn't picked up docker group yet
if groups | grep -qw docker; then
  docker compose up -d --build
else
  sudo docker compose up -d --build
fi

echo "==> Verifying cluster endpoints..."
sleep 5
for ep in \
  "http://127.0.0.1:8081/health|Tier1-Qwen3-4B(node01)" \
  "http://127.0.0.1:8082/health|Tier2-Qwen3-8B(node01)" \
  "http://${WORKER_IP}:8083/health|Tier3-Qwen3-14B(node02)" \
  "http://127.0.0.1:8000/healthz|FastAPI-Gateway(node01)" \
  "http://127.0.0.1:3000/health|Open-WebUI(node01)"; do
  url="${ep%%|*}"
  name="${ep##*|}"
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$url" || echo "000")
  printf "  %-28s -> HTTP %s\n" "$name" "$code"
done

echo "Done! Access Open WebUI on http://$(hostname -I | awk '{print $1}'):80 (via Caddy) or :3000"
