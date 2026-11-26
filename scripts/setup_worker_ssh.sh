#!/usr/bin/env bash
set -euo pipefail

WORKER_IP="${1:-192.168.9.135}"
WORKER_USER="${2:-cluster}"

echo "==> Setting up passwordless SSH from $(hostname) to ${WORKER_USER}@${WORKER_IP}"
if [ ! -f "$HOME/.ssh/id_ed25519" ]; then
  ssh-keygen -t ed25519 -N "" -f "$HOME/.ssh/id_ed25519"
fi

ssh-copy-id -i "$HOME/.ssh/id_ed25519.pub" "${WORKER_USER}@${WORKER_IP}"

echo "==> Verifying passwordless SSH..."
ssh -o BatchMode=yes "${WORKER_USER}@${WORKER_IP}" "hostname && echo 'SSH OK'"
