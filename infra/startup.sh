#!/bin/bash
set -euo pipefail

# Basic startup script: install docker and NVIDIA drivers (best-effort), then pull training image if provided.
# For production use, replace with tested driver install and container runtime.

apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg lsb-release

# Install Docker
if ! command -v docker >/dev/null 2>&1; then
  apt-get install -y docker.io
  systemctl enable --now docker
fi

# Attempt to install NVIDIA drivers (best-effort). On GCE, use the GPU driver installer.
# This step may require GPU-enabled images or additional configuration; treat as informational.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA drivers already present"
else
  echo "Attempting to install NVIDIA driver packages (may require reboot)"
  # Minimal attempt using ubuntu packages; user may prefer NVIDIA's installer.
  apt-get install -y --no-install-recommends nvidia-driver-535 || true
fi

# Pull training container if set
if [ -n "${TRAINING_IMAGE:-}" ]; then
  docker pull "${TRAINING_IMAGE}"
  # Run container detached (user should ssh in/manage training runs)
  docker run -d --gpus all --restart unless-stopped -v /mnt/disks:/mnt/disks "${TRAINING_IMAGE}"
fi

# Signal completion
logger "bdmtx startup script finished"
