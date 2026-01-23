#!/bin/bash
set -euo pipefail

# Default training entrypoint for GCP trainer image.
# Usage: ./train_entrypoint.sh /data /workspace config.yaml

DATA_DIR=${1:-/data}
WORKSPACE_DIR=${2:-/workspace}
CONFIG=${3:-/workspace/dataset.yaml}

cd ${WORKSPACE_DIR}

# Activate virtualenv if present
if [ -d "/workspace/venv" ]; then
  source /workspace/venv/bin/activate
fi

# Run ultralytics train command
python - <<'PY'
from ultralytics import YOLO
import sys
model = 'yolov8n-seg'
print('Starting training with model', model)
YOLO(model).train(data=sys.argv[1], epochs=50) 
PY
