#!/bin/bash
set -euo pipefail

# Preemption hook: find latest checkpoint in the training runs and upload to GCS bucket
# Expected env vars: GCS_BUCKET, PROJECT

GCS_BUCKET=${GCS_BUCKET:-}
PROJECT=${PROJECT:-}
TRAIN_RUN_DIR=${TRAIN_RUN_DIR:-/workspace/runs/train}

if [ -z "$GCS_BUCKET" ]; then
  echo "GCS_BUCKET not set, skipping checkpoint upload"
  exit 0
fi

# find last/latest checkpoint
if [ -d "$TRAIN_RUN_DIR" ]; then
  CKPT=$(find "$TRAIN_RUN_DIR" -type f -name "last*.pt" -o -name "*.pt" | sort -r | head -n 1 || true)
  if [ -n "$CKPT" ]; then
    echo "Uploading checkpoint $CKPT to gs://$GCS_BUCKET/checkpoints/"
    gsutil cp "$CKPT" "gs://$GCS_BUCKET/checkpoints/"
    exit 0
  fi
fi

echo "No checkpoint found to upload"
exit 0
