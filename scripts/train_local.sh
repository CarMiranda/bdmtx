#!/bin/bash
# Train enhancement model locally
# Usage: ./scripts/train_local.sh /path/to/dataset

set -euo pipefail
DATASET=${1:-./data}
python -m bdmtx.train train_enhancement_entry "$DATASET"
