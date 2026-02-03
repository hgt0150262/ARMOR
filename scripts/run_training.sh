#!/bin/bash
# Quick training launcher script
# Usage: bash scripts/run_training.sh [single|multi]

set -e

MODE=${1:-multi}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

if [ "$MODE" == "single" ]; then
    echo "Starting single-GPU training..."
    bash verl_mini/trainer/train_qwen7b_grpo.sh
elif [ "$MODE" == "multi" ]; then
    echo "Starting multi-GPU training..."
    bash verl_mini/trainer/train_qwen7b_grpo_multigpu.sh
else
    echo "Usage: bash scripts/run_training.sh [single|multi]"
    exit 1
fi
