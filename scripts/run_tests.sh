#!/bin/bash
# Run test scripts
# Usage: bash scripts/run_tests.sh [test_name]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

TEST=${1:-all}

if [ "$TEST" == "inference" ]; then
    echo "Testing model inference..."
    python tests/test_model_inference.py
elif [ "$TEST" == "training_flow" ]; then
    echo "Testing training flow..."
    python tests/test_training_flow.py
elif [ "$TEST" == "data" ]; then
    echo "Checking data format..."
    python tests/check_data.py
elif [ "$TEST" == "all" ]; then
    echo "Running all tests..."
    python tests/check_data.py
    python tests/test_model_inference.py
    python tests/test_training_flow.py
else
    echo "Usage: bash scripts/run_tests.sh [inference|training_flow|data|all]"
    exit 1
fi

echo "Tests completed!"
