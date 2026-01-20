#!/bin/bash
# Start Ray worker node on gpu-server1 (2x H100 available)
# Run this on: gpu-server1 (172.16.54.131)

set -e

HEAD_NODE_IP="172.16.54.132"
HEAD_NODE_PORT="6379"

echo "=== Starting Ray Worker Node ==="
echo "Server: gpu-server1 (172.16.54.131)"
echo "GPUs: 2x H100 80GB (devices 1,2)"
echo "Head Node: ${HEAD_NODE_IP}:${HEAD_NODE_PORT}"

# Activate conda
source /data/hgt/miniconda3/bin/activate minimind

# Set visible GPUs (only use device 1 and 2, 0 and 3 are occupied)
export CUDA_VISIBLE_DEVICES=1,2

# Stop any existing Ray instance
ray stop 2>/dev/null || true

# Connect to head node
ray start --address="${HEAD_NODE_IP}:${HEAD_NODE_PORT}" \
    --num-cpus=16 \
    --num-gpus=2

echo ""
echo "=== Ray Worker Node Connected ==="
echo "Connected to head: ${HEAD_NODE_IP}:${HEAD_NODE_PORT}"
echo ""
echo "To verify cluster, run on head node:"
echo "  ray status"
