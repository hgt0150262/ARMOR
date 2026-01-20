#!/bin/bash
# Start Ray head node on gpu-server (4x H100)
# Run this on: gpu-server (172.16.54.132)

set -e

echo "=== Starting Ray Head Node ==="
echo "Server: gpu-server (172.16.54.132)"
echo "GPUs: 4x H100 80GB (all available)"

# Activate conda
source /data/hgt/miniconda3/bin/activate minimind

# Stop any existing Ray instance
ray stop 2>/dev/null || true

# Start head node
ray start --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-cpus=32 \
    --num-gpus=4

echo ""
echo "=== Ray Head Node Started ==="
echo "Dashboard: http://172.16.54.132:8265"
echo "Address for workers: 172.16.54.132:6379"
echo ""
echo "To connect worker nodes, run on gpu-server1:"
echo "  ./start_ray_worker.sh"
