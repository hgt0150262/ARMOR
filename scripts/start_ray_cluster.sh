#!/bin/bash
# Ray Cluster Setup for Multi-Node Training
# Head Node: gpu-server (4x H100)
# Worker Node: gpu-server1 (4x H100)

set -e

MODE=${1:-"head"}  # head or worker
HEAD_IP=${2:-"172.16.54.132"}  # gpu-server IP
HEAD_PORT=${3:-6379}

# Activate conda environment
source /data/hgt/miniconda3/bin/activate minimind

# NCCL settings for multi-node
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

if [ "$MODE" == "head" ]; then
    echo "Starting Ray HEAD node..."
    echo "IP: $HEAD_IP, Port: $HEAD_PORT"
    
    # Stop existing Ray
    ray stop --force 2>/dev/null || true
    
    # Start head node
    ray start --head \
        --port=$HEAD_PORT \
        --num-gpus=4 \
        --num-cpus=32 \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265
    
    echo ""
    echo "============================================"
    echo "Ray HEAD started!"
    echo "Dashboard: http://$HEAD_IP:8265"
    echo "Connect workers with:"
    echo "  bash start_ray_cluster.sh worker $HEAD_IP $HEAD_PORT"
    echo "============================================"

elif [ "$MODE" == "worker" ]; then
    echo "Starting Ray WORKER node..."
    echo "Connecting to HEAD at $HEAD_IP:$HEAD_PORT"
    
    # Stop existing Ray
    ray stop --force 2>/dev/null || true
    
    # Start worker node
    ray start --address="$HEAD_IP:$HEAD_PORT" \
        --num-gpus=4 \
        --num-cpus=32
    
    echo ""
    echo "============================================"
    echo "Ray WORKER connected to $HEAD_IP:$HEAD_PORT"
    echo "============================================"

elif [ "$MODE" == "stop" ]; then
    echo "Stopping Ray..."
    ray stop --force
    echo "Ray stopped."

elif [ "$MODE" == "status" ]; then
    echo "Ray cluster status:"
    ray status

else
    echo "Usage: $0 [head|worker|stop|status] [head_ip] [head_port]"
    echo ""
    echo "Examples:"
    echo "  $0 head                    # Start head node"
    echo "  $0 worker 172.16.54.132    # Connect worker to head"
    echo "  $0 stop                    # Stop Ray"
    echo "  $0 status                  # Check status"
fi
