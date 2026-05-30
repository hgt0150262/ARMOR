# Ray Multi-Server Cluster Setup Guide

## Hardware Overview

| Server | GPUs | Available | IP |
|--------|------|-----------|-----|
| gpu-server | 4x H100 80GB | All (0,1,2,3) | 172.16.54.132 |
| gpu-server1 | 4x H100 80GB | 2 (1,2) | 172.16.54.131 |

**Total Available: 6x H100 80GB**

## Step 1: Start Ray Head Node (on gpu-server)

```bash
# SSH to gpu-server (4x H100 all available)
ssh gpu-server

# Activate conda environment
source /data/hgt/miniconda3/bin/activate minimind

# Start Ray head node
ray start --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-cpus=32 \
    --num-gpus=4 \
    --block
```

## Step 2: Connect Worker Node (on gpu-server1)

```bash
# SSH to gpu-server1
ssh gpu-server1

# Activate conda environment
source /data/hgt/miniconda3/bin/activate minimind

# Set visible GPUs (only use device 1 and 2)
export CUDA_VISIBLE_DEVICES=1,2

# Connect to head node
ray start --address='172.16.54.132:6379' \
    --num-cpus=16 \
    --num-gpus=2 \
    --block
```

## Step 3: Verify Cluster Status

```bash
# On head node (gpu-server)
ray status
```

Expected output:
```
======== Cluster Resources ========
CPUs: 48
GPUs: 6
```

## Step 4: Run Distributed Training

```python
# training_script.py
import ray
from ARMOR import (
    RayPPOConfig,
    RayPPOTrainer,
    RayResourcePool,
    ResourcePoolManager,
    Role,
)

# Connect to existing Ray cluster
ray.init(address="auto")

# Configure for 6 GPUs across 2 nodes
config = RayPPOConfig(
    total_epochs=10,
    batch_size=32,
    num_actor_workers=4,  # Use 4 GPUs for actor
    num_critic_workers=2,  # Use 2 GPUs for critic
    adv_estimator="grpo",
)

# Create trainer
trainer = RayPPOTrainer(config=config)

# Setup resource pools for multi-node
resource_manager = ResourcePoolManager()
actor_pool = resource_manager.create_pool(
    "actor_pool",
    process_on_nodes=[4],  # 4 GPUs on gpu-server
)
critic_pool = resource_manager.create_pool(
    "critic_pool", 
    process_on_nodes=[2],  # 2 GPUs on gpu-server1
)

resource_manager.register_role(Role.ActorRollout, "actor_pool")
resource_manager.register_role(Role.Critic, "critic_pool")

trainer.resource_pool_manager = resource_manager

# Run training
trainer.train()
```

## Alternative: Quick Start Script

```bash
# On gpu-server (head node)
cd /data/hgt/projects/verl_reproduction
python -c "
import ray
ray.init()
print('Cluster resources:', ray.cluster_resources())
"
```

## Troubleshooting

### Port Already in Use
```bash
ray stop  # Stop existing Ray instance
ray start --head --port=6379
```

### Check Node Status
```bash
ray status
```

### Dashboard Access
Open in browser: `http://172.16.54.132:8265`
