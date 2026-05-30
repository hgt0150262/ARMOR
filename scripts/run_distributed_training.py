"""
Multi-server distributed training with Ray.
Runs on: gpu-server (head, 4x H100) + gpu-server1 (worker, 2x H100)
Total: 6x H100 80GB
"""

import os
import sys
sys.path.insert(0, '/data/hgt/projects/verl_reproduction')

import ray
import torch
import numpy as np

from ARMOR import (
    DataProto,
    RayPPOConfig,
    RayPPOTrainer,
    RayResourcePool,
    ResourcePoolManager,
    Role,
    AdvantageEstimator,
)


def check_cluster_resources():
    """Check and display cluster resources."""
    resources = ray.cluster_resources()
    print("\n=== Ray Cluster Resources ===")
    print(f"  CPUs: {resources.get('CPU', 0)}")
    print(f"  GPUs: {resources.get('GPU', 0)}")
    print(f"  Memory: {resources.get('memory', 0) / 1e9:.1f} GB")
    
    nodes = ray.nodes()
    print(f"\n  Active Nodes: {len([n for n in nodes if n['Alive']])}")
    for node in nodes:
        if node['Alive']:
            node_resources = node.get('Resources', {})
            print(f"    - {node['NodeManagerAddress']}: "
                  f"{node_resources.get('GPU', 0)} GPUs, "
                  f"{node_resources.get('CPU', 0)} CPUs")
    print()
    return resources


def create_multi_node_trainer():
    """Create trainer configured for multi-node setup."""
    
    # Configuration for 6 GPU training
    config = RayPPOConfig(
        total_epochs=5,
        batch_size=48,  # 6 GPUs * 8 samples per GPU
        mini_batch_size=8,
        ppo_epochs=2,
        
        # PPO hyperparameters
        clip_range=0.2,
        gamma=1.0,
        gae_lambda=0.95,
        
        # Algorithm
        adv_estimator="grpo",
        normalize_advantage=True,
        
        # Workers: 4 actor (gpu-server) + 2 critic (gpu-server1)
        num_actor_workers=4,
        num_critic_workers=2,
        num_reward_workers=1,
        
        # Learning rates
        actor_lr=1e-6,
        critic_lr=5e-6,
    )
    
    # Create resource manager
    resource_manager = ResourcePoolManager()
    
    # Actor pool on gpu-server (4 GPUs)
    actor_pool = resource_manager.create_pool(
        name="actor_pool",
        process_on_nodes=[4],  # 4 GPUs on first node
    )
    
    # Critic pool on gpu-server1 (2 GPUs)
    critic_pool = resource_manager.create_pool(
        name="critic_pool",
        process_on_nodes=[2],  # 2 GPUs on second node
    )
    
    # Register roles to pools
    resource_manager.register_role(Role.ActorRollout, "actor_pool")
    resource_manager.register_role(Role.Critic, "critic_pool")
    resource_manager.register_role(Role.RewardModel, "actor_pool")
    
    # Create trainer
    trainer = RayPPOTrainer(
        config=config,
        resource_pool_manager=resource_manager,
    )
    
    return trainer


def run_training():
    """Run distributed training."""
    print("="*60)
    print("Multi-Server Distributed PPO Training")
    print("="*60)
    
    # Check cluster
    resources = check_cluster_resources()
    
    total_gpus = resources.get('GPU', 0)
    if total_gpus < 2:
        print(f"Warning: Only {total_gpus} GPU(s) available. "
              "For full multi-node training, ensure both nodes are connected.")
    
    # Create trainer
    print("Creating distributed trainer...")
    trainer = create_multi_node_trainer()
    
    # Setup workers
    print("Setting up worker groups...")
    trainer.setup_worker_groups()
    
    print(f"\nWorker Configuration:")
    print(f"  Actor/Rollout workers: {len(trainer.actor_rollout_group.workers)}")
    print(f"  Critic workers: {len(trainer.critic_group.workers)}")
    print(f"  Reward workers: {len(trainer.reward_group.workers)}")
    
    # Run training
    print("\nStarting training...")
    metrics = trainer.train()
    
    # Print results
    print("\n=== Training Results ===")
    for epoch_metrics in metrics:
        print(f"Epoch {epoch_metrics.get('epoch', 0)}: {epoch_metrics}")
    
    # Cleanup
    trainer.shutdown()
    print("\nTraining complete!")
    
    return metrics


def main():
    # Initialize Ray (connect to existing cluster)
    if not ray.is_initialized():
        try:
            # Try to connect to existing cluster
            ray.init(address="auto")
            print("Connected to existing Ray cluster")
        except Exception as e:
            # Fall back to local cluster
            print(f"Could not connect to cluster: {e}")
            print("Starting local Ray instance...")
            ray.init()
    
    try:
        run_training()
    finally:
        # Don't shutdown Ray if we connected to existing cluster
        pass


if __name__ == "__main__":
    main()
