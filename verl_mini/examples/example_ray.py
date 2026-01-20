"""
Example demonstrating Ray distributed training in verl_mini.
"""

import torch
import numpy as np

from verl_mini.ray_worker import (
    Role,
    RayResourcePool,
    ResourcePoolManager,
    RayWorkerGroup,
    RayWorker,
    RayActorRolloutWorker,
    RayCriticWorker,
    init_ray_cluster,
    shutdown_ray,
    RAY_AVAILABLE,
)
from verl_mini.ray_trainer import (
    RayPPOConfig,
    RayPPOTrainer,
    create_ray_ppo_trainer,
)
from verl_mini.protocol import DataProto


def print_separator(title: str):
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60 + "\n")


def demo_resource_pool():
    """Demonstrate RayResourcePool resource management."""
    print_separator("RayResourcePool Demo")
    
    # Create resource pool
    pool = RayResourcePool(
        process_on_nodes=[2, 2],  # 2 GPUs on 2 nodes = 4 total
        name="demo_pool"
    )
    
    print(f"1. Resource Pool Configuration:")
    print(f"   Name: {pool.name}")
    print(f"   Total GPUs: {pool.total_gpus}")
    print(f"   Number of nodes: {pool.num_nodes}")
    print(f"   GPUs per node: {pool.process_on_nodes}")
    
    print(f"\n2. Worker Placement:")
    for worker_idx in range(pool.total_gpus):
        pg_idx, bundle_idx = pool.get_bundle_indices(worker_idx)
        print(f"   Worker {worker_idx} -> Node {pg_idx}, GPU {bundle_idx}")


def demo_resource_pool_manager():
    """Demonstrate ResourcePoolManager."""
    print_separator("ResourcePoolManager Demo")
    
    manager = ResourcePoolManager()
    
    # Create pools
    global_pool = manager.create_pool("global_pool", process_on_nodes=[4])
    critic_pool = manager.create_pool("critic_pool", process_on_nodes=[2])
    
    print("1. Created Pools:")
    for name, pool in manager.pools.items():
        print(f"   {name}: {pool.total_gpus} GPUs")
    
    # Register roles
    manager.register_role(Role.ActorRollout, "global_pool")
    manager.register_role(Role.Critic, "critic_pool")
    manager.register_role(Role.RewardModel, "global_pool")
    
    print("\n2. Role to Pool Mapping:")
    for role, pool_name in manager.role_to_pool.items():
        pool = manager.get_pool_for_role(role)
        print(f"   {role.value}: {pool_name} ({pool.total_gpus} GPUs)")


def demo_worker_group():
    """Demonstrate RayWorkerGroup (local mode)."""
    print_separator("RayWorkerGroup Demo (Local Mode)")
    
    # Create resource pool
    pool = RayResourcePool(process_on_nodes=[2], name="test_pool")
    
    # Create worker group
    group = RayWorkerGroup(
        worker_cls=RayActorRolloutWorker,
        resource_pool=pool,
        num_workers=2,
        worker_cls_kwargs={"config": {"hidden_size": 256}}
    )
    
    print("1. Creating Workers...")
    group.create_workers()
    print(f"   Created {len(group.workers)} workers")
    
    print("\n2. Worker Info:")
    infos = group.execute_all("get_info")
    for i, info in enumerate(infos):
        print(f"   Worker {i}: rank={info['rank']}, device={info['device']}")
    
    print("\n3. Execute on Rank 0:")
    rank_zero_info = group.execute_rank_zero("get_info")
    print(f"   Rank 0 info: {rank_zero_info}")
    
    # Cleanup
    group.shutdown()
    print("\n4. Workers shutdown successfully")


def demo_data_distribution():
    """Demonstrate DataProto distribution across workers."""
    print_separator("DataProto Distribution Demo")
    
    # Create sample data
    batch_size = 8
    seq_len = 32
    
    data = DataProto.from_dict({
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "rewards": torch.randn(batch_size, seq_len) * 0.5,
    })
    data.non_tensor_batch["prompts"] = np.array([f"prompt_{i}" for i in range(batch_size)])
    
    print(f"1. Original Data:")
    print(f"   Batch size: {len(data)}")
    print(f"   Keys: {list(data.batch.keys())}")
    
    # Create worker group
    pool = RayResourcePool(process_on_nodes=[4], name="test_pool")
    group = RayWorkerGroup(
        worker_cls=RayWorker,
        resource_pool=pool,
        num_workers=4,
    )
    group.create_workers()
    
    print(f"\n2. Broadcasting to {group.num_workers} workers:")
    chunks = group.broadcast_data(data)
    for i, chunk in enumerate(chunks):
        print(f"   Worker {i}: {len(chunk)} samples")
    
    print(f"\n3. Gathering from workers:")
    gathered = group.gather_data(chunks)
    print(f"   Gathered size: {len(gathered)}")
    print(f"   Keys preserved: {list(gathered.batch.keys())}")
    
    group.shutdown()


def demo_ray_ppo_config():
    """Demonstrate RayPPOConfig."""
    print_separator("RayPPOConfig Demo")
    
    # Default config
    default_config = RayPPOConfig()
    print("1. Default Configuration:")
    print(f"   total_epochs: {default_config.total_epochs}")
    print(f"   batch_size: {default_config.batch_size}")
    print(f"   clip_range: {default_config.clip_range}")
    print(f"   adv_estimator: {default_config.adv_estimator}")
    print(f"   num_actor_workers: {default_config.num_actor_workers}")
    
    # Custom config
    custom_config = RayPPOConfig(
        total_epochs=5,
        batch_size=16,
        adv_estimator="grpo",
        num_actor_workers=4,
    )
    print("\n2. Custom Configuration:")
    print(f"   total_epochs: {custom_config.total_epochs}")
    print(f"   batch_size: {custom_config.batch_size}")
    print(f"   adv_estimator: {custom_config.adv_estimator}")
    print(f"   num_actor_workers: {custom_config.num_actor_workers}")


def demo_ray_ppo_trainer():
    """Demonstrate RayPPOTrainer (local mode)."""
    print_separator("RayPPOTrainer Demo (Local Mode)")
    
    # Create config
    config = RayPPOConfig(
        total_epochs=2,
        batch_size=4,
        num_actor_workers=1,
        num_critic_workers=1,
        num_reward_workers=1,
        adv_estimator="gae",
    )
    
    print("1. Creating Trainer...")
    trainer = create_ray_ppo_trainer(config=config)
    print(f"   Device: {trainer.device}")
    print(f"   Advantage estimator: {config.adv_estimator}")
    
    print("\n2. Setting up Worker Groups...")
    trainer.setup_worker_groups()
    print(f"   Actor workers: {len(trainer.actor_rollout_group.workers)}")
    print(f"   Critic workers: {len(trainer.critic_group.workers)}")
    print(f"   Reward workers: {len(trainer.reward_group.workers)}")
    
    print("\n3. Running Training (with dummy data)...")
    metrics = trainer.train()
    
    print("\n4. Training Results:")
    for epoch_metrics in metrics:
        epoch = epoch_metrics.get("epoch", 0)
        print(f"   Epoch {epoch}: {epoch_metrics}")
    
    print("\n5. Shutting down...")
    trainer.shutdown()
    print("   Done!")


def demo_advantage_estimators():
    """Demonstrate different advantage estimators with trainer."""
    print_separator("Advantage Estimators in Trainer")
    
    for estimator in ["gae", "grpo", "rloo", "remax"]:
        print(f"\n--- {estimator.upper()} ---")
        
        config = RayPPOConfig(
            total_epochs=1,
            batch_size=4,
            num_actor_workers=1,
            num_critic_workers=1,
            adv_estimator=estimator,
        )
        
        trainer = create_ray_ppo_trainer(config=config)
        trainer.setup_worker_groups()
        
        # Create test data
        batch_size = 8
        seq_len = 16
        
        data = DataProto.from_dict({
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "response_mask": torch.ones(batch_size, seq_len),
            "rewards": torch.randn(batch_size, seq_len) * 0.5,
            "values": torch.randn(batch_size, seq_len) * 0.1,
            "log_probs": torch.randn(batch_size, seq_len) * 0.1 - 2.0,
            "ref_log_probs": torch.randn(batch_size, seq_len) * 0.1 - 2.0,
        })
        data.non_tensor_batch["index"] = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        
        # Compute advantages
        result = trainer.compute_advantages(data)
        advantages = result.batch.get("advantages")
        
        if advantages is not None:
            print(f"   Advantages shape: {advantages.shape}")
            print(f"   Advantages mean: {advantages.mean().item():.4f}")
            print(f"   Advantages std: {advantages.std().item():.4f}")
        
        trainer.shutdown()


def demo_custom_reward():
    """Demonstrate trainer with custom reward function."""
    print_separator("Custom Reward Function Demo")
    
    # Custom reward: length bonus + diversity penalty
    def custom_reward_fn(data: DataProto) -> torch.Tensor:
        response_mask = data.batch.get("response_mask")
        if response_mask is None:
            return torch.zeros(len(data), 16)
        
        # Length reward
        lengths = response_mask.sum(dim=-1, keepdim=True)
        length_reward = lengths / 100.0  # Normalize
        
        # Random diversity component
        diversity = torch.randn_like(response_mask) * 0.1
        
        rewards = length_reward.expand_as(response_mask) + diversity
        return rewards * response_mask
    
    config = RayPPOConfig(
        total_epochs=2,
        batch_size=4,
        adv_estimator="grpo",
    )
    
    print("1. Creating trainer with custom reward...")
    trainer = create_ray_ppo_trainer(
        config=config,
        reward_fn=custom_reward_fn,
    )
    
    print("2. Training with custom rewards...")
    metrics = trainer.train()
    
    print("\n3. Results:")
    for m in metrics:
        print(f"   Epoch {m.get('epoch', 0)}: {m}")
    
    trainer.shutdown()


if __name__ == "__main__":
    print("="*60)
    print("verl_mini - Ray Distributed Training Examples")
    print("="*60)
    print(f"\nRay Available: {RAY_AVAILABLE}")
    print("(Running in local mode if Ray not available)")
    
    demo_resource_pool()
    demo_resource_pool_manager()
    demo_worker_group()
    demo_data_distribution()
    demo_ray_ppo_config()
    demo_ray_ppo_trainer()
    demo_advantage_estimators()
    demo_custom_reward()
    
    print("\n" + "="*60)
    print("All Ray distributed training examples completed!")
    print("="*60)
