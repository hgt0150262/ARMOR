"""
Ray-based distributed PPO trainer for ARMOR.
Implements the main training loop with distributed workers.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Union
from enum import Enum
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ARMOR.protocol import DataProto
from .core_algos import (
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    compute_remax_advantage,
    compute_rloo_advantage,
    compute_policy_loss_ppo,
    compute_value_loss,
    compute_entropy_loss,
    AdvantageEstimator,
    ADV_ESTIMATOR_REGISTRY,
)
from ARMOR.single_controller.ray import (
    Role,
    RayResourcePool,
    ResourcePoolManager,
    RayWorkerGroup,
    RayActorRolloutWorker,
    RayCriticWorker,
    RayRewardWorker,
    init_ray_cluster,
    RAY_AVAILABLE,
)


@dataclass
class RayPPOConfig:
    """Configuration for Ray PPO Trainer."""
    # Training
    total_epochs: int = 10
    ppo_epochs: int = 1
    batch_size: int = 8
    mini_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    
    # PPO hyperparameters
    clip_range: float = 0.2
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 1.0
    gae_lambda: float = 0.95
    
    # KL penalty
    kl_coef: float = 0.1
    kl_target: float = 0.01
    
    # Advantage estimation
    adv_estimator: str = "gae"
    normalize_advantage: bool = True
    
    # Learning rates
    actor_lr: float = 1e-6
    critic_lr: float = 1e-5
    
    # Distributed
    num_actor_workers: int = 1
    num_critic_workers: int = 1
    num_reward_workers: int = 1
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100
    
    # Misc
    seed: int = 42
    max_grad_norm: float = 1.0


class RayPPOTrainer:
    """
    Distributed PPO trainer using Ray for scalable RLHF.
    
    Orchestrates training across multiple workers:
    - Actor/Rollout workers for policy and generation
    - Critic workers for value estimation
    - Reward workers for reward computation
    
    Supports multiple advantage estimators (GAE, GRPO, RLOO, ReMax).
    """
    
    def __init__(
        self,
        config: RayPPOConfig,
        tokenizer: Any = None,
        resource_pool_manager: Optional[ResourcePoolManager] = None,
        reward_fn: Optional[Callable] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
    ):
        """
        Args:
            config: Training configuration
            tokenizer: Tokenizer for text processing
            resource_pool_manager: Manager for GPU resources
            reward_fn: Custom reward function
            train_dataset: Training dataset
            val_dataset: Validation dataset
        """
        self.config = config
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Resource management
        self.resource_pool_manager = resource_pool_manager or ResourcePoolManager()
        
        # Worker groups
        self.actor_rollout_group: Optional[RayWorkerGroup] = None
        self.critic_group: Optional[RayWorkerGroup] = None
        self.reward_group: Optional[RayWorkerGroup] = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.kl_coef = config.kl_coef
        
        # Metrics
        self.metrics_history: List[Dict[str, float]] = []
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def setup_resource_pools(self):
        """Setup resource pools for workers."""
        # Create global pool
        global_pool = self.resource_pool_manager.create_pool(
            name="global_pool",
            process_on_nodes=[self.config.num_actor_workers],
        )
        
        # Register roles
        self.resource_pool_manager.register_role(Role.ActorRollout, "global_pool")
        self.resource_pool_manager.register_role(Role.Critic, "global_pool")
        self.resource_pool_manager.register_role(Role.RewardModel, "global_pool")
        
        # Initialize pools
        self.resource_pool_manager.initialize_all_pools()
    
    def setup_worker_groups(self):
        """Create worker groups for distributed training."""
        global_pool = self.resource_pool_manager.pools.get("global_pool")
        
        if global_pool is None:
            self.setup_resource_pools()
            global_pool = self.resource_pool_manager.pools["global_pool"]
        
        # Actor/Rollout workers
        self.actor_rollout_group = RayWorkerGroup(
            worker_cls=RayActorRolloutWorker,
            resource_pool=global_pool,
            num_workers=self.config.num_actor_workers,
            worker_cls_kwargs={"config": {"lr": self.config.actor_lr}}
        )
        self.actor_rollout_group.create_workers()
        
        # Critic workers
        self.critic_group = RayWorkerGroup(
            worker_cls=RayCriticWorker,
            resource_pool=global_pool,
            num_workers=self.config.num_critic_workers,
            worker_cls_kwargs={"config": {"lr": self.config.critic_lr}}
        )
        self.critic_group.create_workers()
        
        # Reward workers
        self.reward_group = RayWorkerGroup(
            worker_cls=RayRewardWorker,
            resource_pool=global_pool,
            num_workers=self.config.num_reward_workers,
            worker_cls_kwargs={"config": {}}
        )
        self.reward_group.create_workers()
    
    def compute_advantages(self, data: DataProto) -> DataProto:
        """
        Compute advantages using configured estimator.
        
        Args:
            data: DataProto with rewards and values
        
        Returns:
            DataProto with advantages and returns
        """
        rewards = data.batch.get("rewards")
        values = data.batch.get("values")
        response_mask = data.batch.get("response_mask")
        
        if rewards is None or response_mask is None:
            return data
        
        adv_type = self.config.adv_estimator
        
        if adv_type == "gae" and values is not None:
            advantages, returns = compute_gae_advantage_return(
                token_level_rewards=rewards,
                values=values,
                response_mask=response_mask,
                gamma=self.config.gamma,
                lam=self.config.gae_lambda,
            )
        elif adv_type == "grpo":
            index = data.non_tensor_batch.get("index", np.arange(len(data)))
            advantages, returns = compute_grpo_outcome_advantage(
                token_level_rewards=rewards,
                response_mask=response_mask,
                index=index,
            )
        elif adv_type == "rloo":
            index = data.non_tensor_batch.get("index", np.arange(len(data)))
            advantages, returns = compute_rloo_advantage(
                token_level_rewards=rewards,
                response_mask=response_mask,
                index=index,
            )
        elif adv_type == "remax":
            index = data.non_tensor_batch.get("index", np.arange(len(data)))
            advantages, returns = compute_remax_advantage(
                token_level_rewards=rewards,
                response_mask=response_mask,
                index=index,
            )
        else:
            # Default: outcome-level advantage
            outcome_rewards = (rewards * response_mask).sum(dim=-1, keepdim=True)
            advantages = outcome_rewards.expand_as(rewards) * response_mask
            returns = advantages.clone()
        
        # Normalize advantages
        if self.config.normalize_advantage:
            adv_mean = advantages[response_mask.bool()].mean()
            adv_std = advantages[response_mask.bool()].std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std
            advantages = advantages * response_mask
        
        # Update data
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        
        return data
    
    def apply_kl_penalty(self, data: DataProto) -> DataProto:
        """Apply KL penalty to rewards."""
        log_probs = data.batch.get("log_probs")
        ref_log_probs = data.batch.get("ref_log_probs")
        rewards = data.batch.get("rewards")
        response_mask = data.batch.get("response_mask")
        
        if log_probs is None or ref_log_probs is None or rewards is None:
            return data
        
        # Compute KL divergence
        kl = log_probs - ref_log_probs
        
        # Apply penalty
        penalized_rewards = rewards - self.kl_coef * kl
        data.batch["rewards"] = penalized_rewards
        data.batch["kl"] = kl
        
        return data
    
    def training_step(self, batch_data: DataProto) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Steps:
        1. Generate responses (rollout)
        2. Compute rewards
        3. Compute values
        4. Compute advantages
        5. Update actor and critic
        """
        metrics = {}
        
        # Step 1: Generate responses (using actor/rollout workers)
        if self.actor_rollout_group:
            rollout_data = self.actor_rollout_group.execute_with_data(
                "generate", batch_data
            )
        else:
            rollout_data = batch_data
        
        # Step 2: Compute rewards
        if self.reward_fn is not None:
            # Custom reward function
            rewards = self.reward_fn(rollout_data)
            rollout_data.batch["rewards"] = rewards
        elif self.reward_group:
            rollout_data = self.reward_group.execute_with_data(
                "compute_rewards", rollout_data
            )
        
        # Step 3: Compute values
        if self.critic_group:
            rollout_data = self.critic_group.execute_with_data(
                "compute_values", rollout_data
            )
        
        # Step 4: Apply KL penalty and compute advantages
        rollout_data = self.apply_kl_penalty(rollout_data)
        rollout_data = self.compute_advantages(rollout_data)
        
        # Step 5: Update models
        if self.actor_rollout_group:
            actor_metrics = self.actor_rollout_group.execute_rank_zero(
                "update", rollout_data
            )
            if actor_metrics:
                metrics.update(actor_metrics)
        
        if self.critic_group:
            critic_metrics = self.critic_group.execute_rank_zero(
                "update", rollout_data
            )
            if critic_metrics:
                metrics.update(critic_metrics)
        
        return metrics
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = {}
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        for batch in pbar:
            # Convert batch to DataProto
            if isinstance(batch, DataProto):
                batch_data = batch
            else:
                batch_data = DataProto.from_dict(batch)
            
            # Training step
            step_metrics = self.training_step(batch_data)
            
            # Accumulate metrics
            for key, value in step_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value
            
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            if step_metrics:
                pbar.set_postfix({k: f"{v:.4f}" for k, v in step_metrics.items()})
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)
        
        return epoch_metrics
    
    def train(self) -> List[Dict[str, float]]:
        """
        Main training loop.
        
        Returns:
            List of metrics for each epoch
        """
        print(f"Starting Ray PPO training for {self.config.total_epochs} epochs")
        print(f"  Advantage estimator: {self.config.adv_estimator}")
        print(f"  Actor workers: {self.config.num_actor_workers}")
        print(f"  Critic workers: {self.config.num_critic_workers}")
        print(f"  Device: {self.device}")
        
        # Setup workers if not already done
        if self.actor_rollout_group is None:
            self.setup_worker_groups()
        
        # Create dataloader
        if self.train_dataset is not None:
            dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
            )
        else:
            # Create dummy dataloader for demo
            dataloader = self._create_dummy_dataloader()
        
        all_metrics = []
        
        for epoch in range(self.config.total_epochs):
            self.epoch = epoch
            epoch_metrics = self.train_epoch(dataloader)
            epoch_metrics["epoch"] = epoch
            all_metrics.append(epoch_metrics)
            self.metrics_history.append(epoch_metrics)
            
            print(f"Epoch {epoch}: {epoch_metrics}")
        
        return all_metrics
    
    def _create_dummy_dataloader(self) -> List[DataProto]:
        """Create dummy dataloader for testing."""
        batches = []
        for _ in range(5):
            batch_size = self.config.batch_size
            seq_len = 32
            
            data = DataProto.from_dict({
                "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
                "attention_mask": torch.ones(batch_size, seq_len),
                "response_mask": torch.ones(batch_size, seq_len - 1),
                "log_probs": torch.randn(batch_size, seq_len - 1) * 0.1 - 2.0,
                "ref_log_probs": torch.randn(batch_size, seq_len - 1) * 0.1 - 2.0,
                "rewards": torch.randn(batch_size, seq_len - 1) * 0.5,
                "values": torch.randn(batch_size, seq_len - 1) * 0.1,
            })
            data.non_tensor_batch["index"] = np.arange(batch_size) // 2
            batches.append(data)
        
        return batches
    
    def shutdown(self):
        """Shutdown all workers."""
        if self.actor_rollout_group:
            self.actor_rollout_group.shutdown()
        if self.critic_group:
            self.critic_group.shutdown()
        if self.reward_group:
            self.reward_group.shutdown()


def create_ray_ppo_trainer(
    config: Optional[RayPPOConfig] = None,
    **kwargs
) -> RayPPOTrainer:
    """
    Factory function to create Ray PPO trainer.
    
    Args:
        config: Training configuration
        **kwargs: Additional arguments for trainer
    
    Returns:
        Configured RayPPOTrainer
    """
    if config is None:
        config = RayPPOConfig()
    
    return RayPPOTrainer(config=config, **kwargs)
