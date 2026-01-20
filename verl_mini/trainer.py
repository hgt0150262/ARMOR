"""
Simplified reproduction of verl's PPO Trainer.
Demonstrates the core training loop for RLHF with PPO algorithm.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from .protocol import DataProto
from .core_algos import (
    AdvantageEstimator,
    AdaptiveKLController,
    FixedKLController,
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    compute_policy_loss_ppo,
    compute_value_loss,
    compute_entropy_loss,
    compute_total_loss,
    kl_penalty,
    masked_mean,
    get_adv_estimator_fn,
)

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # Training
    total_epochs: int = 3
    mini_batch_size: int = 8
    ppo_epochs: int = 4
    
    # PPO hyperparameters
    clip_range: float = 0.2
    clip_range_value: float = 0.2
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    
    # GAE parameters
    gamma: float = 1.0
    lam: float = 0.95
    
    # KL penalty
    use_kl_penalty: bool = False
    init_kl_coef: float = 0.1
    target_kl: float = 0.01
    kl_horizon: int = 10000
    
    # Advantage estimation
    adv_estimator: str = "gae"
    norm_adv: bool = True
    
    # Learning rates
    actor_lr: float = 1e-5
    critic_lr: float = 1e-5


class ActorModel(nn.Module):
    """Simple actor model for demonstration."""
    
    def __init__(self, vocab_size: int, hidden_size: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """Forward pass returning logits."""
        embeds = self.embedding(input_ids)
        output, _ = self.lstm(embeds)
        logits = self.head(output)
        return logits
    
    def get_log_probs(self, input_ids: torch.Tensor, 
                      attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Get log probabilities for given input_ids."""
        logits = self.forward(input_ids, attention_mask)
        log_probs = torch.log_softmax(logits, dim=-1)
        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs[:, :-1, :], 
            dim=2, 
            index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        return token_log_probs


class CriticModel(nn.Module):
    """Simple critic model (value function) for demonstration."""
    
    def __init__(self, vocab_size: int, hidden_size: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass returning values."""
        embeds = self.embedding(input_ids)
        output, _ = self.lstm(embeds)
        values = self.head(output).squeeze(-1)
        return values


class ReferenceModel(nn.Module):
    """Reference model (frozen copy of actor for KL computation)."""
    
    def __init__(self, actor_model: ActorModel):
        super().__init__()
        # Create a copy
        self.model = ActorModel(
            vocab_size=actor_model.embedding.num_embeddings,
            hidden_size=actor_model.embedding.embedding_dim
        )
        self.model.load_state_dict(actor_model.state_dict())
        # Freeze
        for param in self.model.parameters():
            param.requires_grad = False
    
    def get_log_probs(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Get log probabilities (no grad)."""
        with torch.no_grad():
            return self.model.get_log_probs(input_ids, attention_mask)


class RewardModel(nn.Module):
    """Simple reward model for demonstration."""
    
    def __init__(self, vocab_size: int, hidden_size: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass returning reward scores."""
        embeds = self.embedding(input_ids)
        output, _ = self.lstm(embeds)
        # Use last hidden state for reward
        if attention_mask is not None:
            # Get last non-padded position
            lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
            last_hidden = output[batch_indices, lengths.long()]
        else:
            last_hidden = output[:, -1]
        rewards = self.head(last_hidden).squeeze(-1)
        return rewards


class PPOTrainer:
    """
    PPO Trainer for RLHF.
    
    This is a simplified single-process trainer that demonstrates the core
    PPO training loop. The real verl uses Ray for distributed training.
    """
    
    def __init__(
        self,
        config: PPOConfig,
        actor_model: ActorModel,
        critic_model: CriticModel,
        ref_model: Optional[ReferenceModel] = None,
        reward_model: Optional[RewardModel] = None,
        reward_fn: Optional[Callable] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.config = config
        self.device = device
        
        # Models
        self.actor = actor_model.to(device)
        self.critic = critic_model.to(device)
        self.ref_model = ref_model.to(device) if ref_model else None
        self.reward_model = reward_model.to(device) if reward_model else None
        self.reward_fn = reward_fn
        
        # Optimizers
        self.actor_optimizer = AdamW(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = AdamW(self.critic.parameters(), lr=config.critic_lr)
        
        # KL controller
        if config.use_kl_penalty:
            self.kl_ctrl = AdaptiveKLController(
                init_kl_coef=config.init_kl_coef,
                target_kl=config.target_kl,
                horizon=config.kl_horizon
            )
        else:
            self.kl_ctrl = None
        
        # Metrics
        self.metrics_history = []
    
    def compute_rewards(self, data: DataProto) -> DataProto:
        """
        Compute rewards for generated responses.
        
        This can use either a reward model or a reward function.
        """
        input_ids = data.batch["input_ids"]
        attention_mask = data.batch.get("attention_mask")
        
        if self.reward_model is not None:
            with torch.no_grad():
                rewards = self.reward_model(input_ids, attention_mask)
        elif self.reward_fn is not None:
            # Call custom reward function
            rewards = self.reward_fn(data)
        else:
            # Default: random rewards for demonstration
            rewards = torch.randn(input_ids.size(0), device=self.device)
        
        # Convert to token-level rewards (reward at last token)
        seq_len = input_ids.size(1)
        token_level_rewards = torch.zeros(input_ids.size(0), seq_len, device=self.device)
        
        if attention_mask is not None:
            # Put reward at last valid position
            lengths = attention_mask.sum(dim=1) - 1
            for i, length in enumerate(lengths):
                token_level_rewards[i, int(length)] = rewards[i]
        else:
            token_level_rewards[:, -1] = rewards
        
        data.batch["token_level_rewards"] = token_level_rewards
        data.batch["rewards"] = rewards
        
        return data
    
    def compute_log_probs(self, data: DataProto) -> DataProto:
        """Compute log probabilities from actor and reference models."""
        input_ids = data.batch["input_ids"]
        attention_mask = data.batch.get("attention_mask")
        
        # Actor log probs
        with torch.no_grad():
            old_log_probs = self.actor.get_log_probs(input_ids, attention_mask)
            data.batch["old_log_probs"] = old_log_probs
        
        # Reference log probs (for KL penalty)
        if self.ref_model is not None:
            ref_log_probs = self.ref_model.get_log_probs(input_ids, attention_mask)
            data.batch["ref_log_prob"] = ref_log_probs
        
        return data
    
    def compute_values(self, data: DataProto) -> DataProto:
        """Compute value estimates from critic."""
        input_ids = data.batch["input_ids"]
        attention_mask = data.batch.get("attention_mask")
        
        with torch.no_grad():
            values = self.critic(input_ids, attention_mask)
            # Align with response length
            if "response_mask" in data.batch:
                response_length = data.batch["response_mask"].size(1)
                values = values[:, -response_length:]
            data.batch["values"] = values
            data.batch["old_values"] = values.clone()
        
        return data
    
    def compute_advantages(self, data: DataProto) -> DataProto:
        """Compute advantages using specified estimator."""
        config = self.config
        
        # Get response mask
        if "response_mask" not in data.batch:
            # Default: all tokens are response
            seq_len = data.batch["input_ids"].size(1)
            data.batch["response_mask"] = torch.ones(
                data.batch["input_ids"].size(0), seq_len - 1,
                device=self.device
            )
        
        response_mask = data.batch["response_mask"]
        token_level_rewards = data.batch["token_level_rewards"]
        
        # Apply KL penalty if enabled
        if self.kl_ctrl is not None and "ref_log_prob" in data.batch:
            kld = kl_penalty(
                data.batch["old_log_probs"],
                data.batch["ref_log_prob"]
            )
            kld = kld * response_mask[:, :kld.size(1)]
            beta = self.kl_ctrl.value
            
            # Adjust rewards
            reward_len = min(token_level_rewards.size(1), kld.size(1))
            token_level_rewards = token_level_rewards.clone()
            token_level_rewards[:, :reward_len] = token_level_rewards[:, :reward_len] - beta * kld[:, :reward_len]
            data.batch["token_level_rewards"] = token_level_rewards
        
        # Compute advantages based on estimator type
        if config.adv_estimator == "gae":
            advantages, returns = compute_gae_advantage_return(
                token_level_rewards=token_level_rewards[:, :response_mask.size(1)],
                values=data.batch["values"],
                response_mask=response_mask,
                gamma=config.gamma,
                lam=config.lam
            )
        elif config.adv_estimator == "grpo":
            advantages, returns = compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards[:, :response_mask.size(1)],
                response_mask=response_mask,
                index=data.non_tensor_batch.get("uid", np.arange(len(data)))
            )
        else:
            # Try registry
            adv_fn = get_adv_estimator_fn(config.adv_estimator)
            advantages, returns = adv_fn(
                token_level_rewards=token_level_rewards[:, :response_mask.size(1)],
                response_mask=response_mask,
                index=data.non_tensor_batch.get("uid", np.arange(len(data)))
            )
        
        # Normalize advantages
        if config.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        
        return data
    
    def ppo_step(self, data: DataProto) -> Dict[str, float]:
        """Perform one PPO optimization step."""
        config = self.config
        metrics = {}
        
        input_ids = data.batch["input_ids"]
        attention_mask = data.batch.get("attention_mask")
        response_mask = data.batch["response_mask"]
        old_log_probs = data.batch["old_log_probs"]
        advantages = data.batch["advantages"]
        returns = data.batch["returns"]
        old_values = data.batch["old_values"]
        
        # Current log probs
        log_probs = self.actor.get_log_probs(input_ids, attention_mask)
        
        # Align dimensions
        min_len = min(log_probs.size(1), response_mask.size(1), advantages.size(1))
        log_probs = log_probs[:, :min_len]
        old_log_probs = old_log_probs[:, :min_len]
        response_mask = response_mask[:, :min_len]
        advantages = advantages[:, :min_len]
        returns = returns[:, :min_len]
        old_values = old_values[:, :min_len]
        
        # Policy loss
        policy_loss, policy_metrics = compute_policy_loss_ppo(
            old_log_probs=old_log_probs,
            log_probs=log_probs,
            advantages=advantages,
            response_mask=response_mask,
            clip_range=config.clip_range
        )
        metrics.update(policy_metrics)
        
        # Value loss
        values = self.critic(input_ids, attention_mask)[:, -min_len:]
        value_loss, value_metrics = compute_value_loss(
            values=values,
            returns=returns,
            old_values=old_values,
            response_mask=response_mask,
            clip_range=config.clip_range_value
        )
        metrics.update(value_metrics)
        
        # Entropy loss
        entropy_loss, entropy_metrics = compute_entropy_loss(
            log_probs=log_probs,
            response_mask=response_mask
        )
        metrics.update(entropy_metrics)
        
        # Total loss
        total_loss = compute_total_loss(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
            vf_coef=config.vf_coef,
            entropy_coef=config.entropy_coef
        )
        metrics["total_loss"] = total_loss.item()
        
        # Backward
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), config.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), config.max_grad_norm)
        
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        return metrics
    
    def train_step(self, data: DataProto) -> Dict[str, float]:
        """
        Perform a complete training step on a batch of data.
        
        Steps:
        1. Compute rewards
        2. Compute log probs and values
        3. Compute advantages
        4. PPO optimization (multiple epochs)
        """
        # Move data to device
        data = data.to(self.device)
        
        # Step 1: Compute rewards
        data = self.compute_rewards(data)
        
        # Step 2: Compute log probs and values
        data = self.compute_log_probs(data)
        data = self.compute_values(data)
        
        # Step 3: Compute advantages
        data = self.compute_advantages(data)
        
        # Step 4: PPO optimization epochs
        all_metrics = []
        for _ in range(self.config.ppo_epochs):
            # Create mini-batch iterator
            iterator = data.make_iterator(
                mini_batch_size=self.config.mini_batch_size,
                epochs=1
            )
            
            for mini_batch in iterator:
                mini_batch = mini_batch.to(self.device)
                metrics = self.ppo_step(mini_batch)
                all_metrics.append(metrics)
        
        # Aggregate metrics
        agg_metrics = {}
        for key in all_metrics[0].keys():
            agg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        # Update KL controller
        if self.kl_ctrl is not None and "policy/approx_kl" in agg_metrics:
            self.kl_ctrl.update(
                current_kl=agg_metrics["policy/approx_kl"],
                n_steps=len(data)
            )
            agg_metrics["kl/coef"] = self.kl_ctrl.value
        
        self.metrics_history.append(agg_metrics)
        return agg_metrics
    
    def fit(self, train_data: List[DataProto], 
            val_data: Optional[List[DataProto]] = None,
            num_epochs: int = None) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            train_data: List of DataProto batches for training
            val_data: Optional validation data
            num_epochs: Number of training epochs
        
        Returns:
            Dictionary of training history
        """
        num_epochs = num_epochs or self.config.total_epochs
        
        logger.info(f"Starting PPO training for {num_epochs} epochs")
        logger.info(f"Training batches: {len(train_data)}")
        
        history = {"train": [], "val": []}
        
        for epoch in range(num_epochs):
            epoch_metrics = []
            
            # Training
            self.actor.train()
            self.critic.train()
            
            pbar = tqdm(train_data, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)
                pbar.set_postfix({"loss": metrics.get("total_loss", 0)})
            
            # Aggregate epoch metrics
            epoch_agg = {}
            for key in epoch_metrics[0].keys():
                epoch_agg[key] = np.mean([m[key] for m in epoch_metrics])
            
            history["train"].append(epoch_agg)
            
            logger.info(f"Epoch {epoch+1} - Loss: {epoch_agg.get('total_loss', 0):.4f}")
        
        return history
