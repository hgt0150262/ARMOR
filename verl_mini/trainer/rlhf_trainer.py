"""
RLHF Trainer for verl_mini.
Implements complete RLHF training loop with PPO/GRPO algorithms.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from verl_mini.protocol import DataProto
from .ppo.core_algos import (
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    compute_rloo_advantage,
    compute_remax_advantage,
    AdvantageEstimator,
)
from verl_mini.utils.logging_utils import TrainingLogger, LoggingConfig, create_logger
from verl_mini.utils.model_utils import ModelConfig, ModelManager


@dataclass
class RLHFConfig:
    """Configuration for RLHF training."""
    
    # Training settings
    total_epochs: int = 3
    batch_size: int = 8
    mini_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    
    # PPO settings
    ppo_epochs: int = 4
    clip_range: float = 0.2
    clip_range_value: float = 0.2
    
    # Advantage estimation
    adv_estimator: str = "grpo"  # "gae", "grpo", "rloo", "remax"
    gamma: float = 1.0
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    
    # KL penalty
    use_kl_penalty: bool = True
    kl_coef: float = 0.1
    target_kl: Optional[float] = None
    
    # Loss coefficients
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Learning rate
    actor_lr: float = 1e-6
    critic_lr: float = 5e-6
    max_grad_norm: float = 1.0
    
    # Generation settings
    max_prompt_length: int = 256
    max_response_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Checkpointing
    save_steps: int = 100
    save_dir: str = "./checkpoints"
    
    # Logging
    log_interval: int = 10
    use_wandb: bool = False
    use_tensorboard: bool = True
    project_name: str = "verl_mini_rlhf"


class RLHFTrainer:
    """Complete RLHF trainer with PPO/GRPO algorithms."""
    
    def __init__(
        self,
        config: RLHFConfig,
        model_manager: ModelManager,
        reward_fn: Optional[Callable] = None,
        tokenizer=None,
    ):
        self.config = config
        self.model_manager = model_manager
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer or model_manager.tokenizer
        
        self.model = model_manager.model
        self.ref_model = model_manager.ref_model
        
        # Create optimizer
        self.optimizer = None
        self.scheduler = None
        
        # Initialize logger
        self.logger = create_logger(
            project_name=config.project_name,
            use_wandb=config.use_wandb,
            use_tensorboard=config.use_tensorboard,
            log_dir=config.save_dir,
        )
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
    def setup(self, num_training_steps: int):
        """Setup training components."""
        # Create reference model if needed
        if self.ref_model is None and self.config.use_kl_penalty:
            self.ref_model = self.model_manager.create_reference_model()
            
        # Create optimizer and scheduler
        self.optimizer, self.scheduler = self.model_manager.create_optimizer(
            num_training_steps=num_training_steps,
        )
        
        # Log configuration
        self.logger.log_config({
            "total_epochs": self.config.total_epochs,
            "batch_size": self.config.batch_size,
            "adv_estimator": self.config.adv_estimator,
            "ppo_epochs": self.config.ppo_epochs,
            "clip_range": self.config.clip_range,
            "actor_lr": self.config.actor_lr,
            **self.model_manager.get_model_info(),
        })
        
    def generate_responses(
        self,
        prompts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Generate responses for prompts."""
        self.model.eval()
        
        # Tokenize prompts
        prompt_encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length,
            return_tensors="pt",
        )
        
        input_ids = prompt_encodings["input_ids"].to(self.model.device)
        attention_mask = prompt_encodings["attention_mask"].to(self.model.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_response_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
        # Extract responses
        response_ids = output_ids[:, input_ids.shape[1]:]
        responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        
        return input_ids, response_ids, responses
    
    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> torch.Tensor:
        """Compute rewards for prompt-response pairs."""
        if self.reward_fn is None:
            # Default: use response length as reward (for testing)
            rewards = torch.tensor([len(r) / 100.0 for r in responses])
        else:
            rewards = self.reward_fn(prompts, responses)
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards)
                
        return rewards.to(self.model.device)
    
    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        response_ids: torch.Tensor,
        model: nn.Module,
    ) -> torch.Tensor:
        """Compute log probabilities for responses."""
        # Concatenate input and response
        full_ids = torch.cat([input_ids, response_ids], dim=1)
        
        # Forward pass
        with torch.no_grad() if model != self.model else torch.enable_grad():
            outputs = model(input_ids=full_ids, return_dict=True)
            
        logits = outputs.logits
        
        # Get log probs for response tokens only
        response_start = input_ids.shape[1]
        response_logits = logits[:, response_start-1:-1, :]  # Shift by 1
        
        log_probs = F.log_softmax(response_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=response_ids.unsqueeze(-1),
        ).squeeze(-1)
        
        return token_log_probs
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages using configured estimator."""
        batch_size = rewards.shape[0]
        
        if self.config.adv_estimator == "gae":
            # GAE requires token-level rewards and values
            seq_len = response_mask.shape[1] if response_mask is not None else 1
            token_rewards = rewards.unsqueeze(-1).expand(-1, seq_len)
            if response_mask is not None:
                token_rewards = token_rewards * response_mask
            if values is None:
                values = token_rewards.clone()  # Use rewards as value estimate
            advantages, returns = compute_gae_advantage_return(
                token_level_rewards=token_rewards,
                values=values,
                response_mask=response_mask if response_mask is not None else torch.ones_like(token_rewards),
                gamma=self.config.gamma,
                lam=self.config.gae_lambda,
            )
        elif self.config.adv_estimator == "grpo":
            # GRPO requires token-level rewards and response mask
            import numpy as np
            batch_size = rewards.shape[0]
            if response_mask is not None:
                token_rewards = rewards.unsqueeze(-1).expand(-1, response_mask.shape[1]) * response_mask
            else:
                token_rewards = rewards.unsqueeze(-1)
            index = np.arange(batch_size)  # Each sample is its own group
            advantages, returns = compute_grpo_outcome_advantage(
                token_level_rewards=token_rewards,
                response_mask=response_mask if response_mask is not None else torch.ones_like(token_rewards),
                index=index,
            )
        elif self.config.adv_estimator == "rloo":
            advantages = compute_rloo_advantage(
                rewards=rewards,
            )
            returns = rewards
        elif self.config.adv_estimator == "remax":
            advantages = compute_remax_advantage(
                rewards=rewards,
            )
            returns = rewards
        else:
            raise ValueError(f"Unknown advantage estimator: {self.config.adv_estimator}")
            
        # Normalize advantages
        if self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        return advantages, returns
    
    def compute_kl_penalty(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute KL divergence penalty."""
        kl = log_probs - ref_log_probs
        
        if mask is not None:
            kl = kl * mask
            kl = kl.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        else:
            kl = kl.mean(dim=-1)
            
        return kl
    
    def ppo_step(
        self,
        input_ids: torch.Tensor,
        response_ids: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """Perform a PPO optimization step."""
        self.model.train()
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        
        batch_size = input_ids.shape[0]
        indices = torch.randperm(batch_size)
        
        for start in range(0, batch_size, self.config.mini_batch_size):
            end = min(start + self.config.mini_batch_size, batch_size)
            mb_indices = indices[start:end]
            
            mb_input_ids = input_ids[mb_indices]
            mb_response_ids = response_ids[mb_indices]
            mb_old_log_probs = old_log_probs[mb_indices]
            mb_advantages = advantages[mb_indices]
            
            # Forward pass
            full_ids = torch.cat([mb_input_ids, mb_response_ids], dim=1)
            outputs = self.model(input_ids=full_ids, return_dict=True)
            logits = outputs.logits
            
            # Compute new log probs
            response_start = mb_input_ids.shape[1]
            response_logits = logits[:, response_start-1:-1, :]
            log_probs = F.log_softmax(response_logits, dim=-1)
            new_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=mb_response_ids.unsqueeze(-1),
            ).squeeze(-1)
            
            # Compute entropy
            probs = F.softmax(response_logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            
            # Sum log probs over sequence
            new_log_probs_sum = new_log_probs.sum(dim=-1)
            old_log_probs_sum = mb_old_log_probs.sum(dim=-1)
            
            # Policy loss with clipping (add numerical stability)
            log_ratio = new_log_probs_sum - old_log_probs_sum
            log_ratio = torch.clamp(log_ratio, -10, 10)  # Prevent extreme ratios
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range)
            
            # Ensure advantages match ratio shape (reduce to batch dimension if needed)
            if mb_advantages.dim() > 1:
                mb_adv_scalar = mb_advantages.sum(dim=-1)  # Sum over sequence
            else:
                mb_adv_scalar = mb_advantages
            
            policy_loss = -torch.min(
                ratio * mb_adv_scalar,
                clipped_ratio * mb_adv_scalar,
            ).mean()
            
            # Total loss
            loss = policy_loss - self.config.entropy_coef * entropy
            
            # KL penalty
            if self.config.use_kl_penalty and self.ref_model is not None:
                with torch.no_grad():
                    ref_outputs = self.ref_model(input_ids=full_ids, return_dict=True)
                    ref_logits = ref_outputs.logits
                    ref_response_logits = ref_logits[:, response_start-1:-1, :]
                    ref_log_probs = F.log_softmax(ref_response_logits, dim=-1)
                    ref_token_log_probs = torch.gather(
                        ref_log_probs,
                        dim=-1,
                        index=mb_response_ids.unsqueeze(-1),
                    ).squeeze(-1)
                    
                kl = self.compute_kl_penalty(new_log_probs, ref_token_log_probs)
                loss = loss + self.config.kl_coef * kl.mean()
                total_kl += kl.mean().item()
            
            # Backward pass
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            total_policy_loss += policy_loss.item()
            total_entropy += entropy.item()
            
        # Optimizer step
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        num_minibatches = (batch_size + self.config.mini_batch_size - 1) // self.config.mini_batch_size
        
        return {
            "loss": total_loss / num_minibatches,
            "policy_loss": total_policy_loss / num_minibatches,
            "entropy": total_entropy / num_minibatches,
            "kl": total_kl / num_minibatches if self.config.use_kl_penalty else 0.0,
            "lr": self.scheduler.get_last_lr()[0],
        }
    
    def train_step(
        self,
        prompts: List[str],
    ) -> Dict[str, float]:
        """Perform a complete training step."""
        # Generate responses
        input_ids, response_ids, responses = self.generate_responses(prompts)
        
        # Compute rewards
        rewards = self.compute_rewards(prompts, responses)
        
        # Compute old log probs
        with torch.no_grad():
            old_log_probs = self.compute_log_probs(input_ids, response_ids, self.model)
            
        # Compute advantages
        response_mask = (response_ids != self.tokenizer.pad_token_id).float()
        advantages, returns = self.compute_advantages(rewards, response_mask=response_mask)
        
        # Expand advantages to match sequence length if needed
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(-1).expand(-1, response_ids.shape[1])
            returns = returns.unsqueeze(-1).expand(-1, response_ids.shape[1]) if returns.dim() == 1 else returns
        
        # PPO updates
        all_metrics = []
        for ppo_epoch in range(self.config.ppo_epochs):
            metrics = self.ppo_step(
                input_ids=input_ids,
                response_ids=response_ids,
                old_log_probs=old_log_probs,
                advantages=advantages,
                returns=returns,
            )
            all_metrics.append(metrics)
            
        # Average metrics across PPO epochs
        avg_metrics = {
            key: sum(m[key] for m in all_metrics) / len(all_metrics)
            for key in all_metrics[0].keys()
        }
        
        # Add reward stats
        avg_metrics["reward_mean"] = rewards.mean().item()
        avg_metrics["reward_std"] = rewards.std().item()
        avg_metrics["response_length"] = response_ids.shape[1]
        
        return avg_metrics
    
    def train(
        self,
        train_prompts: List[str],
        eval_prompts: Optional[List[str]] = None,
        logger: Optional[Any] = None,
    ) -> List[Dict[str, float]]:
        """Run full training loop."""
        num_batches = len(train_prompts) // self.config.batch_size
        num_training_steps = num_batches * self.config.total_epochs * self.config.ppo_epochs
        
        self.setup(num_training_steps)
        
        # Use external logger if provided
        external_logger = logger
        
        all_metrics = []
        
        print(f"\nStarting RLHF training")
        print(f"  Epochs: {self.config.total_epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Total batches per epoch: {num_batches}")
        print(f"  Advantage estimator: {self.config.adv_estimator}")
        print()
        
        for epoch in range(self.config.total_epochs):
            self.current_epoch = epoch
            epoch_metrics = []
            
            # Shuffle prompts
            import random
            shuffled_prompts = train_prompts.copy()
            random.shuffle(shuffled_prompts)
            
            pbar = tqdm(range(0, len(shuffled_prompts), self.config.batch_size), 
                       desc=f"Epoch {epoch}")
            
            for batch_start in pbar:
                batch_prompts = shuffled_prompts[batch_start:batch_start + self.config.batch_size]
                
                if len(batch_prompts) < self.config.batch_size:
                    continue  # Skip incomplete batches
                    
                # Training step
                metrics = self.train_step(batch_prompts)
                epoch_metrics.append(metrics)
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "reward": f"{metrics['reward_mean']:.4f}",
                })
                
                # Logging
                self.global_step += 1
                self.logger.log_metrics(metrics, step=self.global_step)
                
                # Also log to external logger (e.g., SwanLab)
                if external_logger is not None:
                    try:
                        external_logger.log_metrics(metrics, step=self.global_step)
                    except Exception:
                        pass  # Ignore logging errors
                        
                # Direct SwanLab logging (minimind style)
                try:
                    import swanlab
                    swanlab.log(metrics, step=self.global_step)
                except Exception:
                    pass
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")
                    
            # Epoch summary
            if epoch_metrics:
                epoch_summary = {
                    key: sum(m[key] for m in epoch_metrics) / len(epoch_metrics)
                    for key in epoch_metrics[0].keys()
                }
                self.logger.log_epoch(epoch, epoch_summary)
                all_metrics.append(epoch_summary)
                
            # Evaluation
            if eval_prompts:
                eval_metrics = self.evaluate(eval_prompts)
                self.logger.log_metrics({f"eval/{k}": v for k, v in eval_metrics.items()})
                
        # Final save
        self.save_checkpoint("final")
        self.logger.close()
        
        return all_metrics
    
    def evaluate(
        self,
        prompts: List[str],
    ) -> Dict[str, float]:
        """Evaluate on a set of prompts."""
        self.model.eval()
        
        all_rewards = []
        all_lengths = []
        
        with torch.no_grad():
            for i in range(0, len(prompts), self.config.batch_size):
                batch_prompts = prompts[i:i + self.config.batch_size]
                
                _, response_ids, responses = self.generate_responses(batch_prompts)
                rewards = self.compute_rewards(batch_prompts, responses)
                
                all_rewards.extend(rewards.tolist())
                all_lengths.extend([len(r) for r in responses])
                
        return {
            "reward_mean": sum(all_rewards) / len(all_rewards),
            "reward_std": (sum((r - sum(all_rewards)/len(all_rewards))**2 for r in all_rewards) / len(all_rewards)) ** 0.5,
            "response_length_mean": sum(all_lengths) / len(all_lengths),
        }
    
    def save_checkpoint(self, name: str):
        """Save training checkpoint."""
        save_path = Path(self.config.save_dir) / name
        self.model_manager.save_model(str(save_path))
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
        }
        torch.save(state, save_path / "training_state.pt")
        
        print(f"Checkpoint saved: {save_path}")


def create_rlhf_trainer(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    use_lora: bool = True,
    adv_estimator: str = "grpo",
    reward_fn: Optional[Callable] = None,
    **kwargs,
) -> RLHFTrainer:
    """Factory function to create RLHF trainer."""
    from .model_utils import ModelConfig, ModelManager
    
    model_config = ModelConfig(
        model_name_or_path=model_name,
        use_lora=use_lora,
    )
    
    model_manager = ModelManager(model_config)
    model_manager.load_model()
    
    if use_lora:
        model_manager.apply_lora()
        
    rlhf_config = RLHFConfig(
        adv_estimator=adv_estimator,
        **kwargs,
    )
    
    trainer = RLHFTrainer(
        config=rlhf_config,
        model_manager=model_manager,
        reward_fn=reward_fn,
    )
    
    return trainer
