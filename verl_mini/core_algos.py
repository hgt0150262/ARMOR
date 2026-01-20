"""
Simplified reproduction of verl's core PPO algorithms.
Implements advantage estimation, KL penalty, and policy loss functions.
"""

from enum import Enum
from typing import Optional, Callable, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F


class AdvantageEstimator(str, Enum):
    """Enumeration of supported advantage estimation methods."""
    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    RLOO = "rloo"
    REMAX = "remax"


# Registry for custom advantage estimators
ADV_ESTIMATOR_REGISTRY: Dict[str, Callable] = {}


def register_adv_est(name_or_enum):
    """Decorator to register an advantage estimator function."""
    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn
    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function by name."""
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdaptiveKLController:
    """
    Adaptive KL controller as described in:
    https://arxiv.org/pdf/1909.08593.pdf
    
    Dynamically adjusts the KL penalty coefficient based on the current KL divergence.
    """
    
    def __init__(self, init_kl_coef: float, target_kl: float, horizon: int):
        """
        Args:
            init_kl_coef: Initial KL penalty coefficient
            target_kl: Target KL divergence
            horizon: Horizon for adaptation
        """
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon
    
    def update(self, current_kl: float, n_steps: int):
        """Update KL coefficient based on current KL divergence."""
        proportional_error = np.clip(current_kl / self.target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller that maintains constant coefficient."""
    
    def __init__(self, kl_coef: float):
        self.value = kl_coef
    
    def update(self, current_kl: float, n_steps: int):
        """No-op for fixed controller."""
        pass


def get_kl_controller(kl_ctrl_config):
    """Factory function to create KL controller."""
    if kl_ctrl_config.type == "fixed":
        return FixedKLController(kl_ctrl_config.kl_coef)
    elif kl_ctrl_config.type == "adaptive":
        return AdaptiveKLController(
            init_kl_coef=kl_ctrl_config.kl_coef,
            target_kl=kl_ctrl_config.target_kl,
            horizon=kl_ctrl_config.horizon
        )
    else:
        raise ValueError(f"Unknown KL controller type: {kl_ctrl_config.type}")


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Compute mean over masked elements."""
    mask = mask.float()
    masked = tensor * mask
    return masked.sum(dim=axis) / (mask.sum(dim=axis) + 1e-8)


def kl_penalty(log_probs: torch.Tensor, 
               ref_log_probs: torch.Tensor,
               kl_penalty_type: str = "kl") -> torch.Tensor:
    """
    Compute KL divergence penalty between policy and reference policy.
    
    Args:
        log_probs: Log probabilities from current policy
        ref_log_probs: Log probabilities from reference policy
        kl_penalty_type: Type of KL penalty ("kl", "abs", "mse", "full")
    
    Returns:
        KL penalty values
    """
    if kl_penalty_type == "kl":
        # Standard KL: exp(log_ref - log_pi) - 1 - (log_ref - log_pi)
        # Simplified: log_pi - log_ref (approximation)
        return log_probs - ref_log_probs
    elif kl_penalty_type == "abs":
        return torch.abs(log_probs - ref_log_probs)
    elif kl_penalty_type == "mse":
        return 0.5 * (log_probs - ref_log_probs) ** 2
    elif kl_penalty_type == "full":
        # Full KL divergence
        return torch.exp(ref_log_probs) * (ref_log_probs - log_probs)
    else:
        raise ValueError(f"Unknown KL penalty type: {kl_penalty_type}")


@register_adv_est(AdvantageEstimator.GAE)
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    GAE provides a way to balance bias and variance in advantage estimation
    using a weighted combination of n-step returns.
    
    Args:
        token_level_rewards: Rewards at each token position [batch, seq_len]
        values: Value estimates at each position [batch, seq_len]
        response_mask: Mask for valid response tokens [batch, seq_len]
        gamma: Discount factor
        lam: GAE lambda parameter
    
    Returns:
        advantages: GAE advantages [batch, seq_len]
        returns: Value targets [batch, seq_len]
    """
    batch_size, seq_len = token_level_rewards.shape
    device = token_level_rewards.device
    
    # Initialize
    advantages = torch.zeros_like(token_level_rewards)
    last_gae = torch.zeros(batch_size, device=device)
    
    # Compute GAE backwards
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_value = torch.zeros(batch_size, device=device)
        else:
            next_value = values[:, t + 1]
        
        # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = token_level_rewards[:, t] + gamma * next_value - values[:, t]
        
        # GAE: A_t = delta_t + gamma * lam * A_{t+1}
        last_gae = delta + gamma * lam * last_gae * response_mask[:, t]
        advantages[:, t] = last_gae
    
    # Returns = advantages + values
    returns = advantages + values
    
    # Apply mask
    advantages = advantages * response_mask
    returns = returns * response_mask
    
    return advantages, returns


@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    norm_adv_by_std_in_grpo: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GRPO (Group Relative Policy Optimization) advantages.
    
    GRPO normalizes advantages within groups of samples sharing the same prompt,
    which helps with training stability.
    
    Args:
        token_level_rewards: Token-level rewards [batch, seq_len]
        response_mask: Response mask [batch, seq_len]
        index: Group indices (samples with same prompt have same index)
        norm_adv_by_std_in_grpo: Whether to normalize by std within groups
    
    Returns:
        advantages: GRPO advantages [batch, seq_len]
        returns: Returns (same as advantages for GRPO)
    """
    # Compute outcome rewards (sum over sequence)
    outcome_rewards = (token_level_rewards * response_mask).sum(dim=-1)
    
    # Group by index and normalize
    unique_indices = np.unique(index)
    advantages = torch.zeros_like(outcome_rewards)
    
    for idx in unique_indices:
        mask = torch.tensor(index == idx, device=outcome_rewards.device)
        group_rewards = outcome_rewards[mask]
        
        # Normalize within group
        mean = group_rewards.mean()
        std = group_rewards.std() + 1e-8
        
        if norm_adv_by_std_in_grpo:
            advantages[mask] = (group_rewards - mean) / std
        else:
            advantages[mask] = group_rewards - mean
    
    # Broadcast to token level
    advantages = advantages.unsqueeze(-1).expand_as(token_level_rewards)
    advantages = advantages * response_mask
    
    returns = advantages.clone()
    
    return advantages, returns


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)
def compute_reinforce_pp_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute REINFORCE++ advantages with token-level baseline.
    
    REINFORCE++ uses a running baseline at each token position to reduce variance.
    """
    # Compute cumulative rewards from each position to end
    batch_size, seq_len = token_level_rewards.shape
    device = token_level_rewards.device
    
    # Reverse cumsum for future rewards
    reversed_rewards = torch.flip(token_level_rewards * response_mask, dims=[1])
    cumsum_rewards = torch.cumsum(reversed_rewards, dim=1)
    returns = torch.flip(cumsum_rewards, dims=[1])
    
    # Baseline: mean return at each position
    baseline = returns.mean(dim=0, keepdim=True)
    advantages = returns - baseline
    
    # Normalize
    std = advantages.std() + 1e-8
    advantages = advantages / std
    advantages = advantages * response_mask
    
    return advantages, returns


@register_adv_est(AdvantageEstimator.RLOO)
def compute_rloo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute RLOO (Reinforce Leave-One-Out) advantages.
    
    For each sample, the baseline is the mean reward of other samples
    in the same group (leave-one-out).
    """
    outcome_rewards = (token_level_rewards * response_mask).sum(dim=-1)
    unique_indices = np.unique(index)
    advantages = torch.zeros_like(outcome_rewards)
    
    for idx in unique_indices:
        mask = torch.tensor(index == idx, device=outcome_rewards.device)
        group_rewards = outcome_rewards[mask]
        n = len(group_rewards)
        
        if n > 1:
            # Leave-one-out baseline
            total = group_rewards.sum()
            baseline = (total - group_rewards) / (n - 1)
            advantages[mask] = group_rewards - baseline
        else:
            advantages[mask] = 0.0
    
    # Broadcast to token level
    advantages = advantages.unsqueeze(-1).expand_as(token_level_rewards)
    advantages = advantages * response_mask
    returns = advantages.clone()
    
    return advantages, returns


def compute_policy_loss_ppo(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_range: float = 0.2,
    clip_range_high: Optional[float] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute PPO clipped policy loss.
    
    The PPO objective clips the probability ratio to prevent too large policy updates.
    
    Args:
        old_log_probs: Log probs from old policy [batch, seq_len]
        log_probs: Log probs from current policy [batch, seq_len]
        advantages: Advantage estimates [batch, seq_len]
        response_mask: Response mask [batch, seq_len]
        clip_range: PPO clip range (epsilon)
        clip_range_high: Upper clip range (if different from lower)
    
    Returns:
        loss: Policy loss scalar
        metrics: Dictionary of metrics
    """
    if clip_range_high is None:
        clip_range_high = clip_range
    
    # Compute probability ratio
    ratio = torch.exp(log_probs - old_log_probs)
    
    # Clipped ratio
    clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range_high)
    
    # Policy gradient loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    
    # Take max (more conservative)
    pg_loss = torch.max(pg_loss1, pg_loss2)
    
    # Apply mask and average
    loss = masked_mean(pg_loss, response_mask, axis=-1).mean()
    
    # Compute metrics
    with torch.no_grad():
        clip_frac = ((ratio < 1 - clip_range) | (ratio > 1 + clip_range_high)).float()
        clip_frac = masked_mean(clip_frac, response_mask, axis=-1).mean()
        
        approx_kl = masked_mean((ratio - 1) - torch.log(ratio), response_mask, axis=-1).mean()
    
    metrics = {
        "policy/loss": loss.item(),
        "policy/clip_fraction": clip_frac.item(),
        "policy/approx_kl": approx_kl.item(),
        "policy/ratio_mean": ratio.mean().item(),
    }
    
    return loss, metrics


def compute_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor,
    response_mask: torch.Tensor,
    clip_range: float = 0.2
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute clipped value function loss.
    
    Args:
        values: Current value estimates [batch, seq_len]
        returns: Target returns [batch, seq_len]
        old_values: Old value estimates [batch, seq_len]
        response_mask: Response mask [batch, seq_len]
        clip_range: Value clip range
    
    Returns:
        loss: Value loss scalar
        metrics: Dictionary of metrics
    """
    # Clipped value
    values_clipped = old_values + torch.clamp(
        values - old_values, -clip_range, clip_range
    )
    
    # Value losses
    vf_loss1 = (values - returns) ** 2
    vf_loss2 = (values_clipped - returns) ** 2
    
    # Take max
    vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2)
    
    # Apply mask and average
    loss = masked_mean(vf_loss, response_mask, axis=-1).mean()
    
    metrics = {
        "value/loss": loss.item(),
        "value/mean": values.mean().item(),
    }
    
    return loss, metrics


def compute_entropy_loss(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute entropy bonus for exploration.
    
    Higher entropy encourages more exploration.
    """
    # Entropy approximation: -sum(p * log(p)) ≈ -mean(log_p)
    # Since we have log_probs, entropy ≈ -log_probs
    entropy = -log_probs
    
    # Negative because we want to maximize entropy (minimize negative entropy)
    loss = -masked_mean(entropy, response_mask, axis=-1).mean()
    
    metrics = {
        "entropy/mean": (-loss).item(),
    }
    
    return loss, metrics


def compute_total_loss(
    policy_loss: torch.Tensor,
    value_loss: torch.Tensor,
    entropy_loss: torch.Tensor,
    vf_coef: float = 0.5,
    entropy_coef: float = 0.01
) -> torch.Tensor:
    """Compute total PPO loss."""
    return policy_loss + vf_coef * value_loss + entropy_coef * entropy_loss
