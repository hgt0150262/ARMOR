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
    """Enumeration of supported advantage estimation methods.
    
    Note: Users can register custom estimators via @register_adv_est decorator.
    """
    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"  # With baseline
    RLOO = "rloo"
    RLOO_VECTORIZED = "rloo_vectorized"  # Vectorized RLOO
    REMAX = "remax"
    DPO = "dpo"
    OPO = "opo"  # Optimal Policy Optimization (length-weighted baseline)
    GRPO_PASSK = "grpo_passk"  # GRPO with pass@k
    GRPO_VECTORIZED = "grpo_vectorized"  # Vectorized GRPO
    GPG = "gpg"  # Group Policy Gradient


class AlgorithmType(str, Enum):
    """Enumeration of supported RL algorithms."""
    PPO = "ppo"
    DPO = "dpo"
    REMAX = "remax"
    GRPO = "grpo"


# Registry for algorithm implementations
ALGORITHM_REGISTRY: Dict[str, Callable] = {}

# Registry for custom advantage estimators
ADV_ESTIMATOR_REGISTRY: Dict[str, Callable] = {}

# Registry for policy loss functions (from verl official)
POLICY_LOSS_REGISTRY: Dict[str, Callable] = {}


def register_policy_loss(name: str):
    """Decorator to register a policy loss function with the given name."""
    def decorator(fn):
        POLICY_LOSS_REGISTRY[name] = fn
        return fn
    return decorator


def get_policy_loss_fn(name: str):
    """Get the policy loss function by name."""
    if name not in POLICY_LOSS_REGISTRY:
        raise ValueError(f"Unknown policy loss: {name}. Available: {list(POLICY_LOSS_REGISTRY.keys())}")
    return POLICY_LOSS_REGISTRY[name]


def register_adv_est(name_or_enum):
    """Decorator to register an advantage estimator function."""
    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn
    return decorator


def register_algorithm(name_or_enum):
    """Decorator to register an algorithm implementation."""
    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        ALGORITHM_REGISTRY[name] = fn
        return fn
    return decorator


def get_algorithm_fn(name_or_enum):
    """Get the algorithm function by name."""
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm: {name}")
    return ALGORITHM_REGISTRY[name]


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


@register_adv_est(AdvantageEstimator.OPO)
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute OPO (Optimal Policy Optimization) advantages.
    
    OPO uses a length-weighted baseline within each group, which helps
    balance rewards across responses of different lengths.
    Based on: https://arxiv.org/pdf/2505.23585
    
    Args:
        token_level_rewards: Token-level rewards [batch, seq_len]
        response_mask: Response mask [batch, seq_len]
        index: Group indices (samples with same prompt have same index)
    
    Returns:
        advantages: OPO advantages [batch, seq_len]
        returns: Returns (same as advantages for OPO)
    """
    response_length = response_mask.sum(dim=-1)
    outcome_rewards = (token_level_rewards * response_mask).sum(dim=-1)
    
    unique_indices = np.unique(index)
    advantages = torch.zeros_like(outcome_rewards)
    
    for idx in unique_indices:
        mask = torch.tensor(index == idx, device=outcome_rewards.device)
        group_rewards = outcome_rewards[mask]
        group_lengths = response_length[mask]
        n = len(group_rewards)
        
        if n > 1:
            # Length-weighted baseline: B = Σ(len * reward) / Σ(len)
            baseline = (group_lengths * group_rewards).sum() / group_lengths.sum()
            advantages[mask] = group_rewards - baseline
        else:
            advantages[mask] = 0.0
    
    # Broadcast to token level
    advantages = advantages.unsqueeze(-1).expand_as(token_level_rewards)
    advantages = advantages * response_mask
    returns = advantages.clone()
    
    return advantages, returns


@register_policy_loss("vanilla")
@register_policy_loss("ppo")
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


# =============================================================================
# DPO (Direct Preference Optimization) Algorithm
# =============================================================================

@register_algorithm(AlgorithmType.DPO)
def compute_dpo_loss(
    policy_chosen_log_probs: torch.Tensor,
    policy_rejected_log_probs: torch.Tensor,
    ref_chosen_log_probs: torch.Tensor,
    ref_rejected_log_probs: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_mask: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid"
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute DPO (Direct Preference Optimization) loss.
    
    DPO directly optimizes the policy using preference data without a separate
    reward model. It treats the preference optimization as a classification problem.
    
    Reference: https://arxiv.org/abs/2305.18290
    
    Args:
        policy_chosen_log_probs: Log probs for chosen responses [batch, seq_len]
        policy_rejected_log_probs: Log probs for rejected responses [batch, seq_len]
        ref_chosen_log_probs: Reference log probs for chosen [batch, seq_len]
        ref_rejected_log_probs: Reference log probs for rejected [batch, seq_len]
        chosen_mask: Mask for chosen responses [batch, seq_len]
        rejected_mask: Mask for rejected responses [batch, seq_len]
        beta: Temperature parameter controlling deviation from reference
        label_smoothing: Label smoothing for the loss
        loss_type: Type of loss ("sigmoid", "hinge", "ipo")
    
    Returns:
        loss: DPO loss scalar
        metrics: Dictionary of metrics
    """
    # Sum log probs over sequence
    policy_chosen_logps = (policy_chosen_log_probs * chosen_mask).sum(dim=-1)
    policy_rejected_logps = (policy_rejected_log_probs * rejected_mask).sum(dim=-1)
    ref_chosen_logps = (ref_chosen_log_probs * chosen_mask).sum(dim=-1)
    ref_rejected_logps = (ref_rejected_log_probs * rejected_mask).sum(dim=-1)
    
    # Compute log ratios
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    
    # DPO logits: beta * (log(pi(y_w|x)/pi(y_l|x)) - log(pi_ref(y_w|x)/pi_ref(y_l|x)))
    logits = beta * (pi_logratios - ref_logratios)
    
    if loss_type == "sigmoid":
        # Standard DPO loss: -log(sigmoid(logits))
        losses = -F.logsigmoid(logits)
        
        # Apply label smoothing
        if label_smoothing > 0:
            smooth_losses = -F.logsigmoid(-logits)
            losses = (1 - label_smoothing) * losses + label_smoothing * smooth_losses
            
    elif loss_type == "hinge":
        # Hinge loss variant
        losses = torch.relu(1 - logits)
        
    elif loss_type == "ipo":
        # IPO (Identity Preference Optimization) loss
        losses = (logits - 1 / (2 * beta)) ** 2
        
    else:
        raise ValueError(f"Unknown DPO loss type: {loss_type}")
    
    loss = losses.mean()
    
    # Compute metrics
    with torch.no_grad():
        chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        reward_margins = chosen_rewards - rejected_rewards
    
    metrics = {
        "dpo/loss": loss.item(),
        "dpo/chosen_rewards": chosen_rewards.mean().item(),
        "dpo/rejected_rewards": rejected_rewards.mean().item(),
        "dpo/reward_accuracy": reward_accuracies.mean().item(),
        "dpo/reward_margin": reward_margins.mean().item(),
        "dpo/logits": logits.mean().item(),
    }
    
    return loss, metrics


def compute_dpo_loss_simple(
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    preferences: torch.Tensor,
    beta: float = 0.1
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Simplified DPO loss for paired preference data.
    
    Args:
        policy_log_probs: Log probs from policy [batch, seq_len] (alternating chosen/rejected)
        ref_log_probs: Log probs from reference [batch, seq_len]
        response_mask: Response mask [batch, seq_len]
        preferences: Binary labels (1 for chosen, 0 for rejected) [batch]
        beta: Temperature parameter
    
    Returns:
        loss: DPO loss
        metrics: Metrics dictionary
    """
    # Sum log probs
    policy_logps = (policy_log_probs * response_mask).sum(dim=-1)
    ref_logps = (ref_log_probs * response_mask).sum(dim=-1)
    
    # Log ratio
    log_ratio = policy_logps - ref_logps
    
    # Reshape for pairs (assumes batch is [chosen_1, rejected_1, chosen_2, rejected_2, ...])
    batch_size = policy_log_probs.shape[0] // 2
    log_ratio_chosen = log_ratio[::2]  # Even indices
    log_ratio_rejected = log_ratio[1::2]  # Odd indices
    
    # DPO loss
    logits = beta * (log_ratio_chosen - log_ratio_rejected)
    losses = -F.logsigmoid(logits)
    loss = losses.mean()
    
    metrics = {
        "dpo/loss": loss.item(),
        "dpo/accuracy": (logits > 0).float().mean().item(),
    }
    
    return loss, metrics


# =============================================================================
# ReMax Algorithm
# =============================================================================

@register_algorithm(AlgorithmType.REMAX)
@register_adv_est(AdvantageEstimator.REMAX)
def compute_remax_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ReMax advantages.
    
    ReMax uses the maximum reward in each group as the baseline, which provides
    a stronger signal for improvement compared to mean-based baselines.
    
    Reference: ReMax - A Simple, Effective, and Efficient Method for 
    Reinforcement Learning from Human Feedback
    
    Args:
        token_level_rewards: Token-level rewards [batch, seq_len]
        response_mask: Response mask [batch, seq_len]
        index: Group indices
    
    Returns:
        advantages: ReMax advantages [batch, seq_len]
        returns: Returns (same as advantages)
    """
    # Compute outcome rewards
    outcome_rewards = (token_level_rewards * response_mask).sum(dim=-1)
    unique_indices = np.unique(index)
    advantages = torch.zeros_like(outcome_rewards)
    
    for idx in unique_indices:
        mask = torch.tensor(index == idx, device=outcome_rewards.device)
        group_rewards = outcome_rewards[mask]
        
        # ReMax: use max reward as baseline
        max_reward = group_rewards.max()
        advantages[mask] = group_rewards - max_reward
    
    # Broadcast to token level
    advantages = advantages.unsqueeze(-1).expand_as(token_level_rewards)
    advantages = advantages * response_mask
    returns = advantages.clone()
    
    return advantages, returns


def compute_remax_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    beta: float = 0.1,
    normalize_reward: bool = True
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute ReMax policy gradient loss.
    
    ReMax combines REINFORCE with a max-based baseline for variance reduction.
    
    Args:
        log_probs: Policy log probabilities [batch, seq_len]
        ref_log_probs: Reference policy log probabilities [batch, seq_len]
        rewards: Rewards [batch] or [batch, seq_len]
        response_mask: Response mask [batch, seq_len]
        index: Group indices for baseline computation
        beta: KL penalty coefficient
        normalize_reward: Whether to normalize rewards
    
    Returns:
        loss: ReMax loss
        metrics: Metrics dictionary
    """
    # Handle token-level or sequence-level rewards
    if rewards.dim() == 2:
        sequence_rewards = (rewards * response_mask).sum(dim=-1)
    else:
        sequence_rewards = rewards
    
    # Compute advantages using max baseline
    unique_indices = np.unique(index)
    advantages = torch.zeros_like(sequence_rewards)
    
    for idx in unique_indices:
        mask = torch.tensor(index == idx, device=sequence_rewards.device)
        group_rewards = sequence_rewards[mask]
        max_reward = group_rewards.max()
        advantages[mask] = group_rewards - max_reward
    
    # Normalize advantages
    if normalize_reward:
        adv_std = advantages.std() + 1e-8
        advantages = advantages / adv_std
    
    # Compute policy gradient loss
    log_probs_sum = (log_probs * response_mask).sum(dim=-1)
    pg_loss = -(advantages * log_probs_sum).mean()
    
    # KL penalty
    kl = ((log_probs - ref_log_probs) * response_mask).sum(dim=-1)
    kl_loss = beta * kl.mean()
    
    loss = pg_loss + kl_loss
    
    metrics = {
        "remax/loss": loss.item(),
        "remax/pg_loss": pg_loss.item(),
        "remax/kl_loss": kl_loss.item(),
        "remax/advantage_mean": advantages.mean().item(),
        "remax/advantage_std": advantages.std().item(),
        "remax/reward_mean": sequence_rewards.mean().item(),
    }
    
    return loss, metrics


# =============================================================================
# GRPO (Group Relative Policy Optimization) Full Algorithm
# =============================================================================

@register_algorithm(AlgorithmType.GRPO)
def compute_grpo_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    beta: float = 0.1,
    clip_range: float = 0.2,
    normalize_by_std: bool = True
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute GRPO (Group Relative Policy Optimization) loss.
    
    GRPO normalizes advantages within groups and applies PPO-style clipping.
    
    Args:
        log_probs: Current policy log probs [batch, seq_len]
        ref_log_probs: Reference (old) policy log probs [batch, seq_len]
        rewards: Rewards [batch] or [batch, seq_len]
        response_mask: Response mask [batch, seq_len]
        index: Group indices
        beta: KL penalty coefficient
        clip_range: PPO clip range
        normalize_by_std: Whether to normalize by std within groups
    
    Returns:
        loss: GRPO loss
        metrics: Metrics dictionary
    """
    # Handle rewards
    if rewards.dim() == 2:
        sequence_rewards = (rewards * response_mask).sum(dim=-1)
    else:
        sequence_rewards = rewards
    
    # Compute group-normalized advantages
    unique_indices = np.unique(index)
    advantages = torch.zeros_like(sequence_rewards)
    
    for idx in unique_indices:
        mask = torch.tensor(index == idx, device=sequence_rewards.device)
        group_rewards = sequence_rewards[mask]
        mean = group_rewards.mean()
        std = group_rewards.std() + 1e-8
        
        if normalize_by_std:
            advantages[mask] = (group_rewards - mean) / std
        else:
            advantages[mask] = group_rewards - mean
    
    # Compute probability ratio
    log_probs_sum = (log_probs * response_mask).sum(dim=-1)
    ref_log_probs_sum = (ref_log_probs * response_mask).sum(dim=-1)
    ratio = torch.exp(log_probs_sum - ref_log_probs_sum)
    
    # Clipped ratio
    clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    
    # Policy loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    
    # KL penalty
    kl = log_probs_sum - ref_log_probs_sum
    kl_loss = beta * kl.abs().mean()
    
    loss = pg_loss + kl_loss
    
    # Metrics
    with torch.no_grad():
        clip_frac = ((ratio < 1 - clip_range) | (ratio > 1 + clip_range)).float().mean()
    
    metrics = {
        "grpo/loss": loss.item(),
        "grpo/pg_loss": pg_loss.item(),
        "grpo/kl_loss": kl_loss.item(),
        "grpo/clip_fraction": clip_frac.item(),
        "grpo/advantage_mean": advantages.mean().item(),
        "grpo/ratio_mean": ratio.mean().item(),
    }
    
    return loss, metrics
