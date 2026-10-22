"""
ARMOR trainer module.
Contains PPO/GRPO trainers and RLHF training utilities.
"""

from .ppo import (
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    compute_rloo_advantage,
    compute_remax_advantage,
    compute_dpo_loss,
    compute_remax_loss,
    compute_grpo_loss,
    AdvantageEstimator,
    AlgorithmType,
    ALGORITHM_REGISTRY,
    ADV_ESTIMATOR_REGISTRY,
)

__all__ = [
    "compute_gae_advantage_return",
    "compute_grpo_outcome_advantage",
    "compute_rloo_advantage",
    "compute_remax_advantage",
    "compute_dpo_loss",
    "compute_remax_loss",
    "compute_grpo_loss",
    "AdvantageEstimator",
    "AlgorithmType",
    "ALGORITHM_REGISTRY",
    "ADV_ESTIMATOR_REGISTRY",
]
