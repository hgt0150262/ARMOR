"""
ARMOR - A simplified reproduction of the verl RLHF framework.
Implements core components for reinforcement learning from human feedback.
"""

# Core protocol and config
from .protocol import DataProto, DataProtoItem
from .base_config import BaseConfig

# Trainer module - algorithms
from .trainer.ppo import (
    AdvantageEstimator,
    AlgorithmType,
    compute_dpo_loss,
    compute_dpo_loss_simple,
    compute_remax_advantage,
    compute_remax_loss,
    compute_grpo_loss,
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    compute_rloo_advantage,
    get_algorithm_fn,
    get_adv_estimator_fn,
    ALGORITHM_REGISTRY,
    ADV_ESTIMATOR_REGISTRY,
)

# Single controller - Ray distributed
from .single_controller.ray import (
    Role,
    RayResourcePool,
    ResourcePoolManager,
    RayWorkerGroup,
    RayWorker,
    init_ray_cluster,
    shutdown_ray,
    RAY_AVAILABLE,
)

# Utils - logging and model
from .utils import (
    LoggingConfig,
    TrainingLogger,
    MetricsTracker,
    ProgressLogger,
    create_logger,
    WANDB_AVAILABLE,
    TENSORBOARD_AVAILABLE,
    ModelConfig,
    ModelManager,
    load_model_and_tokenizer,
    TRANSFORMERS_AVAILABLE,
    PEFT_AVAILABLE,
)

# Trainer - RLHF
from .trainer.rlhf_trainer import (
    RLHFConfig,
    RLHFTrainer,
    create_rlhf_trainer,
)

# Trainer - Ray PPO
from .trainer.ppo.ray_trainer import (
    RayPPOConfig,
    RayPPOTrainer,
    create_ray_ppo_trainer,
)

__version__ = "0.2.0"

__all__ = [
    # Core
    "DataProto", 
    "DataProtoItem", 
    "BaseConfig", 
    "__version__",
    # Algorithm types
    "AdvantageEstimator",
    "AlgorithmType",
    # DPO
    "compute_dpo_loss",
    "compute_dpo_loss_simple",
    # ReMax
    "compute_remax_advantage",
    "compute_remax_loss",
    # GRPO
    "compute_grpo_loss",
    "compute_grpo_outcome_advantage",
    # GAE/RLOO
    "compute_gae_advantage_return",
    "compute_rloo_advantage",
    # Registry
    "get_algorithm_fn",
    "get_adv_estimator_fn",
    "ALGORITHM_REGISTRY",
    "ADV_ESTIMATOR_REGISTRY",
    # Ray distributed
    "Role",
    "RayResourcePool",
    "ResourcePoolManager",
    "RayWorkerGroup",
    "RayWorker",
    "init_ray_cluster",
    "shutdown_ray",
    "RAY_AVAILABLE",
    # Ray trainer
    "RayPPOConfig",
    "RayPPOTrainer",
    "create_ray_ppo_trainer",
    # Logging
    "LoggingConfig",
    "TrainingLogger",
    "MetricsTracker",
    "ProgressLogger",
    "create_logger",
    "WANDB_AVAILABLE",
    "TENSORBOARD_AVAILABLE",
    # Model utilities
    "ModelConfig",
    "ModelManager",
    "load_model_and_tokenizer",
    "TRANSFORMERS_AVAILABLE",
    "PEFT_AVAILABLE",
    # RLHF trainer
    "RLHFConfig",
    "RLHFTrainer",
    "create_rlhf_trainer",
]
