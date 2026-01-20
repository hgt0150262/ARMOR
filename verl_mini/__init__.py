# Copyright 2024 - verl_mini reproduction
# A simplified reproduction of ByteDance's verl framework
# for understanding RLHF training concepts

from .protocol import DataProto, DataProtoItem
from .base_config import BaseConfig
from .core_algos import (
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
from .ray_worker import (
    Role,
    RayResourcePool,
    ResourcePoolManager,
    RayWorkerGroup,
    RayWorker,
    init_ray_cluster,
    shutdown_ray,
    RAY_AVAILABLE,
)
from .ray_trainer import (
    RayPPOConfig,
    RayPPOTrainer,
    create_ray_ppo_trainer,
)
from .logging_utils import (
    LoggingConfig,
    TrainingLogger,
    MetricsTracker,
    ProgressLogger,
    create_logger,
    WANDB_AVAILABLE,
    TENSORBOARD_AVAILABLE,
)
from .model_utils import (
    ModelConfig,
    ModelManager,
    load_model_and_tokenizer,
    TRANSFORMERS_AVAILABLE,
    PEFT_AVAILABLE,
)
from .rlhf_trainer import (
    RLHFConfig,
    RLHFTrainer,
    create_rlhf_trainer,
)

__version__ = "0.1.0"
__all__ = [
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
