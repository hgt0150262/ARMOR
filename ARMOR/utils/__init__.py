"""
ARMOR utilities module.
Contains logging, model loading, and training utilities.
"""

from .logging_utils import (
    LoggingConfig,
    TrainingLogger,
    MetricsTracker,
    ProgressLogger,
    create_logger,
    WANDB_AVAILABLE,
    TENSORBOARD_AVAILABLE,
    SWANLAB_AVAILABLE,
)

from .model_utils import (
    ModelConfig,
    ModelManager,
    load_model_and_tokenizer,
    TRANSFORMERS_AVAILABLE,
    PEFT_AVAILABLE,
)

from .data_utils import (
    DataConfig,
    RLHFDataset,
    GSM8KDataset,
    AlpacaDataset,
    PreferenceDataset,
    load_dataset,
    create_dataloader,
    PromptSampler,
)

__all__ = [
    # Logging
    "LoggingConfig",
    "TrainingLogger",
    "MetricsTracker",
    "ProgressLogger",
    "create_logger",
    "WANDB_AVAILABLE",
    "TENSORBOARD_AVAILABLE",
    "SWANLAB_AVAILABLE",
    # Model
    "ModelConfig",
    "ModelManager",
    "load_model_and_tokenizer",
    "TRANSFORMERS_AVAILABLE",
    "PEFT_AVAILABLE",
    # Data
    "DataConfig",
    "RLHFDataset",
    "GSM8KDataset",
    "AlpacaDataset",
    "PreferenceDataset",
    "load_dataset",
    "create_dataloader",
    "PromptSampler",
]
