"""
FSDP (Fully Sharded Data Parallel) utilities for ARMOR.
Provides model parallelism support for training large models.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Type, Callable
import functools

import torch
import torch.nn as nn
import torch.distributed as dist

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
        CPUOffload,
        BackwardPrefetch,
    )
    from torch.distributed.fsdp.wrap import (
        transformer_auto_wrap_policy,
        size_based_auto_wrap_policy,
    )
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False
    FSDP = None


@dataclass
class FSDPConfig:
    """Configuration for FSDP training."""
    
    # Sharding strategy
    sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD
    
    # Mixed precision
    use_mixed_precision: bool = True
    param_dtype: str = "bfloat16"
    reduce_dtype: str = "float32"
    buffer_dtype: str = "float32"
    
    # CPU offload
    cpu_offload: bool = False
    
    # Backward prefetch
    backward_prefetch: str = "BACKWARD_PRE"  # BACKWARD_PRE, BACKWARD_POST, None
    
    # Auto wrap policy
    auto_wrap_policy: str = "transformer"  # transformer, size_based, none
    min_num_params: int = 1e6  # For size_based policy
    
    # Transformer layer class names (for transformer policy)
    transformer_layer_cls_names: List[str] = field(default_factory=lambda: [
        "LlamaDecoderLayer",
        "Qwen2DecoderLayer", 
        "GPT2Block",
        "TransformerBlock",
    ])
    
    # Activation checkpointing
    activation_checkpointing: bool = True
    
    # State dict type
    state_dict_type: str = "FULL_STATE_DICT"  # FULL_STATE_DICT, SHARDED_STATE_DICT
    
    def get_sharding_strategy(self) -> "ShardingStrategy":
        """Get ShardingStrategy enum."""
        if not FSDP_AVAILABLE:
            raise ImportError("FSDP not available")
            
        strategies = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
            "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
        }
        return strategies.get(self.sharding_strategy, ShardingStrategy.FULL_SHARD)
    
    def get_mixed_precision(self) -> Optional["MixedPrecision"]:
        """Get MixedPrecision policy."""
        if not self.use_mixed_precision or not FSDP_AVAILABLE:
            return None
            
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        
        return MixedPrecision(
            param_dtype=dtype_map.get(self.param_dtype, torch.bfloat16),
            reduce_dtype=dtype_map.get(self.reduce_dtype, torch.float32),
            buffer_dtype=dtype_map.get(self.buffer_dtype, torch.float32),
        )
    
    def get_cpu_offload(self) -> Optional["CPUOffload"]:
        """Get CPUOffload policy."""
        if not self.cpu_offload or not FSDP_AVAILABLE:
            return None
        return CPUOffload(offload_params=True)
    
    def get_backward_prefetch(self) -> Optional["BackwardPrefetch"]:
        """Get BackwardPrefetch policy."""
        if not FSDP_AVAILABLE:
            return None
            
        prefetch_map = {
            "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
            "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
        }
        return prefetch_map.get(self.backward_prefetch)


def init_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
) -> int:
    """Initialize distributed training."""
    if dist.is_initialized():
        return dist.get_rank()
        
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if init_method is None:
        init_method = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")
        init_method = f"tcp://{init_method}:{port}"
    
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    
    torch.cuda.set_device(local_rank)
    
    return rank


def get_auto_wrap_policy(
    config: FSDPConfig,
    model: nn.Module,
) -> Optional[Callable]:
    """Get auto wrap policy for FSDP."""
    if not FSDP_AVAILABLE:
        return None
        
    if config.auto_wrap_policy == "none":
        return None
        
    if config.auto_wrap_policy == "size_based":
        return functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=int(config.min_num_params),
        )
        
    if config.auto_wrap_policy == "transformer":
        # Find transformer layer classes in model
        layer_classes = []
        for name in config.transformer_layer_cls_names:
            for module in model.modules():
                if type(module).__name__ == name:
                    layer_classes.append(type(module))
                    break
                    
        if not layer_classes:
            # Fallback to common patterns
            for module in model.modules():
                module_name = type(module).__name__
                if any(x in module_name.lower() for x in ["decoder", "layer", "block"]):
                    if hasattr(module, "self_attn") or hasattr(module, "attention"):
                        layer_classes.append(type(module))
                        break
                        
        if layer_classes:
            return functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=set(layer_classes),
            )
            
    return None


def wrap_model_with_fsdp(
    model: nn.Module,
    config: FSDPConfig,
) -> nn.Module:
    """Wrap model with FSDP."""
    if not FSDP_AVAILABLE:
        raise ImportError("FSDP not available. Requires PyTorch >= 1.12")
        
    # Get auto wrap policy
    auto_wrap_policy = get_auto_wrap_policy(config, model)
    
    # Wrap with FSDP
    fsdp_model = FSDP(
        model,
        sharding_strategy=config.get_sharding_strategy(),
        mixed_precision=config.get_mixed_precision(),
        cpu_offload=config.get_cpu_offload(),
        backward_prefetch=config.get_backward_prefetch(),
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )
    
    # Apply activation checkpointing if enabled
    if config.activation_checkpointing:
        apply_activation_checkpointing(fsdp_model, config)
        
    return fsdp_model


def apply_activation_checkpointing(
    model: nn.Module,
    config: FSDPConfig,
):
    """Apply activation checkpointing to model."""
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            CheckpointImpl,
            apply_activation_checkpointing as torch_apply_checkpointing,
        )
        
        # Find checkpointable modules
        check_fn = lambda module: any(
            name in type(module).__name__ 
            for name in config.transformer_layer_cls_names
        )
        
        torch_apply_checkpointing(
            model,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=check_fn,
        )
        
    except ImportError:
        # Fallback for older PyTorch versions
        pass


def get_fsdp_state_dict(
    model: FSDP,
    config: FSDPConfig,
) -> Dict[str, torch.Tensor]:
    """Get state dict from FSDP model."""
    if not FSDP_AVAILABLE:
        return model.state_dict()
        
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    
    if config.state_dict_type == "FULL_STATE_DICT":
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            return model.state_dict()
    else:
        return model.state_dict()


def load_fsdp_state_dict(
    model: FSDP,
    state_dict: Dict[str, torch.Tensor],
    config: FSDPConfig,
):
    """Load state dict to FSDP model."""
    if not FSDP_AVAILABLE:
        model.load_state_dict(state_dict)
        return
        
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    
    if config.state_dict_type == "FULL_STATE_DICT":
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


class FSDPTrainingContext:
    """Context manager for FSDP training."""
    
    def __init__(
        self,
        model: nn.Module,
        config: FSDPConfig,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        optimizer_kwargs: Optional[Dict] = None,
    ):
        self.original_model = model
        self.config = config
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-5}
        
        self.fsdp_model: Optional[FSDP] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.rank: int = 0
        
    def __enter__(self):
        # Initialize distributed
        self.rank = init_distributed()
        
        # Wrap model with FSDP
        self.fsdp_model = wrap_model_with_fsdp(self.original_model, self.config)
        
        # Create optimizer
        self.optimizer = self.optimizer_cls(
            self.fsdp_model.parameters(),
            **self.optimizer_kwargs,
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()
        return False
    
    @property
    def model(self) -> FSDP:
        return self.fsdp_model
    
    def save_checkpoint(self, path: str):
        """Save checkpoint."""
        if self.rank == 0:
            state_dict = get_fsdp_state_dict(self.fsdp_model, self.config)
            torch.save({
                "model": state_dict,
                "optimizer": self.optimizer.state_dict(),
            }, path)
            
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        load_fsdp_state_dict(self.fsdp_model, checkpoint["model"], self.config)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
