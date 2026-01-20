"""
Simplified reproduction of verl's Worker system.
Demonstrates the basic worker abstraction for distributed training.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
from enum import Enum


class Dispatch(str, Enum):
    """Dispatch modes for worker methods."""
    ONE_TO_ALL = "one_to_all"       # Same input to all workers
    DP_COMPUTE = "dp_compute"       # Data parallel computation
    ALL_TO_ALL = "all_to_all"       # Broadcast to all workers


class Execute(str, Enum):
    """Execution modes for worker methods."""
    ALL = "all"                     # Execute on all workers
    RANK_ZERO = "rank_zero"         # Execute only on rank 0


def register(dispatch_mode: Dispatch = Dispatch.ONE_TO_ALL,
             execute_mode: Execute = Execute.ALL,
             blocking: bool = True):
    """
    Decorator to register a method with dispatch/execute semantics.
    
    In the real verl, this integrates with Ray for distributed execution.
    Here it's a simplified demonstration.
    """
    def decorator(func):
        func._dispatch_mode = dispatch_mode
        func._execute_mode = execute_mode
        func._blocking = blocking
        return func
    return decorator


@dataclass
class DistRankInfo:
    """Distributed rank information."""
    tp_rank: int = 0    # Tensor parallel rank
    dp_rank: int = 0    # Data parallel rank
    pp_rank: int = 0    # Pipeline parallel rank
    cp_rank: int = 0    # Context parallel rank


@dataclass
class DistGlobalInfo:
    """Global distributed configuration."""
    tp_size: int = 1
    dp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1


class Worker:
    """
    Base Worker class for distributed training.
    
    In verl, workers are Ray actors that handle different parts of the training:
    - ActorRolloutRefWorker: Handles actor model, rollout generation, and reference policy
    - CriticWorker: Handles critic/value model
    - RewardModelWorker: Handles reward model inference
    
    This simplified version demonstrates the core abstractions.
    """
    
    def __init__(self):
        """Initialize worker with distributed configuration from environment."""
        self._world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self._rank = int(os.environ.get("RANK", "0"))
        self._local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self._local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        self._master_addr = os.environ.get("MASTER_ADDR", "localhost")
        self._master_port = os.environ.get("MASTER_PORT", "29500")
        
        # Dispatch info storage
        self._dispatch_dp_rank: Dict[str, int] = {}
        self._collect_dp_rank: Dict[str, bool] = {}
    
    @property
    def world_size(self) -> int:
        """Get total number of workers."""
        return self._world_size
    
    @property
    def rank(self) -> int:
        """Get this worker's rank."""
        return self._rank
    
    @property
    def local_rank(self) -> int:
        """Get this worker's local rank (within node)."""
        return self._local_rank
    
    def get_master_addr_port(self):
        """Get master address and port for distributed communication."""
        return self._master_addr, self._master_port
    
    def _register_dispatch_collect_info(self, mesh_name: str, dp_rank: int, is_collect: bool):
        """Register dispatch/collect info for a named mesh."""
        self._dispatch_dp_rank[mesh_name] = dp_rank
        self._collect_dp_rank[mesh_name] = is_collect
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_worker_info(self) -> Dict[str, Any]:
        """Get information about this worker."""
        return {
            "rank": self._rank,
            "world_size": self._world_size,
            "local_rank": self._local_rank,
            "master_addr": self._master_addr,
            "master_port": self._master_port,
        }


class ActorRolloutRefWorker(Worker):
    """
    Combined worker for Actor, Rollout, and Reference policy.
    
    This is the main worker in verl's PPO training that handles:
    1. Actor model training
    2. Response generation (rollout)
    3. Reference policy for KL computation
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.actor_model = None
        self.ref_model = None
        self.rollout_engine = None
        
        # Role flags
        self._is_actor = True
        self._is_rollout = True
        self._is_ref = True
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def init_model(self, model_config: Dict[str, Any]):
        """Initialize the actor model."""
        # In real verl, this would load a HuggingFace model
        # and set up FSDP/Megatron for distributed training
        pass
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=True)
    def generate_sequences(self, prompts):
        """Generate response sequences for given prompts."""
        # In real verl, this uses vLLM or SGLang for efficient generation
        pass
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=True)
    def compute_log_probs(self, data):
        """Compute log probabilities for sequences."""
        pass
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=True)
    def compute_ref_log_probs(self, data):
        """Compute reference policy log probabilities."""
        pass
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=True)
    def update_actor(self, data):
        """Perform actor model update step."""
        pass


class CriticWorker(Worker):
    """
    Worker for Critic/Value model.
    
    Handles value function training for advantage estimation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.critic_model = None
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def init_model(self, model_config: Dict[str, Any]):
        """Initialize the critic model."""
        pass
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=True)
    def compute_values(self, data):
        """Compute value estimates for sequences."""
        pass
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=True)
    def update_critic(self, data):
        """Perform critic model update step."""
        pass


class RewardModelWorker(Worker):
    """
    Worker for Reward Model inference.
    
    Computes rewards for generated sequences.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.reward_model = None
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def init_model(self, model_config: Dict[str, Any]):
        """Initialize the reward model."""
        pass
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=True)
    def compute_rewards(self, data):
        """Compute rewards for sequences."""
        pass


class WorkerGroup:
    """
    Manager for a group of distributed workers.
    
    In verl, WorkerGroup manages Ray actors and handles:
    - Worker creation and placement
    - Method dispatch to workers
    - Result collection
    """
    
    def __init__(self, worker_cls, num_workers: int = 1, **kwargs):
        self.worker_cls = worker_cls
        self.num_workers = num_workers
        self.workers = []
        self.kwargs = kwargs
    
    def create_workers(self):
        """Create worker instances."""
        # In real verl, this creates Ray actors
        for i in range(self.num_workers):
            os.environ["RANK"] = str(i)
            os.environ["WORLD_SIZE"] = str(self.num_workers)
            worker = self.worker_cls(**self.kwargs)
            self.workers.append(worker)
    
    def execute(self, method_name: str, *args, **kwargs):
        """Execute a method on all workers."""
        results = []
        for worker in self.workers:
            method = getattr(worker, method_name)
            result = method(*args, **kwargs)
            results.append(result)
        return results
    
    def execute_rank_zero(self, method_name: str, *args, **kwargs):
        """Execute a method only on rank 0."""
        if self.workers:
            method = getattr(self.workers[0], method_name)
            return method(*args, **kwargs)
        return None
