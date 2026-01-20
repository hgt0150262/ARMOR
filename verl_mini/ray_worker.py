"""
Ray-based distributed training support for verl_mini.
Implements RayResourcePool, RayWorkerGroup, and distributed DataProto utilities.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Union, Type
from enum import Enum
import numpy as np

try:
    import ray
    from ray.util.placement_group import PlacementGroup
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None
    PlacementGroup = None

import torch

from .protocol import DataProto


class Role(str, Enum):
    """Roles for different worker types in distributed training."""
    ActorRollout = "actor_rollout"
    ActorRolloutRef = "actor_rollout_ref"
    Critic = "critic"
    RefPolicy = "ref_policy"
    RewardModel = "reward_model"
    Actor = "actor"
    Rollout = "rollout"


@dataclass
class ResourcePoolConfig:
    """Configuration for a resource pool."""
    num_gpus: float = 1.0
    num_cpus: float = 1.0
    memory: Optional[int] = None
    custom_resources: Dict[str, float] = field(default_factory=dict)


class RayResourcePool:
    """
    Manages GPU resources and placement groups for Ray actors.
    
    In verl, this handles:
    - GPU allocation across nodes
    - Placement group creation for co-located workers
    - Resource scheduling
    """
    
    def __init__(self, 
                 process_on_nodes: Optional[List[int]] = None,
                 num_gpus_per_node: int = 1,
                 name: str = "default"):
        """
        Args:
            process_on_nodes: List of GPU counts per node
            num_gpus_per_node: Default GPUs per node if not specified
            name: Pool name for identification
        """
        self.name = name
        self.process_on_nodes = process_on_nodes or [num_gpus_per_node]
        self.num_gpus_per_node = num_gpus_per_node
        self.placement_groups: List[PlacementGroup] = []
        self._initialized = False
        
    @property
    def total_gpus(self) -> int:
        """Total GPUs in this pool."""
        return sum(self.process_on_nodes)
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in this pool."""
        return len(self.process_on_nodes)
    
    def create_placement_groups(self) -> List[PlacementGroup]:
        """Create Ray placement groups for resource allocation."""
        if not RAY_AVAILABLE:
            print("Warning: Ray not available, using mock placement groups")
            return []
        
        if self._initialized:
            return self.placement_groups
        
        self.placement_groups = []
        for node_idx, num_gpus in enumerate(self.process_on_nodes):
            bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
            pg = ray.util.placement_group(
                bundles,
                strategy="PACK",
                name=f"{self.name}_node_{node_idx}"
            )
            self.placement_groups.append(pg)
        
        self._initialized = True
        return self.placement_groups
    
    def get_bundle_indices(self, worker_idx: int) -> tuple:
        """Get placement group and bundle index for a worker."""
        cumsum = 0
        for pg_idx, num_gpus in enumerate(self.process_on_nodes):
            if worker_idx < cumsum + num_gpus:
                bundle_idx = worker_idx - cumsum
                return pg_idx, bundle_idx
            cumsum += num_gpus
        raise ValueError(f"Worker index {worker_idx} out of range")


class ResourcePoolManager:
    """
    Manages multiple resource pools for different worker roles.
    
    Handles mapping between roles and resource pools.
    """
    
    def __init__(self):
        self.pools: Dict[str, RayResourcePool] = {}
        self.role_to_pool: Dict[Role, str] = {}
    
    def add_pool(self, name: str, pool: RayResourcePool):
        """Add a resource pool."""
        self.pools[name] = pool
    
    def create_pool(self, name: str, 
                    process_on_nodes: List[int] = None,
                    num_gpus_per_node: int = 1) -> RayResourcePool:
        """Create and add a new resource pool."""
        pool = RayResourcePool(
            process_on_nodes=process_on_nodes,
            num_gpus_per_node=num_gpus_per_node,
            name=name
        )
        self.pools[name] = pool
        return pool
    
    def register_role(self, role: Role, pool_name: str):
        """Register a role to use a specific pool."""
        if pool_name not in self.pools:
            raise ValueError(f"Pool '{pool_name}' not found")
        self.role_to_pool[role] = pool_name
    
    def get_pool_for_role(self, role: Role) -> Optional[RayResourcePool]:
        """Get the resource pool for a role."""
        pool_name = self.role_to_pool.get(role)
        if pool_name:
            return self.pools.get(pool_name)
        return None
    
    def initialize_all_pools(self):
        """Initialize all resource pools."""
        for pool in self.pools.values():
            pool.create_placement_groups()


class RayWorkerGroup:
    """
    Manages a group of Ray actor workers.
    
    Handles:
    - Worker creation with proper resource allocation
    - Method dispatch across workers
    - Result collection and aggregation
    - DataProto distribution and gathering
    """
    
    def __init__(self,
                 worker_cls: Type,
                 resource_pool: RayResourcePool,
                 num_workers: Optional[int] = None,
                 worker_cls_args: tuple = (),
                 worker_cls_kwargs: Dict[str, Any] = None):
        """
        Args:
            worker_cls: Worker class to instantiate
            resource_pool: Resource pool for allocation
            num_workers: Number of workers (defaults to total GPUs in pool)
            worker_cls_args: Positional args for worker constructor
            worker_cls_kwargs: Keyword args for worker constructor
        """
        self.worker_cls = worker_cls
        self.resource_pool = resource_pool
        self.num_workers = num_workers or resource_pool.total_gpus
        self.worker_cls_args = worker_cls_args
        self.worker_cls_kwargs = worker_cls_kwargs or {}
        
        self.workers: List[Any] = []
        self._initialized = False
    
    def create_workers(self):
        """Create Ray actor workers."""
        if self._initialized:
            return
        
        if not RAY_AVAILABLE:
            # Fallback to local workers
            self._create_local_workers()
            return
        
        # Create placement groups
        pgs = self.resource_pool.create_placement_groups()
        
        # Create remote worker class
        RemoteWorker = ray.remote(num_gpus=1)(self.worker_cls)
        
        for worker_idx in range(self.num_workers):
            pg_idx, bundle_idx = self.resource_pool.get_bundle_indices(worker_idx)
            
            # Set environment variables for the worker
            env_vars = {
                "RANK": str(worker_idx),
                "WORLD_SIZE": str(self.num_workers),
                "LOCAL_RANK": str(bundle_idx),
            }
            
            # Create worker with placement
            worker = RemoteWorker.options(
                placement_group=pgs[pg_idx] if pgs else None,
                placement_group_bundle_index=bundle_idx if pgs else None,
                runtime_env={"env_vars": env_vars}
            ).remote(*self.worker_cls_args, **self.worker_cls_kwargs)
            
            self.workers.append(worker)
        
        self._initialized = True
    
    def _create_local_workers(self):
        """Create local workers when Ray is not available."""
        for worker_idx in range(self.num_workers):
            os.environ["RANK"] = str(worker_idx)
            os.environ["WORLD_SIZE"] = str(self.num_workers)
            os.environ["LOCAL_RANK"] = str(worker_idx)
            
            worker = self.worker_cls(*self.worker_cls_args, **self.worker_cls_kwargs)
            self.workers.append(worker)
        
        self._initialized = True
    
    def execute_all(self, method_name: str, *args, **kwargs) -> List[Any]:
        """Execute a method on all workers."""
        if not RAY_AVAILABLE:
            return [getattr(w, method_name)(*args, **kwargs) for w in self.workers]
        
        refs = [getattr(w, method_name).remote(*args, **kwargs) for w in self.workers]
        return ray.get(refs)
    
    def execute_rank_zero(self, method_name: str, *args, **kwargs) -> Any:
        """Execute a method on rank 0 only."""
        if not self.workers:
            return None
        
        if not RAY_AVAILABLE:
            return getattr(self.workers[0], method_name)(*args, **kwargs)
        
        ref = getattr(self.workers[0], method_name).remote(*args, **kwargs)
        return ray.get(ref)
    
    def broadcast_data(self, data: DataProto) -> List[DataProto]:
        """Broadcast DataProto to all workers."""
        # Chunk data for each worker
        chunks = data.chunk(self.num_workers)
        return chunks
    
    def gather_data(self, data_list: List[DataProto]) -> DataProto:
        """Gather DataProto from all workers."""
        return DataProto.concat(data_list)
    
    def execute_with_data(self, 
                          method_name: str, 
                          data: DataProto,
                          **kwargs) -> DataProto:
        """
        Execute a method on all workers with distributed data.
        
        1. Chunks data across workers
        2. Executes method on each worker
        3. Gathers results
        """
        chunks = self.broadcast_data(data)
        
        if not RAY_AVAILABLE:
            results = []
            for worker, chunk in zip(self.workers, chunks):
                result = getattr(worker, method_name)(chunk, **kwargs)
                results.append(result)
        else:
            refs = []
            for worker, chunk in zip(self.workers, chunks):
                ref = getattr(worker, method_name).remote(chunk, **kwargs)
                refs.append(ref)
            results = ray.get(refs)
        
        return self.gather_data(results)
    
    def shutdown(self):
        """Shutdown all workers."""
        if RAY_AVAILABLE and self.workers:
            for worker in self.workers:
                ray.kill(worker)
        self.workers = []
        self._initialized = False


def init_ray_cluster(
    address: str = "auto",
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
    namespace: str = "verl_mini",
    log_to_driver: bool = True,
    **kwargs
) -> bool:
    """
    Initialize Ray cluster.
    
    Args:
        address: Ray cluster address ("auto" for existing cluster, "local" for new)
        num_cpus: Number of CPUs (for local mode)
        num_gpus: Number of GPUs (for local mode)
        namespace: Ray namespace
        log_to_driver: Whether to log to driver
    
    Returns:
        True if initialization successful
    """
    if not RAY_AVAILABLE:
        print("Warning: Ray not available. Install with: pip install ray")
        return False
    
    if ray.is_initialized():
        return True
    
    try:
        ray.init(
            address=address,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            namespace=namespace,
            log_to_driver=log_to_driver,
            **kwargs
        )
        return True
    except Exception as e:
        print(f"Failed to initialize Ray: {e}")
        return False


def shutdown_ray():
    """Shutdown Ray cluster."""
    if RAY_AVAILABLE and ray.is_initialized():
        ray.shutdown()


# =============================================================================
# Distributed DataProto Utilities
# =============================================================================

def all_gather_data_proto(data: DataProto, world_size: int) -> DataProto:
    """
    All-gather DataProto across processes.
    
    In the real verl, this uses torch.distributed for efficient gathering.
    """
    if not torch.distributed.is_initialized():
        return data
    
    # Gather batch data
    gathered_batches = [None] * world_size
    torch.distributed.all_gather_object(gathered_batches, data.batch)
    
    # Gather non-tensor data
    gathered_non_tensor = [None] * world_size
    torch.distributed.all_gather_object(gathered_non_tensor, data.non_tensor_batch)
    
    # Merge
    merged_batch = {}
    for key in data.batch.keys():
        tensors = [b[key] for b in gathered_batches if b is not None]
        merged_batch[key] = torch.cat(tensors, dim=0)
    
    merged_non_tensor = {}
    for key in data.non_tensor_batch.keys():
        arrays = [b[key] for b in gathered_non_tensor if b is not None]
        merged_non_tensor[key] = np.concatenate(arrays, axis=0)
    
    return DataProto(
        batch=merged_batch,
        non_tensor_batch=merged_non_tensor,
        meta_info=data.meta_info
    )


def scatter_data_proto(data: DataProto, world_size: int, rank: int) -> DataProto:
    """
    Scatter DataProto to specific rank.
    """
    chunks = data.chunk(world_size)
    if rank < len(chunks):
        return chunks[rank]
    return DataProto()


# =============================================================================
# Ray Actor Base Classes
# =============================================================================

class RayWorker:
    """
    Base class for Ray actor workers.
    
    Provides common functionality for distributed workers.
    """
    
    def __init__(self):
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        
        self._dispatch_info: Dict[str, Dict[str, Any]] = {}
    
    def get_info(self) -> Dict[str, Any]:
        """Get worker information."""
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "device": self.device,
        }
    
    def register_dispatch_info(self, name: str, dp_rank: int, is_collect: bool):
        """Register dispatch/collect info for a named operation."""
        self._dispatch_info[name] = {
            "dp_rank": dp_rank,
            "is_collect": is_collect
        }
    
    def process_data(self, data: DataProto) -> DataProto:
        """Process DataProto (to be overridden)."""
        return data


class RayActorRolloutWorker(RayWorker):
    """Ray actor worker for actor rollout."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model = None
    
    def init_model(self):
        """Initialize model."""
        pass
    
    def generate(self, data: DataProto) -> DataProto:
        """Generate responses."""
        return data
    
    def compute_log_probs(self, data: DataProto) -> DataProto:
        """Compute log probabilities."""
        return data
    
    def update(self, data: DataProto) -> Dict[str, float]:
        """Update model."""
        return {"loss": 0.0}


class RayCriticWorker(RayWorker):
    """Ray actor worker for critic."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model = None
    
    def init_model(self):
        """Initialize model."""
        pass
    
    def compute_values(self, data: DataProto) -> DataProto:
        """Compute values."""
        return data
    
    def update(self, data: DataProto) -> Dict[str, float]:
        """Update model."""
        return {"loss": 0.0}


class RayRewardWorker(RayWorker):
    """Ray actor worker for reward model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model = None
    
    def init_model(self):
        """Initialize model."""
        pass
    
    def compute_rewards(self, data: DataProto) -> DataProto:
        """Compute rewards."""
        return data
