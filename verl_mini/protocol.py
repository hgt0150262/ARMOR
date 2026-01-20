"""
Simplified reproduction of verl's DataProto - the core data transfer protocol.
DataProto provides a standard way to exchange data between different components
in the RLHF training pipeline.
"""

import copy
import pickle
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader


def list_of_dict_to_dict_of_list(list_of_dict: List[Dict]) -> Dict:
    """Convert a list of dicts to a dict of lists."""
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()
    output = {key: [] for key in keys}
    for data in list_of_dict:
        for key, item in data.items():
            assert key in output
            output[key].append(item)
    return output


def union_tensor_dict(tensor_dict1: Dict[str, torch.Tensor], 
                      tensor_dict2: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Union two tensor dicts."""
    for key in tensor_dict2.keys():
        if key not in tensor_dict1.keys():
            tensor_dict1[key] = tensor_dict2[key]
        else:
            assert torch.equal(tensor_dict1[key], tensor_dict2[key]), \
                f"{key} in tensor_dict1 and tensor_dict2 are not the same"
    return tensor_dict1


@dataclass
class DataProtoItem:
    """A single item from DataProto, returned when indexing with integer."""
    batch: Dict[str, torch.Tensor] = None
    non_tensor_batch: Dict[str, Any] = field(default_factory=dict)
    meta_info: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class DataProto:
    """
    A DataProto is a data structure that provides a standard protocol for data 
    exchange between functions in RLHF training.
    
    It contains:
    - batch: Dict of torch.Tensors with the same batch size (dim 0)
    - non_tensor_batch: Dict of numpy arrays (for non-tensor data like strings)
    - meta_info: Dict for metadata that doesn't have batch dimension
    
    This is a simplified version of verl's DataProto which uses TensorDict.
    """
    
    batch: Dict[str, torch.Tensor] = None
    non_tensor_batch: Dict[str, Any] = field(default_factory=dict)
    meta_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Perform consistency checking after initialization."""
        self.check_consistency()
    
    def __len__(self) -> int:
        """Return the batch size."""
        if self.batch is not None and len(self.batch) > 0:
            first_key = list(self.batch.keys())[0]
            return self.batch[first_key].shape[0]
        elif self.non_tensor_batch is not None and len(self.non_tensor_batch) > 0:
            first_key = list(self.non_tensor_batch.keys())[0]
            return self.non_tensor_batch[first_key].shape[0]
        return 0
    
    def __getitem__(self, item):
        """Enhanced indexing for DataProto objects."""
        # Single integer - return DataProtoItem
        if isinstance(item, (int, np.integer)):
            tensor_data = {k: v[item] for k, v in self.batch.items()} if self.batch else None
            non_tensor_data = {k: v[item] for k, v in self.non_tensor_batch.items()}
            return DataProtoItem(
                batch=tensor_data, 
                non_tensor_batch=non_tensor_data, 
                meta_info=self.meta_info
            )
        
        # Slice - return new DataProto
        if isinstance(item, slice):
            return self.slice(item.start, item.stop, item.step)
        
        # List/array indexing
        if isinstance(item, (list, np.ndarray, torch.Tensor)):
            return self.select_idxs(item)
        
        raise TypeError(f"Indexing with {type(item)} is not supported")
    
    def check_consistency(self):
        """Check the consistency of batch and non_tensor_batch."""
        batch_size = None
        
        if self.batch is not None:
            for key, val in self.batch.items():
                if batch_size is None:
                    batch_size = val.shape[0]
                else:
                    assert val.shape[0] == batch_size, \
                        f"Tensor {key} has different batch size"
        
        if self.non_tensor_batch is not None and batch_size is not None:
            for key, val in self.non_tensor_batch.items():
                assert isinstance(val, np.ndarray), \
                    f"non_tensor_batch[{key}] must be numpy array"
                assert val.shape[0] == batch_size, \
                    f"non_tensor_batch[{key}] has different batch size"
    
    @classmethod
    def from_dict(cls, 
                  tensors: Optional[Dict[str, torch.Tensor]] = None,
                  non_tensors: Optional[Dict[str, Any]] = None,
                  meta_info: Optional[Dict] = None) -> "DataProto":
        """Create a DataProto from dictionaries."""
        if tensors is None:
            tensors = {}
        if non_tensors is None:
            non_tensors = {}
        if meta_info is None:
            meta_info = {}
            
        # Convert non-tensors to numpy arrays
        for key, val in non_tensors.items():
            if not isinstance(val, np.ndarray):
                non_tensors[key] = np.array(val, dtype=object)
        
        return cls(batch=tensors, non_tensor_batch=non_tensors, meta_info=meta_info)
    
    @classmethod
    def from_single_dict(cls, data: Dict[str, Any], meta_info=None) -> "DataProto":
        """Create DataProto from a single dict, auto-separating tensors and non-tensors."""
        tensors = {}
        non_tensors = {}
        
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key] = val
            elif isinstance(val, np.ndarray):
                non_tensors[key] = val
            else:
                raise ValueError(f"Unsupported type {type(val)} for key {key}")
        
        return cls.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)
    
    def to(self, device) -> "DataProto":
        """Move batch tensors to device."""
        if self.batch is not None:
            self.batch = {k: v.to(device) for k, v in self.batch.items()}
        return self
    
    def select(self, batch_keys=None, non_tensor_batch_keys=None, 
               meta_info_keys=None) -> "DataProto":
        """Select a subset of keys."""
        new_batch = None
        if batch_keys is not None and self.batch is not None:
            new_batch = {k: v for k, v in self.batch.items() if k in batch_keys}
        else:
            new_batch = self.batch
            
        new_non_tensor = None
        if non_tensor_batch_keys is not None:
            new_non_tensor = {k: v for k, v in self.non_tensor_batch.items() 
                            if k in non_tensor_batch_keys}
        else:
            new_non_tensor = self.non_tensor_batch
            
        new_meta = None
        if meta_info_keys is not None:
            new_meta = {k: v for k, v in self.meta_info.items() if k in meta_info_keys}
        else:
            new_meta = self.meta_info
            
        return DataProto(batch=new_batch, non_tensor_batch=new_non_tensor, meta_info=new_meta)
    
    def select_idxs(self, idxs) -> "DataProto":
        """Select specific indices from the DataProto."""
        if isinstance(idxs, list):
            idxs_torch = torch.tensor(idxs)
            idxs_np = np.array(idxs)
        elif isinstance(idxs, np.ndarray):
            idxs_torch = torch.from_numpy(idxs)
            idxs_np = idxs
        else:
            idxs_torch = idxs
            idxs_np = idxs.detach().cpu().numpy()
        
        new_batch = None
        if self.batch is not None:
            new_batch = {k: v[idxs_torch] for k, v in self.batch.items()}
            
        new_non_tensor = {k: v[idxs_np] for k, v in self.non_tensor_batch.items()}
        
        return DataProto(batch=new_batch, non_tensor_batch=new_non_tensor, 
                        meta_info=self.meta_info)
    
    def slice(self, start=None, end=None, step=None) -> "DataProto":
        """Slice the DataProto."""
        slice_obj = slice(start, end, step)
        
        new_batch = None
        if self.batch is not None:
            new_batch = {k: v[slice_obj] for k, v in self.batch.items()}
            
        new_non_tensor = {k: v[slice_obj] for k, v in self.non_tensor_batch.items()}
        
        return DataProto(batch=new_batch, non_tensor_batch=new_non_tensor, 
                        meta_info=self.meta_info)
    
    def chunk(self, chunks: int) -> List["DataProto"]:
        """Split the batch into chunks along dim=0."""
        batch_size = len(self)
        assert batch_size % chunks == 0, \
            f"Batch size {batch_size} not divisible by {chunks}"
        
        chunk_size = batch_size // chunks
        return [self[i*chunk_size:(i+1)*chunk_size] for i in range(chunks)]
    
    def split(self, split_size: int) -> List["DataProto"]:
        """Split the batch into pieces of split_size."""
        return [self[i:i+split_size] for i in range(0, len(self), split_size)]
    
    @staticmethod
    def concat(data: List["DataProto"]) -> "DataProto":
        """Concatenate a list of DataProtos along dim=0."""
        if len(data) == 0:
            return DataProto()
        
        # Concat batch tensors
        new_batch = None
        if data[0].batch is not None:
            keys = data[0].batch.keys()
            new_batch = {
                k: torch.cat([d.batch[k] for d in data], dim=0)
                for k in keys
            }
        
        # Concat non_tensor_batch
        non_tensor_keys = data[0].non_tensor_batch.keys()
        new_non_tensor = {
            k: np.concatenate([d.non_tensor_batch[k] for d in data], axis=0)
            for k in non_tensor_keys
        }
        
        # Merge meta_info
        merged_meta = {}
        for d in data:
            for k, v in d.meta_info.items():
                if k not in merged_meta:
                    merged_meta[k] = v
        
        return DataProto(batch=new_batch, non_tensor_batch=new_non_tensor, 
                        meta_info=merged_meta)
    
    def union(self, other: "DataProto") -> "DataProto":
        """Union with another DataProto."""
        if self.batch is not None and other.batch is not None:
            self.batch = union_tensor_dict(self.batch, other.batch)
        
        for key, val in other.non_tensor_batch.items():
            if key not in self.non_tensor_batch:
                self.non_tensor_batch[key] = val
                
        for key, val in other.meta_info.items():
            if key not in self.meta_info:
                self.meta_info[key] = val
                
        return self
    
    def repeat(self, repeat_times: int, interleave: bool = True) -> "DataProto":
        """Repeat the batch data."""
        new_batch = None
        if self.batch is not None:
            if interleave:
                new_batch = {
                    k: v.repeat_interleave(repeat_times, dim=0)
                    for k, v in self.batch.items()
                }
            else:
                new_batch = {
                    k: v.unsqueeze(0).expand(repeat_times, *v.shape).reshape(-1, *v.shape[1:])
                    for k, v in self.batch.items()
                }
        
        new_non_tensor = {}
        for k, v in self.non_tensor_batch.items():
            if interleave:
                new_non_tensor[k] = np.repeat(v, repeat_times, axis=0)
            else:
                new_non_tensor[k] = np.tile(v, (repeat_times,) + (1,) * (v.ndim - 1))
        
        return DataProto(batch=new_batch, non_tensor_batch=new_non_tensor, 
                        meta_info=self.meta_info)
    
    def make_iterator(self, mini_batch_size: int, epochs: int, seed=None):
        """Create an iterator for mini-batch training."""
        assert len(self) % mini_batch_size == 0
        
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = None
        
        def collate_fn(items):
            batch = {}
            non_tensor_batch = {}
            
            if items[0].batch is not None:
                for key in items[0].batch.keys():
                    batch[key] = torch.stack([item.batch[key] for item in items])
            
            for key in items[0].non_tensor_batch.keys():
                non_tensor_batch[key] = np.array([item.non_tensor_batch[key] for item in items], dtype=object)
            
            return DataProto(batch=batch if batch else None, 
                           non_tensor_batch=non_tensor_batch,
                           meta_info=self.meta_info)
        
        # Create a simple dataset wrapper
        class DataProtoDataset:
            def __init__(self, proto):
                self.proto = proto
            def __len__(self):
                return len(self.proto)
            def __getitem__(self, idx):
                return self.proto[idx]
        
        dataset = DataProtoDataset(self)
        dataloader = DataLoader(
            dataset, 
            batch_size=mini_batch_size, 
            shuffle=True,
            collate_fn=collate_fn,
            generator=generator
        )
        
        def get_data():
            for _ in range(epochs):
                for batch in dataloader:
                    yield batch
        
        return iter(get_data())
    
    def save_to_disk(self, filepath: str):
        """Save DataProto to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load_from_disk(filepath: str) -> "DataProto":
        """Load DataProto from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
    
    def print_size(self, prefix=""):
        """Print the size of the DataProto."""
        tensor_size = 0
        if self.batch is not None:
            for _, tensor in self.batch.items():
                tensor_size += tensor.element_size() * tensor.numel()
        
        non_tensor_size = 0
        for _, arr in self.non_tensor_batch.items():
            non_tensor_size += arr.nbytes
        
        tensor_size_gb = tensor_size / (1024**3)
        non_tensor_size_gb = non_tensor_size / (1024**3)
        
        msg = f"Tensor size: {tensor_size_gb:.4f} GB, Non-tensor size: {non_tensor_size_gb:.4f} GB"
        if prefix:
            msg = f"{prefix}: {msg}"
        print(msg)
    
    def get_data_info(self) -> str:
        """Return formatted information about stored data."""
        info = ["=== DataProto Info ==="]
        info.append("batch:")
        if self.batch:
            for key, tensor in self.batch.items():
                info.append(f"  {key}: {tuple(tensor.shape)} ({tensor.dtype})")
        else:
            info.append("  (empty)")
        
        info.append("non_tensor_batch:")
        for key, arr in self.non_tensor_batch.items():
            info.append(f"  {key}: {arr.shape} ({arr.dtype})")
        
        info.append("meta_info:")
        for key, val in self.meta_info.items():
            info.append(f"  {key}: {type(val).__name__}")
        
        return "\n".join(info)
