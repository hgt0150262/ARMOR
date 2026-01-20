"""
vLLM-based rollout for high-performance inference in verl_mini.
Provides fast generation using vLLM engine.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import logging

import torch

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    LoRARequest = None

logger = logging.getLogger(__name__)


@dataclass
class VLLMConfig:
    """Configuration for vLLM rollout."""
    
    # Model settings
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B"
    tokenizer_name_or_path: Optional[str] = None
    trust_remote_code: bool = True
    
    # vLLM engine settings
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    enforce_eager: bool = False
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1
    repetition_penalty: float = 1.0
    
    # LoRA settings
    enable_lora: bool = False
    max_lora_rank: int = 64
    lora_path: Optional[str] = None
    
    # Batch settings
    max_num_seqs: int = 256
    
    def get_sampling_params(self, **kwargs) -> "SamplingParams":
        """Get vLLM SamplingParams."""
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is required. Install with: pip install vllm")
            
        params = {
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k if self.top_k > 0 else -1,
            "repetition_penalty": self.repetition_penalty,
        }
        params.update(kwargs)
        return SamplingParams(**params)


class VLLMRollout:
    """vLLM-based rollout for fast inference."""
    
    def __init__(self, config: VLLMConfig):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is required. Install with: pip install vllm")
            
        self.config = config
        self.llm: Optional[LLM] = None
        self.tokenizer = None
        self._initialized = False
        
    def init_engine(self):
        """Initialize vLLM engine."""
        if self._initialized:
            return
            
        logger.info(f"Initializing vLLM engine with model: {self.config.model_name_or_path}")
        
        # Determine dtype
        dtype_map = {
            "float16": "float16",
            "bfloat16": "bfloat16",
            "float32": "float32",
            "auto": "auto",
        }
        dtype = dtype_map.get(self.config.dtype, "auto")
        
        # Initialize LLM
        self.llm = LLM(
            model=self.config.model_name_or_path,
            tokenizer=self.config.tokenizer_name_or_path or self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
            tensor_parallel_size=self.config.tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            enforce_eager=self.config.enforce_eager,
            enable_lora=self.config.enable_lora,
            max_lora_rank=self.config.max_lora_rank if self.config.enable_lora else None,
            max_num_seqs=self.config.max_num_seqs,
        )
        
        self.tokenizer = self.llm.get_tokenizer()
        self._initialized = True
        
        logger.info("vLLM engine initialized successfully")
        
    def generate(
        self,
        prompts: Union[List[str], List[List[int]]],
        sampling_params: Optional["SamplingParams"] = None,
        lora_request: Optional["LoRARequest"] = None,
        return_tokens: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate responses for prompts.
        
        Args:
            prompts: List of prompt strings or token IDs
            sampling_params: vLLM SamplingParams (uses config defaults if None)
            lora_request: Optional LoRA request for adapter
            return_tokens: Whether to return token IDs
            
        Returns:
            Dictionary with generated outputs
        """
        if not self._initialized:
            self.init_engine()
            
        if sampling_params is None:
            sampling_params = self.config.get_sampling_params()
            
        # Handle token IDs vs strings
        if prompts and isinstance(prompts[0], list):
            # Token IDs provided
            prompt_token_ids = prompts
            outputs = self.llm.generate(
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
                lora_request=lora_request,
            )
        else:
            # String prompts provided
            outputs = self.llm.generate(
                prompts=prompts,
                sampling_params=sampling_params,
                lora_request=lora_request,
            )
            
        # Process outputs
        results = {
            "responses": [],
            "response_ids": [],
            "prompt_ids": [],
            "logprobs": [],
        }
        
        for output in outputs:
            # Get generated text
            generated_text = output.outputs[0].text
            results["responses"].append(generated_text)
            
            # Get token IDs
            if return_tokens:
                response_ids = output.outputs[0].token_ids
                results["response_ids"].append(list(response_ids))
                results["prompt_ids"].append(list(output.prompt_token_ids))
                
            # Get log probabilities if available
            if output.outputs[0].logprobs:
                results["logprobs"].append(output.outputs[0].logprobs)
                
        return results
    
    def generate_with_logprobs(
        self,
        prompts: List[str],
        sampling_params: Optional["SamplingParams"] = None,
    ) -> Dict[str, Any]:
        """Generate with log probabilities for policy gradient."""
        if sampling_params is None:
            sampling_params = self.config.get_sampling_params(logprobs=1)
        else:
            # Ensure logprobs is enabled
            sampling_params.logprobs = max(sampling_params.logprobs or 0, 1)
            
        return self.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            return_tokens=True,
        )
    
    def update_weights(self, state_dict: Dict[str, torch.Tensor]):
        """
        Update model weights from training.
        Used to sync actor weights to rollout engine.
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized")
            
        # vLLM weight update (requires specific vLLM version)
        try:
            # For newer vLLM versions with weight update support
            model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            model.load_state_dict(state_dict, strict=False)
            logger.info("Model weights updated successfully")
        except Exception as e:
            logger.warning(f"Weight update not supported in this vLLM version: {e}")
            logger.info("Consider reinitializing engine with new weights")
            
    def load_lora(self, lora_path: str, lora_name: str = "default") -> "LoRARequest":
        """Load a LoRA adapter."""
        if not self.config.enable_lora:
            raise ValueError("LoRA not enabled in config")
            
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is required for LoRA")
            
        return LoRARequest(
            lora_name=lora_name,
            lora_int_id=1,
            lora_path=lora_path,
        )
    
    def shutdown(self):
        """Shutdown the vLLM engine."""
        if self.llm is not None:
            del self.llm
            self.llm = None
            self._initialized = False
            torch.cuda.empty_cache()
            logger.info("vLLM engine shutdown")


def create_vllm_rollout(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    tensor_parallel_size: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    **kwargs,
) -> VLLMRollout:
    """Factory function to create vLLM rollout."""
    config = VLLMConfig(
        model_name_or_path=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        **kwargs,
    )
    return VLLMRollout(config)
