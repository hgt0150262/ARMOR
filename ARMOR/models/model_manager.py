"""Model manager for ARMOR - handles model loading, LoRA, etc."""
import torch
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


class ModelManager:
    """Manages model loading, LoRA adaptation, and inference."""
    
    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ):
        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.tokenizer = None
        
    def load_model(self) -> "ModelManager":
        """Load the base model."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        )
        return self
        
    def load_tokenizer(self, padding_side: str = "left") -> "ModelManager":
        """Load the tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        self.tokenizer.padding_side = padding_side
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self
    
    def apply_lora(
        self,
        rank: int = 32,
        alpha: int = 32,
        dropout: float = 0.05,
        target_modules: Optional[list] = None,
    ) -> "ModelManager":
        """Apply LoRA adaptation to the model."""
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
        )
        self.model = get_peft_model(self.model, lora_config)
        return self
    
    def enable_gradient_checkpointing(self) -> "ModelManager":
        """Enable gradient checkpointing for memory efficiency."""
        self.model.gradient_checkpointing_enable()
        return self
    
    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer
