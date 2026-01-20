"""
Model utilities for verl_mini.
Provides model loading, configuration, and training utilities.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
        GenerationConfig,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        PeftModel,
        TaskType,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


@dataclass
class ModelConfig:
    """Configuration for model loading and training."""
    
    # Model identification
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B"
    tokenizer_name_or_path: Optional[str] = None
    
    # Model settings
    torch_dtype: str = "bfloat16"  # "float16", "bfloat16", "float32"
    trust_remote_code: bool = True
    use_flash_attention: bool = False  # Requires flash_attn package
    
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # Training settings
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    def get_torch_dtype(self) -> torch.dtype:
        """Get torch dtype from string."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype, torch.bfloat16)


class ModelManager:
    """Manages model loading, saving, and training utilities."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.ref_model: Optional[PreTrainedModel] = None
        
    def load_model(
        self,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required. Install with: pip install transformers")
            
        print(f"Loading model: {self.config.model_name_or_path}")
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Model loading kwargs
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": self.config.get_torch_dtype(),
        }
        
        # Flash attention
        if self.config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            
        # Quantization
        if load_in_8bit or load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit,
                )
                model_kwargs["quantization_config"] = quant_config
            except ImportError:
                print("Warning: bitsandbytes not available for quantization")
        else:
            model_kwargs["device_map"] = device
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            **model_kwargs,
        )
        
        # Load tokenizer
        tokenizer_path = self.config.tokenizer_name_or_path or self.config.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        print(f"Model loaded: {self.get_model_info()}")
        
        return self.model, self.tokenizer
    
    def apply_lora(self) -> PreTrainedModel:
        """Apply LoRA to the model."""
        if not PEFT_AVAILABLE:
            raise ImportError("peft is required for LoRA. Install with: pip install peft")
            
        if self.model is None:
            raise ValueError("Model must be loaded before applying LoRA")
            
        print(f"Applying LoRA with r={self.config.lora_r}, alpha={self.config.lora_alpha}")
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        return self.model
    
    def create_reference_model(self) -> PreTrainedModel:
        """Create a frozen reference model for KL penalty."""
        if self.model is None:
            raise ValueError("Model must be loaded before creating reference")
            
        print("Creating frozen reference model...")
        
        # Always load a separate model for reference to avoid shared parameters
        device = next(self.model.parameters()).device
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=self.config.get_torch_dtype(),
            device_map=device,
        )
            
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        self.ref_model.eval()
        
        return self.ref_model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.model is None:
            return {}
            
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.config.model_name_or_path,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
            "dtype": str(self.config.get_torch_dtype()),
            "device": str(next(self.model.parameters()).device) if total_params > 0 else "unknown",
        }
    
    def create_optimizer(
        self,
        num_training_steps: int,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Create optimizer and learning rate scheduler."""
        if self.model is None:
            raise ValueError("Model must be loaded before creating optimizer")
            
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Warmup + Cosine decay scheduler
        warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        
        decay_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=self.config.learning_rate * 0.1,
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps],
        )
        
        return optimizer, scheduler
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate sequences from the model."""
        if self.model is None:
            raise ValueError("Model must be loaded before generation")
            
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "do_sample": self.config.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        gen_kwargs.update(kwargs)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
            
        return outputs
    
    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities for sequences."""
        if self.model is None:
            raise ValueError("Model must be loaded")
            
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        logits = outputs.logits
        
        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = (labels if labels is not None else input_ids)[..., 1:].contiguous()
        
        # Compute log probabilities
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)
        
        return token_log_probs, logits
    
    def save_model(self, save_path: str):
        """Save model and tokenizer."""
        if self.model is None:
            raise ValueError("No model to save")
            
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(save_path)
        else:
            torch.save(self.model.state_dict(), save_path / "model.pt")
            
        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_path)
            
        # Save config
        import json
        config_dict = {
            "model_name_or_path": self.config.model_name_or_path,
            "use_lora": self.config.use_lora,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
        }
        with open(save_path / "verl_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
            
        print(f"Model saved to: {save_path}")
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if (checkpoint_path / "adapter_config.json").exists():
            # Load LoRA adapter
            if not PEFT_AVAILABLE:
                raise ImportError("peft is required to load LoRA checkpoints")
            self.model = PeftModel.from_pretrained(
                self.model,
                checkpoint_path,
            )
        elif (checkpoint_path / "model.pt").exists():
            # Load state dict
            state_dict = torch.load(checkpoint_path / "model.pt")
            self.model.load_state_dict(state_dict)
        else:
            # Load full model
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                trust_remote_code=self.config.trust_remote_code,
                torch_dtype=self.config.get_torch_dtype(),
            )
            
        print(f"Loaded checkpoint from: {checkpoint_path}")


def load_model_and_tokenizer(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    use_lora: bool = True,
    lora_r: int = 16,
    device: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, ModelManager]:
    """Convenience function to load model and tokenizer."""
    config = ModelConfig(
        model_name_or_path=model_name,
        use_lora=use_lora,
        lora_r=lora_r,
    )
    
    manager = ModelManager(config)
    model, tokenizer = manager.load_model(device=device)
    
    if use_lora:
        model = manager.apply_lora()
        
    return model, tokenizer, manager
