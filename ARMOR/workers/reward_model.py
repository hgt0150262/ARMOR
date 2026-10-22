"""
Reward model for RLHF in ARMOR.
Provides reward model training and inference.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        AutoConfig,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class RewardModelConfig:
    """Configuration for reward model."""
    
    # Model
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B"
    num_labels: int = 1  # Scalar reward
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    
    # Training
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_length: int = 512
    
    # Loss
    loss_type: str = "ranking"  # ranking, regression, margin
    margin: float = 1.0
    
    def get_torch_dtype(self) -> torch.dtype:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype, torch.bfloat16)


class RewardModel(nn.Module):
    """Reward model for scoring responses."""
    
    def __init__(self, config: RewardModelConfig):
        super().__init__()
        self.config = config
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required for RewardModel")
            
        # Load base model
        model_config = AutoConfig.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
            num_labels=config.num_labels,
        )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name_or_path,
            config=model_config,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=config.get_torch_dtype(),
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass returning scalar rewards."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits.squeeze(-1)
    
    def score(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> torch.Tensor:
        """Score prompt-response pairs."""
        # Combine prompt and response
        texts = [f"{p}{r}" for p, r in zip(prompts, responses)]
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        
        device = next(self.model.parameters()).device
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        
        with torch.no_grad():
            rewards = self.forward(input_ids, attention_mask)
            
        return rewards
    
    def compute_ranking_loss(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ranking loss (Bradley-Terry model)."""
        return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    
    def compute_margin_loss(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute margin ranking loss."""
        margin = self.config.margin
        loss = F.relu(margin - (chosen_rewards - rejected_rewards))
        return loss.mean()


class RewardModelTrainer:
    """Trainer for reward model."""
    
    def __init__(
        self,
        model: RewardModel,
        config: RewardModelConfig,
    ):
        self.model = model
        self.config = config
        self.optimizer = None
        self.scheduler = None
        
    def setup(self, num_training_steps: int):
        """Setup optimizer and scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps,
        )
        
    def train_step(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Forward pass
        chosen_rewards = self.model(chosen_input_ids, chosen_attention_mask)
        rejected_rewards = self.model(rejected_input_ids, rejected_attention_mask)
        
        # Compute loss
        if self.config.loss_type == "ranking":
            loss = self.model.compute_ranking_loss(chosen_rewards, rejected_rewards)
        elif self.config.loss_type == "margin":
            loss = self.model.compute_margin_loss(chosen_rewards, rejected_rewards)
        else:
            loss = self.model.compute_ranking_loss(chosen_rewards, rejected_rewards)
            
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Metrics
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "chosen_reward_mean": chosen_rewards.mean().item(),
            "rejected_reward_mean": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
        }
    
    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int = 3,
    ) -> List[Dict[str, float]]:
        """Full training loop."""
        num_steps = len(train_dataloader) * num_epochs
        self.setup(num_steps)
        
        all_metrics = []
        
        for epoch in range(num_epochs):
            epoch_metrics = []
            
            for batch in train_dataloader:
                metrics = self.train_step(
                    chosen_input_ids=batch["chosen_input_ids"],
                    chosen_attention_mask=batch["chosen_attention_mask"],
                    rejected_input_ids=batch["rejected_input_ids"],
                    rejected_attention_mask=batch["rejected_attention_mask"],
                )
                epoch_metrics.append(metrics)
                
            # Average metrics
            avg_metrics = {
                k: sum(m[k] for m in epoch_metrics) / len(epoch_metrics)
                for k in epoch_metrics[0].keys()
            }
            avg_metrics["epoch"] = epoch
            all_metrics.append(avg_metrics)
            
            print(f"Epoch {epoch}: loss={avg_metrics['loss']:.4f}, acc={avg_metrics['accuracy']:.4f}")
            
        return all_metrics
    
    def save(self, path: str):
        """Save model."""
        self.model.model.save_pretrained(path)
        self.model.tokenizer.save_pretrained(path)
        
    def load(self, path: str):
        """Load model."""
        self.model.model = AutoModelForSequenceClassification.from_pretrained(
            path,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=self.config.get_torch_dtype(),
        )


def create_reward_model(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    device: Optional[str] = None,
) -> RewardModel:
    """Factory function to create reward model."""
    config = RewardModelConfig(model_name_or_path=model_name)
    model = RewardModel(config)
    
    if device:
        model = model.to(device)
    elif torch.cuda.is_available():
        model = model.to("cuda")
        
    return model
