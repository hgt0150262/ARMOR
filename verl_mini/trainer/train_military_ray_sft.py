"""
Ray Distributed SFT Training for Military Domain.
Master: gpu-server (4x H100)
Worker: gpu-server1 (4x H100)
Dataset: US Army Field Manuals
"""
import os

# Environment variables for multi-node training
NCCL_ENV = {
    "NCCL_IB_DISABLE": "1",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_P2P_LEVEL": "NVL",
    "NCCL_SHM_DISABLE": "1",
    "NCCL_SOCKET_IFNAME": "ens65f0",
    "NCCL_DEBUG": "WARN",
    "GLOO_SOCKET_IFNAME": "ens65f0",
    "GLOO_SOCKET_TIMEOUT_MS": "300000",  # 5 min timeout
    "RAY_DEDUP_LOGS": "1",
}
for k, v in NCCL_ENV.items():
    os.environ[k] = v

# Suppress verbose Ray logging
import logging
logging.getLogger("ray").setLevel(logging.WARNING)
logging.getLogger("ray.train").setLevel(logging.WARNING)
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

try:
    import ray
    from ray import train
    from ray.train import ScalingConfig, RunConfig, CheckpointConfig
    from ray.train.torch import TorchTrainer
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Warning: Ray not available")


class MilitaryDataset(Dataset):
    """Dataset for US Army Field Manuals."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load JSONL files
        data_dir = Path(data_path)
        for jsonl_file in data_dir.glob("*.jsonl"):
            print(f"Loading {jsonl_file.name}...")
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        if 'conversations' in item:
                            self.data.append(item)
                    except json.JSONDecodeError:
                        continue
        
        print(f"Loaded {len(self.data)} conversations")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        conversations = item['conversations']
        
        # Format as Qwen chat template
        text = ""
        for conv in conversations:
            role = conv.get('from', '')
            content = conv.get('value', '')
            if role == 'human':
                text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == 'gpt':
                text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Labels = input_ids (causal LM)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


def train_func(config: Dict[str, Any]):
    """Ray Train worker function."""
    # Setup distributed
    if RAY_AVAILABLE:
        import ray.train as train_module
        rank = train_module.get_context().get_world_rank()
        world_size = train_module.get_context().get_world_size()
        local_rank = train_module.get_context().get_local_rank()
    else:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    print(f"[Rank {rank}/{world_size}] Starting on device {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_path'],
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    print(f"[Rank {rank}] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config['model_path'],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    
    # Apply LoRA
    print(f"[Rank {rank}] Applying LoRA (rank={config['lora_rank']})...")
    lora_config = LoraConfig(
        r=config['lora_rank'],
        lora_alpha=config['lora_alpha'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    
    if rank == 0:
        model.print_trainable_parameters()
    
    # Wrap with DDP
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    
    # Load dataset
    print(f"[Rank {rank}] Loading dataset...")
    dataset = MilitaryDataset(
        config['data_path'],
        tokenizer,
        max_length=config['max_length']
    )
    
    # Distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01,
    )
    
    total_steps = len(dataloader) * config['num_epochs']
    warmup_steps = int(total_steps * 0.03)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Training loop
    global_step = 0
    for epoch in range(config['num_epochs']):
        sampler.set_epoch(epoch)
        model.train()
        
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=(rank != 0))
        
        for batch in progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            
            loss.backward()
            
            # Gradient accumulation
            if (global_step + 1) % config['gradient_accumulation'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            global_step += 1
            
            if rank == 0:
                progress.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"[Rank {rank}] Epoch {epoch+1} avg_loss: {avg_loss:.4f}")
        
        # Save checkpoint (rank 0 only)
        if rank == 0:
            checkpoint_dir = Path(config['output_dir']) / f"epoch_{epoch+1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save LoRA weights
            model.module.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"Saved checkpoint to {checkpoint_dir}")
        
        # Report metrics - ALL workers must call train.report for synchronization
        if RAY_AVAILABLE:
            train.report({"loss": avg_loss, "epoch": epoch + 1})
    
    # Final save
    if rank == 0:
        final_dir = Path(config['output_dir']) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        model.module.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"Final model saved to {final_dir}")


def main():
    parser = argparse.ArgumentParser(description="Ray Distributed Military SFT")
    parser.add_argument("--model_path", type=str, 
                       default="/data/hgt/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", type=str,
                       default="/data/hgt/datasets/us-army-fm-instruct")
    parser.add_argument("--output_dir", type=str,
                       default="/data/hgt/projects/verl_reproduction/checkpoints/military_ray_sft")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)  # Reduced for stability
    parser.add_argument("--gradient_accumulation", type=int, default=8)  # Increased to compensate
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Total GPU workers across all nodes (4 + 4)")
    parser.add_argument("--ray_address", type=str, default="auto",
                       help="Ray cluster address")
    args = parser.parse_args()
    
    # Training config
    train_config = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
    }
    
    if RAY_AVAILABLE:
        # Initialize Ray with runtime_env for NCCL
        print(f"Connecting to Ray cluster at {args.ray_address}...")
        ray.init(
            address=args.ray_address,
            runtime_env={"env_vars": NCCL_ENV}
        )
        
        # Print cluster info
        print(f"Ray cluster resources: {ray.cluster_resources()}")
        
        # Create trainer with Gloo backend (works better for multi-node without NVLink)
        trainer = TorchTrainer(
            train_loop_per_worker=train_func,
            train_loop_config=train_config,
            scaling_config=ScalingConfig(
                num_workers=args.num_workers,
                use_gpu=True,
                resources_per_worker={"GPU": 1, "CPU": 4},
            ),
            run_config=RunConfig(
                name="military_sft",
                storage_path=args.output_dir,
                checkpoint_config=CheckpointConfig(
                    num_to_keep=2,
                ),
            ),
            torch_config=ray.train.torch.TorchConfig(
                backend="gloo",  # Use Gloo instead of NCCL to avoid NVLink issues
            ),
        )
        
        # Run training
        print("Starting Ray distributed training...")
        result = trainer.fit()
        print(f"Training complete! Results: {result}")
    else:
        # Fallback to single GPU
        print("Ray not available, running single GPU training...")
        train_func(train_config)


if __name__ == "__main__":
    main()
