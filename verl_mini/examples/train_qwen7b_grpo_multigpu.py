"""
Qwen2.5-7B GRPO Multi-GPU Training Script for verl_mini
Supports 4x H100 distributed training with accelerate

Usage:
    # Single command (auto-detect GPUs)
    accelerate launch --multi_gpu --num_processes 4 verl_mini/examples/train_qwen7b_grpo_multigpu.py

    # Or with torchrun
    torchrun --nproc_per_node=4 verl_mini/examples/train_qwen7b_grpo_multigpu.py
"""

import os
import sys
import argparse
import re
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from verl_mini import (
    ModelConfig,
    ModelManager,
    RLHFConfig,
    AdvantageEstimator,
    TRANSFORMERS_AVAILABLE,
    PEFT_AVAILABLE,
)


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract numerical answer from GSM8K format response with multiple patterns."""
    # Pattern 1: #### format (standard GSM8K)
    match = re.search(r'####\s*(\-?[\d,\.]+)', text)
    if match:
        return match.group(1).replace(',', '')
    
    # Pattern 2: "answer is X" or "answer: X"
    match = re.search(r'answer\s*(?:is|:|=)\s*(\-?[\d,\.]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')
    
    # Pattern 3: "= X" at end of line (calculation result)
    match = re.search(r'=\s*(\-?[\d,\.]+)\s*$', text, re.MULTILINE)
    if match:
        return match.group(1).replace(',', '')
    
    # Pattern 4: Last number in the text (fallback)
    numbers = re.findall(r'\-?[\d,\.]+', text)
    if numbers:
        # Filter out very small numbers (likely not answers)
        valid_numbers = [n for n in numbers if len(n.replace(',', '').replace('.', '')) <= 10]
        return valid_numbers[-1].replace(',', '') if valid_numbers else None
    
    return None


def gsm8k_reward_fn(prompts: List[str], responses: List[str], ground_truths: Optional[List[str]] = None, debug: bool = False) -> torch.Tensor:
    """
    GSM8K reward function with format reward shaping.
    
    Reward structure:
    - 1.0: Correct answer
    - 0.5: Close answer (within 10%)
    - 0.3: Has reasoning steps + some number
    - 0.2: Has reasoning steps
    - 0.1: Has any mathematical content
    - 0.0: No valid response
    """
    rewards = []
    
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        pred_answer = extract_gsm8k_answer(response)
        gt_answer = ground_truths[i] if ground_truths and i < len(ground_truths) else None
        
        reward = 0.0
        
        # Check for correct answer
        if pred_answer and gt_answer:
            try:
                pred_num = float(pred_answer)
                gt_num = float(gt_answer)
                
                if abs(pred_num - gt_num) < 1e-6:
                    reward = 1.0  # Exact match
                elif gt_num != 0 and abs(pred_num - gt_num) / abs(gt_num) < 0.1:
                    reward = 0.5  # Within 10%
                elif gt_num != 0 and abs(pred_num - gt_num) / abs(gt_num) < 0.2:
                    reward = 0.3  # Within 20%
            except (ValueError, ZeroDivisionError):
                if pred_answer == gt_answer:
                    reward = 1.0
        
        # Format reward shaping (if no correct answer)
        if reward < 0.3:
            has_steps = any(op in response for op in ['=', '+', '-', '*', '/'])
            has_numbers = bool(re.search(r'\d+', response))
            has_structure = '####' in response or 'answer' in response.lower()
            
            format_reward = 0.0
            if has_steps and has_numbers:
                format_reward = 0.2
            elif has_steps or has_numbers:
                format_reward = 0.1
            if has_structure:
                format_reward += 0.1
            
            reward = max(reward, format_reward)
        
        # Debug logging (only first batch)
        if debug and i == 0:
            print(f"[DEBUG] GT: {gt_answer}, Pred: {pred_answer}, Reward: {reward:.2f}")
            print(f"[DEBUG] Response (first 200 chars): {response[:200]}...")
        
        rewards.append(reward)
    
    return torch.tensor(rewards, dtype=torch.float32)


def load_gsm8k_data(data_path: str) -> tuple:
    """Load preprocessed GSM8K data from parquet."""
    import pandas as pd
    
    df = pd.read_parquet(data_path)
    prompts = []
    ground_truths = []
    
    for _, row in df.iterrows():
        prompt_data = row['prompt']
        if isinstance(prompt_data, list) and len(prompt_data) > 0:
            prompt = prompt_data[0].get('content', '')
        else:
            prompt = str(prompt_data)
        prompts.append(prompt)
        
        reward_model = row.get('reward_model', {})
        if isinstance(reward_model, dict):
            gt = reward_model.get('ground_truth', '')
        else:
            gt = ''
        ground_truths.append(gt)
    
    return prompts, ground_truths


class PromptDataset(torch.utils.data.Dataset):
    """Simple dataset for prompts."""
    def __init__(self, prompts, ground_truths):
        self.prompts = prompts
        self.ground_truths = ground_truths
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "ground_truth": self.ground_truths[idx]}


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B GRPO Multi-GPU Training")
    
    # Model
    parser.add_argument("--model_path", type=str, 
                        default="/data/hgt/models/Qwen2.5-7B-Instruct")
    
    # Data
    parser.add_argument("--train_data", type=str, default="data/gsm8k/train.parquet")
    parser.add_argument("--val_data", type=str, default="data/gsm8k/test.parquet")
    parser.add_argument("--max_samples", type=int, default=-1)
    
    # Training
    parser.add_argument("--total_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)  # Per GPU
    parser.add_argument("--mini_batch_size", type=int, default=4)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # Generation
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_response_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    
    # Algorithm
    parser.add_argument("--adv_estimator", type=str, default="grpo")
    parser.add_argument("--grpo_n", type=int, default=5)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--kl_coef", type=float, default=0.001)
    
    # Optimization
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    
    # Logging
    parser.add_argument("--project_name", type=str, default="verl_mini_qwen7b_multigpu")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    if is_main_process(rank):
        print("="*60)
        print("Qwen2.5-7B GRPO Multi-GPU Training")
        print("="*60)
        print(f"World size: {world_size} GPUs")
        print(f"Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
    
    # Model config
    model_config = ModelConfig(
        model_name_or_path=args.model_path,
        use_lora=True,
        lora_r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        torch_dtype="bfloat16",
        gradient_checkpointing=args.gradient_checkpointing,
    )
    
    # Load model
    manager = ModelManager(model_config)
    model, tokenizer = manager.load_model()
    
    # Fix padding side for decoder-only models
    tokenizer.padding_side = 'left'
    if is_main_process(rank):
        print(f"Tokenizer padding_side set to: {tokenizer.padding_side}")
    
    if PEFT_AVAILABLE and model_config.use_lora:
        model = manager.apply_lora()
        if is_main_process(rank):
            print(f"LoRA applied: rank={args.lora_rank}")
    
    model = model.to(device)
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        # Fix for gradient_checkpointing + DDP compatibility
        model._set_static_graph()
        if is_main_process(rank):
            print("Model wrapped with DistributedDataParallel (static_graph=True)")
    
    # Create reference model (on each GPU)
    ref_model = manager.create_reference_model()
    ref_model = ref_model.to(device)
    ref_model.eval()
    
    # Load data
    if is_main_process(rank):
        print(f"\nLoading data from {args.train_data}...")
    
    train_prompts, train_ground_truths = load_gsm8k_data(args.train_data)
    
    if args.max_samples > 0:
        train_prompts = train_prompts[:args.max_samples]
        train_ground_truths = train_ground_truths[:args.max_samples]
    
    if is_main_process(rank):
        print(f"Training samples: {len(train_prompts)}")
    
    # Create dataset and distributed sampler
    dataset = PromptDataset(train_prompts, train_ground_truths)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler,
        collate_fn=lambda x: x,  # Keep as list of dicts
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    
    # Scheduler
    num_training_steps = len(dataloader) * args.total_epochs * args.ppo_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_training_steps, eta_min=args.learning_rate * 0.1
    )
    
    # SwanLab (main process only)
    if is_main_process(rank):
        try:
            import swanlab
            exp_name = args.experiment_name or f"qwen7b_grpo_{world_size}gpu"
            swanlab.init(
                project=args.project_name,
                name=exp_name,
                mode="local",
                logdir="./swanlog",
            )
            print("SwanLab initialized")
        except Exception as e:
            print(f"SwanLab init failed: {e}")
    
    # Training loop
    if is_main_process(rank):
        print("\n" + "="*60)
        print("Starting Multi-GPU GRPO Training...")
        print("="*60)
    
    global_step = 0
    
    for epoch in range(args.total_epochs):
        sampler.set_epoch(epoch)
        model.train()
        
        if is_main_process(rank):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        else:
            pbar = dataloader
        
        for batch_idx, batch in enumerate(pbar):
            prompts = [item["prompt"] for item in batch]
            ground_truths = [item["ground_truth"] for item in batch]
            
            # Tokenize prompts
            prompt_encodings = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=args.max_prompt_length,
                return_tensors="pt",
            ).to(device)
            
            # Generate responses
            raw_model = model.module if hasattr(model, 'module') else model
            with torch.no_grad():
                output_ids = raw_model.generate(
                    **prompt_encodings,
                    max_new_tokens=args.max_response_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    repetition_penalty=args.repetition_penalty,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            response_ids = output_ids[:, prompt_encodings["input_ids"].shape[1]:]
            responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            
            # Compute rewards (debug=True for first batch to see outputs)
            debug_mode = (batch_idx == 0 and epoch == 0 and is_main_process(rank))
            rewards = gsm8k_reward_fn(prompts, responses, ground_truths, debug=debug_mode).to(device)
            
            # Forward pass for log probs
            full_ids = torch.cat([prompt_encodings["input_ids"], response_ids], dim=1)
            outputs = model(input_ids=full_ids, return_dict=True)
            logits = outputs.logits
            
            # Compute log probs
            response_start = prompt_encodings["input_ids"].shape[1]
            response_logits = logits[:, response_start-1:-1, :]
            log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs, dim=-1, index=response_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            # Reference log probs
            with torch.no_grad():
                ref_outputs = ref_model(input_ids=full_ids, return_dict=True)
                ref_logits = ref_outputs.logits
                ref_response_logits = ref_logits[:, response_start-1:-1, :]
                ref_log_probs = torch.nn.functional.log_softmax(ref_response_logits, dim=-1)
                ref_token_log_probs = torch.gather(
                    ref_log_probs, dim=-1, index=response_ids.unsqueeze(-1)
                ).squeeze(-1)
            
            # Compute advantages (simplified GRPO)
            advantages = rewards - rewards.mean()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Policy loss
            ratio = torch.exp(token_log_probs.sum(-1) - token_log_probs.detach().sum(-1))
            clipped_ratio = torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            
            # KL penalty
            kl = (token_log_probs - ref_token_log_probs).mean()
            
            # Total loss
            loss = policy_loss + args.kl_coef * kl
            loss = loss / args.gradient_accumulation_steps
            
            loss.backward()
            
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging (main process only)
                if is_main_process(rank) and global_step % args.log_interval == 0:
                    metrics = {
                        "loss": loss.item() * args.gradient_accumulation_steps,
                        "policy_loss": policy_loss.item(),
                        "kl": kl.item(),
                        "reward_mean": rewards.mean().item(),
                        "lr": scheduler.get_last_lr()[0],
                    }
                    
                    try:
                        import swanlab
                        swanlab.log(metrics, step=global_step)
                    except:
                        pass
                    
                    pbar.set_postfix(loss=f"{metrics['loss']:.4f}", reward=f"{metrics['reward_mean']:.4f}")
                
                # Save checkpoint
                if is_main_process(rank) and global_step % args.save_steps == 0:
                    save_path = os.path.join(args.save_dir, f"step_{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    raw_model = model.module if hasattr(model, 'module') else model
                    raw_model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"Checkpoint saved: {save_path}")
    
    # Final save
    if is_main_process(rank):
        save_path = os.path.join(args.save_dir, "final")
        os.makedirs(save_path, exist_ok=True)
        raw_model = model.module if hasattr(model, 'module') else model
        raw_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"\nTraining complete! Model saved to {save_path}")
        
        try:
            import swanlab
            swanlab.finish()
        except:
            pass
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
