"""
Qwen2.5-7B GRPO Training Script for verl_mini
Based on official verl examples/tuning/7b configuration

Usage:
    python verl_mini/examples/train_qwen7b_grpo.py --model_path /path/to/Qwen2.5-7B-Instruct

Reference: Official verl GRPO-LoRA configuration for 7B models
"""

import os
import sys
import argparse
import re
import torch
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from verl_mini import (
    ModelConfig,
    ModelManager,
    RLHFConfig,
    RLHFTrainer,
    AdvantageEstimator,
    TRANSFORMERS_AVAILABLE,
    PEFT_AVAILABLE,
)


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract numerical answer from GSM8K format response."""
    # Look for #### pattern
    match = re.search(r'####\s*(\-?[\d,\.]+)', text)
    if match:
        return match.group(1).replace(',', '')
    # Fallback: look for last number
    numbers = re.findall(r'\-?[\d,\.]+', text)
    return numbers[-1].replace(',', '') if numbers else None


def gsm8k_reward_fn(prompts: List[str], responses: List[str], ground_truths: Optional[List[str]] = None) -> torch.Tensor:
    """
    GSM8K reward function based on answer correctness.
    
    Args:
        prompts: Input prompts
        responses: Model responses
        ground_truths: Expected answers (from extra_info)
    
    Returns:
        Tensor of rewards (1.0 for correct, 0.0 for incorrect)
    """
    rewards = []
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        pred_answer = extract_gsm8k_answer(response)
        
        # Get ground truth from closure or default
        if ground_truths and i < len(ground_truths):
            gt_answer = ground_truths[i]
        else:
            gt_answer = None
        
        if pred_answer and gt_answer:
            # Exact match
            try:
                pred_num = float(pred_answer)
                gt_num = float(gt_answer)
                reward = 1.0 if abs(pred_num - gt_num) < 1e-6 else 0.0
            except ValueError:
                reward = 1.0 if pred_answer == gt_answer else 0.0
        else:
            # Partial reward for structured response
            has_steps = '=' in response or 'step' in response.lower()
            has_answer = '####' in response
            reward = 0.3 * has_steps + 0.2 * has_answer
        
        rewards.append(reward)
    
    return torch.tensor(rewards, dtype=torch.float32)


def load_gsm8k_data(data_path: str) -> tuple:
    """Load preprocessed GSM8K data from parquet."""
    import pandas as pd
    
    df = pd.read_parquet(data_path)
    prompts = []
    ground_truths = []
    
    for _, row in df.iterrows():
        # Extract prompt from chat format
        prompt_data = row['prompt']
        if isinstance(prompt_data, list) and len(prompt_data) > 0:
            prompt = prompt_data[0].get('content', '')
        else:
            prompt = str(prompt_data)
        prompts.append(prompt)
        
        # Extract ground truth
        reward_model = row.get('reward_model', {})
        if isinstance(reward_model, dict):
            gt = reward_model.get('ground_truth', '')
        else:
            gt = ''
        ground_truths.append(gt)
    
    return prompts, ground_truths


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B GRPO Training")
    
    # Model
    parser.add_argument("--model_path", type=str, 
                        default="/data/hgt/models/Qwen2.5-7B-Instruct",
                        help="Path to model")
    
    # Data
    parser.add_argument("--train_data", type=str, 
                        default="data/gsm8k/train.parquet",
                        help="Training data path")
    parser.add_argument("--val_data", type=str, 
                        default="data/gsm8k/test.parquet",
                        help="Validation data path")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max training samples (-1 for all)")
    
    # Training
    parser.add_argument("--total_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mini_batch_size", type=int, default=16)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # Generation
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_response_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    
    # Algorithm
    parser.add_argument("--adv_estimator", type=str, default="grpo",
                        choices=["gae", "grpo", "reinforce_plus_plus", "rloo"])
    parser.add_argument("--grpo_n", type=int, default=5,
                        help="Number of samples per prompt for GRPO")
    parser.add_argument("--clip_range", type=float, default=0.2)
    
    # KL control
    parser.add_argument("--kl_coef", type=float, default=0.001)
    parser.add_argument("--kl_type", type=str, default="low_var_kl",
                        choices=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"])
    
    # Optimization
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for generation")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--vllm_gpu_memory", type=float, default=0.2)
    
    # Logging
    parser.add_argument("--project_name", type=str, default="verl_mini_qwen7b")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    
    args = parser.parse_args()
    
    # ============================================
    # Configuration
    # ============================================
    print("="*60)
    print("Qwen2.5-7B GRPO Training with verl_mini")
    print("="*60)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
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
    
    # RLHF config
    rlhf_config = RLHFConfig(
        total_epochs=args.total_epochs,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        learning_rate=args.learning_rate,
        
        # Algorithm
        adv_estimator=args.adv_estimator,
        clip_range=args.clip_range,
        kl_coef=args.kl_coef,
        
        # Generation
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        temperature=args.temperature,
        top_p=args.top_p,
        
        # vLLM
        use_vllm=args.use_vllm,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory,
        
        # Logging
        project_name=args.project_name,
        run_name=args.experiment_name,
        log_interval=args.log_interval,
        save_steps=args.save_steps,
        save_dir=args.save_dir,
        use_wandb=False,
        use_tensorboard=True,
    )
    
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_path}")
    print(f"  LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"  Algorithm: {args.adv_estimator}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  KL: {args.kl_type} (coef={args.kl_coef})")
    print(f"  vLLM: {args.use_vllm}")
    
    # ============================================
    # Load Data
    # ============================================
    print(f"\nLoading training data from {args.train_data}...")
    train_prompts, train_ground_truths = load_gsm8k_data(args.train_data)
    
    print(f"Loading validation data from {args.val_data}...")
    val_prompts, val_ground_truths = load_gsm8k_data(args.val_data)
    
    if args.max_samples > 0:
        train_prompts = train_prompts[:args.max_samples]
        train_ground_truths = train_ground_truths[:args.max_samples]
    
    print(f"  Training samples: {len(train_prompts)}")
    print(f"  Validation samples: {len(val_prompts)}")
    
    # Create reward function with ground truths
    def reward_fn(prompts, responses):
        return gsm8k_reward_fn(prompts, responses, train_ground_truths)
    
    # ============================================
    # Initialize SwanLab
    # ============================================
    try:
        import swanlab
        swanlab.init(
            project=args.project_name,
            name=args.experiment_name or f"qwen7b_grpo_lora{args.lora_rank}",
            mode="local",
            logdir="./swanlog",
        )
        print("SwanLab initialized")
    except Exception as e:
        print(f"SwanLab init failed: {e}")
    
    # ============================================
    # Train
    # ============================================
    print("\n" + "="*60)
    print("Starting GRPO Training...")
    print("="*60)
    
    try:
        # Create model manager
        manager = ModelManager(model_config)
        model, tokenizer = manager.load_model()
        
        if PEFT_AVAILABLE and model_config.use_lora:
            model = manager.apply_lora()
            print(f"LoRA applied: rank={args.lora_rank}")
        
        # Create trainer
        trainer = RLHFTrainer(
            config=rlhf_config,
            model_manager=manager,
            reward_fn=reward_fn,
        )
        
        # Run training
        metrics = trainer.train(
            train_prompts=train_prompts,
            eval_prompts=val_prompts[:100],  # Use subset for eval
        )
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        if metrics:
            print(f"Final metrics: {metrics[-1]}")
        
        # Save final model
        save_path = os.path.join(args.save_dir, "final")
        manager.save_model(save_path)
        print(f"Model saved to {save_path}")
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Finish SwanLab
    try:
        import swanlab
        swanlab.finish()
    except:
        pass


if __name__ == "__main__":
    main()
