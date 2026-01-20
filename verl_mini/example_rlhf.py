"""
Example: RLHF Training with verl_mini
Demonstrates complete RLHF training workflow with Qwen2.5-0.5B.
"""

import os
import sys
import torch
from typing import List

# Add project root to path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl_mini import (
    # Model utilities
    ModelConfig,
    ModelManager,
    TRANSFORMERS_AVAILABLE,
    PEFT_AVAILABLE,
    # Logging
    LoggingConfig,
    TrainingLogger,
    create_logger,
    WANDB_AVAILABLE,
    TENSORBOARD_AVAILABLE,
    # RLHF training
    RLHFConfig,
    RLHFTrainer,
    create_rlhf_trainer,
    # Algorithms
    AdvantageEstimator,
)


def check_dependencies():
    """Check and report available dependencies."""
    print("="*60)
    print("verl_mini RLHF Training - Dependency Check")
    print("="*60)
    
    deps = {
        "Transformers": TRANSFORMERS_AVAILABLE,
        "PEFT (LoRA)": PEFT_AVAILABLE,
        "WandB": WANDB_AVAILABLE,
        "TensorBoard": TENSORBOARD_AVAILABLE,
        "CUDA": torch.cuda.is_available(),
    }
    
    for name, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {name}")
        
    if torch.cuda.is_available():
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
    print("="*60 + "\n")
    
    return all([TRANSFORMERS_AVAILABLE])


def demo_logging():
    """Demonstrate logging utilities."""
    print("\n" + "="*60)
    print("Logging Demo")
    print("="*60)
    
    # Create logger
    logger = create_logger(
        project_name="verl_mini_demo",
        run_name="logging_demo",
        log_dir="./logs_demo",
        use_wandb=False,
        use_tensorboard=True,
    )
    
    # Log configuration
    logger.log_config({
        "batch_size": 8,
        "learning_rate": 1e-5,
        "epochs": 3,
    })
    
    # Simulate training metrics
    for step in range(50):
        metrics = {
            "loss": 2.0 - step * 0.02,
            "reward": 0.5 + step * 0.01,
            "entropy": 1.0 - step * 0.005,
        }
        logger.log_metrics(metrics, step=step)
        
    # Log epoch summary
    logger.log_epoch(0, {"avg_loss": 1.5, "avg_reward": 0.75})
    
    logger.close()
    print("Logging demo complete!\n")


def demo_model_loading():
    """Demonstrate model loading utilities."""
    print("\n" + "="*60)
    print("Model Loading Demo")
    print("="*60)
    
    if not TRANSFORMERS_AVAILABLE:
        print("Skipping: transformers not available")
        return None
        
    # Configure model
    config = ModelConfig(
        model_name_or_path="Qwen/Qwen2.5-0.5B",
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        torch_dtype="bfloat16",
    )
    
    # Create manager
    manager = ModelManager(config)
    
    # Load model (this will download if not cached)
    print("\nLoading model (may take a moment if downloading)...")
    try:
        model, tokenizer = manager.load_model()
        
        # Apply LoRA if available
        if PEFT_AVAILABLE and config.use_lora:
            model = manager.apply_lora()
            
        # Show model info
        info = manager.get_model_info()
        print(f"\nModel Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
            
        return manager
        
    except Exception as e:
        print(f"Model loading failed: {e}")
        print("This may be due to network issues or missing model files.")
        return None


def simple_reward_fn(prompts: List[str], responses: List[str]) -> torch.Tensor:
    """Simple reward function for demonstration."""
    rewards = []
    for prompt, response in zip(prompts, responses):
        # Reward based on:
        # 1. Response length (moderate length is good)
        # 2. Diversity (avoid repetition)
        # 3. Relevance (contains keywords from prompt)
        
        length_score = min(len(response) / 100, 1.0) - max(0, (len(response) - 200) / 200)
        
        # Penalize repetition
        words = response.lower().split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        diversity_score = unique_ratio
        
        # Check for prompt keywords
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        relevance_score = len(prompt_words & response_words) / max(len(prompt_words), 1)
        
        reward = 0.4 * length_score + 0.3 * diversity_score + 0.3 * relevance_score
        rewards.append(reward)
        
    return torch.tensor(rewards)


def demo_rlhf_training():
    """Demonstrate RLHF training workflow."""
    print("\n" + "="*60)
    print("RLHF Training Demo")
    print("="*60)
    
    if not TRANSFORMERS_AVAILABLE:
        print("Skipping: transformers not available")
        return
        
    # Sample prompts for training
    train_prompts = [
        "Explain the concept of machine learning in simple terms.",
        "What are the benefits of renewable energy?",
        "How does photosynthesis work?",
        "Describe the water cycle.",
        "What is the difference between AI and ML?",
        "Explain how computers store data.",
        "What causes earthquakes?",
        "How do vaccines work?",
        "What is climate change?",
        "Explain the theory of evolution.",
        "How does the internet work?",
        "What is quantum computing?",
        "Explain blockchain technology.",
        "What are neural networks?",
        "How do self-driving cars work?",
        "What is the greenhouse effect?",
    ]
    
    eval_prompts = [
        "What is artificial intelligence?",
        "How does electricity work?",
    ]
    
    # Configure RLHF training
    config = RLHFConfig(
        total_epochs=1,  # Just 1 epoch for demo
        batch_size=4,
        mini_batch_size=2,
        ppo_epochs=2,
        
        # Algorithm
        adv_estimator="grpo",
        clip_range=0.2,
        
        # Generation
        max_prompt_length=128,
        max_response_length=64,
        temperature=0.7,
        
        # Logging
        log_interval=1,
        save_steps=100,
        save_dir="./checkpoints_demo",
        use_tensorboard=True,
        use_wandb=False,
        project_name="verl_mini_rlhf_demo",
    )
    
    # Model configuration
    model_config = ModelConfig(
        model_name_or_path="Qwen/Qwen2.5-0.5B",
        use_lora=True,
        lora_r=8,
        torch_dtype="bfloat16",
    )
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_config.model_name_or_path}")
    print(f"  LoRA: r={model_config.lora_r}")
    print(f"  Epochs: {config.total_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Advantage estimator: {config.adv_estimator}")
    
    try:
        # Create model manager
        manager = ModelManager(model_config)
        model, tokenizer = manager.load_model()
        
        if PEFT_AVAILABLE and model_config.use_lora:
            model = manager.apply_lora()
            
        # Create trainer
        trainer = RLHFTrainer(
            config=config,
            model_manager=manager,
            reward_fn=simple_reward_fn,
        )
        
        # Run training
        print("\nStarting RLHF training...")
        metrics = trainer.train(
            train_prompts=train_prompts,
            eval_prompts=eval_prompts,
        )
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Final metrics: {metrics[-1] if metrics else 'N/A'}")
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()


def demo_quick_test():
    """Quick test without full training."""
    print("\n" + "="*60)
    print("Quick Test (No Model Loading)")
    print("="*60)
    
    # Test logging
    logger = create_logger(
        project_name="test",
        log_dir="./logs_test",
        use_wandb=False,
        use_tensorboard=False,
    )
    
    logger.log_config({"test": True})
    logger.log_metrics({"loss": 1.0, "reward": 0.5}, step=0)
    logger.close()
    
    print("Quick test passed!")


def main():
    print("\n" + "="*60)
    print("verl_mini RLHF Training Examples")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        print("Missing required dependencies. Running quick test only.")
        demo_quick_test()
        return
        
    # Run demos
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", choices=["all", "logging", "model", "train", "quick"],
                       default="all", help="Which demo to run")
    args = parser.parse_args()
    
    if args.demo in ["all", "logging"]:
        demo_logging()
        
    if args.demo in ["all", "model"]:
        demo_model_loading()
        
    if args.demo == "train":
        demo_rlhf_training()
        
    if args.demo == "quick":
        demo_quick_test()
        
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
