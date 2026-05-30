"""
Example usage of ARMOR - demonstrating core RLHF/PPO training concepts.

This example shows how to:
1. Create DataProto for data management
2. Set up models (Actor, Critic, Reference, Reward)
3. Train with PPO algorithm
"""

import numpy as np
import torch

from ARMOR import DataProto, BaseConfig
from ARMOR.trainer import (
    PPOConfig,
    PPOTrainer,
    ActorModel,
    CriticModel,
    ReferenceModel,
    RewardModel,
)
from ARMOR.core_algos import AdvantageEstimator


def create_dummy_data(batch_size: int = 16, seq_len: int = 32, vocab_size: int = 1000):
    """Create dummy training data for demonstration."""
    
    # Generate random input sequences
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Response mask (assume last half is response)
    response_mask = torch.zeros(batch_size, seq_len - 1)
    response_mask[:, seq_len//2:] = 1.0
    
    # Create DataProto
    data = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
        },
        non_tensors={
            "uid": np.arange(batch_size),  # Unique IDs for GRPO grouping
            "prompts": np.array([f"prompt_{i}" for i in range(batch_size)], dtype=object),
        },
        meta_info={
            "task": "demo",
            "batch_idx": 0,
        }
    )
    
    return data


def example_dataproto_usage():
    """Demonstrate DataProto functionality."""
    print("=" * 60)
    print("DataProto Usage Example")
    print("=" * 60)
    
    # Create data
    data = create_dummy_data(batch_size=16, seq_len=32)
    
    # Print info
    print("\n1. Data Info:")
    print(data.get_data_info())
    
    # Print size
    print("\n2. Memory Size:")
    data.print_size()
    
    # Indexing examples
    print("\n3. Indexing Examples:")
    
    # Single item
    item = data[0]
    print(f"   Single item type: {type(item).__name__}")
    
    # Slice
    sliced = data[5:10]
    print(f"   Sliced (5:10) length: {len(sliced)}")
    
    # List indexing
    selected = data[[0, 2, 4, 6]]
    print(f"   Selected [0,2,4,6] length: {len(selected)}")
    
    # Chunk
    chunks = data.chunk(4)
    print(f"   Chunked into 4: lengths = {[len(c) for c in chunks]}")
    
    # Concat
    concatenated = DataProto.concat(chunks)
    print(f"   Concatenated back: length = {len(concatenated)}")
    
    # Repeat
    repeated = data[:4].repeat(3)
    print(f"   Repeated 3x: length = {len(repeated)}")
    
    # Iterator
    print("\n4. Mini-batch Iterator:")
    iterator = data.make_iterator(mini_batch_size=4, epochs=1)
    for i, batch in enumerate(iterator):
        print(f"   Batch {i}: length = {len(batch)}")
    
    print("\n" + "=" * 60)


def example_ppo_training():
    """Demonstrate PPO training loop."""
    print("=" * 60)
    print("PPO Training Example")
    print("=" * 60)
    
    # Config
    config = PPOConfig(
        total_epochs=2,
        mini_batch_size=4,
        ppo_epochs=2,
        clip_range=0.2,
        adv_estimator="gae",
        actor_lr=1e-4,
        critic_lr=1e-4,
    )
    
    vocab_size = 1000
    hidden_size = 128
    
    # Create models
    print("\n1. Creating Models...")
    actor = ActorModel(vocab_size=vocab_size, hidden_size=hidden_size)
    critic = CriticModel(vocab_size=vocab_size, hidden_size=hidden_size)
    ref_model = ReferenceModel(actor)
    reward_model = RewardModel(vocab_size=vocab_size, hidden_size=hidden_size)
    
    print(f"   Actor params: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"   Critic params: {sum(p.numel() for p in critic.parameters()):,}")
    
    # Create trainer
    print("\n2. Creating Trainer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    trainer = PPOTrainer(
        config=config,
        actor_model=actor,
        critic_model=critic,
        ref_model=ref_model,
        reward_model=reward_model,
        device=device,
    )
    
    # Create training data
    print("\n3. Creating Training Data...")
    train_data = [create_dummy_data(batch_size=16, seq_len=32, vocab_size=vocab_size) 
                  for _ in range(3)]
    print(f"   Number of batches: {len(train_data)}")
    
    # Train
    print("\n4. Training...")
    history = trainer.fit(train_data, num_epochs=config.total_epochs)
    
    # Print results
    print("\n5. Training Results:")
    for epoch, metrics in enumerate(history["train"]):
        print(f"   Epoch {epoch + 1}:")
        print(f"      Total Loss: {metrics.get('total_loss', 0):.4f}")
        print(f"      Policy Loss: {metrics.get('policy/loss', 0):.4f}")
        print(f"      Value Loss: {metrics.get('value/loss', 0):.4f}")
        print(f"      Clip Fraction: {metrics.get('policy/clip_fraction', 0):.4f}")
    
    print("\n" + "=" * 60)


def example_advantage_estimators():
    """Demonstrate different advantage estimation methods."""
    print("=" * 60)
    print("Advantage Estimator Examples")
    print("=" * 60)
    
    from ARMOR.core_algos import (
        compute_gae_advantage_return,
        compute_grpo_outcome_advantage,
        compute_rloo_advantage,
    )
    
    # Create dummy data
    batch_size = 8
    seq_len = 16
    
    token_level_rewards = torch.randn(batch_size, seq_len)
    values = torch.randn(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)
    index = np.array([0, 0, 1, 1, 2, 2, 3, 3])  # Group pairs
    
    print("\n1. GAE (Generalized Advantage Estimation):")
    advantages, returns = compute_gae_advantage_return(
        token_level_rewards=token_level_rewards,
        values=values,
        response_mask=response_mask,
        gamma=0.99,
        lam=0.95,
    )
    print(f"   Advantages shape: {advantages.shape}")
    print(f"   Advantages mean: {advantages.mean():.4f}")
    print(f"   Advantages std: {advantages.std():.4f}")
    
    print("\n2. GRPO (Group Relative Policy Optimization):")
    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    print(f"   Advantages shape: {advantages.shape}")
    print(f"   Advantages mean: {advantages.mean():.4f}")
    print(f"   Advantages std: {advantages.std():.4f}")
    
    print("\n3. RLOO (Reinforce Leave-One-Out):")
    advantages, returns = compute_rloo_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    print(f"   Advantages shape: {advantages.shape}")
    print(f"   Advantages mean: {advantages.mean():.4f}")
    print(f"   Advantages std: {advantages.std():.4f}")
    
    print("\n" + "=" * 60)


def example_custom_reward_function():
    """Demonstrate using a custom reward function."""
    print("=" * 60)
    print("Custom Reward Function Example")
    print("=" * 60)
    
    vocab_size = 1000
    
    # Define custom reward function
    def custom_reward_fn(data: DataProto) -> torch.Tensor:
        """
        Custom reward: prefer shorter responses.
        In real applications, this could be:
        - Rule-based rewards (format checking, length penalties)
        - Code execution results
        - External API calls
        """
        attention_mask = data.batch["attention_mask"]
        lengths = attention_mask.sum(dim=1)
        # Reward = negative length (prefer shorter)
        rewards = -lengths / 100.0
        return rewards
    
    # Create trainer with custom reward
    config = PPOConfig(mini_batch_size=4, ppo_epochs=1)
    actor = ActorModel(vocab_size=vocab_size, hidden_size=64)
    critic = CriticModel(vocab_size=vocab_size, hidden_size=64)
    
    trainer = PPOTrainer(
        config=config,
        actor_model=actor,
        critic_model=critic,
        reward_fn=custom_reward_fn,
    )
    
    # Train
    data = [create_dummy_data(batch_size=8, seq_len=32, vocab_size=vocab_size)]
    history = trainer.fit(data, num_epochs=1)
    
    print("\n1. Training with custom reward function completed!")
    print(f"   Final loss: {history['train'][0].get('total_loss', 0):.4f}")
    
    print("\n" + "=" * 60)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ARMOR - Simplified RLHF Framework Demo")
    print("=" * 60 + "\n")
    
    # Run examples
    example_dataproto_usage()
    print("\n")
    
    example_advantage_estimators()
    print("\n")
    
    example_custom_reward_function()
    print("\n")
    
    example_ppo_training()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
