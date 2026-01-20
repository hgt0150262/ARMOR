"""
Example demonstrating DPO and ReMax algorithms in verl_mini.
"""

import torch
import numpy as np
from verl_mini.core_algos import (
    AlgorithmType,
    AdvantageEstimator,
    compute_dpo_loss,
    compute_remax_loss,
    compute_remax_advantage,
    compute_grpo_loss,
    get_algorithm_fn,
    ALGORITHM_REGISTRY,
)


def print_separator(title: str):
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60 + "\n")


def demo_dpo():
    """Demonstrate DPO (Direct Preference Optimization) algorithm."""
    print_separator("DPO (Direct Preference Optimization) Demo")
    
    batch_size = 4
    seq_len = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Simulate preference data (chosen vs rejected responses)
    # In practice, these come from preference dataset
    policy_chosen_log_probs = torch.randn(batch_size, seq_len, device=device) * 0.1 - 2.0
    policy_rejected_log_probs = torch.randn(batch_size, seq_len, device=device) * 0.1 - 2.5
    ref_chosen_log_probs = torch.randn(batch_size, seq_len, device=device) * 0.1 - 2.0
    ref_rejected_log_probs = torch.randn(batch_size, seq_len, device=device) * 0.1 - 2.5
    
    # Masks for valid tokens
    chosen_mask = torch.ones(batch_size, seq_len, device=device)
    rejected_mask = torch.ones(batch_size, seq_len, device=device)
    
    # Compute DPO loss with different beta values
    print("1. DPO Loss with different beta values:")
    for beta in [0.1, 0.5, 1.0]:
        loss, metrics = compute_dpo_loss(
            policy_chosen_log_probs=policy_chosen_log_probs,
            policy_rejected_log_probs=policy_rejected_log_probs,
            ref_chosen_log_probs=ref_chosen_log_probs,
            ref_rejected_log_probs=ref_rejected_log_probs,
            chosen_mask=chosen_mask,
            rejected_mask=rejected_mask,
            beta=beta,
        )
        print(f"   beta={beta}: loss={metrics['dpo/loss']:.4f}, "
              f"accuracy={metrics['dpo/reward_accuracy']:.2%}, "
              f"margin={metrics['dpo/reward_margin']:.4f}")
    
    # Different loss types
    print("\n2. DPO with different loss types:")
    for loss_type in ["sigmoid", "hinge", "ipo"]:
        loss, metrics = compute_dpo_loss(
            policy_chosen_log_probs=policy_chosen_log_probs,
            policy_rejected_log_probs=policy_rejected_log_probs,
            ref_chosen_log_probs=ref_chosen_log_probs,
            ref_rejected_log_probs=ref_rejected_log_probs,
            chosen_mask=chosen_mask,
            rejected_mask=rejected_mask,
            beta=0.1,
            loss_type=loss_type,
        )
        print(f"   {loss_type}: loss={loss.item():.4f}")
    
    # Label smoothing
    print("\n3. DPO with label smoothing:")
    for smooth in [0.0, 0.1, 0.2]:
        loss, metrics = compute_dpo_loss(
            policy_chosen_log_probs=policy_chosen_log_probs,
            policy_rejected_log_probs=policy_rejected_log_probs,
            ref_chosen_log_probs=ref_chosen_log_probs,
            ref_rejected_log_probs=ref_rejected_log_probs,
            chosen_mask=chosen_mask,
            rejected_mask=rejected_mask,
            beta=0.1,
            label_smoothing=smooth,
        )
        print(f"   smoothing={smooth}: loss={loss.item():.4f}")


def demo_remax():
    """Demonstrate ReMax algorithm."""
    print_separator("ReMax Algorithm Demo")
    
    batch_size = 8
    seq_len = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Simulate token-level rewards and log probs
    token_rewards = torch.randn(batch_size, seq_len, device=device) * 0.5
    log_probs = torch.randn(batch_size, seq_len, device=device) * 0.1 - 2.0
    ref_log_probs = torch.randn(batch_size, seq_len, device=device) * 0.1 - 2.0
    response_mask = torch.ones(batch_size, seq_len, device=device)
    
    # Group indices (4 samples per group)
    index = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    # Compute ReMax advantages
    print("1. ReMax Advantage Estimation:")
    advantages, returns = compute_remax_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=index,
    )
    print(f"   Advantages shape: {advantages.shape}")
    print(f"   Advantages mean: {advantages.mean().item():.4f}")
    print(f"   Advantages std: {advantages.std().item():.4f}")
    print(f"   Note: ReMax uses max reward as baseline, so advantages are <= 0")
    
    # Compute ReMax loss
    print("\n2. ReMax Loss Computation:")
    for beta in [0.01, 0.1, 0.5]:
        loss, metrics = compute_remax_loss(
            log_probs=log_probs,
            ref_log_probs=ref_log_probs,
            rewards=token_rewards,
            response_mask=response_mask,
            index=index,
            beta=beta,
        )
        print(f"   beta={beta}: loss={metrics['remax/loss']:.4f}, "
              f"pg_loss={metrics['remax/pg_loss']:.4f}, "
              f"kl_loss={metrics['remax/kl_loss']:.4f}")
    
    # Compare with/without reward normalization
    print("\n3. ReMax with/without reward normalization:")
    for normalize in [True, False]:
        loss, metrics = compute_remax_loss(
            log_probs=log_probs,
            ref_log_probs=ref_log_probs,
            rewards=token_rewards,
            response_mask=response_mask,
            index=index,
            beta=0.1,
            normalize_reward=normalize,
        )
        print(f"   normalize={normalize}: adv_mean={metrics['remax/advantage_mean']:.4f}, "
              f"adv_std={metrics['remax/advantage_std']:.4f}")


def demo_grpo():
    """Demonstrate GRPO algorithm."""
    print_separator("GRPO (Group Relative Policy Optimization) Demo")
    
    batch_size = 8
    seq_len = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Simulate data
    log_probs = torch.randn(batch_size, seq_len, device=device) * 0.1 - 2.0
    ref_log_probs = torch.randn(batch_size, seq_len, device=device) * 0.1 - 2.0
    rewards = torch.randn(batch_size, seq_len, device=device) * 0.5
    response_mask = torch.ones(batch_size, seq_len, device=device)
    index = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    print("1. GRPO Loss with different configurations:")
    for clip_range in [0.1, 0.2, 0.3]:
        loss, metrics = compute_grpo_loss(
            log_probs=log_probs,
            ref_log_probs=ref_log_probs,
            rewards=rewards,
            response_mask=response_mask,
            index=index,
            clip_range=clip_range,
        )
        print(f"   clip={clip_range}: loss={metrics['grpo/loss']:.4f}, "
              f"clip_frac={metrics['grpo/clip_fraction']:.2%}")
    
    print("\n2. GRPO with/without std normalization:")
    for normalize in [True, False]:
        loss, metrics = compute_grpo_loss(
            log_probs=log_probs,
            ref_log_probs=ref_log_probs,
            rewards=rewards,
            response_mask=response_mask,
            index=index,
            normalize_by_std=normalize,
        )
        print(f"   normalize_by_std={normalize}: "
              f"adv_mean={metrics['grpo/advantage_mean']:.4f}")


def demo_algorithm_registry():
    """Demonstrate algorithm registry system."""
    print_separator("Algorithm Registry System")
    
    print("1. Registered Algorithms:")
    for name, fn in ALGORITHM_REGISTRY.items():
        print(f"   - {name}: {fn.__name__}")
    
    print("\n2. Using get_algorithm_fn:")
    dpo_fn = get_algorithm_fn(AlgorithmType.DPO)
    print(f"   DPO function: {dpo_fn.__name__}")
    
    remax_fn = get_algorithm_fn("remax")
    print(f"   ReMax function: {remax_fn.__name__}")


def demo_training_loop():
    """Demonstrate a simple training loop with DPO."""
    print_separator("Simple DPO Training Loop Demo")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    seq_len = 32
    hidden_size = 64
    num_steps = 5
    
    # Simple policy network (for demo)
    policy = torch.nn.Sequential(
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, 1),
    ).to(device)
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    print(f"Training on {device} for {num_steps} steps...")
    
    for step in range(num_steps):
        # Simulate embeddings
        chosen_emb = torch.randn(batch_size, seq_len, hidden_size, device=device)
        rejected_emb = torch.randn(batch_size, seq_len, hidden_size, device=device)
        
        # Get log probs from policy
        policy_chosen_logps = policy(chosen_emb).squeeze(-1)
        policy_rejected_logps = policy(rejected_emb).squeeze(-1)
        
        # Reference (frozen, simulated)
        with torch.no_grad():
            ref_chosen_logps = torch.randn_like(policy_chosen_logps) * 0.1
            ref_rejected_logps = torch.randn_like(policy_rejected_logps) * 0.1
        
        masks = torch.ones(batch_size, seq_len, device=device)
        
        # Compute DPO loss
        loss, metrics = compute_dpo_loss(
            policy_chosen_log_probs=policy_chosen_logps,
            policy_rejected_log_probs=policy_rejected_logps,
            ref_chosen_log_probs=ref_chosen_logps,
            ref_rejected_log_probs=ref_rejected_logps,
            chosen_mask=masks,
            rejected_mask=masks,
            beta=0.1,
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"   Step {step+1}: loss={metrics['dpo/loss']:.4f}, "
              f"accuracy={metrics['dpo/reward_accuracy']:.2%}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    print("="*60)
    print("verl_mini - DPO & ReMax Algorithm Examples")
    print("="*60)
    
    demo_dpo()
    demo_remax()
    demo_grpo()
    demo_algorithm_registry()
    demo_training_loop()
    
    print("\n" + "="*60)
    print("All DPO & ReMax examples completed successfully!")
    print("="*60)
