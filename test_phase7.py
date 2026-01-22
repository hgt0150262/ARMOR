"""Test Phase 7: OPO + Policy Loss Registry (aligned with official verl)"""
import torch
import numpy as np

from verl_mini.trainer.ppo.core_algos import (
    compute_opo_outcome_advantage,
    ADV_ESTIMATOR_REGISTRY,
    POLICY_LOSS_REGISTRY,
    get_adv_estimator_fn,
    get_policy_loss_fn,
    AdvantageEstimator
)

print("=== Phase 7 Test: OPO + Registry (aligned with official verl) ===")
print(f"AdvantageEstimator enums: {[e.value for e in AdvantageEstimator]}")
print(f"ADV_ESTIMATOR_REGISTRY: {list(ADV_ESTIMATOR_REGISTRY.keys())}")
print(f"POLICY_LOSS_REGISTRY: {list(POLICY_LOSS_REGISTRY.keys())}")

# Test OPO
batch_size, seq_len = 8, 16
rewards = torch.randn(batch_size, seq_len)
mask = torch.ones(batch_size, seq_len)
index = np.array([0, 0, 1, 1, 2, 2, 3, 3])

adv, ret = compute_opo_outcome_advantage(rewards, mask, index)
print(f"OPO advantages shape: {adv.shape}")
print(f"OPO advantages mean: {adv.mean():.4f}")

# Test registry
opo_fn = get_adv_estimator_fn("opo")
print(f"get_adv_estimator_fn(opo): {opo_fn.__name__}")

ppo_loss_fn = get_policy_loss_fn("ppo")
print(f"get_policy_loss_fn(ppo): {ppo_loss_fn.__name__}")

# Test vanilla policy loss alias
vanilla_loss_fn = get_policy_loss_fn("vanilla")
print(f"get_policy_loss_fn(vanilla): {vanilla_loss_fn.__name__}")

# Test KL penalty types
from verl_mini.trainer.ppo.core_algos import kl_penalty
log_probs = torch.randn(4, 8)
ref_log_probs = torch.randn(4, 8)
for kl_type in ["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"]:
    kl = kl_penalty(log_probs, ref_log_probs, kl_type)
    print(f"kl_penalty({kl_type}): shape={kl.shape}")

print("=== All tests passed! ===")
