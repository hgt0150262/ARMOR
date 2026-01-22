"""Test Phase 7: OPO + Policy Loss Registry"""
import torch
import numpy as np

from verl_mini.trainer.ppo.core_algos import (
    compute_opo_outcome_advantage,
    ADV_ESTIMATOR_REGISTRY,
    POLICY_LOSS_REGISTRY,
    get_adv_estimator_fn,
    get_policy_loss_fn
)

print("=== Phase 7 Test: OPO + Registry ===")
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

print("=== All tests passed! ===")
