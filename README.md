# ARMOR - Simplified RLHF Framework

A simplified reproduction of ByteDance's [verl](https://github.com/volcengine/verl) framework for understanding RLHF (Reinforcement Learning from Human Feedback) concepts.

## Overview

ARMOR demonstrates the core concepts of verl without the complexity of distributed training. It includes:

- **DataProto**: Unified data protocol for RLHF training
- **Core Algorithms**: GAE, GRPO, RLOO advantage estimators
- **PPO Trainer**: Complete PPO training loop
- **Worker Abstractions**: Simplified distributed worker concepts

## Original verl Features

The original verl framework from ByteDance provides:

| Feature | Description |
|---------|-------------|
| **Distributed Training** | FSDP, FSDP2, Megatron-LM backends |
| **Rollout Engines** | vLLM, SGLang, HF Transformers |
| **RL Algorithms** | PPO, GRPO, REINFORCE++, RLOO, DAPO |
| **Model Support** | Qwen, Llama, DeepSeek, Gemma |
| **Scalability** | Up to 671B models on 100s of GPUs |

## Installation

```bash
cd verl_reproduction
pip install -r requirements.txt
```

## Quick Start

### 1. DataProto Usage

```python
from ARMOR import DataProto
import torch
import numpy as np

# Create data
data = DataProto.from_dict(
    tensors={
        "input_ids": torch.randint(0, 1000, (16, 32)),
        "attention_mask": torch.ones(16, 32),
    },
    non_tensors={
        "prompts": np.array(["Hello"] * 16, dtype=object),
    },
    meta_info={"task": "chat"}
)

# Indexing
batch = data[0:8]           # Slice
item = data[5]              # Single item
selected = data[[1, 3, 5]]  # List indexing

# Chunking and concatenation
chunks = data.chunk(4)
combined = DataProto.concat(chunks)

# Mini-batch iterator
for mini_batch in data.make_iterator(mini_batch_size=4, epochs=2):
    print(len(mini_batch))
```

### 2. PPO Training

```python
from ARMOR.trainer import PPOConfig, PPOTrainer, ActorModel, CriticModel

# Config
config = PPOConfig(
    total_epochs=3,
    mini_batch_size=8,
    clip_range=0.2,
    adv_estimator="gae",  # or "grpo", "rloo"
)

# Models
actor = ActorModel(vocab_size=32000, hidden_size=256)
critic = CriticModel(vocab_size=32000, hidden_size=256)

# Trainer
trainer = PPOTrainer(
    config=config,
    actor_model=actor,
    critic_model=critic,
)

# Train
history = trainer.fit(train_data)
```

### 3. Custom Reward Function

```python
def my_reward_fn(data):
    # Custom reward logic
    responses = data.non_tensor_batch["responses"]
    rewards = torch.tensor([len(r) for r in responses])
    return rewards

trainer = PPOTrainer(
    config=config,
    actor_model=actor,
    critic_model=critic,
    reward_fn=my_reward_fn,
)
```

## Project Structure

```
verl_reproduction/
├── README.md
├── requirements.txt
└── ARMOR/
    ├── __init__.py          # Package exports
    ├── protocol.py          # DataProto implementation
    ├── base_config.py       # Configuration base class
    ├── core_algos.py        # PPO algorithms (GAE, GRPO, etc.)
    ├── trainer.py           # PPO Trainer
    ├── worker.py            # Worker abstractions
    └── example.py           # Usage examples
```

## Key Concepts

### DataProto
A unified data structure combining:
- `batch`: Dict of torch.Tensors (same batch size)
- `non_tensor_batch`: Dict of numpy arrays (strings, objects)
- `meta_info`: Dict of metadata (no batch dimension)

### Advantage Estimators

| Estimator | Description |
|-----------|-------------|
| **GAE** | Generalized Advantage Estimation |
| **GRPO** | Group Relative Policy Optimization |
| **RLOO** | Reinforce Leave-One-Out |
| **REINFORCE++** | REINFORCE with baselines |

### PPO Training Loop

1. **Rollout**: Generate responses from actor
2. **Reward**: Compute rewards (model or function)
3. **Advantages**: Estimate advantages (GAE/GRPO)
4. **Update**: PPO clipped policy gradient

## Run Examples

```bash
cd verl_reproduction
python -m ARMOR.example
```

## Comparison with Original verl

| Aspect | verl | ARMOR |
|--------|------|-----------|
| Purpose | Production training | Educational |
| Distribution | Ray + FSDP/Megatron | Single process |
| Models | HuggingFace/Custom | Simple LSTM demo |
| Rollout | vLLM/SGLang | Basic generation |
| Scale | 100s of GPUs | Single GPU/CPU |

## References

- [verl GitHub](https://github.com/volcengine/verl)
- [HybridFlow Paper](https://arxiv.org/abs/2409.19256)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)

## License

Educational use. Original verl is Apache 2.0 licensed by ByteDance.
