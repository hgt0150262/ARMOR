# ARMOR: Adaptive and Robust Multi-GPU Optimization for RL Post-Training

ARMOR is an advanced, reliable, and numerically stable framework for reinforcement learning from human feedback (RLHF) and post-training alignment of Large Language Models. Inspired by the architectural elegance of ByteDance's [verl](https://github.com/volcengine/verl), ARMOR introduces key methodological and engineering optimizations designed to address the challenges of RL post-training on multi-GPU systems.

## Overview

ARMOR provides a robust environment for post-training alignment with state-of-the-art algorithms such as Group Relative Policy Optimization (GRPO) and Low-Rank Adaptation (LoRA). It stands out through several original technical contributions:

- **Cross-Domain Verifiable Reward Design**: An elegant design pattern supporting multiple domains (mathematics, safety, military knowledge) with verifiable, multi-dimensional rule-based reward functions, eliminating the need for expensive and vulnerable learned reward models.
- **Per-Process GPU Isolation**: Eliminates device contention during initialization and ensures strict per-worker deterministic execution.
- **StabilityGuard Protocol**: Mitigates numerical instability and policy collapse in GRPO fine-tuning through orthogonal protection layers.
- **DataProto**: A unified, high-performance data protocol that bridges tensor and non-tensor payloads in RL loops.

## Core Features

ARMOR integrates advanced RL post-training pipelines with the following feature set:

| Feature | Description |
|---------|-------------|
| **Distributed Training** | Multi-GPU FSDP & PyTorch DDP integration with Ray clusters |
| **Rollout Engines** | High-throughput generation powered by vLLM |
| **RL Algorithms** | Full GRPO, PPO, GAE, and REINFORCE++ support |
| **Model Support** | Out-of-the-box support for **Qwen** and **DeepSeek** models |

## Installation

```bash
git clone https://github.com/hgt0150262/ARMOR.git
cd ARMOR
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
ARMOR/
├── README.md
├── requirements.txt
├── scripts/             # Evaluation and execution scripts
├── tests/               # Python testing suite
└── ARMOR/               # Core framework code
    ├── __init__.py          # Package exports
    ├── protocol.py          # DataProto implementation
    ├── base_config.py       # Configuration base class
    ├── core_algos.py        # PPO algorithms (GAE, GRPO, etc.)
    ├── trainer/             # GRPO and PPO trainers
    ├── workers/             # Distributed rollout and reward workers
    └── utils/               # Model and logging utilities
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

### PPO/GRPO Training Loop

1. **Rollout**: Generate responses from actor (vLLM)
2. **Reward**: Compute rewards (multi-dimensional verifiable functions)
3. **Advantages**: Estimate relative advantages
4. **Update**: Clipped policy gradient updates

## Run Examples

```bash
python3 -m ARMOR.examples.example_basic
```

## Comparison with Original verl

| Aspect | verl | ARMOR |
|--------|------|-----------|
| **Primary Focus** | General high-throughput scale | Numerical stability, isolation & rule-based verifiable reward patterns |
| **Stability Protection**| Standard clipping | Active StabilityGuard & ProcessIsolator |
| **Supported Models** | Qwen, Llama, DeepSeek, Gemma | **Qwen**, **DeepSeek** |
| **Rollout Backend** | vLLM, SGLang, HF | vLLM Rollout |
| **Reward Paradigm** | Learned Reward Models / RLVR | Cross-Domain Verifiable Reward Design Pattern (Math, Safety, Military) |

## References

- [verl GitHub](https://github.com/volcengine/verl)
- [HybridFlow Paper](https://arxiv.org/abs/2409.19256)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)

## License

Educational and research use. Inspired by verl, which is Apache 2.0 licensed by ByteDance.

