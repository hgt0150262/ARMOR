# ARMOR: Adaptive and Robust Multi-GPU Optimization for RL Post-Training

ARMOR is a research prototype and simplified reproduction inspired by ByteDance's [verl](https://github.com/volcengine/verl). It collects educational RLHF utilities, a LoRA-based multi-GPU post-training script, and deterministic task rewards.

Two post-training paths are provided. The legacy default (`--adv_mode batch`) samples one completion per prompt and standardizes rewards within each local mini-batch; `--adv_mode group` expands each prompt into `grpo_n` completions and normalizes advantages within each group, which is standard multi-completion Group Relative Policy Optimization (GRPO). Both arms are reported in the manuscript. ARMOR is not presented as a production or performance competitor to verl, OpenRLHF, or TRL.

> **Naming note:** An unrelated paper titled ["ARMOR: Stabilizing On-Policy LLM RL with Off-Policy Anchor Samples"](https://arxiv.org/abs/2607.10481) appeared in July 2026. That work introduces Anchor Rollout and Mixed Optimization and is algorithmically unrelated to this repository.

## Study Scope

The repository supports a bounded reliability case study with four evidence-linked components:

- **Reward-design ladder:** exact checks for canonical answers, shaped deterministic rewards for sparse mathematical feedback, and lexical or structural proxy scores for open-ended tasks.
- **Execution controls:** pre-import process-to-GPU visibility, DDP static-graph configuration, and explicit evaluation/training mode transitions.
- **Numerical safeguards:** bounded sequence log ratios, finite-loss rejection, and gradient clipping in the reported training path.
- **Artifact traceability:** the result files behind every reported number are shipped as supplementary material with the manuscript; this repository carries code only.

These mechanisms are established techniques assembled and documented for this repository. They are not individually claimed as new algorithms or runtime primitives. Open-ended scores are deterministic proxies and do not verify semantic correctness, truthfulness, safety, or doctrinal quality.

## Core Features

The codebase includes the following experimental and educational paths:

| Feature | Description |
|---------|-------------|
| **Distributed training** | PyTorch DDP/FSDP and Ray utility paths; retained multi-node evidence is LoRA SFT, not RL |
| **Generation** | Transformers generation in the reported trainer and an experimental vLLM rollout module |
| **RL utilities** | PPO-style losses and GAE, RLOO, ReMax, and GRPO advantage/loss helpers |
| **Reported model path** | Qwen2.5-7B-Instruct with LoRA adapters |

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

### 2. Inspect the Reported Post-Training Entry Point

```bash
python ARMOR/trainer/train_qwen7b_grpo_multigpu.py --help
```

The task reward is selected with `--reward_fn gsm8k`, `--reward_fn truthfulqa`, or `--reward_fn military`. Launch examples are under `ARMOR/trainer/`, and the dataset preparation utilities are under `ARMOR/data_preprocess/`.

The default `--adv_mode batch` path is the legacy reference-normalized loop. `--adv_mode group` expands each prompt into `grpo_n` completions and normalizes advantages within each group (standard multi-completion GRPO). `--ppo_epochs` only scales the number of optimizer steps; it does not repeat inner optimization.

## Project Structure

```
ARMOR/
├── README.md
├── requirements.txt
└── ARMOR/               # Core framework code
    ├── __init__.py          # Package exports
    ├── protocol.py          # DataProto implementation
    ├── base_config.py       # Configuration base class
    ├── data_preprocess/     # GSM8K, TruthfulQA, military, HH-RLHF loaders
    ├── trainer/ppo/         # PPO-style losses and advantage utilities
    ├── trainer/             # GRPO and PPO trainers + launch examples
    ├── workers/             # Distributed rollout and reward workers
    └── utils/               # Model, logging, reproducibility utilities
```

## Key Concepts

### DataProto
A unified data structure combining:
- `batch`: Dict of torch.Tensors (same batch size)
- `non_tensor_batch`: Dict of numpy arrays (strings, objects)
- `meta_info`: Dict of metadata (no batch dimension)

### Advantage and Loss Utilities

| Estimator | Description |
|-----------|-------------|
| **GAE** | Generalized Advantage Estimation |
| **GRPO** | Group Relative Policy Optimization |
| **RLOO** | Reinforce Leave-One-Out |
| **REINFORCE++** | REINFORCE with baselines |

### Reported GRPO-Style Training Loop

1. **Rollout:** Generate one sampled response per local prompt with Transformers.
2. **Reward:** Compute an exact, shaped, or deterministic proxy score.
3. **Normalization:** Standardize rewards within the local mini-batch.
4. **Update:** Apply a clipped sequence-level ratio against a frozen reference model.

This path is documented code-faithfully in the accompanying manuscript; it should not be used as evidence about standard GRPO without replacing it with a multi-completion implementation.

## Run Examples

```bash
python3 -m ARMOR.examples.example_basic
```

## Scope Relative to verl

verl is a mature, high-throughput RLHF framework with broader algorithm, rollout, placement, and scaling support. ARMOR borrows simplified architectural ideas for education and for reconstructing one repository-specific reliability trace. No throughput, scalability, feature-superiority, or production-readiness claim is made against verl.

The manuscript is not included in this repository. Its empirical claims are restricted to the retained artifacts shipped with the submission.

## References

- [verl GitHub](https://github.com/volcengine/verl)
- [HybridFlow Paper](https://arxiv.org/abs/2409.19256)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)

## License

This repository does not currently include a standalone license file. Add an explicit license before inviting redistribution or reuse. The upstream verl project is Apache-2.0 licensed; that upstream license does not automatically license this repository's original files.
