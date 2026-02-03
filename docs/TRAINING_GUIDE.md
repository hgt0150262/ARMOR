# verl_mini Training Guide

## Quick Start

### Prerequisites
- CUDA-enabled GPU (tested on H100)
- Python 3.10+
- PyTorch 2.0+

### Environment Setup
```bash
conda activate minimind
pip install -r requirements.txt
```

### Data Preparation
```bash
python verl_mini/data_preprocess/gsm8k.py
```

### Training

#### Single GPU
```bash
bash scripts/run_training.sh single
```

#### Multi-GPU (4x GPU)
```bash
bash scripts/run_training.sh multi
```

## Configuration

### Key Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | `/data/hgt/models/Qwen2.5-7B-Instruct` | Base model path |
| `--batch_size` | 4 | Per-GPU batch size |
| `--lora_rank` | 32 | LoRA rank |
| `--temperature` | 0.8 | Generation temperature |
| `--repetition_penalty` | 1.1 | Repetition penalty |

### NCCL Settings (Multi-GPU)
Required environment variables for multi-GPU training:
```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
```

## Troubleshooting

### NVLink Errors
If you see `CUDA error: NVLink peer GPU memory` errors, ensure NCCL P2P is disabled.

### Garbled Output
If model outputs are garbled during training, ensure `model.eval()` is called before `generate()` when using gradient checkpointing.

## Output Structure
```
checkpoints/
└── verl_mini_qwen7b_grpo_4gpu_YYYYMMDD_HHMMSS/
    ├── step_100/
    └── final/

logs/
└── verl_mini_qwen7b_grpo_4gpu_YYYYMMDD_HHMMSS.log

swanlog/
└── run-YYYYMMDD_HHMMSS-xxxxx/
```
