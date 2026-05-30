# ARMOR Training Guide

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
python ARMOR/data_preprocess/gsm8k.py
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

## Safety Alignment Training (TruthfulQA)

ARMOR supports cross-domain RL post-training. Beyond GSM8K (math), you can train on TruthfulQA for safety alignment.

### Data Preparation
```bash
# Download TruthfulQA on a machine with internet access
python ARMOR/data_preprocess/truthfulqa.py --input data/truthfulqa/raw.parquet --output_dir data/truthfulqa
```

### Training
```bash
bash ARMOR/trainer/train_qwen7b_grpo_safety.sh
```

### Key Differences from GSM8K
| Aspect | GSM8K | TruthfulQA |
|--------|-------|------------|
| Domain | Math reasoning | Safety alignment |
| Reward | Binary (correct/incorrect) | Multi-dimensional (truthfulness + misinfo rejection + format) |
| `--reward_fn` | `gsm8k` | `truthfulqa` |
| Response length | 512 | 256 |

### Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_safety.py \
    --base_model /data/hgt/models/Qwen2.5-7B-Instruct \
    --lora_path checkpoints/<experiment>/final \
    --test_data data/truthfulqa/test.parquet \
    --output results/safety_eval_results.json
```

## Offline Server Setup (GPU Server without Internet)

The gpu-server has no internet access. Use SSH reverse tunnels to proxy package registries.

### PyPI Packages
```bash
# On local machine (has internet):
python docs/pypi-registry-proxy.py --listen 127.0.0.1 --port 7891 --registry https://pypi.tuna.tsinghua.edu.cn

# In another terminal, create reverse tunnel:
ssh -N -R 127.0.0.1:7891:127.0.0.1:7891 gpu-server

# On gpu-server:
pip install --index-url http://127.0.0.1:7891/simple/ --trusted-host 127.0.0.1 <package>
```

### HuggingFace Datasets
Download datasets on the local machine and `scp` to gpu-server:
```bash
# Local: download and save as parquet
python -c "from datasets import load_dataset; ds = load_dataset('truthfulqa/truthful_qa', 'generation', split='validation'); ds.to_parquet('/tmp/raw.parquet')"

# Transfer to gpu-server
scp /tmp/raw.parquet gpu-server:/data/hgt/projects/verl_reproduction/data/truthfulqa/raw.parquet
```

## Troubleshooting

### NVLink Errors
If you see `CUDA error: NVLink peer GPU memory` errors, ensure NCCL P2P is disabled.

### Garbled Output
If model outputs are garbled during training, ensure `model.eval()` is called before `generate()` when using gradient checkpointing.

## Output Structure
```
checkpoints/
└── ARMOR_qwen7b_grpo_4gpu_YYYYMMDD_HHMMSS/
    ├── step_100/
    └── final/

logs/
└── ARMOR_qwen7b_grpo_4gpu_YYYYMMDD_HHMMSS.log

swanlog/
└── run-YYYYMMDD_HHMMSS-xxxxx/
```
