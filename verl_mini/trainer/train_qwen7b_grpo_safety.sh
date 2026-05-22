#!/bin/bash
# Qwen2.5-7B GRPO Safety Alignment Training (TruthfulQA)
# Uses ARMOR framework with multi-dimensional safety reward function
# Usage: bash verl_mini/trainer/train_qwen7b_grpo_safety.sh

set -e

# Configuration - GPU isolation handled per-process in Python
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NVLS_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

NUM_GPUS=3
NOW=$(date +%Y%m%d_%H%M%S)
PROJECT_NAME="armor_safety_grpo"
EXPERIMENT_NAME="${PROJECT_NAME}_${NUM_GPUS}gpu_${NOW}"

# Paths
MODEL_PATH="/data/hgt/models/Qwen2.5-7B-Instruct"
TRAIN_DATA="data/truthfulqa/train.parquet"
VAL_DATA="data/truthfulqa/test.parquet"

# Training params
BATCH_SIZE_PER_GPU=2       # Smaller batch for longer safety responses
GRADIENT_ACCUMULATION=8    # Effective batch = 3*2*8 = 48
LORA_RANK=32
LEARNING_RATE=1e-5
KL_COEF=0.05
EPOCHS=3

cd /data/hgt/projects/verl_reproduction
source /data/hgt/miniconda3/bin/activate minimind

# Run GPU health check first
echo "============================================"
echo "ARMOR GPU Health Pre-Check"
echo "============================================"
python -c "
from verl_mini.utils.gpu_health import run_health_check
results = run_health_check()
for r in results:
    status = 'HEALTHY' if r.is_healthy else 'DEGRADED'
    print(f'  GPU {r.gpu_id} ({r.name}): {status} (score={r.health_score:.2f}, temp={r.temperature}C)')
" 2>/dev/null || echo "  GPU health check skipped (module not available in this env)"

# Create directories
mkdir -p logs checkpoints

echo "============================================"
echo "ARMOR Safety Alignment GRPO Training"
echo "============================================"
echo "Dataset: TruthfulQA (694 train / 123 test)"
echo "Reward: Multi-dimensional safety (truthfulness + misinformation rejection + format)"
echo "GPUs: $NUM_GPUS x H100"
echo "LoRA rank: $LORA_RANK"
echo "Effective batch size: $((NUM_GPUS * BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION))"
echo "Experiment: $EXPERIMENT_NAME"
echo "============================================"

# Launch distributed training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    verl_mini/trainer/train_qwen7b_grpo_multigpu.py \
    --model_path "$MODEL_PATH" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --mini_batch_size $BATCH_SIZE_PER_GPU \
    --ppo_epochs 2 \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_RANK \
    --learning_rate $LEARNING_RATE \
    --max_prompt_length 256 \
    --max_response_length 256 \
    --temperature 0.7 \
    --top_p 0.9 \
    --repetition_penalty 1.1 \
    --kl_coef $KL_COEF \
    --gradient_checkpointing \
    --reward_fn truthfulqa \
    --project_name "$PROJECT_NAME" \
    --experiment_name "$EXPERIMENT_NAME" \
    --save_dir "checkpoints/${EXPERIMENT_NAME}" \
    --save_steps 50 \
    --log_interval 5 \
    --total_epochs $EPOCHS \
    2>&1 | tee "logs/${EXPERIMENT_NAME}.log"

echo ""
echo "============================================"
echo "Training complete!"
echo "Log: logs/${EXPERIMENT_NAME}.log"
echo "Checkpoints: checkpoints/${EXPERIMENT_NAME}/"
echo "============================================"
