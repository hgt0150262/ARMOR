#!/bin/bash
# Qwen2.5-7B GRPO Military Domain Training
# Uses ARMOR framework with multi-dimensional military reward function
# Usage: bash ARMOR/trainer/train_qwen7b_grpo_military.sh

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
PROJECT_NAME="armor_military_grpo"
EXPERIMENT_NAME="${PROJECT_NAME}_${NUM_GPUS}gpu_${NOW}"

# Paths
MODEL_PATH="/data/hgt/models/Qwen2.5-7B-Instruct"
TRAIN_DATA="data/military/train.parquet"
VAL_DATA="data/military/test.parquet"

# Training params — optimized based on GSM8K rank scaling findings
BATCH_SIZE_PER_GPU=2       # Small batch for longer military responses
GRADIENT_ACCUMULATION=8    # Effective batch = 3*2*8 = 48
LORA_RANK=64               # High rank for faster convergence
LEARNING_RATE=1e-5
KL_COEF=0.01               # Low KL + LoRA implicit regularization
EPOCHS=3

cd /data/hgt/projects/verl_reproduction
source /data/hgt/miniconda3/bin/activate minimind

# Create directories
mkdir -p logs checkpoints

echo "============================================"
echo "ARMOR Military Domain GRPO Training"
echo "============================================"
echo "Dataset: US Army Field Manuals"
echo "Reward: Multi-dimensional (terminology + factual + structure)"
echo "GPUs: $NUM_GPUS x H100"
echo "LoRA rank: $LORA_RANK"
echo "KL coef: $KL_COEF"
echo "Effective batch size: $((NUM_GPUS * BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION))"
echo "Experiment: $EXPERIMENT_NAME"
echo "============================================"

# Launch distributed training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29502 \
    ARMOR/trainer/train_qwen7b_grpo_multigpu.py \
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
    --max_prompt_length 512 \
    --max_response_length 512 \
    --temperature 0.7 \
    --top_p 0.9 \
    --repetition_penalty 1.1 \
    --kl_coef $KL_COEF \
    --gradient_checkpointing \
    --reward_fn military \
    --project_name "$PROJECT_NAME" \
    --experiment_name "$EXPERIMENT_NAME" \
    --save_dir "checkpoints/${EXPERIMENT_NAME}" \
    --save_steps 100 \
    --log_interval 10 \
    --total_epochs $EPOCHS \
    2>&1 | tee "logs/${EXPERIMENT_NAME}.log"

echo ""
echo "============================================"
echo "Training complete!"
echo "Log: logs/${EXPERIMENT_NAME}.log"
echo "Checkpoints: checkpoints/${EXPERIMENT_NAME}/"
echo "============================================"
