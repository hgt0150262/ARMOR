#!/bin/bash
# Qwen2.5-7B GRPO Multi-GPU Training (4x H100)
# Usage: bash verl_mini/examples/train_qwen7b_grpo_multigpu.sh

set -e

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NVLS_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
NUM_GPUS=4
NOW=$(date +%Y%m%d_%H%M%S)
PROJECT_NAME="verl_mini_qwen7b_grpo"
EXPERIMENT_NAME="${PROJECT_NAME}_${NUM_GPUS}gpu_${NOW}"

# Paths
MODEL_PATH="/data/hgt/models/Qwen2.5-7B-Instruct"
TRAIN_DATA="data/gsm8k/train.parquet"
VAL_DATA="data/gsm8k/test.parquet"

# Training params (per-GPU batch size)
BATCH_SIZE_PER_GPU=4
GRADIENT_ACCUMULATION=4
# Effective batch size = 4 GPUs * 4 batch * 4 accum = 64

cd /data/hgt/projects/verl_reproduction
source /data/hgt/miniconda3/bin/activate minimind

# Create directories
mkdir -p logs checkpoints

echo "============================================"
echo "Qwen2.5-7B GRPO Multi-GPU Training"
echo "============================================"
echo "GPUs: $NUM_GPUS x H100"
echo "Effective batch size: $((NUM_GPUS * BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION))"
echo "Experiment: $EXPERIMENT_NAME"
echo "============================================"

# Launch distributed training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    verl_mini/trainer/train_qwen7b_grpo_multigpu.py \
    --model_path "$MODEL_PATH" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --mini_batch_size $BATCH_SIZE_PER_GPU \
    --ppo_epochs 2 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --learning_rate 1e-5 \
    --max_prompt_length 512 \
    --max_response_length 512 \
    --temperature 0.7 \
    --top_p 0.9 \
    --repetition_penalty 1.1 \
    --kl_coef 0.05 \
    --gradient_checkpointing \
    --project_name "$PROJECT_NAME" \
    --experiment_name "$EXPERIMENT_NAME" \
    --save_dir "checkpoints/${EXPERIMENT_NAME}" \
    --total_epochs 3 \
    2>&1 | tee "logs/${EXPERIMENT_NAME}.log"

echo "Training complete! Log saved to logs/${EXPERIMENT_NAME}.log"
