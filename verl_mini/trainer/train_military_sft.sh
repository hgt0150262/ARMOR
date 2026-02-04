#!/bin/bash
# Military Domain LoRA SFT Training (Multi-GPU)
# Usage: bash verl_mini/trainer/train_military_sft.sh

set -e

# Configuration - Use GPUs 1,2,3 (GPU 0 is occupied)
export CUDA_VISIBLE_DEVICES=1,2,3
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
NUM_GPUS=3

NOW=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="military_sft_qwen7b_${NOW}"

# Paths
MODEL_PATH="/data/hgt/models/Qwen2.5-7B-Instruct"
OUTPUT_DIR="checkpoints/${EXPERIMENT_NAME}"

cd /data/hgt/projects/verl_reproduction
source /data/hgt/miniconda3/bin/activate minimind

# Create directories
mkdir -p logs checkpoints

echo "============================================"
echo "Military Domain LoRA SFT Training"
echo "============================================"
echo "Model: Qwen2.5-7B-Instruct"
echo "Dataset: US Army Field Manuals"
echo "GPUs: $NUM_GPUS x H100"
echo "Experiment: $EXPERIMENT_NAME"
echo "============================================"

# Launch training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    verl_mini/trainer/train_military_sft.py \
    --model_path "$MODEL_PATH" \
    --dataset_name "Heralax/us-army-fm-instruct" \
    --output_dir "$OUTPUT_DIR" \
    --max_length 2048 \
    --batch_size 2 \
    --gradient_accumulation 8 \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --lora_rank 64 \
    --lora_alpha 128 \
    2>&1 | tee "logs/${EXPERIMENT_NAME}.log"

echo "Training complete! Log: logs/${EXPERIMENT_NAME}.log"
