#!/bin/bash
# -*- coding: utf-8 -*-
# Qwen2.5-7B GRPO Training with verl_mini
# Reference: Official verl examples/tuning/7b/qwen2-7b_grpo-lora_1_h100_fsdp_vllm.sh

# ============================================
# Environment Configuration
# ============================================
export CUDA_VISIBLE_DEVICES=0
NOW=$(date +%Y%m%d_%H%M%S)
export PROJECT_NAME="verl_mini_qwen7b_grpo"
export EXPERIMENT_NAME="${PROJECT_NAME}_${NOW}"

# Model path (modify to your local path)
MODEL_PATH="${VERL_MODEL_PATH:-/data/hgt/models/Qwen2.5-7B-Instruct}"

# Data paths (using preprocessed GSM8K)
TRAIN_DATA="data/gsm8k/train.parquet"
VAL_DATA="data/gsm8k/test.parquet"

# ============================================
# Training Parameters (Official verl aligned)
# ============================================
# Batch sizes
BATCH_SIZE=16
MINI_BATCH_SIZE=16
PPO_EPOCHS=4

# LoRA configuration
LORA_RANK=32
LORA_ALPHA=32

# Learning rate
LEARNING_RATE=3e-5

# Generation settings
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=1024
TEMPERATURE=0.7
TOP_P=0.9

# KL control (low_var_kl = k3, recommended)
KL_COEF=0.001
KL_TYPE="low_var_kl"

# GRPO specific
GRPO_N=5  # samples per prompt

# ============================================
# Run Training
# ============================================
set -x

cd /data/hgt/projects/verl_reproduction

# Activate environment
source /data/hgt/miniconda3/bin/activate minimind

python verl_mini/trainer/train_qwen7b_grpo.py \
    --model_path "$MODEL_PATH" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --batch_size $BATCH_SIZE \
    --mini_batch_size $MINI_BATCH_SIZE \
    --ppo_epochs $PPO_EPOCHS \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --learning_rate $LEARNING_RATE \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --max_response_length $MAX_RESPONSE_LENGTH \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --kl_coef $KL_COEF \
    --kl_type $KL_TYPE \
    --grpo_n $GRPO_N \
    --use_vllm \
    --gradient_checkpointing \
    --project_name "$PROJECT_NAME" \
    --experiment_name "$EXPERIMENT_NAME" \
    --save_dir "checkpoints/${EXPERIMENT_NAME}" \
    --total_epochs 1 \
    2>&1 | tee "logs/${EXPERIMENT_NAME}.log"
