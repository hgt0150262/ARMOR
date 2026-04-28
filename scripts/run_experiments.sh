#!/bin/bash
# VeRL-Mini Extended Experiments for Paper
# Runs multiple experiment configurations on 4x H100

set -e

cd /data/hgt/projects/verl_reproduction
source /data/hgt/miniconda3/bin/activate minimind

# Common env vars for GPU isolation
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NVLS_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

MODEL_PATH="/data/hgt/models/Qwen2.5-7B-Instruct"
TRAIN_DATA="data/gsm8k/train.parquet"
VAL_DATA="data/gsm8k/test.parquet"
NOW=$(date +%Y%m%d_%H%M%S)

mkdir -p logs checkpoints

run_experiment() {
    local EXP_NAME=$1
    local NUM_GPUS=$2
    local LORA_RANK=$3
    local LORA_ALPHA=$4
    local LR=$5
    local BATCH_SIZE=$6
    local GRAD_ACCUM=$7
    local EPOCHS=$8
    local EXTRA_ARGS=$9

    echo "============================================"
    echo "Experiment: $EXP_NAME"
    echo "GPUs: $NUM_GPUS, LoRA r=$LORA_RANK, LR=$LR, BS=$BATCH_SIZE"
    echo "============================================"

    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        verl_mini/trainer/train_qwen7b_grpo_multigpu.py \
        --model_path "$MODEL_PATH" \
        --train_data "$TRAIN_DATA" \
        --val_data "$VAL_DATA" \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --mini_batch_size $BATCH_SIZE \
        --ppo_epochs 2 \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --learning_rate $LR \
        --max_prompt_length 512 \
        --max_response_length 512 \
        --temperature 0.7 \
        --top_p 0.9 \
        --repetition_penalty 1.1 \
        --kl_coef 0.05 \
        --gradient_checkpointing \
        --project_name "verl_mini_experiments" \
        --experiment_name "$EXP_NAME" \
        --save_dir "checkpoints/${EXP_NAME}" \
        --total_epochs $EPOCHS \
        $EXTRA_ARGS \
        2>&1 | tee "logs/${EXP_NAME}.log"

    echo "Experiment $EXP_NAME completed!"
    echo ""
}

# =====================================================
# Experiment 1: 4-GPU baseline (re-test GPU 3)
# =====================================================
echo ">>> Exp 1: 4-GPU Baseline with GPU 3 re-test"
run_experiment "exp1_4gpu_r32_${NOW}" 4 32 32 1e-5 4 4 1

# =====================================================
# Experiment 2: LoRA rank scaling (r=8)
# =====================================================
echo ">>> Exp 2: LoRA rank=8"
run_experiment "exp2_r8_${NOW}" 3 8 8 1e-5 4 4 1

# =====================================================
# Experiment 3: LoRA rank scaling (r=64)
# =====================================================
echo ">>> Exp 3: LoRA rank=64"
run_experiment "exp3_r64_${NOW}" 3 64 64 1e-5 4 4 1

# =====================================================
# Experiment 4: Higher learning rate (5e-5)
# =====================================================
echo ">>> Exp 4: Higher LR=5e-5"
run_experiment "exp4_lr5e5_${NOW}" 3 32 32 5e-5 4 4 1

# =====================================================
# Experiment 5: Lower KL coefficient (0.001)
# =====================================================
echo ">>> Exp 5: KL coef=0.001"
run_experiment "exp5_lowkl_${NOW}" 3 32 32 1e-5 4 4 1 "--kl_coef 0.001"

echo "============================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "============================================"
