#!/bin/bash
# Military Domain Ray Distributed SFT Training
# Requires: Ray cluster running on gpu-server (head) + gpu-server1 (worker)

set -e

# Configuration
MODEL_PATH="/data/hgt/models/Qwen2.5-7B-Instruct"
DATA_PATH="/data/hgt/datasets/us-army-fm-instruct"
OUTPUT_DIR="/data/hgt/projects/verl_reproduction/checkpoints/military_ray_sft_$(date +%Y%m%d_%H%M%S)"
NUM_WORKERS=8  # 4 from gpu-server + 4 from gpu-server1

# Activate environment
cd /data/hgt/projects/verl_reproduction
source /data/hgt/miniconda3/bin/activate minimind

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "Military Domain Ray Distributed SFT Training"
echo "============================================"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Workers: $NUM_WORKERS GPUs"
echo "============================================"

# Check Ray cluster
echo "Checking Ray cluster..."
ray status || {
    echo "ERROR: Ray cluster not running!"
    echo "Start with: bash scripts/start_ray_cluster.sh head"
    exit 1
}

# Run training
python verl_mini/trainer/train_military_ray_sft.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_length 2048 \
    --batch_size 2 \
    --gradient_accumulation 4 \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --num_workers $NUM_WORKERS \
    --ray_address auto \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "============================================"
echo "Training complete!"
echo "Checkpoint: $OUTPUT_DIR/final"
echo "Log: $OUTPUT_DIR/training.log"
echo "============================================"
