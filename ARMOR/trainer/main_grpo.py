"""Main GRPO training entry point for ARMOR.

This script provides a clean entry point for GRPO training,
following the official verl trainer structure.

Usage:
    python -m ARMOR.trainer.main_grpo --config config.yaml
    
Or with torchrun for multi-GPU:
    torchrun --nproc_per_node=4 -m ARMOR.trainer.main_grpo --config config.yaml
"""
import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ARMOR.examples.train_qwen7b_grpo_multigpu import main as train_main


def parse_args():
    parser = argparse.ArgumentParser(description="ARMOR GRPO Training")
    parser.add_argument("--config", type=str, help="Path to config file (optional)")
    return parser.parse_known_args()


def main():
    args, remaining = parse_args()
    
    # If config provided, load it
    if args.config:
        print(f"Loading config from {args.config}")
        # TODO: Implement config loading
    
    # Call the actual training function
    train_main()


if __name__ == "__main__":
    main()
