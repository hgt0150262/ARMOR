#!/usr/bin/env python3
"""
Military Domain Data Preprocessor for ARMOR GRPO Training.

Converts US Army Field Manual JSONL instruction data into ARMOR's
standardized RLHF training format (parquet with prompt/reward_model fields).

Usage:
    python ARMOR/data_preprocess/military.py \
        --input /data/hgt/datasets/us-army-fm-instruct \
        --output_dir data/military
"""
import argparse
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd


def preprocess_military(
    input_dir: str,
    output_dir: str = "data/military",
    instruction: str = "Answer the following military question accurately, "
                       "referencing specific doctrine (FM/AR numbers) where applicable. "
                       "Provide a structured, detailed response.",
    test_ratio: float = 0.15,
    max_samples: Optional[int] = None,
) -> dict:
    """
    Preprocess US Army Field Manual JSONL data into ARMOR training format.

    Args:
        input_dir: Directory containing JSONL files with 'conversations' field
        output_dir: Output directory for train/test parquet files
        instruction: System instruction appended to questions
        test_ratio: Fraction of data for test split
        max_samples: Optional cap on total samples

    Returns:
        dict with paths and counts
    """
    data_source = "military_fm"
    input_path = Path(input_dir)

    # Load all JSONL files
    raw_data = []
    for jsonl_file in sorted(input_path.glob("*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if "conversations" in item:
                        raw_data.append(item)
                except json.JSONDecodeError:
                    continue

    print(f"Loaded {len(raw_data)} conversations from {input_path}")

    if max_samples and len(raw_data) > max_samples:
        raw_data = raw_data[:max_samples]

    # Convert to ARMOR format
    processed = []
    for idx, item in enumerate(raw_data):
        convs = item["conversations"]
        if len(convs) < 2:
            continue

        # Extract first QA pair
        question = None
        answer = None
        for conv in convs:
            if conv["from"] == "human" and question is None:
                question = conv["value"]
            elif conv["from"] == "gpt" and answer is None:
                answer = conv["value"]
            if question and answer:
                break

        if not question or not answer:
            continue

        # Add instruction
        full_question = f"{question}\n\n{instruction}"

        processed.append({
            "data_source": data_source,
            "prompt": [{"role": "user", "content": full_question}],
            "ability": "military_domain",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer,
            },
            "extra_info": {
                "split": "",  # filled below
                "index": idx,
                "question": question,
                "answer_length": len(answer.split()),
                "num_turns": len(convs),
            },
        })

    print(f"Processed {len(processed)} QA pairs")

    # Train/test split
    import random
    random.seed(42)
    random.shuffle(processed)

    n_test = max(1, int(len(processed) * test_ratio))
    test_data = processed[:n_test]
    train_data = processed[n_test:]

    for item in train_data:
        item["extra_info"]["split"] = "train"
    for item in test_data:
        item["extra_info"]["split"] = "test"

    # Save
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    result = {
        "train_path": train_path,
        "test_path": test_path,
        "train_count": len(train_data),
        "test_count": len(test_data),
    }

    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train_data)} samples)")
    print(f"  Test:  {test_path} ({len(test_data)} samples)")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess military JSONL data for ARMOR")
    parser.add_argument("--input", type=str, required=True,
                        help="Directory containing JSONL files")
    parser.add_argument("--output_dir", type=str, default="data/military",
                        help="Output directory for parquet files")
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    preprocess_military(
        input_dir=args.input,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        max_samples=args.max_samples,
    )
