"""
TruthfulQA数据集预处理脚本
将TruthfulQA数据转换为ARMOR GRPO训练格式，用于安全对齐RL后训练

输出格式 (Parquet):
{
    "data_source": "truthfulqa",
    "prompt": [{"role": "user", "content": "问题"}],
    "ability": "safety",
    "reward_model": {
        "style": "rule",
        "ground_truth": "最佳答案",
        "correct_answers": ["正确答案1", ...],
        "incorrect_answers": ["错误答案1", ...]
    },
    "extra_info": {"category": "...", "type": "...", "index": 0}
}
"""

import argparse
import json
import os
from typing import Optional

import pandas as pd


def preprocess_truthfulqa(
    input_path: str,
    output_dir: str = "data/truthfulqa",
    train_ratio: float = 0.85,
    instruction: str = "Please answer the following question truthfully and concisely. If the question contains a false premise, point it out.",
) -> dict:
    """
    预处理TruthfulQA数据集

    Args:
        input_path: raw.parquet 路径
        output_dir: 输出目录
        train_ratio: 训练集比例 (817条: ~694 train, ~123 test)
        instruction: 附加指令

    Returns:
        dict: 输出路径和数据统计
    """
    df = pd.read_parquet(input_path)
    print(f"加载TruthfulQA数据: {len(df)} 条, 列: {list(df.columns)}")
    print(f"类别分布: {df['category'].value_counts().head(10).to_dict()}")

    processed_data = []
    for idx, row in df.iterrows():
        question = row["question"]
        best_answer = row["best_answer"]
        correct_answers = row.get("correct_answers", [])
        incorrect_answers = row.get("incorrect_answers", [])

        # Ensure list types
        if isinstance(correct_answers, str):
            correct_answers = [correct_answers]
        if isinstance(incorrect_answers, str):
            incorrect_answers = [incorrect_answers]

        # Build prompt with safety instruction
        full_question = f"{question}\n\n{instruction}"

        processed_item = {
            "data_source": "truthfulqa",
            "prompt": [{"role": "user", "content": full_question}],
            "ability": "safety",
            "reward_model": {
                "style": "rule",
                "ground_truth": best_answer,
                "correct_answers": list(correct_answers) if hasattr(correct_answers, '__iter__') else [str(correct_answers)],
                "incorrect_answers": list(incorrect_answers) if hasattr(incorrect_answers, '__iter__') else [str(incorrect_answers)],
            },
            "extra_info": {
                "index": int(idx),
                "category": row.get("category", ""),
                "type": row.get("type", ""),
                "source": row.get("source", ""),
            },
        }
        processed_data.append(processed_item)

    # Shuffle deterministically then split
    import random
    random.seed(42)
    random.shuffle(processed_data)

    split_idx = int(len(processed_data) * train_ratio)
    train_data = processed_data[:split_idx]
    test_data = processed_data[split_idx:]

    # Save
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    print(f"\nTruthfulQA数据预处理完成:")
    print(f"  训练集: {len(train_data)} 条 -> {train_path}")
    print(f"  测试集: {len(test_data)} 条 -> {test_path}")

    return {
        "train_path": train_path,
        "test_path": test_path,
        "train_size": len(train_data),
        "test_size": len(test_data),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TruthfulQA数据预处理")
    parser.add_argument(
        "--input", default="data/truthfulqa/raw.parquet", help="raw.parquet路径"
    )
    parser.add_argument("--output_dir", default="data/truthfulqa", help="输出目录")
    parser.add_argument("--train_ratio", type=float, default=0.85, help="训练集比例")

    args = parser.parse_args()
    preprocess_truthfulqa(
        input_path=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
    )
