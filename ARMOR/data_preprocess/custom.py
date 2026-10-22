"""
自定义数据集预处理工具
支持将任意JSON/JSONL/CSV格式转换为ARMOR RLHF训练格式

示例输入格式:
- JSON/JSONL: {"question": "...", "answer": "..."}
- CSV: question,answer 列

输出格式 (Parquet):
{
    "data_source": "custom",
    "prompt": [{"role": "user", "content": "..."}],
    "ability": "general",
    "reward_model": {"style": "rule", "ground_truth": "..."},
    "extra_info": {"index": 0}
}
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Union

import pandas as pd


def load_custom_data(
    input_path: str,
    question_key: str = "question",
    answer_key: str = "answer",
) -> List[Dict]:
    """
    加载自定义数据文件
    
    支持格式:
    - .json: JSON数组
    - .jsonl: JSON Lines
    - .csv: CSV文件
    """
    ext = os.path.splitext(input_path)[1].lower()
    
    if ext == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
    elif ext == ".jsonl":
        data = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    elif ext == ".csv":
        df = pd.read_csv(input_path)
        data = df.to_dict(orient="records")
    else:
        raise ValueError(f"不支持的文件格式: {ext}. 支持: .json, .jsonl, .csv")
    
    # 验证字段
    if data and question_key not in data[0]:
        raise KeyError(f"找不到问题字段 '{question_key}'. 可用字段: {list(data[0].keys())}")
    
    return data


def preprocess_custom(
    input_path: str,
    output_dir: str = "~/data/custom",
    question_key: str = "question",
    answer_key: str = "answer",
    data_source: str = "custom",
    ability: str = "general",
    reward_style: str = "rule",
    instruction: Optional[str] = None,
    train_ratio: float = 0.9,
) -> Dict:
    """
    预处理自定义数据集
    
    Args:
        input_path: 输入文件路径
        output_dir: 输出目录
        question_key: 问题字段名
        answer_key: 答案字段名
        data_source: 数据来源标识
        ability: 能力类型
        reward_style: 奖励类型 ("rule" 或 "model")
        instruction: 附加指令 (可选)
        train_ratio: 训练集比例
    
    Returns:
        dict: 输出路径和数据统计
    """
    # 加载数据
    raw_data = load_custom_data(input_path, question_key, answer_key)
    print(f"加载数据: {len(raw_data)} 条")
    
    # 转换格式
    processed_data = []
    for idx, item in enumerate(raw_data):
        question = item.get(question_key, "")
        answer = item.get(answer_key, "")
        
        if instruction:
            question = f"{question}\n\n{instruction}"
        
        processed_item = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": ability,
            "reward_model": {
                "style": reward_style,
                "ground_truth": str(answer) if reward_style == "rule" else ""
            },
            "extra_info": {
                "index": idx,
                **{k: v for k, v in item.items() if k not in [question_key, answer_key]}
            }
        }
        processed_data.append(processed_item)
    
    # 划分训练/测试集
    split_idx = int(len(processed_data) * train_ratio)
    train_data = processed_data[:split_idx]
    test_data = processed_data[split_idx:]
    
    # 保存
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    train_df.to_parquet(train_path)
    if len(test_data) > 0:
        test_df.to_parquet(test_path)
    
    print(f"数据预处理完成:")
    print(f"  训练集: {len(train_data)} 条 -> {train_path}")
    print(f"  测试集: {len(test_data)} 条 -> {test_path}")
    
    return {
        "train_path": train_path,
        "test_path": test_path if len(test_data) > 0 else None,
        "train_size": len(train_data),
        "test_size": len(test_data),
    }


def create_sample_data(output_path: str = "sample_data.jsonl"):
    """创建示例数据文件"""
    samples = [
        {"question": "什么是人工智能?", "answer": "人工智能是计算机科学的一个分支,旨在创建能够模拟人类智能的系统。"},
        {"question": "1+1等于多少?", "answer": "2"},
        {"question": "请解释机器学习和深度学习的区别。", "answer": "机器学习是人工智能的子集,深度学习是机器学习的子集,使用多层神经网络。"},
    ]
    
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"示例数据已创建: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自定义数据预处理")
    parser.add_argument("--input", required=True, help="输入文件路径 (.json/.jsonl/.csv)")
    parser.add_argument("--output_dir", default="~/data/custom", help="输出目录")
    parser.add_argument("--question_key", default="question", help="问题字段名")
    parser.add_argument("--answer_key", default="answer", help="答案字段名")
    parser.add_argument("--data_source", default="custom", help="数据来源标识")
    parser.add_argument("--ability", default="general", help="能力类型")
    parser.add_argument("--reward_style", default="rule", choices=["rule", "model"])
    parser.add_argument("--instruction", default=None, help="附加指令")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="训练集比例")
    parser.add_argument("--create_sample", action="store_true", help="创建示例数据")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data()
    else:
        preprocess_custom(
            input_path=args.input,
            output_dir=args.output_dir,
            question_key=args.question_key,
            answer_key=args.answer_key,
            data_source=args.data_source,
            ability=args.ability,
            reward_style=args.reward_style,
            instruction=args.instruction,
            train_ratio=args.train_ratio,
        )
