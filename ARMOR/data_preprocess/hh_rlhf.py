"""
HH-RLHF数据集预处理脚本
将Anthropic HH-RLHF数据转换为ARMOR训练格式

支持三种模式:
- sft: 监督微调数据 (prompt + response)
- rm: 奖励模型数据 (prompt + chosen + rejected)
- rl: 强化学习数据 (prompt only)
"""

import argparse
import os
from typing import Optional

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm


def preprocess_hh_rlhf_sft(
    output_dir: str = "~/data/full_hh_rlhf/sft",
    local_dataset_path: Optional[str] = None,
) -> dict:
    """
    生成SFT训练数据
    同时使用chosen和rejected作为训练样本
    """
    if local_dataset_path is not None:
        dataset = load_dataset(local_dataset_path)
    else:
        dataset = load_dataset("Dahoas/full-hh-rlhf")
    
    output = {"prompt": [], "response": []}
    for data in tqdm(dataset["train"], desc="Processing SFT data"):
        # 添加chosen
        output["prompt"].append(data["prompt"])
        output["response"].append(data["chosen"])
        # 添加rejected
        output["prompt"].append(data["prompt"])
        output["response"].append(data["rejected"])
    
    df = pd.DataFrame(output)
    
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "train.parquet")
    df.to_parquet(path=output_path)
    
    print(f"SFT数据预处理完成: {len(df)} 条 -> {output_path}")
    return {"path": output_path, "size": len(df)}


def preprocess_hh_rlhf_rm(
    output_dir: str = "~/data/full_hh_rlhf/rm",
    local_dataset_path: Optional[str] = None,
    train_ratio: float = 0.75,
) -> dict:
    """
    生成奖励模型训练数据
    按比例划分训练集和测试集
    """
    train_split = f"train[:{int(train_ratio * 100)}%]"
    test_split = f"train[-{int((1 - train_ratio) * 100)}%:]"
    
    if local_dataset_path is not None:
        train_dataset = load_dataset(local_dataset_path, split=train_split)
        test_dataset = load_dataset(local_dataset_path, split=test_split)
    else:
        train_dataset = load_dataset("Dahoas/full-hh-rlhf", split=train_split)
        test_dataset = load_dataset("Dahoas/full-hh-rlhf", split=test_split)
    
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    for dataset, name in [(train_dataset, "train"), (test_dataset, "test")]:
        output = {"prompt": [], "chosen": [], "rejected": []}
        for data in tqdm(dataset, desc=f"Processing RM {name} data"):
            output["prompt"].append(data["prompt"])
            output["chosen"].append(data["chosen"])
            output["rejected"].append(data["rejected"])
        
        df = pd.DataFrame(output)
        output_path = os.path.join(output_dir, f"{name}.parquet")
        df.to_parquet(path=output_path)
        
        print(f"RM {name}数据: {len(df)} 条 -> {output_path}")
        results[f"{name}_path"] = output_path
        results[f"{name}_size"] = len(df)
    
    return results


def preprocess_hh_rlhf_rl(
    output_dir: str = "~/data/full_hh_rlhf/rl",
    local_dataset_path: Optional[str] = None,
) -> dict:
    """
    生成RL训练数据 (仅prompt)
    """
    data_source = "Dahoas/full-hh-rlhf"
    
    if local_dataset_path is not None:
        dataset = load_dataset(local_dataset_path)
    else:
        dataset = load_dataset(data_source)
    
    train_dataset = dataset["train"]
    
    def make_map_fn(split: str):
        def process_fn(example, idx):
            prompt = example.pop("prompt")
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "alignment",
                "reward_model": {
                    "style": "model",
                    "ground_truth": "",  # RM模型评分
                },
                "extra_info": {"split": split, "index": idx},
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "train.parquet")
    train_dataset.to_parquet(output_path)
    
    print(f"RL数据预处理完成: {len(train_dataset)} 条 -> {output_path}")
    return {"path": output_path, "size": len(train_dataset)}


def preprocess_hh_rlhf(
    split: str,
    output_dir: str = "~/data/full_hh_rlhf",
    local_dataset_path: Optional[str] = None,
) -> dict:
    """
    HH-RLHF数据预处理入口
    
    Args:
        split: "sft", "rm", 或 "rl"
        output_dir: 输出目录
        local_dataset_path: 本地数据集路径
    """
    output_dir = os.path.join(output_dir, split)
    
    if split == "sft":
        return preprocess_hh_rlhf_sft(output_dir, local_dataset_path)
    elif split == "rm":
        return preprocess_hh_rlhf_rm(output_dir, local_dataset_path)
    elif split == "rl":
        return preprocess_hh_rlhf_rl(output_dir, local_dataset_path)
    else:
        raise ValueError(f"Unknown split: {split}. Must be one of: sft, rm, rl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HH-RLHF数据预处理")
    parser.add_argument("--split", type=str, choices=["sft", "rm", "rl"], required=True)
    parser.add_argument("--output_dir", default="~/data/full_hh_rlhf", help="输出目录")
    parser.add_argument("--local_dataset_path", default=None, help="本地数据集路径")
    
    args = parser.parse_args()
    preprocess_hh_rlhf(
        split=args.split,
        output_dir=args.output_dir,
        local_dataset_path=args.local_dataset_path,
    )
