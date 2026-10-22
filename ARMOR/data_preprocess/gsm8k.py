"""
GSM8K数据集预处理脚本
将GSM8K数据转换为ARMOR RLHF训练格式

输出格式 (Parquet):
{
    "data_source": "openai/gsm8k",
    "prompt": [{"role": "user", "content": "问题"}],
    "ability": "math",
    "reward_model": {"style": "rule", "ground_truth": "答案"},
    "extra_info": {"split": "train", "index": 0, "answer": "完整答案"}
}
"""

import argparse
import os
import re
from typing import Optional

import datasets


def extract_solution(solution_str: str) -> str:
    """从GSM8K答案字符串中提取最终数字答案"""
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    if solution is None:
        return ""
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def preprocess_gsm8k(
    output_dir: str = "~/data/gsm8k",
    local_dataset_path: Optional[str] = None,
    instruction: str = 'Let\'s think step by step and output the final answer after "####".',
) -> dict:
    """
    预处理GSM8K数据集
    
    Args:
        output_dir: 输出目录
        local_dataset_path: 本地数据集路径(可选，支持save_to_disk格式)
        instruction: 附加指令
    
    Returns:
        dict: {"train_path": str, "test_path": str, "train_size": int, "test_size": int}
    """
    data_source = "openai/gsm8k"
    
    # 加载数据集
    if local_dataset_path is not None:
        # 支持 save_to_disk 格式 (DatasetDict)
        try:
            dataset = datasets.load_from_disk(local_dataset_path)
        except Exception:
            # 回退到 load_dataset
            dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source, "main")
    
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    
    def make_map_fn(split: str):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            question = question_raw + " " + instruction
            
            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {"role": "user", "content": question}
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data
        return process_fn
    
    # 转换数据
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    
    # 保存到Parquet
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)
    
    print(f"GSM8K数据预处理完成:")
    print(f"  训练集: {len(train_dataset)} 条 -> {train_path}")
    print(f"  测试集: {len(test_dataset)} 条 -> {test_path}")
    
    return {
        "train_path": train_path,
        "test_path": test_path,
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSM8K数据预处理")
    parser.add_argument("--output_dir", default="~/data/gsm8k", help="输出目录")
    parser.add_argument("--local_dataset_path", default=None, help="本地数据集路径")
    parser.add_argument(
        "--instruction",
        default='Let\'s think step by step and output the final answer after "####".',
        help="附加指令"
    )
    
    args = parser.parse_args()
    preprocess_gsm8k(
        output_dir=args.output_dir,
        local_dataset_path=args.local_dataset_path,
        instruction=args.instruction,
    )
