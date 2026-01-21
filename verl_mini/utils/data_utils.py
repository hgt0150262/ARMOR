"""
Data utilities for RLHF training in verl_mini.
Provides dataset loading, preprocessing, and sampling.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Iterator, Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


@dataclass 
class DataConfig:
    """Configuration for RLHF data loading."""
    
    # Dataset settings
    dataset_name: str = "gsm8k"  # gsm8k, alpaca, custom
    data_path: Optional[str] = None
    split: str = "train"
    
    # Preprocessing
    max_prompt_length: int = 512
    max_response_length: int = 512
    
    # Sampling
    batch_size: int = 8
    shuffle: bool = True
    num_workers: int = 4
    
    # Format
    prompt_key: str = "prompt"
    response_key: str = "response"
    reward_key: str = "reward"


class RLHFDataset(Dataset):
    """Dataset for RLHF training with prompts and optional responses/rewards."""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        config: DataConfig,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # Get prompt
        prompt = item.get(self.config.prompt_key, "")
        
        # Tokenize prompt
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.config.max_prompt_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        result = {
            "prompt": prompt,
            "input_ids": prompt_encoding["input_ids"],
            "attention_mask": prompt_encoding["attention_mask"],
        }
        
        # Optional: response (for SFT or preference data)
        if self.config.response_key in item:
            response = item[self.config.response_key]
            result["response"] = response
            
        # Optional: reward (for offline RL)
        if self.config.reward_key in item:
            result["reward"] = item[self.config.reward_key]
            
        return result
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        """Collate batch with padding."""
        # Find max length
        max_len = max(len(item["input_ids"]) for item in batch)
        
        # Pad sequences
        input_ids = []
        attention_mask = []
        
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [0] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            
        result = {
            "prompts": [item["prompt"] for item in batch],
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }
        
        # Optional fields
        if "response" in batch[0]:
            result["responses"] = [item["response"] for item in batch]
        if "reward" in batch[0]:
            result["rewards"] = torch.tensor([item["reward"] for item in batch])
            
        return result


class GSM8KDataset:
    """GSM8K math reasoning dataset."""
    
    PROMPT_TEMPLATE = """Solve the following math problem step by step.

Problem: {question}

Solution:"""

    @staticmethod
    def load(split: str = "train", data_path: Optional[str] = None) -> List[Dict]:
        """Load GSM8K dataset."""
        try:
            from datasets import load_dataset
            
            if data_path:
                dataset = load_dataset("json", data_files=data_path, split=split)
            else:
                dataset = load_dataset("openai/gsm8k", "main", split=split)
                
            data = []
            for item in dataset:
                prompt = GSM8KDataset.PROMPT_TEMPLATE.format(question=item["question"])
                data.append({
                    "prompt": prompt,
                    "response": item.get("answer", ""),
                    "question": item["question"],
                })
            return data
            
        except ImportError:
            print("datasets library required. Install with: pip install datasets")
            return []


class AlpacaDataset:
    """Alpaca instruction-following dataset."""
    
    PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""

    PROMPT_TEMPLATE_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

    @staticmethod
    def load(split: str = "train", data_path: Optional[str] = None) -> List[Dict]:
        """Load Alpaca dataset."""
        try:
            from datasets import load_dataset
            
            if data_path:
                dataset = load_dataset("json", data_files=data_path, split=split)
            else:
                dataset = load_dataset("tatsu-lab/alpaca", split=split)
                
            data = []
            for item in dataset:
                if item.get("input", "").strip():
                    prompt = AlpacaDataset.PROMPT_TEMPLATE.format(
                        instruction=item["instruction"],
                        input=item["input"],
                    )
                else:
                    prompt = AlpacaDataset.PROMPT_TEMPLATE_NO_INPUT.format(
                        instruction=item["instruction"],
                    )
                    
                data.append({
                    "prompt": prompt,
                    "response": item.get("output", ""),
                    "instruction": item["instruction"],
                })
            return data
            
        except ImportError:
            print("datasets library required. Install with: pip install datasets")
            return []


class PreferenceDataset:
    """Dataset for preference learning (DPO/KTO)."""
    
    @staticmethod
    def load(data_path: str) -> List[Dict]:
        """Load preference dataset from JSON."""
        with open(data_path, 'r') as f:
            data = json.load(f)
            
        # Expected format: {prompt, chosen, rejected}
        processed = []
        for item in data:
            processed.append({
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
            })
        return processed


def load_dataset(
    dataset_name: str,
    split: str = "train",
    data_path: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict]:
    """Load dataset by name."""
    
    loaders = {
        "gsm8k": GSM8KDataset.load,
        "alpaca": AlpacaDataset.load,
    }
    
    if dataset_name in loaders:
        data = loaders[dataset_name](split=split, data_path=data_path)
    elif data_path:
        # Custom JSON dataset
        with open(data_path, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    if limit:
        data = data[:limit]
        
    return data


def create_dataloader(
    data: List[Dict],
    tokenizer,
    config: DataConfig,
) -> DataLoader:
    """Create DataLoader for RLHF training."""
    dataset = RLHFDataset(data, tokenizer, config)
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        collate_fn=RLHFDataset.collate_fn,
        pin_memory=True,
    )


class PromptSampler:
    """Sampler for generating prompts during training."""
    
    def __init__(
        self,
        prompts: List[str],
        batch_size: int = 8,
        shuffle: bool = True,
    ):
        self.prompts = prompts
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._indices = list(range(len(prompts)))
        self._current = 0
        
        if shuffle:
            import random
            random.shuffle(self._indices)
            
    def __iter__(self) -> Iterator[List[str]]:
        return self
    
    def __next__(self) -> List[str]:
        if self._current >= len(self._indices):
            self._current = 0
            if self.shuffle:
                import random
                random.shuffle(self._indices)
                
        batch_indices = self._indices[self._current:self._current + self.batch_size]
        self._current += self.batch_size
        
        return [self.prompts[i] for i in batch_indices]
    
    def __len__(self) -> int:
        return (len(self.prompts) + self.batch_size - 1) // self.batch_size


class GSM8KEvaluator:
    """Evaluator for GSM8K math reasoning benchmark."""
    
    @staticmethod
    def extract_answer(text: str) -> Optional[str]:
        """Extract numerical answer from model response."""
        import re
        
        # Pattern 1: #### followed by number
        match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
        if match:
            return match.group(1).replace(',', '')
        
        # Pattern 2: "The answer is X"
        match = re.search(r'[Tt]he answer is[:\s]*(-?[\d,]+\.?\d*)', text)
        if match:
            return match.group(1).replace(',', '')
        
        # Pattern 3: Last number in text
        numbers = re.findall(r'-?[\d,]+\.?\d*', text)
        if numbers:
            return numbers[-1].replace(',', '')
        
        return None
    
    @staticmethod
    def extract_ground_truth(answer_text: str) -> Optional[str]:
        """Extract ground truth answer from GSM8K format."""
        import re
        match = re.search(r'####\s*(-?[\d,]+\.?\d*)', answer_text)
        if match:
            return match.group(1).replace(',', '')
        return None
    
    @staticmethod
    def check_answer(prediction: str, ground_truth: str) -> bool:
        """Check if prediction matches ground truth."""
        if prediction is None or ground_truth is None:
            return False
        try:
            pred_num = float(prediction)
            gt_num = float(ground_truth)
            return abs(pred_num - gt_num) < 1e-5
        except ValueError:
            return prediction.strip() == ground_truth.strip()
    
    @classmethod
    def evaluate(
        cls,
        model,
        tokenizer,
        data: List[Dict],
        batch_size: int = 8,
        max_new_tokens: int = 256,
    ) -> Dict[str, float]:
        """Evaluate model on GSM8K dataset."""
        correct = 0
        total = 0
        results = []
        
        model.eval()
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            prompts = [item["prompt"] for item in batch]
            ground_truths = [cls.extract_ground_truth(item.get("response", "")) for item in batch]
            
            # Tokenize
            inputs = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"].to(model.device),
                    attention_mask=inputs["attention_mask"].to(model.device),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Decode and evaluate
            for j, output in enumerate(outputs):
                response = tokenizer.decode(output[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                pred = cls.extract_answer(response)
                gt = ground_truths[j]
                is_correct = cls.check_answer(pred, gt)
                
                results.append({
                    "prompt": prompts[j],
                    "response": response,
                    "prediction": pred,
                    "ground_truth": gt,
                    "correct": is_correct,
                })
                
                if is_correct:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
        }
