"""Reward functions for verl_mini training."""
import re
import torch
from typing import List, Optional


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract numerical answer from GSM8K format response."""
    # Pattern 1: #### followed by number
    match = re.search(r'####\s*(\-?[\d,\.]+)', text)
    if match:
        return match.group(1).replace(',', '')
    
    # Pattern 2: "answer is X" or "= X"
    match = re.search(r'(?:answer is|=)\s*\$?(\-?[\d,\.]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')
    
    # Pattern 3: Boxed answer \boxed{X}
    match = re.search(r'\\boxed\{(\-?[\d,\.]+)\}', text)
    if match:
        return match.group(1).replace(',', '')
    
    # Fallback: last number in text
    numbers = re.findall(r'\-?[\d,\.]+', text)
    return numbers[-1].replace(',', '') if numbers else None


def gsm8k_reward_fn(
    prompts: List[str],
    responses: List[str],
    ground_truths: Optional[List[str]] = None,
    debug: bool = False
) -> torch.Tensor:
    """GSM8K reward function based on answer correctness."""
    rewards = []
    
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        pred_answer = extract_gsm8k_answer(response)
        gt_answer = ground_truths[i] if ground_truths and i < len(ground_truths) else None
        
        if pred_answer and gt_answer:
            try:
                pred_num = float(pred_answer)
                gt_num = float(gt_answer)
                # Exact match reward
                if abs(pred_num - gt_num) < 1e-6:
                    reward = 1.0
                # Partial credit for close answers
                elif abs(pred_num - gt_num) / max(abs(gt_num), 1e-6) < 0.01:
                    reward = 0.8
                else:
                    reward = 0.0
            except ValueError:
                reward = 1.0 if pred_answer == gt_answer else 0.0
        else:
            # Format reward for having structure
            has_steps = '=' in response or 'step' in response.lower()
            has_answer = '####' in response or 'answer' in response.lower()
            reward = 0.1 * has_steps + 0.1 * has_answer
        
        rewards.append(reward)
        
        if debug and i == 0:
            print(f"[DEBUG] GT: {gt_answer}, Pred: {pred_answer}, Reward: {reward:.2f}")
            print(f"[DEBUG] Response (first 200 chars): {response[:200]}...")
    
    return torch.tensor(rewards, dtype=torch.float32)


def simple_reward_fn(
    prompts: List[str],
    responses: List[str],
    **kwargs
) -> torch.Tensor:
    """Simple reward function based on response length and structure."""
    rewards = []
    for response in responses:
        # Penalize very short or very long responses
        length = len(response)
        if length < 10:
            reward = 0.1
        elif length > 2000:
            reward = 0.5
        else:
            reward = min(1.0, length / 500)
        rewards.append(reward)
    return torch.tensor(rewards, dtype=torch.float32)
