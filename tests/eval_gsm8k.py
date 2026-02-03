"""Evaluate on official GSM8K test set."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
import re
from tqdm import tqdm
import argparse

BASE_MODEL_PATH = "/data/hgt/models/Qwen2.5-7B-Instruct"
CHECKPOINT_PATH = "/data/hgt/projects/verl_reproduction/checkpoints/verl_mini_qwen7b_grpo_4gpu_20260203_143758/final"
TEST_DATA_PATH = "/data/hgt/projects/verl_reproduction/data/gsm8k/test.parquet"

def extract_answer(text):
    """Extract numerical answer from response."""
    # Try #### format first
    match = re.search(r'####\s*(\-?[\d,\.]+)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    # Try "answer is X" pattern
    match = re.search(r'(?:answer|total|result|equals?|is)\s*(?:is|=|:)?\s*\$?(\-?[\d,\.]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '').strip()
    # Fall back to last number
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    return numbers[-1] if numbers else None

def normalize_answer(ans):
    """Normalize answer for comparison."""
    if ans is None:
        return None
    try:
        return str(int(float(ans)))
    except:
        return ans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=100, help="Max samples to evaluate")
    parser.add_argument("--use_lora", action="store_true", default=True)
    args = parser.parse_args()
    
    print(f"Loading model (LoRA={args.use_lora})...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    
    if args.use_lora:
        model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
        print(f"LoRA loaded from {CHECKPOINT_PATH}")
    else:
        model = base_model
        print("Using base model (no LoRA)")
    model.eval()
    
    # Load test data
    print(f"\nLoading test data from {TEST_DATA_PATH}...")
    df = pd.read_parquet(TEST_DATA_PATH)
    if args.max_samples > 0:
        df = df.head(args.max_samples)
    print(f"Evaluating on {len(df)} samples")
    
    correct = 0
    total = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        prompt = row['prompt']
        if isinstance(prompt, list):
            question = prompt[0]['content'] if prompt else ""
        else:
            question = str(prompt)
        
        ground_truth = str(row['ground_truth'])
        
        # Generate
        messages = [{"role": "user", "content": question + " Let's think step by step."}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512, do_sample=False, 
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Extract and compare
        pred = extract_answer(response)
        pred_norm = normalize_answer(pred)
        gt_norm = normalize_answer(ground_truth)
        
        is_correct = pred_norm == gt_norm
        correct += int(is_correct)
        total += 1
        
        if total <= 5 or (not is_correct and total <= 20):
            status = "✅" if is_correct else "❌"
            print(f"\n[{total}] {status} GT={gt_norm}, Pred={pred_norm}")
            print(f"  Q: {question[:60]}...")
            print(f"  A: {response[:80].replace(chr(10), ' ')}...")
    
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"Results: {correct}/{total} correct ({accuracy:.1f}%)")
    print(f"Model: {'LoRA v10' if args.use_lora else 'Base'}")

if __name__ == "__main__":
    main()
