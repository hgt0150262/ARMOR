#!/usr/bin/env python3
"""Evaluate base vs GRPO model on GSM8K test set."""

import argparse
import json
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def extract_answer(text):
    """Extract numerical answer from model output."""
    patterns = [
        r'####\s*([\-\d\.,]+)',
        r'(?:answer|Answer|ANSWER)\s*(?:is|:)\s*([\-\d\.,]+)',
        r'(?:=|equals)\s*([\-\d\.,]+)\s*$',
        r'\\boxed\{([\-\d\.,]+)\}',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).replace(',', '').strip()
    # Last number in text
    nums = re.findall(r'[\-]?\d[\d,]*\.?\d*', text)
    return nums[-1].replace(',', '') if nums else ""


def evaluate_model(model, tokenizer, test_data, max_samples=200, label="Model"):
    correct = 0
    total = 0
    for i, row in test_data.iterrows():
        if total >= max_samples:
            break
        prompt_data = row['prompt']
        if isinstance(prompt_data, list):
            text = tokenizer.apply_chat_template(prompt_data, tokenize=False, add_generation_prompt=True)
        else:
            text = str(prompt_data)

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True, top_p=0.95)
        response = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        gt_data = row.get('reward_model', {})
        if isinstance(gt_data, dict):
            gt = str(gt_data.get('ground_truth', ''))
        else:
            gt = ''

        pred = extract_answer(response)
        gt_clean = gt.replace(',', '').strip()
        is_correct = (pred == gt_clean)
        if is_correct:
            correct += 1
        total += 1

        if total <= 3:
            print(f"  [{label}] Q: {text[-80:]}...")
            print(f"    Pred={pred} GT={gt_clean} {'✓' if is_correct else '✗'}")
        if total % 50 == 0:
            print(f"  [{label}] {total}/{max_samples} done, acc={correct/total:.4f}")

    acc = correct / total if total > 0 else 0
    return {"accuracy": acc, "correct": correct, "total": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output", default="results/gsm8k_eval.json")
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()

    test_df = pd.read_parquet(args.test_data)
    print(f"Test samples: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base model
    print("\n--- Base Model ---")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    base_results = evaluate_model(base_model, tokenizer, test_df, args.max_samples, "Base")
    print(f"  Base Accuracy: {base_results['accuracy']:.4f} ({base_results['correct']}/{base_results['total']})")
    del base_model
    torch.cuda.empty_cache()

    # GRPO model
    if args.lora_path:
        print("\n--- GRPO Model ---")
        grpo_base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        grpo_model = PeftModel.from_pretrained(grpo_base, args.lora_path)
        grpo_model = grpo_model.merge_and_unload()
        grpo_results = evaluate_model(grpo_model, tokenizer, test_df, args.max_samples, "GRPO")
        print(f"  GRPO Accuracy: {grpo_results['accuracy']:.4f} ({grpo_results['correct']}/{grpo_results['total']})")
        del grpo_model
        torch.cuda.empty_cache()
    else:
        grpo_results = None

    # Summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"Base Accuracy:  {base_results['accuracy']:.4f}")
    if grpo_results:
        delta = grpo_results['accuracy'] - base_results['accuracy']
        print(f"GRPO Accuracy:  {grpo_results['accuracy']:.4f}")
        print(f"Delta:          {delta:+.4f}")

    results = {"base": base_results, "grpo": grpo_results}
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
