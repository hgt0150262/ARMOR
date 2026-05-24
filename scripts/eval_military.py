#!/usr/bin/env python3
"""
ARMOR Military Domain Evaluation Script
Evaluates base model vs GRPO-finetuned model on military test set.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_military.py \
        --base_model /data/hgt/models/Qwen2.5-7B-Instruct \
        --lora_path checkpoints/<experiment>/final \
        --test_data data/military/test.parquet \
        --output results/military_eval_results.json
"""
import argparse
import json
import os
import re
import sys
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MILITARY_TERMS = {
    "fm", "ar", "atp", "adp", "tc",
    "mdmp", "lsco", "ipb", "opord", "frago", "warno",
    "roe", "tlp", "mett-tc", "oakoc", "ascope",
    "platoon", "squad", "company", "battalion", "brigade",
    "maneuver", "fires", "reconnaissance", "security", "logistics",
    "commander", "nco", "operations", "intelligence", "sustainment",
    "offensive", "defensive", "stability", "tactical", "operational",
    "mission command", "unified land operations", "decisive action",
}


def load_test_data(data_path):
    """Load military test data."""
    df = pd.read_parquet(data_path)
    items = []
    for _, row in df.iterrows():
        prompt_data = row["prompt"]
        if hasattr(prompt_data, "tolist"):
            prompt_data = prompt_data.tolist()

        reward_model = row.get("reward_model", {})
        gt = reward_model.get("ground_truth", "") if isinstance(reward_model, dict) else ""
        extra = row.get("extra_info", {})
        question = extra.get("question", "") if isinstance(extra, dict) else ""

        items.append({
            "prompt": prompt_data,
            "ground_truth": gt,
            "question": question,
        })
    return items


def compute_military_scores(response, ground_truth):
    """Compute multi-dimensional military domain scores."""
    response_lower = response.lower().strip()
    resp_words = set(response_lower.split())

    # --- Terminology accuracy (0.4) ---
    term_hits = sum(1 for t in MILITARY_TERMS if t in response_lower)
    fm_refs = len(re.findall(r'\b(?:fm|ar|atp|adp|tc)\s*\d+[\-\.]\d+', response_lower))
    term_score = min(1.0, (term_hits * 0.1) + (fm_refs * 0.2))

    # --- Factual matching (0.4) ---
    fact_score = 0.0
    gt_text = ground_truth if isinstance(ground_truth, str) else str(ground_truth)
    if gt_text.strip():
        gt_lower = gt_text.lower().strip()
        gt_words = set(gt_lower.split())
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
                     "and", "or", "for", "on", "with", "that", "this", "it", "as", "by"}
        gt_content = gt_words - stopwords
        resp_content = resp_words - stopwords
        if gt_content:
            overlap = len(gt_content & resp_content) / len(gt_content)
            fact_score = min(1.0, overlap * 1.2)
        gt_trigrams = [" ".join(gt_lower.split()[j:j+3]) for j in range(len(gt_lower.split())-2)]
        for tri in gt_trigrams[:10]:
            if tri in response_lower:
                fact_score = min(1.0, fact_score + 0.1)

    # --- Structure quality (0.2) ---
    struct_score = 0.0
    if re.search(r'(?:\d+[\.\)]\s|\-\s|\*\s|•)', response):
        struct_score += 0.4
    sentence_count = len(re.findall(r'[.!?]\s', response))
    if sentence_count >= 3:
        struct_score += 0.3
    word_count = len(response.split())
    if 50 <= word_count <= 500:
        struct_score += 0.3
    elif word_count >= 20:
        struct_score += 0.1
    struct_score = min(struct_score, 1.0)

    combined = 0.4 * term_score + 0.4 * fact_score + 0.2 * struct_score

    return {
        "terminology": term_score,
        "factual": fact_score,
        "structure": struct_score,
        "combined": combined,
    }


def evaluate_model(model, tokenizer, test_items, max_new_tokens=384, label="Model"):
    """Evaluate a model on military test items."""
    all_scores = []

    for i, item in enumerate(test_items):
        prompt = item["prompt"]
        text = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
            )
        response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        scores = compute_military_scores(response, item["ground_truth"])
        all_scores.append(scores)

        if i < 3:
            q = item.get("question", str(item["prompt"])[:100])
            print(f"  [{label}] Q: {q[:80]}...")
            print(f"    A: {response[:150]}...")
            print(f"    Scores: term={scores['terminology']:.2f} fact={scores['factual']:.2f} struct={scores['structure']:.2f} combined={scores['combined']:.2f}")
            print()

        if (i + 1) % 50 == 0:
            print(f"  [{label}] Evaluated {i+1}/{len(test_items)}...")

    # Aggregate
    avg = {}
    for key in all_scores[0]:
        avg[key] = sum(s[key] for s in all_scores) / len(all_scores)

    return avg, all_scores


def main():
    parser = argparse.ArgumentParser(description="ARMOR Military Domain Evaluation")
    parser.add_argument("--base_model", type=str, default="/data/hgt/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA checkpoint")
    parser.add_argument("--test_data", type=str, default="data/military/test.parquet")
    parser.add_argument("--output", type=str, default="results/military_eval_results.json")
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print("ARMOR Military Domain Evaluation")
    print("=" * 60)

    test_items = load_test_data(args.test_data)
    if args.max_samples > 0:
        test_items = test_items[:args.max_samples]
    print(f"Test samples: {len(test_items)}")

    # Evaluate base model
    print("\n--- Base Model ---")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    base_model.eval()
    base_avg, base_scores = evaluate_model(base_model, tokenizer, test_items, label="Base")
    del base_model
    torch.cuda.empty_cache()

    # Evaluate GRPO model
    grpo_avg = None
    grpo_scores = None
    if args.lora_path:
        print("\n--- GRPO Model ---")
        from peft import PeftModel
        grpo_base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        grpo_model = PeftModel.from_pretrained(grpo_base, args.lora_path)
        grpo_model.eval()
        grpo_avg, grpo_scores = evaluate_model(grpo_model, tokenizer, test_items, label="GRPO")
        del grpo_model
        torch.cuda.empty_cache()

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} {'Base':>8}", end="")
    if grpo_avg:
        print(f" {'GRPO':>8} {'Delta':>8}")
    else:
        print()

    for key in base_avg:
        line = f"{key:<20} {base_avg[key]:>8.3f}"
        if grpo_avg:
            delta = grpo_avg[key] - base_avg[key]
            sign = "+" if delta >= 0 else ""
            line += f" {grpo_avg[key]:>8.3f} {sign}{delta:>7.3f}"
        print(line)

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results = {
        "base": base_avg,
        "base_per_sample": base_scores,
    }
    if grpo_avg:
        results["grpo"] = grpo_avg
        results["grpo_per_sample"] = grpo_scores

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
