"""Test military domain LoRA fine-tuned model inference."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse


def test_inference(base_model_path, adapter_path, prompts=None):
    print(f"Loading base model from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    if prompts is None:
        prompts = [
            "What is the purpose of FM 7-8 Infantry Rifle Platoon and Squad?",
            "Explain the principles of offensive operations according to US Army doctrine.",
            "What are the key elements of a defensive position?",
            "Describe the role of a squad leader in combat operations.",
            "What is the difference between a movement to contact and a hasty attack?",
        ]
    
    print("\n" + "="*80)
    print("MILITARY DOMAIN MODEL INFERENCE TEST")
    print("="*80)
    
    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        print(f"\n--- Question {i+1} ---")
        print(f"Q: {prompt}")
        print(f"A: {response}")
        print("-"*60)
    
    # Compare with base model (no LoRA)
    print("\n" + "="*80)
    print("BASE MODEL (without LoRA) COMPARISON")
    print("="*80)
    
    base_model_raw = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model_raw.eval()
    
    compare_prompt = prompts[0]
    messages = [{"role": "user", "content": compare_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(base_model_raw.device)
    
    with torch.no_grad():
        outputs = base_model_raw.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"\nQ: {compare_prompt}")
    print(f"A (base): {response}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="/data/hgt/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", type=str, default="/data/hgt/projects/verl_reproduction/checkpoints/military_ray_sft/final")
    args = parser.parse_args()
    
    test_inference(args.base_model, args.adapter)
