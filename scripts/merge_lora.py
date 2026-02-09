"""Merge LoRA adapter weights into base model and save as a standalone model."""
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_and_save(base_model_path, adapter_path, output_path):
    print(f"Loading base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    
    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path, safe_serialization=True)
    
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"Done! Merged model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument("--base_model", type=str, default="/data/hgt/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", type=str, default="/data/hgt/projects/verl_reproduction/checkpoints/military_ray_sft/final")
    parser.add_argument("--output", type=str, default="/data/hgt/models/Qwen2.5-7B-Military")
    args = parser.parse_args()
    
    merge_and_save(args.base_model, args.adapter, args.output)
