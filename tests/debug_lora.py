"""Debug LoRA loading issue."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

BASE_MODEL_PATH = "/data/hgt/models/Qwen2.5-7B-Instruct"
CHECKPOINT_PATH = "/data/hgt/projects/verl_reproduction/checkpoints/verl_mini_qwen7b_grpo_4gpu_20260203_143758/final"

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    base_model.eval()
    
    # Test base model
    question = "What is 2+3?"
    messages = [{"role": "user", "content": question}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(base_model.device)
    
    print("\n=== Base Model Test ===")
    with torch.no_grad():
        out = base_model.generate(**inputs, max_new_tokens=50, do_sample=False)
    print(f"Response: {tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)}")
    
    # Check LoRA weights
    print(f"\n=== Loading LoRA from {CHECKPOINT_PATH} ===")
    import safetensors.torch as st
    lora_weights = st.load_file(os.path.join(CHECKPOINT_PATH, "adapter_model.safetensors"))
    
    print(f"LoRA weight keys: {list(lora_weights.keys())[:5]}...")
    print(f"Total LoRA layers: {len(lora_weights)}")
    
    # Check weight statistics
    for key in list(lora_weights.keys())[:3]:
        w = lora_weights[key]
        print(f"  {key}: shape={w.shape}, mean={w.mean().item():.6f}, std={w.std().item():.6f}, min={w.min().item():.6f}, max={w.max().item():.6f}")
    
    # Load LoRA
    print("\n=== Applying LoRA ===")
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    model.eval()
    
    # Check if LoRA is merged or separate
    print(f"LoRA model type: {type(model)}")
    print(f"Active adapters: {model.active_adapters}")
    
    # Test LoRA model
    print("\n=== LoRA Model Test ===")
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    response = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"Response: {response[:100]}")
    
    # Check output logits
    print("\n=== Checking Logits ===")
    with torch.no_grad():
        base_out = base_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        lora_out = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
    print(f"Base logits: mean={base_out.logits.mean().item():.4f}, std={base_out.logits.std().item():.4f}")
    print(f"LoRA logits: mean={lora_out.logits.mean().item():.4f}, std={lora_out.logits.std().item():.4f}")
    
    # Check if logits have NaN or Inf
    print(f"Base has NaN: {torch.isnan(base_out.logits).any()}, Inf: {torch.isinf(base_out.logits).any()}")
    print(f"LoRA has NaN: {torch.isnan(lora_out.logits).any()}, Inf: {torch.isinf(lora_out.logits).any()}")

if __name__ == "__main__":
    main()
