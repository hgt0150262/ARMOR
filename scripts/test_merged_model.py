"""Test the merged military model (no adapter needed)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/data/hgt/models/Qwen2.5-7B-Military"

print("Loading merged model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()

prompts = [
    "What is the purpose of FM 7-8 Infantry Rifle Platoon and Squad?",
    "Describe the role of a squad leader in combat operations.",
    "What are the key principles of mission command?",
]

print("=" * 80)
print("MERGED MODEL INFERENCE TEST (Qwen2.5-7B-Military)")
print("=" * 80)

for i, prompt in enumerate(prompts):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=256, temperature=0.7,
            top_p=0.9, repetition_penalty=1.1, do_sample=True,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n--- Question {i+1} ---")
    print(f"Q: {prompt}")
    print(f"A: {response[:600]}")
    print("-" * 60)

print("\nDone!")
