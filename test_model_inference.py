"""Test Qwen2.5-7B inference to diagnose garbled output issue."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/data/hgt/models/Qwen2.5-7B-Instruct"

def test_basic_inference():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    # Test prompt
    messages = [{"role": "user", "content": "What is 2 + 3? Answer with just the number."}]
    
    # Method 1: apply_chat_template
    print("\n=== Test 1: apply_chat_template ===")
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"Formatted prompt:\n{formatted}\n")
    
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Response: {response}")
    
    # Test 2: With gradient checkpointing enabled
    print("\n=== Test 2: With gradient_checkpointing ===")
    model.gradient_checkpointing_enable()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Response: {response}")
    
    # Test 3: GSM8K style prompt
    print("\n=== Test 3: GSM8K style prompt ===")
    gsm8k_messages = [{
        "role": "user", 
        "content": "Gerald wants to buy a meat pie that costs 2 pfennigs. Gerald has 54 farthings, and there are 6 farthings to a pfennig. How many pfennigs will Gerald have left after buying the pie? Let's think step by step and output the final answer after \"####\"."
    }]
    formatted_gsm8k = tokenizer.apply_chat_template(gsm8k_messages, tokenize=False, add_generation_prompt=True)
    inputs_gsm8k = tokenizer(formatted_gsm8k, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs_gsm8k,
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs_gsm8k["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Response:\n{response}")

if __name__ == "__main__":
    test_basic_inference()
