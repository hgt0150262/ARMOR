"""Test training flow to diagnose garbled output issue."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

MODEL_PATH = "/data/hgt/models/Qwen2.5-7B-Instruct"

def test_training_flow():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = 'left'  # As in training code
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    
    # Test 1: Before LoRA
    print("\n=== Test 1: Before LoRA ===")
    test_generate(model, tokenizer)
    
    # Apply LoRA (as in training code)
    print("\n=== Applying LoRA ===")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Test 2: After LoRA, eval mode
    print("\n=== Test 2: After LoRA (eval mode) ===")
    model.eval()
    test_generate(model, tokenizer)
    
    # Test 3: After LoRA, train mode
    print("\n=== Test 3: After LoRA (train mode) ===")
    model.train()
    test_generate(model, tokenizer)
    
    # Test 4: Enable gradient checkpointing
    print("\n=== Test 4: With gradient_checkpointing (train mode) ===")
    model.gradient_checkpointing_enable()
    test_generate(model, tokenizer)
    
    # Test 5: Back to eval mode with gradient_checkpointing
    print("\n=== Test 5: With gradient_checkpointing (eval mode) ===")
    model.eval()
    test_generate(model, tokenizer)

def test_generate(model, tokenizer):
    messages = [{
        "role": "user", 
        "content": "What is 2 + 3? Just give the number."
    }]
    
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt", padding=True).to(model.device)
    
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

if __name__ == "__main__":
    test_training_flow()
