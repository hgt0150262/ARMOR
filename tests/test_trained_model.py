"""Test the trained model checkpoint."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# Paths
BASE_MODEL_PATH = "/data/hgt/models/Qwen2.5-7B-Instruct"
CHECKPOINT_PATH = "/data/hgt/projects/verl_reproduction/checkpoints/verl_mini_qwen7b_grpo_4gpu_20260202_195906/final"

def test_trained_model():
    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    print(f"Loading LoRA adapter from {CHECKPOINT_PATH}...")
    if os.path.exists(CHECKPOINT_PATH):
        model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
        print("LoRA adapter loaded successfully!")
    else:
        print(f"Checkpoint not found at {CHECKPOINT_PATH}")
        print("Using base model instead...")
        model = base_model
    
    model.eval()
    
    # Test prompts
    test_cases = [
        {
            "question": "Gerald wants to buy a meat pie that costs 2 pfennigs. Gerald has 54 farthings, and there are 6 farthings to a pfennig. How many pfennigs will Gerald have left after buying the pie?",
            "expected": "7"
        },
        {
            "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "expected": "72"
        },
        {
            "question": "If a train travels 60 miles per hour for 2.5 hours, how far does it travel?",
            "expected": "150"
        }
    ]
    
    print("\n" + "="*60)
    print("Testing trained model on GSM8K-style questions")
    print("="*60)
    
    for i, tc in enumerate(test_cases, 1):
        messages = [{
            "role": "user",
            "content": f"{tc['question']} Let's think step by step and output the final answer after \"####\"."
        }]
        
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        print(f"\n--- Test {i} ---")
        print(f"Question: {tc['question'][:80]}...")
        print(f"Expected: {tc['expected']}")
        print(f"Response:\n{response[:500]}")
        print("-" * 40)

if __name__ == "__main__":
    test_trained_model()
