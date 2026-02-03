"""Test more GSM8K questions."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re

BASE_MODEL_PATH = "/data/hgt/models/Qwen2.5-7B-Instruct"
CHECKPOINT_PATH = "/data/hgt/projects/verl_reproduction/checkpoints/verl_mini_qwen7b_grpo_4gpu_20260203_143758/final"

TEST_CASES = [
    {"q": "A farmer has 17 sheep. All but 9 die. How many sheep are left?", "a": "9"},
    {"q": "If you have 3 apples and get 2 more, how many apples do you have?", "a": "5"},
    {"q": "A store sells notebooks for $3 each. If Tom buys 4 notebooks, how much does he spend?", "a": "12"},
    {"q": "Sarah has 24 candies. She gives 8 to her brother and 6 to her sister. How many candies does she have left?", "a": "10"},
    {"q": "A rectangle has a length of 8 cm and a width of 5 cm. What is its area?", "a": "40"},
    {"q": "If a car travels at 60 km/h for 3 hours, how far does it travel?", "a": "180"},
    {"q": "John has 15 marbles. He loses 7 and then finds 4. How many marbles does he have?", "a": "12"},
    {"q": "A pizza is cut into 8 slices. If 3 people each eat 2 slices, how many slices are left?", "a": "2"},
]

def extract_answer(text):
    match = re.search(r'####\s*(\-?[\d,\.]+)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    numbers = re.findall(r'\b(\d+)\b', text.split('\n')[-1] if text else "")
    return numbers[-1] if numbers else None

def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    model.eval()
    print("Model loaded!\n")
    
    correct = 0
    for i, tc in enumerate(TEST_CASES, 1):
        messages = [{"role": "user", "content": f"{tc['q']} Let's think step by step."}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        response = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        pred = extract_answer(response)
        is_correct = pred == tc['a']
        correct += int(is_correct)
        
        status = "✅" if is_correct else "❌"
        print(f"[{i}] {status} Expected: {tc['a']}, Got: {pred}")
        print(f"    Q: {tc['q'][:60]}...")
        print(f"    A: {response[:100].replace(chr(10), ' ')}...")
        print()
    
    print(f"\n{'='*40}")
    print(f"Accuracy: {correct}/{len(TEST_CASES)} ({100*correct/len(TEST_CASES):.1f}%)")

if __name__ == "__main__":
    main()
