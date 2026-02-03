"""Check specific answer."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained('/data/hgt/models/Qwen2.5-7B-Instruct', trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained('/data/hgt/models/Qwen2.5-7B-Instruct', torch_dtype=torch.bfloat16, device_map='cuda:0', trust_remote_code=True)
model = PeftModel.from_pretrained(base, '/data/hgt/projects/verl_reproduction/checkpoints/verl_mini_qwen7b_grpo_4gpu_20260203_143758/final')
model.eval()

questions = [
    "A store sells notebooks for $3 each. If Tom buys 4 notebooks, how much does he spend?",
    "A rectangle has a length of 8 cm and a width of 5 cm. What is its area?",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    msgs = [{'role': 'user', 'content': q + " Let's think step by step."}]
    fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(fmt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=400, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    print(tokenizer.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True))
