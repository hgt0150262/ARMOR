"""Military Domain LoRA SFT Training Script for Qwen2.5-7B.

Uses US Army Field Manuals dataset for domain adaptation.
"""
import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
import warnings
warnings.filterwarnings("ignore")


def format_instruction(example):
    """Format dataset example into instruction format."""
    # US Army FM dataset has 'conversations' field
    if 'conversations' in example:
        convs = example['conversations']
        text = ""
        for conv in convs:
            role = conv.get('from', conv.get('role', ''))
            content = conv.get('value', conv.get('content', ''))
            if role in ['human', 'user']:
                text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role in ['gpt', 'assistant']:
                text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        return {"text": text}
    # Fallback for other formats
    elif 'instruction' in example and 'output' in example:
        instruction = example['instruction']
        inp = example.get('input', '')
        output = example['output']
        if inp:
            prompt = f"{instruction}\n\nInput: {inp}"
        else:
            prompt = instruction
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>\n"
        return {"text": text}
    return {"text": ""}


def main():
    parser = argparse.ArgumentParser(description="Military Domain LoRA SFT")
    parser.add_argument("--model_path", type=str, default="/data/hgt/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="Heralax/us-army-fm-instruct")
    parser.add_argument("--output_dir", type=str, default="checkpoints/military_sft")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()
    
    # Setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    print(f"[Rank {local_rank}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print(f"[Rank {local_rank}] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank} if world_size > 1 else "auto",
        trust_remote_code=True,
    )
    
    # LoRA config
    print(f"[Rank {local_rank}] Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print(f"[Rank {local_rank}] Loading dataset: {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name, split="train")
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    print(f"[Rank {local_rank}] Dataset size: {len(dataset)}")
    
    # Format and tokenize
    print(f"[Rank {local_rank}] Formatting dataset...")
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
    
    print(f"[Rank {local_rank}] Tokenizing...")
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        report_to="none",
        dataloader_num_workers=4,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    
    # Train
    print(f"[Rank {local_rank}] Starting training...")
    trainer.train()
    
    # Save
    if local_rank == 0:
        print("Saving model...")
        trainer.save_model(os.path.join(args.output_dir, "final"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
        print(f"Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
