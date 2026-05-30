"""
Qwen2.5-7B GRPO Multi-GPU Training Script for ARMOR
Supports 4x H100 distributed training with accelerate

Usage:
    # Single command (auto-detect GPUs)
    accelerate launch --multi_gpu --num_processes 4 ARMOR/examples/train_qwen7b_grpo_multigpu.py

    # Or with torchrun
    torchrun --nproc_per_node=4 ARMOR/examples/train_qwen7b_grpo_multigpu.py
"""

import os
import sys

# CRITICAL: Isolate each process to its own GPU BEFORE importing torch.
# This prevents CUDA from enabling NVLink P2P access between GPUs,
# which causes 'Invalid access of peer GPU memory over nvlink' errors.
_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
os.environ["CUDA_VISIBLE_DEVICES"] = str(_local_rank)

import argparse
import re
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ARMOR import (
    ModelConfig,
    ModelManager,
    RLHFConfig,
    AdvantageEstimator,
    TRANSFORMERS_AVAILABLE,
    PEFT_AVAILABLE,
)


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        # Each process only sees 1 GPU (set via CUDA_VISIBLE_DEVICES at top of file)
        torch.cuda.set_device(0)
        dist.init_process_group(backend="nccl")
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract numerical answer from GSM8K format response with multiple patterns."""
    # Pattern 1: #### format (standard GSM8K)
    match = re.search(r'####\s*(\-?[\d,\.]+)', text)
    if match:
        return match.group(1).replace(',', '')
    
    # Pattern 2: "answer is X" or "answer: X"
    match = re.search(r'answer\s*(?:is|:|=)\s*(\-?[\d,\.]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')
    
    # Pattern 3: "= X" at end of line (calculation result)
    match = re.search(r'=\s*(\-?[\d,\.]+)\s*$', text, re.MULTILINE)
    if match:
        return match.group(1).replace(',', '')
    
    # Pattern 4: Last number in the text (fallback)
    numbers = re.findall(r'\-?[\d,\.]+', text)
    if numbers:
        # Filter out very small numbers (likely not answers)
        valid_numbers = [n for n in numbers if len(n.replace(',', '').replace('.', '')) <= 10]
        return valid_numbers[-1].replace(',', '') if valid_numbers else None
    
    return None


def gsm8k_reward_fn(prompts: List[str], responses: List[str], ground_truths: Optional[List[str]] = None, debug: bool = False) -> torch.Tensor:
    """
    GSM8K reward function with format reward shaping.
    
    Reward structure:
    - 1.0: Correct answer
    - 0.5: Close answer (within 10%)
    - 0.3: Has reasoning steps + some number
    - 0.2: Has reasoning steps
    - 0.1: Has any mathematical content
    - 0.0: No valid response
    """
    rewards = []
    
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        pred_answer = extract_gsm8k_answer(response)
        gt_answer = ground_truths[i] if ground_truths and i < len(ground_truths) else None
        
        reward = 0.0
        
        # Check for correct answer
        if pred_answer and gt_answer:
            try:
                pred_num = float(pred_answer)
                gt_num = float(gt_answer)
                
                if abs(pred_num - gt_num) < 1e-6:
                    reward = 1.0  # Exact match
                elif gt_num != 0 and abs(pred_num - gt_num) / abs(gt_num) < 0.1:
                    reward = 0.5  # Within 10%
                elif gt_num != 0 and abs(pred_num - gt_num) / abs(gt_num) < 0.2:
                    reward = 0.3  # Within 20%
            except (ValueError, ZeroDivisionError):
                if pred_answer == gt_answer:
                    reward = 1.0
        
        # Format reward shaping (if no correct answer)
        if reward < 0.3:
            has_steps = any(op in response for op in ['=', '+', '-', '*', '/'])
            has_numbers = bool(re.search(r'\d+', response))
            has_structure = '####' in response or 'answer' in response.lower()
            
            format_reward = 0.0
            if has_steps and has_numbers:
                format_reward = 0.2
            elif has_steps or has_numbers:
                format_reward = 0.1
            if has_structure:
                format_reward += 0.1
            
            reward = max(reward, format_reward)
        
        # Debug logging (only first batch)
        if debug and i == 0:
            print(f"[DEBUG] GT: {gt_answer}, Pred: {pred_answer}, Reward: {reward:.2f}")
            print(f"[DEBUG] Response (first 200 chars): {response[:200]}...")
        
        rewards.append(reward)
    
    return torch.tensor(rewards, dtype=torch.float32)


def load_gsm8k_data(data_path: str) -> tuple:
    """Load preprocessed GSM8K data from parquet.
    
    Returns:
        prompts: List of chat messages (for apply_chat_template)
        ground_truths: List of answer strings
    """
    import pandas as pd
    import numpy as np
    
    df = pd.read_parquet(data_path)
    prompts = []  # Will store chat format: [{"role": "user", "content": ...}]
    ground_truths = []
    
    for _, row in df.iterrows():
        prompt_data = row['prompt']
        
        # Convert numpy array to list if needed
        if isinstance(prompt_data, np.ndarray):
            prompt_data = prompt_data.tolist()
        
        # Ensure proper chat format for apply_chat_template
        if isinstance(prompt_data, list) and len(prompt_data) > 0:
            # Already in chat format: [{"role": "user", "content": "..."}]
            prompts.append(prompt_data)
        elif isinstance(prompt_data, str):
            # Plain text, wrap in chat format
            prompts.append([{"role": "user", "content": prompt_data}])
        else:
            # Fallback
            prompts.append([{"role": "user", "content": str(prompt_data)}])
        
        reward_model = row.get('reward_model', {})
        if isinstance(reward_model, dict):
            gt = reward_model.get('ground_truth', '')
        else:
            gt = ''
        ground_truths.append(gt)
    
    return prompts, ground_truths


class PromptDataset(torch.utils.data.Dataset):
    """Simple dataset for prompts."""
    def __init__(self, prompts, ground_truths, reward_meta=None):
        self.prompts = prompts
        self.ground_truths = ground_truths
        self.reward_meta = reward_meta  # Extra metadata for safety reward
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        item = {"prompt": self.prompts[idx], "ground_truth": self.ground_truths[idx]}
        if self.reward_meta is not None:
            item["reward_meta"] = self.reward_meta[idx]
        return item


def truthfulqa_reward_fn(prompts: List[str], responses: List[str],
                         ground_truths: Optional[List[str]] = None,
                         reward_meta: Optional[List[dict]] = None,
                         debug: bool = False) -> torch.Tensor:
    """
    TruthfulQA safety reward function with multi-dimensional scoring.

    Reward structure:
    - Truthfulness (0.5 weight): Does response match correct answers?
    - Misinformation rejection (0.3 weight): Does response avoid incorrect answers?
    - Format quality (0.2 weight): Does response show reasoning/hedging?
    """
    rewards = []

    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        gt = ground_truths[i] if ground_truths and i < len(ground_truths) else ""
        meta = reward_meta[i] if reward_meta and i < len(reward_meta) else {}

        correct_answers = meta.get("correct_answers", [gt] if gt else [])
        incorrect_answers = meta.get("incorrect_answers", [])

        response_lower = response.lower().strip()

        # --- Dimension 1: Truthfulness (0.5 weight) ---
        truth_score = 0.0
        # Check if response contains any correct answer keyword
        for ca in correct_answers:
            ca_lower = ca.lower().strip()
            if not ca_lower:
                continue
            # Exact containment or high keyword overlap
            ca_words = set(ca_lower.split())
            resp_words = set(response_lower.split())
            if ca_lower in response_lower:
                truth_score = 1.0
                break
            overlap = len(ca_words & resp_words) / max(len(ca_words), 1)
            if overlap > 0.6:
                truth_score = max(truth_score, 0.7)
            elif overlap > 0.3:
                truth_score = max(truth_score, 0.4)

        # --- Dimension 2: Misinformation rejection (0.3 weight) ---
        misinfo_score = 1.0  # Start perfect, penalize for incorrect matches
        for ia in incorrect_answers:
            ia_lower = ia.lower().strip()
            if not ia_lower:
                continue
            if ia_lower in response_lower:
                misinfo_score = 0.0
                break
            # Enhanced ngram overlap scoring
            ia_words = ia_lower.split()
            resp_words = response_lower.split()
            # Unigram overlap
            ia_set = set(ia_words)
            resp_set = set(resp_words)
            uni_overlap = len(ia_set & resp_set) / max(len(ia_set), 1)
            # Bigram overlap for better phrase detection
            ia_bigrams = set(zip(ia_words[:-1], ia_words[1:])) if len(ia_words) > 1 else set()
            resp_bigrams = set(zip(resp_words[:-1], resp_words[1:])) if len(resp_words) > 1 else set()
            bi_overlap = len(ia_bigrams & resp_bigrams) / max(len(ia_bigrams), 1) if ia_bigrams else 0
            combined_overlap = 0.5 * uni_overlap + 0.5 * bi_overlap
            if combined_overlap > 0.4:
                misinfo_score = min(misinfo_score, 0.2)
            elif uni_overlap > 0.5:
                misinfo_score = min(misinfo_score, 0.3)

        # --- Dimension 3: Format quality (0.2 weight) ---
        format_score = 0.0
        hedging_phrases = ["actually", "in fact", "the truth is", "it is a common",
                           "misconception", "no,", "not true", "false premise",
                           "there is no evidence", "this is not"]
        for phrase in hedging_phrases:
            if phrase in response_lower:
                format_score = 0.5
                break
        if len(response.split()) >= 10:
            format_score += 0.3
        if any(c in response for c in ['.', '!', '?']):
            format_score += 0.2
        format_score = min(format_score, 1.0)

        # Weighted combination
        reward = 0.5 * truth_score + 0.3 * misinfo_score + 0.2 * format_score

        if debug and i == 0:
            print(f"[DEBUG-Safety] GT: {gt[:80]}")
            print(f"[DEBUG-Safety] Truth={truth_score:.2f} Misinfo={misinfo_score:.2f} Format={format_score:.2f} -> Reward={reward:.2f}")
            print(f"[DEBUG-Safety] Response (first 200 chars): {response[:200]}...")

        rewards.append(reward)

    return torch.tensor(rewards, dtype=torch.float32)


def load_truthfulqa_data(data_path: str) -> tuple:
    """Load preprocessed TruthfulQA data from parquet.

    Returns:
        prompts: List of chat messages
        ground_truths: List of best_answer strings
        reward_meta: List of dicts with correct_answers and incorrect_answers
    """
    import pandas as pd
    import numpy as np

    df = pd.read_parquet(data_path)
    prompts = []
    ground_truths = []
    reward_meta = []

    for _, row in df.iterrows():
        prompt_data = row['prompt']
        if isinstance(prompt_data, np.ndarray):
            prompt_data = prompt_data.tolist()
        if isinstance(prompt_data, list) and len(prompt_data) > 0:
            prompts.append(prompt_data)
        elif isinstance(prompt_data, str):
            prompts.append([{"role": "user", "content": prompt_data}])
        else:
            prompts.append([{"role": "user", "content": str(prompt_data)}])

        reward_model = row.get('reward_model', {})
        if isinstance(reward_model, dict):
            gt = reward_model.get('ground_truth', '')
            correct = reward_model.get('correct_answers', [])
            incorrect = reward_model.get('incorrect_answers', [])
        else:
            gt = ''
            correct = []
            incorrect = []
        ground_truths.append(gt)
        reward_meta.append({"correct_answers": correct, "incorrect_answers": incorrect})

    return prompts, ground_truths, reward_meta


# ============== Military Domain Reward Function (Level 3: Multi-dimensional) ==============

MILITARY_TERMS = {
    "fm", "ar", "atp", "adp", "tc",  # doctrine references
    "mdmp", "lsco", "ipb", "opord", "frago", "warno",  # processes
    "roe", "tlp", "mett-tc", "oakoc", "ascope",  # frameworks
    "platoon", "squad", "company", "battalion", "brigade",  # units
    "maneuver", "fires", "reconnaissance", "security", "logistics",  # warfighting functions
    "commander", "nco", "operations", "intelligence", "sustainment",
    "offensive", "defensive", "stability", "tactical", "operational",
    "mission command", "unified land operations", "decisive action",
}


def military_reward_fn(prompts: List[str], responses: List[str],
                       ground_truths: Optional[List[str]] = None,
                       reward_meta: Optional[List[dict]] = None,
                       debug: bool = False) -> torch.Tensor:
    """
    Military domain reward function with multi-dimensional scoring.

    Reward structure:
    - Terminology accuracy (0.4 weight): Uses military terms and FM references
    - Factual matching (0.4 weight): Keyword overlap with ground truth answer
    - Structure quality (0.2 weight): Organized output with lists/paragraphs
    """
    rewards = []

    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        gt = ground_truths[i] if ground_truths and i < len(ground_truths) else ""
        meta = reward_meta[i] if reward_meta and i < len(reward_meta) else {}
        response_lower = response.lower().strip()
        resp_words = set(response_lower.split())

        # --- Dimension 1: Terminology accuracy (0.4 weight) ---
        term_hits = sum(1 for t in MILITARY_TERMS if t in response_lower)
        # Check for FM/AR number references (e.g., "FM 3-0", "AR 600-20")
        fm_refs = len(re.findall(r'\b(?:fm|ar|atp|adp|tc)\s*\d+[\-\.]\d+', response_lower))
        term_score = min(1.0, (term_hits * 0.1) + (fm_refs * 0.2))

        # --- Dimension 2: Factual matching (0.4 weight) ---
        fact_score = 0.0
        gt_text = gt if isinstance(gt, str) else str(gt)
        if gt_text.strip():
            gt_lower = gt_text.lower().strip()
            gt_words = set(gt_lower.split())
            # Remove common stopwords for better overlap
            stopwords = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
                         "and", "or", "for", "on", "with", "that", "this", "it", "as", "by"}
            gt_content = gt_words - stopwords
            resp_content = resp_words - stopwords
            if gt_content:
                overlap = len(gt_content & resp_content) / len(gt_content)
                fact_score = min(1.0, overlap * 1.2)  # slight boost
            # Bonus for exact phrase matches (3+ word subsequences)
            gt_trigrams = [" ".join(gt_lower.split()[j:j+3]) for j in range(len(gt_lower.split())-2)]
            for tri in gt_trigrams[:10]:  # check first 10 trigrams
                if tri in response_lower:
                    fact_score = min(1.0, fact_score + 0.1)

        # --- Dimension 3: Structure quality (0.2 weight) ---
        struct_score = 0.0
        # Has numbered/bulleted list
        if re.search(r'(?:\d+[\.\)]\s|\-\s|\*\s|•)', response):
            struct_score += 0.4
        # Has paragraph structure (multiple sentences)
        sentence_count = len(re.findall(r'[.!?]\s', response))
        if sentence_count >= 3:
            struct_score += 0.3
        # Reasonable length (50-500 words)
        word_count = len(response.split())
        if 50 <= word_count <= 500:
            struct_score += 0.3
        elif word_count >= 20:
            struct_score += 0.1
        struct_score = min(struct_score, 1.0)

        # Weighted combination
        reward = 0.4 * term_score + 0.4 * fact_score + 0.2 * struct_score

        if debug and i == 0:
            print(f"[DEBUG-Military] GT (first 80): {gt_text[:80]}")
            print(f"[DEBUG-Military] Term={term_score:.2f} Fact={fact_score:.2f} Struct={struct_score:.2f} -> Reward={reward:.2f}")
            print(f"[DEBUG-Military] Response (first 200 chars): {response[:200]}...")

        rewards.append(reward)

    return torch.tensor(rewards, dtype=torch.float32)


def load_military_data(data_path: str) -> tuple:
    """Load preprocessed military data from parquet.

    Returns:
        prompts: List of chat messages
        ground_truths: List of answer strings
        reward_meta: List of dicts (empty for military, kept for interface consistency)
    """
    import pandas as pd
    import numpy as np

    df = pd.read_parquet(data_path)
    prompts = []
    ground_truths = []
    reward_meta = []

    for _, row in df.iterrows():
        prompt_data = row['prompt']
        if isinstance(prompt_data, np.ndarray):
            prompt_data = prompt_data.tolist()
        if isinstance(prompt_data, list) and len(prompt_data) > 0:
            prompts.append(prompt_data)
        elif isinstance(prompt_data, str):
            prompts.append([{"role": "user", "content": prompt_data}])
        else:
            prompts.append([{"role": "user", "content": str(prompt_data)}])

        reward_model = row.get('reward_model', {})
        if isinstance(reward_model, dict):
            gt = reward_model.get('ground_truth', '')
        else:
            gt = ''
        ground_truths.append(gt)
        reward_meta.append({})

    return prompts, ground_truths, reward_meta


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B GRPO Multi-GPU Training")
    
    # Model
    parser.add_argument("--model_path", type=str, 
                        default="/data/hgt/models/Qwen2.5-7B-Instruct")
    
    # Data
    parser.add_argument("--train_data", type=str, default="data/gsm8k/train.parquet")
    parser.add_argument("--val_data", type=str, default="data/gsm8k/test.parquet")
    parser.add_argument("--max_samples", type=int, default=-1)
    
    # Training
    parser.add_argument("--total_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)  # Per GPU
    parser.add_argument("--mini_batch_size", type=int, default=4)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # Generation
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_response_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    
    # Algorithm
    parser.add_argument("--adv_estimator", type=str, default="grpo")
    parser.add_argument("--grpo_n", type=int, default=5)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--kl_coef", type=float, default=0.001)
    
    # Reward function selection
    parser.add_argument("--reward_fn", type=str, default="gsm8k",
                        choices=["gsm8k", "truthfulqa", "military"],
                        help="Reward function: gsm8k (math), truthfulqa (safety), or military (domain)")
    
    # Optimization
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    
    # Logging
    parser.add_argument("--project_name", type=str, default="ARMOR_qwen7b_multigpu")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    
    # Resume from checkpoint
    parser.add_argument("--resume_from", type=str, default=None, help="Path to LoRA checkpoint dir to resume from")
    parser.add_argument("--resume_step", type=int, default=0, help="Global step to resume from (skip batches before this)")
    
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    # Each process sees only 1 GPU (CUDA_VISIBLE_DEVICES set at top of file)
    device = torch.device("cuda:0")
    
    if is_main_process(rank):
        print("="*60)
        print("Qwen2.5-7B GRPO Multi-GPU Training")
        print("="*60)
        print(f"World size: {world_size} GPUs")
        print(f"Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
    
    # Model config
    model_config = ModelConfig(
        model_name_or_path=args.model_path,
        use_lora=True,
        lora_r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        torch_dtype="bfloat16",
        gradient_checkpointing=args.gradient_checkpointing,
    )
    
    # Load model
    manager = ModelManager(model_config)
    model, tokenizer = manager.load_model()
    
    # Fix padding side for decoder-only models
    tokenizer.padding_side = 'left'
    if is_main_process(rank):
        print(f"Tokenizer padding_side set to: {tokenizer.padding_side}")
    
    if PEFT_AVAILABLE and model_config.use_lora:
        model = manager.apply_lora()
        if is_main_process(rank):
            print(f"LoRA applied: rank={args.lora_rank}")
    
    # Resume from checkpoint if specified
    if args.resume_from:
        from peft import PeftModel
        if is_main_process(rank):
            print(f"Resuming from checkpoint: {args.resume_from}")
        # Load the saved LoRA adapter weights
        import safetensors.torch
        adapter_path = os.path.join(args.resume_from, "adapter_model.safetensors")
        if os.path.exists(adapter_path):
            state_dict = safetensors.torch.load_file(adapter_path)
            model.load_state_dict(state_dict, strict=False)
            if is_main_process(rank):
                print(f"Loaded LoRA weights from {adapter_path}")
        else:
            if is_main_process(rank):
                print(f"WARNING: No adapter_model.safetensors found in {args.resume_from}")
    
    model = model.to(device)
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[0], output_device=0, find_unused_parameters=False)
        # Fix for gradient_checkpointing + DDP compatibility
        model._set_static_graph()
        if is_main_process(rank):
            print("Model wrapped with DistributedDataParallel (static_graph=True)")
    
    # Create reference model (on each GPU)
    ref_model = manager.create_reference_model()
    ref_model = ref_model.to(device)
    ref_model.eval()
    
    # Load data
    if is_main_process(rank):
        print(f"\nLoading data from {args.train_data} (reward_fn={args.reward_fn})...")
    
    train_reward_meta = None
    if args.reward_fn == "truthfulqa":
        train_prompts, train_ground_truths, train_reward_meta = load_truthfulqa_data(args.train_data)
    elif args.reward_fn == "military":
        train_prompts, train_ground_truths, train_reward_meta = load_military_data(args.train_data)
    else:
        train_prompts, train_ground_truths = load_gsm8k_data(args.train_data)
    
    if args.max_samples > 0:
        train_prompts = train_prompts[:args.max_samples]
        train_ground_truths = train_ground_truths[:args.max_samples]
        if train_reward_meta is not None:
            train_reward_meta = train_reward_meta[:args.max_samples]
    
    if is_main_process(rank):
        print(f"Training samples: {len(train_prompts)}")
    
    # Create dataset and distributed sampler
    dataset = PromptDataset(train_prompts, train_ground_truths, reward_meta=train_reward_meta)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler,
        collate_fn=lambda x: x,  # Keep as list of dicts
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    
    # Scheduler
    num_training_steps = len(dataloader) * args.total_epochs * args.ppo_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_training_steps, eta_min=args.learning_rate * 0.1
    )
    
    # SwanLab (main process only)
    if is_main_process(rank):
        try:
            import swanlab
            exp_name = args.experiment_name or f"qwen7b_grpo_{world_size}gpu"
            swanlab.init(
                project=args.project_name,
                name=exp_name,
                mode="local",
                logdir="./swanlog",
            )
            print("SwanLab initialized")
        except Exception as e:
            print(f"SwanLab init failed: {e}")
    
    # Training loop
    if is_main_process(rank):
        print("\n" + "="*60)
        print("Starting Multi-GPU GRPO Training...")
        print("="*60)
    
    global_step = 0
    resume_step = args.resume_step if args.resume_from else 0
    
    if resume_step > 0 and is_main_process(rank):
        print(f"Resuming from step {resume_step}, skipping earlier batches...")
    
    for epoch in range(args.total_epochs):
        sampler.set_epoch(epoch)
        model.train()
        
        if is_main_process(rank):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        else:
            pbar = dataloader
        
        for batch_idx, batch in enumerate(pbar):
            # Skip batches before resume_step
            if global_step < resume_step:
                global_step += 1
                continue
            chat_prompts = [item["prompt"] for item in batch]  # List of chat format messages
            ground_truths = [item["ground_truth"] for item in batch]
            batch_reward_meta = [item.get("reward_meta") for item in batch] if args.reward_fn in ("truthfulqa", "military") else None
            
            # Apply chat template to convert chat format to model input
            # This properly formats the prompt with special tokens for Qwen2.5
            formatted_prompts = [
                tokenizer.apply_chat_template(
                    chat_msg,
                    tokenize=False,
                    add_generation_prompt=True  # Add assistant turn start
                )
                for chat_msg in chat_prompts
            ]
            
            # Debug: show formatted prompt on first batch
            if batch_idx == 0 and epoch == 0 and is_main_process(rank):
                print(f"[DEBUG] Formatted prompt (first 300 chars): {formatted_prompts[0][:300]}...")
            
            # Tokenize formatted prompts
            prompt_encodings = tokenizer(
                formatted_prompts,
                padding=True,
                truncation=True,
                max_length=args.max_prompt_length,
                return_tensors="pt",
            ).to(device)
            
            # Generate responses
            # CRITICAL: Switch to eval mode for generate (gradient_checkpointing + train mode causes garbled output)
            raw_model = model.module if hasattr(model, 'module') else model
            raw_model.eval()
            with torch.no_grad():
                output_ids = raw_model.generate(
                    **prompt_encodings,
                    max_new_tokens=args.max_response_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    repetition_penalty=args.repetition_penalty,
                    eos_token_id=tokenizer.eos_token_id,
                )
            raw_model.train()  # Switch back to train mode
            
            response_ids = output_ids[:, prompt_encodings["input_ids"].shape[1]:]
            responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            
            # Compute rewards (debug=True for first batch to see outputs)
            debug_mode = (batch_idx == 0 and epoch == 0 and is_main_process(rank))
            if args.reward_fn == "truthfulqa":
                rewards = truthfulqa_reward_fn(formatted_prompts, responses, ground_truths,
                                              reward_meta=batch_reward_meta, debug=debug_mode).to(device)
            elif args.reward_fn == "military":
                rewards = military_reward_fn(formatted_prompts, responses, ground_truths,
                                            reward_meta=batch_reward_meta, debug=debug_mode).to(device)
            else:
                rewards = gsm8k_reward_fn(formatted_prompts, responses, ground_truths, debug=debug_mode).to(device)
            
            # Forward pass for log probs
            full_ids = torch.cat([prompt_encodings["input_ids"], response_ids], dim=1)
            outputs = model(input_ids=full_ids, return_dict=True)
            logits = outputs.logits
            
            # Compute log probs
            response_start = prompt_encodings["input_ids"].shape[1]
            response_logits = logits[:, response_start-1:-1, :]
            log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs, dim=-1, index=response_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            # Reference log probs
            with torch.no_grad():
                ref_outputs = ref_model(input_ids=full_ids, return_dict=True)
                ref_logits = ref_outputs.logits
                ref_response_logits = ref_logits[:, response_start-1:-1, :]
                ref_log_probs = torch.nn.functional.log_softmax(ref_response_logits, dim=-1)
                ref_token_log_probs = torch.gather(
                    ref_log_probs, dim=-1, index=response_ids.unsqueeze(-1)
                ).squeeze(-1)
            
            # Compute advantages (simplified GRPO)
            advantages = rewards - rewards.mean()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Policy loss (GRPO style: use log prob ratio)
            log_ratio = token_log_probs.sum(-1) - ref_token_log_probs.sum(-1)
            # Clamp to prevent numerical instability
            log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            
            # KL penalty (per-token average)
            kl = (token_log_probs - ref_token_log_probs).mean()
            kl = torch.clamp(kl, -100.0, 100.0)  # Prevent NaN
            
            # Total loss
            loss = policy_loss + args.kl_coef * kl
            
            # NaN check
            if torch.isnan(loss) or torch.isinf(loss):
                if is_main_process(rank):
                    print(f"Warning: NaN/Inf loss detected, skipping batch")
                optimizer.zero_grad()
                continue
            loss = loss / args.gradient_accumulation_steps
            
            loss.backward()
            
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging (main process only)
                if is_main_process(rank) and global_step % args.log_interval == 0:
                    metrics = {
                        "loss": loss.item() * args.gradient_accumulation_steps,
                        "policy_loss": policy_loss.item(),
                        "kl": kl.item(),
                        "reward_mean": rewards.mean().item(),
                        "lr": scheduler.get_last_lr()[0],
                    }
                    
                    try:
                        import swanlab
                        swanlab.log(metrics, step=global_step)
                    except:
                        pass
                    
                    pbar.set_postfix(loss=f"{metrics['loss']:.4f}", reward=f"{metrics['reward_mean']:.4f}")
                
                # Save checkpoint
                if is_main_process(rank) and global_step % args.save_steps == 0:
                    save_path = os.path.join(args.save_dir, f"step_{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    raw_model = model.module if hasattr(model, 'module') else model
                    raw_model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"Checkpoint saved: {save_path}")
    
    # Final save
    if is_main_process(rank):
        save_path = os.path.join(args.save_dir, "final")
        os.makedirs(save_path, exist_ok=True)
        raw_model = model.module if hasattr(model, 'module') else model
        raw_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"\nTraining complete! Model saved to {save_path}")
        
        try:
            import swanlab
            swanlab.finish()
        except:
            pass
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
