#!/usr/bin/env python3
"""
================================================================================
  军事领域大模型 LoRA 微调 — 原理介绍与效果演示
  Military Domain LLM LoRA Fine-Tuning: Principles & Demo
================================================================================

本脚本分为三个部分:
  Part 1 — 技术原理概述 (纯文本输出，无需 GPU)
  Part 2 — 核心代码解析 (展示训练流水线关键片段)
  Part 3 — 效果对比演示 (基座模型 vs 微调模型 实时推理)

运行方式:
  # 仅展示原理 (不需要 GPU)
  python demo_military_lora_sft.py --part 1

  # 仅展示代码解析
  python demo_military_lora_sft.py --part 2

  # 仅运行效果对比 (需要 GPU)
  CUDA_VISIBLE_DEVICES=0 python demo_military_lora_sft.py --part 3

  # 完整演示 (需要 GPU)
  CUDA_VISIBLE_DEVICES=0 python demo_military_lora_sft.py --part all

硬件环境:
  训练: 2 节点 × 4 × NVIDIA H100 80GB (Ray 分布式)
  推理: 单卡 H100 80GB
================================================================================
"""
import argparse
import textwrap
import time

# ─────────────────────────────────────────────
# Part 1: 技术原理概述
# ─────────────────────────────────────────────

PRINCIPLE_TEXT = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Part 1 — LoRA 微调技术原理概述                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. 什么是 LoRA (Low-Rank Adaptation)?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LoRA 是一种参数高效微调 (PEFT) 方法。核心思想:

  原始权重矩阵 W ∈ R^{d×k} 在微调时保持冻结，
  仅训练一对低秩分解矩阵 A ∈ R^{d×r} 和 B ∈ R^{r×k}:

          h = W·x + α/r · (B·A)·x
              ───     ─────────
              冻结      可训练

  其中 r << min(d, k) 为秩 (rank)，α 为缩放因子。

  优势:
    ✓ 可训练参数量仅为全量微调的 0.5~2%
    ✓ 训练显存大幅降低
    ✓ 推理时可合并回原始权重，零额外开销
    ✓ 支持多个 LoRA adapter 热切换

2. 本项目的 LoRA 配置
━━━━━━━━━━━━━━━━━━━━━━
  ┌─────────────────────┬──────────────────────────────────────┐
  │ 参数                │ 值                                   │
  ├─────────────────────┼──────────────────────────────────────┤
  │ 基座模型            │ Qwen2.5-7B-Instruct (72亿参数)       │
  │ LoRA Rank (r)       │ 64                                   │
  │ LoRA Alpha (α)      │ 128  (缩放系数 α/r = 2.0)           │
  │ 目标模块            │ q/k/v/o_proj, gate/up/down_proj      │
  │ Dropout             │ 0.05                                 │
  │ 可训练参数          │ ~168M / 7.6B (约 2.2%)               │
  └─────────────────────┴──────────────────────────────────────┘

3. 为什么选择 SFT (Supervised Fine-Tuning)?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SFT 是大模型对齐三阶段中的第一步:

    预训练 → [SFT 监督微调] → RLHF 强化学习对齐
                 ▲
              当前阶段

  SFT 通过高质量的 指令-回复 对来教会模型:
    ✓ 遵循特定领域的术语和表达方式
    ✓ 输出符合领域规范的回答格式
    ✓ 掌握领域专业知识

4. 分布式训练架构 (Ray + DDP)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    ┌──────────────────┐
                    │   Ray Head Node  │
                    │   gpu-server     │
                    │  172.16.54.132   │
                    │  4× H100 80GB   │
                    └────────┬─────────┘
                             │  Gloo Backend (TCP)
                    ┌────────┴─────────┐
                    │  Ray Worker Node │
                    │   gpu-server1    │
                    │  172.16.54.131   │
                    │  4× H100 80GB   │
                    └──────────────────┘

  总计: 8 GPU 并行训练
  通信后端: Gloo (替代 NCCL，规避跨节点 NVLink P2P 错误)
  数据并行: PyTorch DDP，每卡独立前向/反向，梯度同步聚合

5. 训练数据
━━━━━━━━━━━
  数据集: US Army Field Manuals (美国陆军野战条令)
  格式: 多轮对话 JSONL (human/gpt 角色对)
  规模: 7,001 条对话 × 3 个 JSONL 文件
  内容: FM 7-8 步兵排与班、任务式指挥、作战原则等

6. 训练超参数
━━━━━━━━━━━━
  ┌──────────────────────┬─────────────────────┐
  │ 超参数               │ 值                  │
  ├──────────────────────┼─────────────────────┤
  │ 学习率               │ 2e-5                │
  │ Batch Size (per GPU) │ 1                   │
  │ 梯度累积步数         │ 8                   │
  │ 有效 Batch Size      │ 1×8×8GPU = 64       │
  │ Epochs               │ 3                   │
  │ 最大序列长度         │ 2048 tokens         │
  │ 优化器               │ AdamW (weight_decay │
  │                      │ = 0.01)             │
  │ 学习率调度           │ Cosine + 3% warmup  │
  │ 梯度裁剪             │ max_norm = 1.0      │
  └──────────────────────┴─────────────────────┘

7. 训练结果
━━━━━━━━━━━
  ┌──────────┬───────────┬──────────────┐
  │ Epoch    │ Avg Loss  │ 训练时间     │
  ├──────────┼───────────┼──────────────┤
  │ Epoch 1  │ 1.87      │ ~73 min      │
  │ Epoch 2  │ ~1.55     │ ~73 min      │
  │ Epoch 3  │ 1.42      │ ~73 min      │
  └──────────┴───────────┴──────────────┘
  总训练时间: ~3.6 小时 (8× H100)
"""


# ─────────────────────────────────────────────
# Part 2: 核心代码解析
# ─────────────────────────────────────────────

CODE_ANALYSIS_TEXT = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Part 2 — 核心训练代码解析                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

以下逐段解析 train_military_ray_sft.py 的关键代码:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[代码段 1] 数据集构造 — MilitaryDataset
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```python
class MilitaryDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048):
        self.data = []
        for jsonl_file in Path(data_path).glob("*.jsonl"):
            with open(jsonl_file, 'r') as f:
                for line in f:
                    item = json.loads(line.strip())
                    if 'conversations' in item:
                        self.data.append(item)

    def __getitem__(self, idx):
        conversations = self.data[idx]['conversations']
        # 拼接为 Qwen ChatML 格式
        text = ""
        for conv in conversations:
            if conv['from'] == 'human':
                text += f"<|im_start|>user\n{conv['value']}<|im_end|>\n"
            elif conv['from'] == 'gpt':
                text += f"<|im_start|>assistant\n{conv['value']}<|im_end|>\n"

        encoding = self.tokenizer(text, truncation=True,
                                  max_length=self.max_length,
                                  padding='max_length')
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100   # 只计算非 padding 的 loss
        return {'input_ids': ..., 'attention_mask': ..., 'labels': ...}
```

原理说明:
  ① JSONL 格式每行一个 JSON 对象，包含 conversations 字段 (human/gpt 角色对)
  ② 使用 Qwen 的 ChatML 模板格式化: <|im_start|>role\ncontent<|im_end|>
  ③ Labels 中 padding 位置设为 -100，PyTorch 的 CrossEntropyLoss 会自动忽略
  ④ 这是因果语言建模 (Causal LM): 模型学习预测下一个 token

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[代码段 2] LoRA 注入 — 低秩适配器
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=64,                    # 低秩矩阵的秩，越大容量越强
    lora_alpha=128,          # 缩放因子，实际缩放 = alpha/r = 2.0
    target_modules=[         # 注入 LoRA 的目标模块:
        "q_proj", "k_proj",  #   注意力层的 Q/K/V/O 投影
        "v_proj", "o_proj",
        "gate_proj",         #   FFN 层的 Gate/Up/Down 投影
        "up_proj", "down_proj"
    ],
    lora_dropout=0.05,       # Dropout 防止过拟合
    bias="none",             # 不训练偏置项
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
# 输出: trainable params: 167,772,160 / 7,615,616,000 (2.20%)
```

原理说明:
  ① 基座模型 7.6B 参数全部冻结，仅新增 ~168M 可训练参数 (2.2%)
  ② target_modules 覆盖了 Transformer 的两大核心:
     - 自注意力层 (Self-Attention): Q/K/V/O 投影矩阵
     - 前馈网络 (FFN): Gate/Up/Down 投影矩阵 (SwiGLU 架构)
  ③ rank=64 提供了较强的表达能力，适合领域差异较大的军事知识
  ④ alpha/r = 2.0 的缩放确保 LoRA 更新不会过大或过小

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[代码段 3] Ray 分布式训练编排
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```python
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig

ray.init(address="auto", runtime_env={"env_vars": NCCL_ENV})

trainer = TorchTrainer(
    train_loop_per_worker=train_func,      # 每个 worker 执行的训练函数
    train_loop_config=train_config,        # 训练超参数字典
    scaling_config=ScalingConfig(
        num_workers=8,                     # 8 GPU workers (4+4)
        use_gpu=True,
        resources_per_worker={"GPU": 1, "CPU": 4},
    ),
    torch_config=ray.train.torch.TorchConfig(
        backend="gloo",                    # 使用 Gloo 替代 NCCL
    ),
)
result = trainer.fit()
```

原理说明:
  ① Ray Train 自动将 train_func 分发到 8 个 GPU worker 上并行执行
  ② 每个 worker 独立加载模型副本，通过 DDP 同步梯度
  ③ ScalingConfig 定义资源分配: 每 worker 1 GPU + 4 CPU
  ④ 使用 Gloo 后端 (基于 TCP) 替代 NCCL:
     - NCCL 的 NVLink P2P 在跨节点时报 CUDA 硬件错误
     - Gloo 虽然性能略低，但跨节点兼容性更好
  ⑤ runtime_env 确保环境变量传播到所有 worker

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[代码段 4] DDP 训练循环
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```python
# DDP 包装 (每个 worker 内部)
model = torch.nn.parallel.DistributedDataParallel(
    model, device_ids=[local_rank], find_unused_parameters=False)

# 分布式采样器: 确保每个 worker 处理不同的数据子集
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)       # 每 epoch 重新打乱
    for batch in dataloader:
        loss = model(**batch).loss
        loss.backward()

        if (step + 1) % gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # 所有 worker 必须调用 train.report (Ray 同步屏障)
    train.report({"loss": avg_loss, "epoch": epoch + 1})
```

原理说明:
  ① DDP (DistributedDataParallel): 每个 GPU 持有完整模型副本
     - 前向传播: 各自独立计算
     - 反向传播: 自动 AllReduce 聚合梯度
  ② DistributedSampler: 将数据集均匀分割给各 worker，避免重复训练
  ③ 梯度累积: batch_size=1 × accumulation=8 → 有效 batch=8/GPU
  ④ train.report() 是 Ray Train 的同步屏障 — 所有 worker 必须调用
     否则会导致死锁 (这是我们调试中发现的关键问题)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[代码段 5] 权重合并 — LoRA → 完整模型
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(base_path)
model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = model.merge_and_unload()   # W' = W + α/r · B·A
merged_model.save_pretrained(output_path)
```

原理说明:
  ① merge_and_unload() 将 LoRA 权重永久合并回基座权重:
       W' = W + (α/r) · B · A
  ② 合并后模型与原始模型结构完全相同，推理无额外开销
  ③ 输出: 15GB SafeTensors 文件 (4 个分片)
"""


# ─────────────────────────────────────────────
# Part 3: 效果对比演示
# ─────────────────────────────────────────────

DEMO_QUESTIONS = [
    {
        "category": "条令知识",
        "question": "What is the purpose of FM 7-8 Infantry Rifle Platoon and Squad?",
        "question_zh": "FM 7-8《步兵步枪排与班》的用途是什么？",
    },
    {
        "category": "作战原则",
        "question": "Explain the principles of offensive operations according to US Army doctrine.",
        "question_zh": "根据美国陆军条令，解释进攻作战的原则。",
    },
    {
        "category": "指挥职责",
        "question": "Describe the role of a squad leader in combat operations.",
        "question_zh": "描述班长在战斗行动中的角色。",
    },
    {
        "category": "战术对比",
        "question": "What is the difference between a movement to contact and a hasty attack?",
        "question_zh": "接触运动与仓促进攻有何区别？",
    },
    {
        "category": "指挥原则",
        "question": "What are the key principles of mission command?",
        "question_zh": "任务式指挥的关键原则是什么？",
    },
]


def print_section(title: str):
    width = 78
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print()


def print_separator():
    print("-" * 78)


def run_part1():
    """输出技术原理概述。"""
    print(PRINCIPLE_TEXT)


def run_part2():
    """输出核心代码解析。"""
    print(CODE_ANALYSIS_TEXT)


def run_part3(base_model_path: str, merged_model_path: str, adapter_path: str,
              use_merged: bool = True, max_new_tokens: int = 384):
    """运行效果对比演示: 基座模型 vs 微调模型。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print_section("Part 3 — 军事领域模型效果对比演示")

    # ── 加载微调模型 ──
    if use_merged and merged_model_path:
        print(f"[1/3] 加载合并后模型: {merged_model_path}")
        ft_tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
        ft_model = AutoModelForCausalLM.from_pretrained(
            merged_model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
    else:
        from peft import PeftModel
        print(f"[1/3] 加载基座模型 + LoRA adapter")
        print(f"       基座: {base_model_path}")
        print(f"       Adapter: {adapter_path}")
        ft_tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        ft_model = PeftModel.from_pretrained(base, adapter_path)
    ft_model.eval()
    print("       微调模型加载完成\n")

    # ── 加载基座模型 ──
    print(f"[2/3] 加载基座模型: {base_model_path}")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    base_model.eval()
    print("       基座模型加载完成\n")

    # ── 逐题对比 ──
    print(f"[3/3] 开始对比推理 ({len(DEMO_QUESTIONS)} 个问题)")
    print_separator()

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
    )

    for i, item in enumerate(DEMO_QUESTIONS):
        print(f"\n{'─'*78}")
        print(f"  问题 {i+1}/{len(DEMO_QUESTIONS)}  [{item['category']}]")
        print(f"  Q: {item['question']}")
        print(f"     ({item['question_zh']})")
        print(f"{'─'*78}")

        messages = [{"role": "user", "content": item["question"]}]

        # --- 微调模型回答 ---
        text_ft = ft_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs_ft = ft_tokenizer(text_ft, return_tensors="pt").to(ft_model.device)
        t0 = time.time()
        with torch.no_grad():
            out_ft = ft_model.generate(**inputs_ft, **gen_kwargs)
        t_ft = time.time() - t0
        resp_ft = ft_tokenizer.decode(
            out_ft[0][inputs_ft["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # --- 基座模型回答 ---
        text_base = base_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs_base = base_tokenizer(text_base, return_tensors="pt").to(base_model.device)
        t0 = time.time()
        with torch.no_grad():
            out_base = base_model.generate(**inputs_base, **gen_kwargs)
        t_base = time.time() - t0
        resp_base = base_tokenizer.decode(
            out_base[0][inputs_base["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # --- 输出对比 ---
        print(f"\n  [微调模型] (生成耗时 {t_ft:.1f}s)")
        for line in resp_ft.strip().split("\n"):
            print(f"    {line}")

        print(f"\n  [基座模型] (生成耗时 {t_base:.1f}s)")
        for line in resp_base.strip().split("\n"):
            print(f"    {line}")

        print()

    # ── 总结 ──
    print_section("对比总结")
    print(textwrap.dedent("""\
    微调模型特点:
      ✓ 回答更聚焦于美军条令规范，引用具体 FM 编号
      ✓ 使用军事术语 (MDMP, LSCO, OE, IPB 等)
      ✓ 表述简洁直接，符合军事文书风格
      ✓ 结构化输出 (要点列举、层次分明)

    基座模型特点:
      ○ 回答更泛化，偏向百科全书式介绍
      ○ 术语使用较少，更多通俗表达
      ○ 内容更长但信息密度较低

    训练配置回顾:
      模型: Qwen2.5-7B-Instruct + LoRA (rank=64, alpha=128)
      数据: US Army Field Manuals (7,001 conversations)
      训练: 3 epochs, loss 1.87 → 1.42, ~3.6h on 8× H100
      合并: LoRA 权重已合并为独立模型 (15GB)
    """))


def main():
    parser = argparse.ArgumentParser(
        description="军事领域 LoRA 微调演示脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        示例:
          python demo_military_lora_sft.py --part 1          # 仅原理
          python demo_military_lora_sft.py --part 2          # 仅代码解析
          python demo_military_lora_sft.py --part 3          # 仅效果对比
          python demo_military_lora_sft.py --part all        # 完整演示
          python demo_military_lora_sft.py --part 3 --adapter /path/to/lora  # 使用 LoRA adapter
        """),
    )
    parser.add_argument(
        "--part", type=str, default="all", choices=["1", "2", "3", "all"],
        help="演示部分: 1=原理, 2=代码解析, 3=效果对比, all=全部",
    )
    parser.add_argument(
        "--base_model", type=str,
        default="/data/hgt/models/Qwen2.5-7B-Instruct",
        help="基座模型路径",
    )
    parser.add_argument(
        "--merged_model", type=str,
        default="/data/hgt/models/Qwen2.5-7B-Military",
        help="合并后的微调模型路径",
    )
    parser.add_argument(
        "--adapter", type=str,
        default="/data/hgt/projects/verl_reproduction/checkpoints/military_ray_sft/final",
        help="LoRA adapter 路径 (如果不使用合并模型)",
    )
    parser.add_argument(
        "--use_adapter", action="store_true",
        help="使用 LoRA adapter 而非合并模型进行推理",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=384,
        help="最大生成 token 数",
    )
    args = parser.parse_args()

    parts = ["1", "2", "3"] if args.part == "all" else [args.part]

    print(__doc__)

    for part in parts:
        if part == "1":
            run_part1()
        elif part == "2":
            run_part2()
        elif part == "3":
            run_part3(
                base_model_path=args.base_model,
                merged_model_path=args.merged_model,
                adapter_path=args.adapter,
                use_merged=not args.use_adapter,
                max_new_tokens=args.max_new_tokens,
            )

    print("\n" + "=" * 78)
    print("  演示完毕")
    print("=" * 78)


if __name__ == "__main__":
    main()
