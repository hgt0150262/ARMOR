# Project Archive (develop.md)

## Project Overview

* **Project**: ARMOR - Simplified RLHF Framework Reproduction
* **Repository**: `F:\LLM\reproduce\verl_reproduction`
* **Remote**: `gpu` → `gpu-server:/data/hgt/projects/verl_reproduction`
* **Version**: v0.2.0

## History Archive

### Phase 1: Initial Deployment (2025-01-20)

**Phase Goal**: Deploy ARMOR RLHF framework reproduction to gpu-server1

**Work Steps Completed**:
1. [✓] Install Miniconda3 to `/data/hgt/miniconda3`
2. [✓] Initialize conda and configure `.bashrc`
3. [✓] Update `requirements.txt` with full dependencies
4. [✓] Initialize git repo and add remote `gpu1`
5. [✓] Push code to gpu-server1 via `git push -u gpu1 master`
6. [✓] Run example and verify all tests pass on CUDA

**Key Results**:
| Item | Status | Details |
|------|--------|---------|
| Miniconda3 | ✅ | `/data/hgt/miniconda3` |
| Conda Env | ✅ | `minimind` with CUDA |
| Code Sync | ✅ | `gpu1` remote configured |
| Server Path | ✅ | `/data/hgt/projects/verl_reproduction` |
| Tests | ✅ | All examples passed (DataProto, GAE/GRPO/RLOO, PPO Training) |

**Important Decisions**:
- Used git sync instead of scp due to network constraints
- Used `gpu1` as remote name (gpu was occupied)
- Used `/data/hgt/` as base path on server

---

## Rule Deviation Log

*(No deviations recorded)*

---

### Phase 2: DPO & ReMax Algorithms (2025-01-20)

**Phase Goal**: Extend ARMOR with DPO and ReMax algorithms

**Work Steps Completed**:
1. [✓] Research DPO and ReMax algorithm principles
2. [✓] Implement DPO (Direct Preference Optimization) algorithm
3. [✓] Implement ReMax algorithm
4. [✓] Add algorithm registry mechanism
5. [✓] Create example and tests
6. [✓] Sync to gpu-server1 and verify on CUDA

**Key Results**:
| Algorithm | Function | Features |
|-----------|----------|----------|
| DPO | `compute_dpo_loss()` | sigmoid/hinge/ipo loss types, label smoothing |
| ReMax | `compute_remax_advantage()`, `compute_remax_loss()` | max-based baseline |
| GRPO | `compute_grpo_loss()` | PPO-style clipping |

**New Files**:
- `ARMOR/example_dpo_remax.py`

---

### Phase 3: Ray Distributed Training (2026-01-20)

**Phase Goal**: Add Ray-based distributed training support

**Work Steps Completed**:
1. [✓] Research Ray distributed training architecture
2. [✓] Implement RayResourcePool resource management
3. [✓] Implement RayWorkerGroup worker abstraction
4. [✓] Implement distributed DataProto transmission
5. [✓] Create RayPPOTrainer distributed trainer
6. [✓] Create multi-server Ray cluster scripts
7. [✓] Verify on gpu-server + gpu-server1 (6x H100)

**Key Results**:
| Component | File | Features |
|-----------|------|----------|
| RayResourcePool | `ray_worker.py` | Placement groups, GPU management |
| RayWorkerGroup | `ray_worker.py` | Distributed worker orchestration |
| RayPPOTrainer | `ray_trainer.py` | Distributed PPO training loop |

---

### Phase 4: RLHF Training + Project Restructure (2026-01-20)

**Phase Goal**: Complete RLHF framework with logging and model utilities

**Work Steps Completed**:
1. [✓] Implement logging_utils (WandB/TensorBoard/Console)
2. [✓] Implement model_utils (Qwen2.5 + LoRA)
3. [✓] Implement rlhf_trainer (PPO/GRPO training loop)
4. [✓] Restructure project to match verl folder structure
5. [✓] Remove 12 duplicate files
6. [✓] Verify on gpu-server (v0.2.0)

**Project Structure (v0.2.0)**:
```
ARMOR/
├── __init__.py
├── base_config.py
├── protocol.py
├── trainer/
│   ├── ppo/
│   │   ├── core_algos.py
│   │   └── ray_trainer.py
│   └── rlhf_trainer.py
├── workers/
│   └── worker.py
├── single_controller/
│   └── ray/
│       └── ray_worker.py
├── utils/
│   ├── logging_utils.py
│   └── model_utils.py
└── examples/
    ├── example_basic.py
    ├── example_dpo_remax.py
    ├── example_ray.py
    └── example_rlhf.py
```

---

## Phase History

### Phase 5 Completed (2026-01-20):
1. ✅ vLLM推理后端 - workers/rollout/vllm_rollout.py
2. ✅ 数据管道 GSM8K/Alpaca - utils/data_utils.py
3. ✅ FSDP模型并行 - workers/fsdp_utils.py
4. ✅ 奖励模型训练 - workers/reward_model.py
5. ✅ RLHF训练成功 - Qwen2.5-0.5B + LoRA + GAE算法
6. ✅ SwanLab离线监控集成

### Phase 6 Completed (2026-01-21):
1. ✅ vLLM生成加速集成 - trainer/rlhf_trainer.py (use_vllm选项)
2. ✅ 多epoch训练+checkpoint resume - trainer/rlhf_trainer.py
3. ✅ GSM8K评估流程 - utils/data_utils.py (GSM8KEvaluator)
4. ✅ 7B模型支持 - gradient_checkpointing + FSDP
5. ✅ Ray分布式RLHF - trainer/ppo/ray_trainer.py (已有)

### Phase 7 Completed (2026-01-22):
参照verl官方项目优化:
1. ✅ OPO算法 - 长度加权基线优势估计 (core_algos.py)
2. ✅ Policy Loss注册机制 - POLICY_LOSS_REGISTRY + @register_policy_loss
3. ✅ AdvantageEstimator枚举扩展 (12种): GAE, GRPO, REINFORCE++, REINFORCE++_BASELINE, RLOO, RLOO_VECTORIZED, REMAX, DPO, OPO, GRPO_PASSK, GRPO_VECTORIZED, GPG
4. ✅ vanilla policy loss别名 (官方默认模式)
5. ✅ Dr.GRPO支持 (norm_adv_by_std_in_grpo=False)
6. ✅ KL惩罚类型扩展: kl/k1, abs, mse/k2, low_var_kl/k3
7. ✅ AdaptiveKLController + FixedKLController + get_kl_controller工厂

**当前版本**: v0.4.2 (verl官方完全对齐)

### Phase 8: 数据预处理模块 (2026-01-22)
新增 `ARMOR/data_preprocess/` 模块:
1. ✅ `gsm8k.py` - GSM8K数学推理数据预处理
2. ✅ `hh_rlhf.py` - HH-RLHF对话对齐数据预处理 (SFT/RM/RL)
3. ✅ `custom.py` - 自定义数据集预处理 (JSON/JSONL/CSV)

**数据格式规范**:
```python
{
    "data_source": "数据来源",
    "prompt": [{"role": "user", "content": "问题"}],
    "ability": "math/alignment/general",
    "reward_model": {"style": "rule/model", "ground_truth": "答案"},
    "extra_info": {"index": 0, ...}
}
```

**当前版本**: v0.5.0

### Phase 9: Multi-GPU GRPO Training Debug (2026-02-03)
1. ✅ Fix NVLink peer GPU memory errors - NCCL env vars
2. ✅ Fix prompt format nesting - numpy.ndarray → list conversion
3. ✅ Fix garbled output - model.eval() before generate()
4. ✅ Training v8 successful - reward=1.0, loss=0.004
5. ✅ Server directory cleanup + project restructure

### Phase 10: Training Quality & Extended Training (2026-02-03)
1. ✅ Debug: ratio calculation bug (a - a.detach() = 0, should use ref_log_probs)
2. ✅ Add NaN/Inf protection and log_ratio clamping
3. ✅ Training v10 - LoRA weights valid, GSM8K 74%
4. ✅ Training v11 - 3 epochs, GSM8K 76%
- **v11 Checkpoint**: `/data/hgt/projects/verl_reproduction/checkpoints/ARMOR_qwen7b_grpo_4gpu_20260203_195744/final`

### Phase 11: Military Domain Ray Distributed SFT (2026-02-04 ~ 2026-02-09)

**Phase Goal**: Fine-tune Qwen2.5-7B for military domain using LoRA SFT with Ray distributed training across 2 nodes (8x H100 GPUs)

**Work Steps Completed**:
1. [✓] Ray cluster setup: gpu-server (head, 4 GPUs) + gpu-server1 (worker, 4 GPUs) = 8 GPUs
2. [✓] Sync model and dataset to gpu-server1 via rsync
3. [✓] NCCL network configuration (NCCL_SOCKET_IFNAME=ens65f0, disable IB/P2P/SHM)
4. [✓] Fix CUDA NVLink P2P error → switched distributed backend from NCCL to Gloo
5. [✓] Fix Gloo connection closed error → reduced batch_size 2→1, increased gradient_accumulation 4→8, added GLOO_SOCKET_TIMEOUT_MS=300000
6. [✓] Fix train.report sync deadlock → all workers must call train.report (not just rank 0)
7. [✓] Military LoRA SFT training completed: 3 epochs, loss 1.87→1.42
8. [✓] LoRA model inference test - military domain answers validated
9. [✓] LoRA weights merged into base model → `/data/hgt/models/Qwen2.5-7B-Military`
10. [✓] Merged model inference test passed

**Key Results**:
| Item | Details |
|------|---------|
| Training Script | `ARMOR/trainer/train_military_ray_sft.py` |
| Dataset | US Army FM Instruct (7001 conversations, 3 JSONL files) |
| Training Config | LoRA rank=64, alpha=128, batch_size=1, grad_accum=8, lr=2e-5, max_len=2048 |
| Backend | Gloo (TCP-based, avoids NVLink P2P issues across nodes) |
| Training Loss | Epoch 1: 1.87 → Epoch 3: 1.42 |
| Training Time | ~73 min/epoch, ~3.6h total |
| Checkpoints | `/data/hgt/projects/verl_reproduction/checkpoints/military_ray_sft/epoch_{1,2,3}/` |
| Merged Model | `/data/hgt/models/Qwen2.5-7B-Military` (15GB, SafeTensors) |

**Important Specifications Learned**:
- 【Specification】: All Ray Train workers must call `train.report()`, not just rank 0. → Scenario: Ray TorchTrainer epoch-end reporting. → Reason: Ray uses train.report as a synchronization barrier across all workers.
- 【Specification】: Use Gloo backend instead of NCCL for cross-node distributed training on H100 with NVLink. → Scenario: Multi-node Ray TorchTrainer. → Reason: NCCL NVLink P2P causes CUDA errors across nodes.
- 【Specification】: Set GLOO_SOCKET_IFNAME to correct network interface (e.g., ens65f0). → Scenario: Gloo backend initialization. → Reason: Default resolves to loopback, causing connection failures.

**当前版本**: v0.8.0
