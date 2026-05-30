# Working Document (working.md)

## Rule Compliance

- Follow Work Rules workflow.

## Status and Signals

* **Git Sync Ready**: `git push gpu master` configured for code sync to gpu-server
* **Environment**: `minimind` conda env on both gpu-server and gpu-server1 with CUDA support
* **Mode**: [Standby Mode] → Phase 12 cleanup completed (v0.8.1)
* **Ray Cluster**: gpu-server (head, 172.16.54.132, 4 GPUs) + gpu-server1 (worker, 172.16.54.131, 4 GPUs) = 8x H100 80GB

## User Requirements

* Deploy ARMOR RLHF framework reproduction to gpu-server
* Run multi-GPU training for Qwen2.5-7B with GRPO on GSM8K dataset
* Fine-tune Qwen2.5-7B for military domain using LoRA SFT with Ray distributed training
* Merge LoRA weights into base model for standalone deployment

## 📋 System Standards

* 【Specification】: When using SSH commands for gpu-server, use jump-win proxy. → Scenario: SSH/SCP operations to gpu-server. → Reason: Server requires ProxyJump configuration.
* 【Specification】: Use `gpu` as git remote name for gpu-server. → Scenario: Git push operations.
* 【Specification】: Use `/data/hgt/` as base path on gpu-server. → Scenario: File storage and conda installation. → Reason: User-designated data partition.
* 【Specification】: Switch to `model.eval()` before `generate()` when using gradient_checkpointing. → Scenario: Model inference during training. → Reason: gradient_checkpointing + train mode causes garbled output.
* 【Specification】: Set NCCL_P2P_DISABLE=1, NCCL_IB_DISABLE=1, NCCL_SHM_DISABLE=1 for multi-GPU training. → Scenario: NVLink peer GPU memory access errors. → Reason: Prevents CUDA peer memory access issues.
* 【Specification】: All Ray Train workers must call `train.report()`, not just rank 0. → Scenario: Ray TorchTrainer epoch-end reporting. → Reason: Ray uses train.report as a synchronization barrier across all workers.
* 【Specification】: Use Gloo backend instead of NCCL for cross-node distributed training on H100 with NVLink. → Scenario: Multi-node Ray TorchTrainer. → Reason: NCCL NVLink P2P causes CUDA errors across nodes.
* 【Specification】: Set GLOO_SOCKET_IFNAME to correct network interface (e.g., ens65f0). → Scenario: Gloo backend initialization. → Reason: Default resolves to loopback, causing connection failures.

## Specification Points

* Git sync command: `git push gpu master`
* Server project path: `/data/hgt/projects/verl_reproduction`
* Conda path: `/data/hgt/miniconda3`
* Active env: `minimind`
* Training scripts location: `ARMOR/trainer/`
* Ray head node: gpu-server (172.16.54.132)
* Ray worker node: gpu-server1 (172.16.54.131)
* Network interface: `ens65f0`
* Merged military model: `/data/hgt/models/Qwen2.5-7B-Military`

## Work Plan

* **Phase Goal**: All phases completed through v0.8.0

* **Previous Phases (Archived to develop.md)**:
  - Phase 1-7: Framework development ✅
  - Phase 8: Multi-GPU Training + Restructure ✅
  - Phase 9: Training Quality Fix (v10) ✅
  - Phase 10: Extended Training (v11, GSM8K 76%) ✅
  - Phase 11: Military Domain Ray Distributed SFT ✅
  - Phase 12: Proprietary script cleanup ✅

## Work Task

* **Current Step**: Phase 12 cleanup completed (v0.8.1)
* **Next Action**: Awaiting new user instructions
* **Action Status**: ✅ Successful

* **Action Log/Result (Phase 12 - Cleanup)**:
  - Deleted 10 proprietary/one-off scripts (military-specific, hardcoded checkpoint paths)
  - Backed up to `backup/proprietary_20260209_093357/` (local only, gitignored)
  - Retained 13 general reusable scripts
  - Added `backup/` to `.gitignore`

---

## Work Status and Results

* **Current Overall Status**: ✅ Completed (v0.8.1)

* **Key Results Summary**:

| Phase | Status | Details |
|-------|--------|---------|
| Phase 1-7 | ✅ | Framework development |
| Phase 8 | ✅ | Multi-GPU Training (v8) + Project Restructure |
| Phase 9 | ✅ | Training Quality Fix (v10) - LoRA works correctly |
| Phase 10 | ✅ | Extended Training (v11) - 3 epochs, GSM8K 76% |
| Phase 11 | ✅ | Military SFT (Ray 8 GPUs) - loss 1.87→1.42, merged model |
| Phase 12 | ✅ | Cleanup: removed 10 proprietary scripts, kept 13 general |

* **Models**:

| Model | Path | Details |
|-------|------|---------|
| GRPO v11 | `/data/hgt/projects/verl_reproduction/checkpoints/ARMOR_qwen7b_grpo_4gpu_20260203_195744/final` | GSM8K 76% |
| Military SFT | `/data/hgt/models/Qwen2.5-7B-Military` | 15GB, SafeTensors, merged |
| Military LoRA | `/data/hgt/projects/verl_reproduction/checkpoints/military_ray_sft/final` | adapter only |

* **Remaining Tests** (`tests/`): eval_gsm8k.py, test_import.py, test_model_inference.py, test_training_flow.py
* **Remaining Scripts** (`scripts/`): merge_lora.py, ray_cluster_setup.md, run_distributed_training.py, run_tests.sh, run_training.sh, start_ray_*.sh
* **Trainers** (`ARMOR/trainer/`): rlhf_trainer.py, train_qwen7b_grpo_multigpu.py/.sh, main_grpo.py

* **Sync Command**: `git add -A ; git commit -m "update" ; git push gpu master`