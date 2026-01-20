# Working Document (working.md)

## Rule Compliance

- Follow the 001-workflow rule.

## Status and Signals

* **Git Sync Ready**: `git push gpu1 master` configured for code sync to gpu-server1
* **Environment**: `minimind` conda env on gpu-server1 with CUDA support
* **Mode**: [Work Mode] → Phase 2 completed → DPO & ReMax algorithms implemented

## User Requirements

* Deploy verl_mini RLHF framework reproduction to gpu-server1
* Use git for code synchronization (remote: `gpu1`)
* Ensure all specified Python dependencies are met
* Run and verify example on CUDA

## 📋 System Standards

* 【Specification】: When using SSH commands for gpu-server1, use jump-win proxy. → Scenario: SSH/SCP operations to gpu-server1. → Reason: Server requires ProxyJump configuration.
* 【Specification】: Use `gpu1` as git remote name for gpu-server1. → Scenario: Git push operations. → Reason: `gpu` remote is occupied by another server.
* 【Specification】: Use `/data/hgt/` as base path on gpu-server1. → Scenario: File storage and conda installation. → Reason: User-designated data partition.

## Specification Points

* Git sync command: `git push gpu1 master`
* Server project path: `/data/hgt/projects/verl_reproduction`
* Conda path: `/data/hgt/miniconda3`
* Active env: `minimind`

## Work Plan

* **Phase Goal**: Extend verl_mini with DPO and ReMax algorithms

* **Previous Phases (Archived)**:
  - Phase 1: Deploy verl_mini to gpu-server1 ✅ Completed
  - Phase 2: Add DPO & ReMax algorithms ✅ Completed

* **Work Steps**:

1. [✓] Completed - Research DPO and ReMax algorithm principles
2. [✓] Completed - Implement DPO (Direct Preference Optimization) algorithm
3. [✓] Completed - Implement ReMax algorithm
4. [✓] Completed - Add algorithm registry mechanism (AlgorithmType, ALGORITHM_REGISTRY)
5. [✓] Completed - Update `__init__.py` exports
6. [✓] Completed - Create example and tests (`example_dpo_remax.py`)
7. [✓] Completed - Sync to gpu-server1 and verify on CUDA

## Work Task

* **Current Step**: Phase 2 completed

* **Thought & Strategy**: Implemented DPO, ReMax, and enhanced GRPO with algorithm registry system

* **Next Action**: Ready for Phase 3 or other development

* **Action Status**: Successful

* **Action Log/Result (Phase 2)**:
  - Added `AlgorithmType` enum and `ALGORITHM_REGISTRY` for algorithm registration
  - Implemented `compute_dpo_loss()` with sigmoid/hinge/ipo loss types and label smoothing
  - Implemented `compute_remax_advantage()` and `compute_remax_loss()` with max-based baseline
  - Enhanced `compute_grpo_loss()` with PPO-style clipping
  - Created `example_dpo_remax.py` demonstrating all new algorithms
  - All tests passed on gpu-server1 with CUDA

---

## Work Status and Results

* **Current Overall Status**: ✅ Completed

* **Key Results Summary**:

| Item | Status | Details |
|------|--------|---------|
| Phase 1 | ✅ | Deployment to gpu-server1 |
| Phase 2 | ✅ | DPO & ReMax algorithms |
| DPO | ✅ | sigmoid/hinge/ipo loss types |
| ReMax | ✅ | max-based baseline advantage |
| GRPO | ✅ | PPO-style clipping |
| Registry | ✅ | `ALGORITHM_REGISTRY` system |
| Tests | ✅ | All examples passed on CUDA |

* **New Files**:
  - `verl_mini/example_dpo_remax.py` - Algorithm demos
  - `develop.md` - Project archive

* **Sync Command**: `git add -A && git commit -m "update" && git push gpu1 master`