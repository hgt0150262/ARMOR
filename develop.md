# Project Archive (develop.md)

## Project Overview

* **Project**: verl_mini - Simplified RLHF Framework Reproduction
* **Repository**: `F:\LLM\reproduce\verl_reproduction`
* **Remote**: `gpu1` → `gpu-server1:/data/hgt/projects/verl_reproduction`

## History Archive

### Phase 1: Initial Deployment (2025-01-20)

**Phase Goal**: Deploy verl_mini RLHF framework reproduction to gpu-server1

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

**Phase Goal**: Extend verl_mini with DPO and ReMax algorithms

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
- `verl_mini/example_dpo_remax.py`

---

## Next Phase Planning

**Potential Phase 3 directions**:
1. Integrate with real LLM models (Qwen, LLaMA)
2. Add distributed training support with Ray
3. Implement full training pipeline with real data
4. Add evaluation and benchmarking suite
5. Implement KTO (Kahneman-Tversky Optimization) algorithm
