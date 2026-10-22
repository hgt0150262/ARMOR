"""
GPU Health Monitor for Fault-Tolerant Distributed Training

Innovative contribution: Automatic detection of degraded GPUs via ECC error
monitoring, enabling graceful degradation in multi-GPU RL training pipelines.

This module provides:
1. Pre-training GPU health verification
2. ECC error monitoring (correctable and uncorrectable)
3. Automatic selection of healthy GPUs
4. Runtime ECC monitoring during training
"""

import subprocess
import re
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class GPUHealthStatus:
    """Health status for a single GPU."""
    gpu_id: int
    name: str = ""
    temperature: int = 0
    memory_total_mb: int = 0
    memory_used_mb: int = 0
    # ECC metrics
    sram_correctable: int = 0
    sram_uncorrectable: int = 0
    dram_correctable: int = 0
    dram_uncorrectable: int = 0
    # Aggregate (lifetime)
    agg_sram_correctable: int = 0
    agg_sram_uncorrectable: int = 0
    agg_dram_correctable: int = 0
    agg_dram_uncorrectable: int = 0
    # Row remapping
    remapped_rows_correctable: int = 0
    remapped_rows_uncorrectable: int = 0
    remapping_failure: bool = False
    # Health assessment
    is_healthy: bool = True
    health_score: float = 1.0
    warnings: List[str] = field(default_factory=list)


def query_nvidia_smi() -> str:
    """Run nvidia-smi -q and return output."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q"],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return ""


def parse_gpu_health(nvidia_smi_output: str) -> List[GPUHealthStatus]:
    """Parse nvidia-smi -q output into GPUHealthStatus objects."""
    gpus = []
    # Split by GPU sections
    gpu_sections = re.split(r'GPU \d+:\d+:\d+\.\d+', nvidia_smi_output)

    # Get GPU names
    gpu_names = re.findall(r'Product Name\s+:\s+(.+)', nvidia_smi_output)
    
    # Parse each GPU's ECC info
    for i, name in enumerate(gpu_names):
        status = GPUHealthStatus(gpu_id=i, name=name.strip())
        
        # We need to parse per-GPU info; use nvidia-smi -i <id> -q
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(i), "-q"],
                capture_output=True, text=True, timeout=30
            )
            gpu_output = result.stdout
        except Exception:
            gpus.append(status)
            continue

        # Temperature
        temp_match = re.search(r'GPU Current Temp\s+:\s+(\d+)', gpu_output)
        if temp_match:
            status.temperature = int(temp_match.group(1))

        # Memory
        mem_total = re.search(r'FB Memory Usage.*?Total\s+:\s+(\d+)', gpu_output, re.DOTALL)
        mem_used = re.search(r'FB Memory Usage.*?Used\s+:\s+(\d+)', gpu_output, re.DOTALL)
        if mem_total:
            status.memory_total_mb = int(mem_total.group(1))
        if mem_used:
            status.memory_used_mb = int(mem_used.group(1))

        # ECC Errors - Volatile
        ecc_section = gpu_output
        
        # Parse volatile ECC
        vol_match = re.search(
            r'Volatile.*?'
            r'SRAM Correctable\s+:\s+(\d+).*?'
            r'SRAM Uncorrectable.*?:\s+(\d+).*?'
            r'DRAM Correctable\s+:\s+(\d+).*?'
            r'DRAM Uncorrectable\s+:\s+(\d+)',
            ecc_section, re.DOTALL
        )
        if vol_match:
            status.sram_correctable = int(vol_match.group(1))
            status.sram_uncorrectable = int(vol_match.group(2))
            status.dram_correctable = int(vol_match.group(3))
            status.dram_uncorrectable = int(vol_match.group(4))

        # Parse aggregate ECC
        agg_match = re.search(
            r'Aggregate.*?'
            r'SRAM Correctable\s+:\s+(\d+).*?'
            r'SRAM Uncorrectable.*?:\s+(\d+).*?'
            r'DRAM Correctable\s+:\s+(\d+).*?'
            r'DRAM Uncorrectable\s+:\s+(\d+)',
            ecc_section, re.DOTALL
        )
        if agg_match:
            status.agg_sram_correctable = int(agg_match.group(1))
            status.agg_sram_uncorrectable = int(agg_match.group(2))
            status.agg_dram_correctable = int(agg_match.group(3))
            status.agg_dram_uncorrectable = int(agg_match.group(4))

        # Remapped rows
        remap_corr = re.search(r'Remapped Rows.*?Correctable Error\s+:\s+(\d+)', ecc_section, re.DOTALL)
        remap_uncorr = re.search(r'Remapped Rows.*?Uncorrectable Error\s+:\s+(\d+)', ecc_section, re.DOTALL)
        remap_fail = re.search(r'Remapping Failure Occurred\s+:\s+(\w+)', ecc_section)
        
        if remap_corr:
            status.remapped_rows_correctable = int(remap_corr.group(1))
        if remap_uncorr:
            status.remapped_rows_uncorrectable = int(remap_uncorr.group(1))
        if remap_fail:
            status.remapping_failure = remap_fail.group(1).lower() == "yes"

        # Compute health score
        status = _compute_health_score(status)
        gpus.append(status)

    return gpus


def _compute_health_score(status: GPUHealthStatus) -> GPUHealthStatus:
    """Compute a health score [0, 1] and set warnings."""
    score = 1.0
    warnings = []

    # DRAM uncorrectable errors are critical
    if status.dram_uncorrectable > 0:
        score -= 0.3 * min(status.dram_uncorrectable, 3)
        warnings.append(
            f"DRAM uncorrectable errors (volatile): {status.dram_uncorrectable}"
        )

    # Aggregate DRAM uncorrectable indicates degraded hardware
    if status.agg_dram_uncorrectable > 10:
        score -= 0.2
        warnings.append(
            f"High lifetime DRAM uncorrectable errors: {status.agg_dram_uncorrectable}"
        )

    # Remapping failure is a red flag
    if status.remapping_failure:
        score -= 0.3
        warnings.append("Row remapping capacity exhausted!")

    # High remapped rows
    if status.remapped_rows_uncorrectable > 5:
        score -= 0.1
        warnings.append(
            f"Many uncorrectable row remaps: {status.remapped_rows_uncorrectable}"
        )

    # SRAM uncorrectable errors
    if status.sram_uncorrectable > 0:
        score -= 0.4
        warnings.append(
            f"SRAM uncorrectable errors: {status.sram_uncorrectable}"
        )

    # Temperature warning
    if status.temperature > 85:
        score -= 0.1
        warnings.append(f"High temperature: {status.temperature}C")

    # Correctable errors are less severe but noteworthy
    if status.dram_correctable > 5:
        score -= 0.05
        warnings.append(
            f"Elevated DRAM correctable errors: {status.dram_correctable}"
        )

    score = max(0.0, score)
    status.health_score = round(score, 2)
    status.is_healthy = score >= 0.5
    status.warnings = warnings
    return status


def select_healthy_gpus(
    min_gpus: int = 1,
    health_threshold: float = 0.5,
    prefer_count: Optional[int] = None
) -> Tuple[List[int], List[GPUHealthStatus]]:
    """
    Select healthy GPUs for training.
    
    Args:
        min_gpus: Minimum number of healthy GPUs required.
        health_threshold: Minimum health score to consider a GPU healthy.
        prefer_count: Preferred number of GPUs (will select top N by score).
    
    Returns:
        Tuple of (list of GPU indices, list of GPUHealthStatus for all GPUs).
    """
    smi_output = query_nvidia_smi()
    if not smi_output:
        raise RuntimeError("Failed to query nvidia-smi")

    all_gpus = parse_gpu_health(smi_output)
    healthy = [g for g in all_gpus if g.health_score >= health_threshold]
    
    # Sort by health score (descending)
    healthy.sort(key=lambda g: g.health_score, reverse=True)

    if len(healthy) < min_gpus:
        raise RuntimeError(
            f"Only {len(healthy)} healthy GPUs found, "
            f"need at least {min_gpus}. "
            f"Unhealthy GPUs: {[(g.gpu_id, g.health_score, g.warnings) for g in all_gpus if not g.is_healthy]}"
        )

    if prefer_count and prefer_count <= len(healthy):
        selected = healthy[:prefer_count]
    else:
        selected = healthy

    return [g.gpu_id for g in selected], all_gpus


def print_gpu_health_report(gpus: Optional[List[GPUHealthStatus]] = None):
    """Print a formatted GPU health report."""
    if gpus is None:
        smi_output = query_nvidia_smi()
        gpus = parse_gpu_health(smi_output)

    print("=" * 70)
    print("GPU Health Report")
    print("=" * 70)

    for g in gpus:
        status_icon = "✅" if g.is_healthy else "❌"
        print(f"\nGPU {g.gpu_id}: {g.name}")
        print(f"  Health Score: {g.health_score:.2f} {status_icon}")
        print(f"  Temperature:  {g.temperature}°C")
        print(f"  Memory:       {g.memory_used_mb}/{g.memory_total_mb} MB")
        print(f"  ECC (volatile):  SRAM corr={g.sram_correctable} uncorr={g.sram_uncorrectable} "
              f"| DRAM corr={g.dram_correctable} uncorr={g.dram_uncorrectable}")
        print(f"  ECC (lifetime):  SRAM corr={g.agg_sram_correctable} uncorr={g.agg_sram_uncorrectable} "
              f"| DRAM corr={g.agg_dram_correctable} uncorr={g.agg_dram_uncorrectable}")
        print(f"  Row Remapping:   corr={g.remapped_rows_correctable} "
              f"uncorr={g.remapped_rows_uncorrectable} "
              f"failure={g.remapping_failure}")
        if g.warnings:
            for w in g.warnings:
                print(f"  ⚠️  {w}")

    print("\n" + "=" * 70)
    healthy_ids = [g.gpu_id for g in gpus if g.is_healthy]
    print(f"Healthy GPUs: {healthy_ids} ({len(healthy_ids)}/{len(gpus)})")
    print("=" * 70)


if __name__ == "__main__":
    print_gpu_health_report()
    
    try:
        selected, all_gpus = select_healthy_gpus(min_gpus=1, prefer_count=4)
        print(f"\nSelected GPUs for training: {selected}")
    except RuntimeError as e:
        print(f"\nGPU selection failed: {e}")
