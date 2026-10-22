"""
Reproducibility utilities for ARMOR experiments.

Provides global RNG seeding and run-manifest collection so that every
reported experiment can be traced to its code commit, configuration,
random seed, and runtime environment.
"""

import json
import os
import platform
import random
import socket
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all relevant RNGs (python random, numpy, torch CPU/GPU).

    Applied identically on every distributed rank so that model
    initialization is identical across ranks (required by DDP).
    Rollout diversity still comes from per-rank prompts and sampling.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_repo_root() -> str:
    """Walk up from this file to locate the repository root (.git)."""
    path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.isdir(os.path.join(path, ".git")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    # Fallback: ARMOR/utils/<file> -> repository root
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _git_state(repo_root: str) -> dict:
    """Return the current commit hash and working-tree dirtiness."""
    state = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=10,
        )
        if commit.returncode == 0:
            state["commit"] = commit.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root, capture_output=True, text=True, timeout=10,
        )
        if status.returncode == 0:
            state["dirty"] = bool(status.stdout.strip())
    except Exception:
        pass
    return state


def _package_versions(packages) -> dict:
    """Best-effort version lookup for a list of installed packages."""
    from importlib import metadata
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except Exception:
            versions[name] = None
    return versions


def collect_run_manifest(seed: int, config: dict, extra: dict = None) -> dict:
    """Collect a machine-readable record of the run's provenance.

    Includes timestamp, host, seed, git commit/dirty state, python and
    package versions, CUDA environment, full CLI arguments, and any
    caller-provided extra fields.
    """
    gpu_names = []
    try:
        if torch.cuda.is_available():
            gpu_names = [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]
    except Exception:
        pass

    manifest = {
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "seed": seed,
        "git": _git_state(_find_repo_root()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(
            ["torch", "transformers", "peft", "numpy", "accelerate", "datasets"]
        ),
        "cuda": {
            "available": torch.cuda.is_available(),
            "version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": gpu_names,
        },
        "argv": sys.argv,
        "config": config,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_run_manifest(path: str, seed: int, config: dict, extra: dict = None) -> dict:
    """Collect the run manifest and write it to `path` as JSON."""
    manifest = collect_run_manifest(seed=seed, config=config, extra=extra)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest
