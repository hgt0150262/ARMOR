"""
Logging and monitoring utilities for verl_mini.
Supports WandB, TensorBoard, and console logging.
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    
    # General settings
    project_name: str = "verl_mini"
    run_name: Optional[str] = None
    log_dir: str = "./logs"
    
    # Backend selection
    use_wandb: bool = False
    use_tensorboard: bool = True
    use_swanlab: bool = True  # SwanLab离线监控
    use_console: bool = True
    
    # WandB settings
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    
    # SwanLab settings (离线模式)
    swanlab_mode: str = "local"  # "local" for offline, "cloud" for online
    swanlab_log_dir: str = "./swanlog"
    
    # Logging frequency
    log_interval: int = 10  # Log every N steps
    save_interval: int = 100  # Save checkpoint every N steps
    
    # Metrics to track
    track_lr: bool = True
    track_grad_norm: bool = True
    track_memory: bool = True


class MetricsTracker:
    """Track and aggregate metrics over steps."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.step_count = 0
        
    def add(self, name: str, value: float):
        """Add a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        
    def get_mean(self, name: str) -> float:
        """Get mean of a metric."""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return sum(self.metrics[name]) / len(self.metrics[name])
    
    def get_all_means(self) -> Dict[str, float]:
        """Get means of all metrics."""
        return {name: self.get_mean(name) for name in self.metrics}
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        
    def increment_step(self):
        """Increment step counter."""
        self.step_count += 1


class TrainingLogger:
    """Unified logging interface for training."""
    
    def __init__(self, config: LoggingConfig):
        self.config = config
        self.metrics_tracker = MetricsTracker()
        self._step = 0
        self._epoch = 0
        
        # Initialize run name
        if config.run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config.run_name = f"run_{timestamp}"
        
        # Create log directory
        self.log_dir = Path(config.log_dir) / config.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize backends
        self._wandb_run = None
        self._tb_writer = None
        self._swanlab_run = None
        
        if config.use_wandb and WANDB_AVAILABLE:
            self._init_wandb()
        elif config.use_wandb and not WANDB_AVAILABLE:
            print("Warning: WandB requested but not installed. Install with: pip install wandb")
            
        if config.use_tensorboard and TENSORBOARD_AVAILABLE:
            self._init_tensorboard()
        elif config.use_tensorboard and not TENSORBOARD_AVAILABLE:
            print("Warning: TensorBoard requested but not installed. Install with: pip install tensorboard")
            
        if config.use_swanlab and SWANLAB_AVAILABLE:
            self._init_swanlab()
        elif config.use_swanlab and not SWANLAB_AVAILABLE:
            print("Warning: SwanLab requested but not installed. Install with: pip install swanlab")
            
        # Console log file
        self.console_log_path = self.log_dir / "training.log"
        
    def _init_wandb(self):
        """Initialize WandB logging."""
        try:
            self._wandb_run = wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                entity=self.config.wandb_entity,
                tags=self.config.wandb_tags,
                dir=str(self.log_dir),
                reinit=True,
            )
            print(f"WandB initialized: {wandb.run.url}")
        except Exception as e:
            print(f"Failed to initialize WandB: {e}")
            self._wandb_run = None
            
    def _init_tensorboard(self):
        """Initialize TensorBoard logging."""
        try:
            tb_dir = self.log_dir / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
            print(f"TensorBoard initialized: {tb_dir}")
        except Exception as e:
            print(f"Failed to initialize TensorBoard: {e}")
            self._tb_writer = None
            
    def _init_swanlab(self):
        """Initialize SwanLab logging (offline mode by default)."""
        try:
            swanlab_dir = Path(self.config.swanlab_log_dir)
            swanlab_dir.mkdir(parents=True, exist_ok=True)
            self._swanlab_run = swanlab.init(
                project=self.config.project_name,
                experiment_name=self.config.run_name,
                mode=self.config.swanlab_mode,  # "local" for offline
                logdir=str(swanlab_dir),
            )
            print(f"SwanLab initialized (mode={self.config.swanlab_mode}): {swanlab_dir}")
            print(f"View dashboard: swanlab watch {swanlab_dir} --port 5092")
        except Exception as e:
            print(f"Failed to initialize SwanLab: {e}")
            self._swanlab_run = None
            
    def log_config(self, config: Dict[str, Any]):
        """Log configuration at start of training."""
        # Save to file
        config_path = self.log_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
            
        # Log to WandB
        if self._wandb_run:
            wandb.config.update(config)
            
        # Log to console
        if self.config.use_console:
            print("\n" + "="*60)
            print("Training Configuration")
            print("="*60)
            for key, value in config.items():
                print(f"  {key}: {value}")
            print("="*60 + "\n")
            
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to all backends."""
        if step is not None:
            self._step = step
            
        # Track metrics
        for name, value in metrics.items():
            self.metrics_tracker.add(name, value)
            
        # Log to backends if at log interval
        if self._step % self.config.log_interval == 0:
            self._log_to_backends(metrics)
            
    def _log_to_backends(self, metrics: Dict[str, float]):
        """Log metrics to all configured backends."""
        # WandB
        if self._wandb_run:
            wandb.log(metrics, step=self._step)
            
        # TensorBoard
        if self._tb_writer:
            for name, value in metrics.items():
                self._tb_writer.add_scalar(name, value, self._step)
                
        # SwanLab
        if self._swanlab_run:
            swanlab.log(metrics, step=self._step)
                
        # Console
        if self.config.use_console:
            self._log_to_console(metrics)
            
        # File
        self._log_to_file(metrics)
        
    def _log_to_console(self, metrics: Dict[str, float]):
        """Log metrics to console."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"[{timestamp}] Step {self._step} | {metrics_str}")
        
    def _log_to_file(self, metrics: Dict[str, float]):
        """Log metrics to file."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "step": self._step,
            "epoch": self._epoch,
            "metrics": metrics,
        }
        with open(self.console_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
            
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch-level metrics."""
        self._epoch = epoch
        
        if self._wandb_run:
            epoch_metrics = {f"epoch/{k}": v for k, v in metrics.items()}
            epoch_metrics["epoch"] = epoch
            wandb.log(epoch_metrics)
            
        if self._tb_writer:
            for name, value in metrics.items():
                self._tb_writer.add_scalar(f"epoch/{name}", value, epoch)
                
        if self.config.use_console:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch} Summary")
            print(f"{'='*60}")
            for name, value in metrics.items():
                print(f"  {name}: {value:.4f}")
            print(f"{'='*60}\n")
            
    def log_model_info(self, model_name: str, num_params: int, trainable_params: int):
        """Log model information."""
        info = {
            "model_name": model_name,
            "total_params": num_params,
            "trainable_params": trainable_params,
            "trainable_ratio": trainable_params / num_params if num_params > 0 else 0,
        }
        
        if self._wandb_run:
            wandb.config.update(info)
            
        if self.config.use_console:
            print(f"\nModel: {model_name}")
            print(f"  Total params: {num_params:,}")
            print(f"  Trainable params: {trainable_params:,}")
            print(f"  Trainable ratio: {info['trainable_ratio']:.2%}\n")
            
    def log_gpu_memory(self):
        """Log GPU memory usage."""
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1e9
                    reserved = torch.cuda.memory_reserved(i) / 1e9
                    self.log_metrics({
                        f"gpu{i}/memory_allocated_gb": allocated,
                        f"gpu{i}/memory_reserved_gb": reserved,
                    })
        except Exception:
            pass
            
    def step(self):
        """Increment step counter."""
        self._step += 1
        self.metrics_tracker.increment_step()
        
    def close(self):
        """Close all logging backends."""
        if self._wandb_run:
            wandb.finish()
            
        if self._swanlab_run:
            swanlab.finish()
            
        if self._tb_writer:
            self._tb_writer.close()
            
        print(f"\nTraining logs saved to: {self.log_dir}")


class ProgressLogger:
    """Simple progress logger for training loops."""
    
    def __init__(self, total_steps: int, desc: str = "Training"):
        self.total_steps = total_steps
        self.desc = desc
        self.current_step = 0
        self.start_time = time.time()
        self.metrics: Dict[str, float] = {}
        
    def update(self, n: int = 1, **metrics):
        """Update progress."""
        self.current_step += n
        self.metrics.update(metrics)
        self._print_progress()
        
    def _print_progress(self):
        """Print progress bar."""
        elapsed = time.time() - self.start_time
        progress = self.current_step / self.total_steps
        eta = elapsed / progress - elapsed if progress > 0 else 0
        
        bar_len = 30
        filled = int(bar_len * progress)
        bar = '█' * filled + '░' * (bar_len - filled)
        
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in self.metrics.items()])
        
        print(f"\r{self.desc}: |{bar}| {self.current_step}/{self.total_steps} "
              f"[{elapsed:.0f}s<{eta:.0f}s] {metrics_str}", end="")
        
        if self.current_step >= self.total_steps:
            print()  # New line at end
            
    def reset(self):
        """Reset progress."""
        self.current_step = 0
        self.start_time = time.time()
        self.metrics = {}


def create_logger(
    project_name: str = "verl_mini",
    run_name: Optional[str] = None,
    log_dir: str = "./logs",
    use_wandb: bool = False,
    use_tensorboard: bool = True,
) -> TrainingLogger:
    """Factory function to create a training logger."""
    config = LoggingConfig(
        project_name=project_name,
        run_name=run_name,
        log_dir=log_dir,
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
    )
    return TrainingLogger(config)
