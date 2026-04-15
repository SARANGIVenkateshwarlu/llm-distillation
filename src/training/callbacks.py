"""
Custom callbacks for training.
"""

import os
import time
from typing import Dict, Optional

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class LoggingCallback(TrainerCallback):
    """
    Custom callback for logging training progress.
    
    Logs:
    - Training loss
    - Learning rate
    - Epoch progress
    - Time per step
    """
    
    def __init__(
        self,
        log_every_n_steps: int = 10,
        log_dir: Optional[str] = None,
    ):
        self.log_every_n_steps = log_every_n_steps
        self.log_dir = log_dir
        self.start_time = None
        self.step_start_time = None
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Called at the beginning of training."""
        self.start_time = time.time()
        print("=" * 60)
        print("Training Started")
        print("=" * 60)
        print(f"Total steps: {state.max_steps}")
        print(f"Total epochs: {args.num_train_epochs}")
        print(f"Batch size: {args.per_device_train_batch_size}")
        print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        print("=" * 60)
    
    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Called at the beginning of each step."""
        self.step_start_time = time.time()
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Called at the end of each step."""
        if state.global_step % self.log_every_n_steps == 0:
            step_time = time.time() - self.step_start_time
            elapsed = time.time() - self.start_time
            
            # Get current loss
            loss = state.log_history[-1].get("loss", 0) if state.log_history else 0
            lr = state.log_history[-1].get("learning_rate", 0) if state.log_history else 0
            
            # Compute progress
            progress = state.global_step / state.max_steps * 100 if state.max_steps > 0 else 0
            
            # Estimate remaining time
            if state.global_step > 0:
                avg_step_time = elapsed / state.global_steps
                remaining_steps = state.max_steps - state.global_step
                eta = avg_step_time * remaining_steps
                eta_str = self._format_time(eta)
            else:
                eta_str = "N/A"
            
            print(
                f"Step {state.global_step}/{state.max_steps} "
                f"({progress:.1f}%) | "
                f"Loss: {loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Step time: {step_time:.3f}s | "
                f"ETA: {eta_str}"
            )
    
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """Called after evaluation."""
        if metrics:
            print("\n" + "=" * 60)
            print("Evaluation Results")
            print("=" * 60)
            for key, value in sorted(metrics.items()):
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            print("=" * 60 + "\n")
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Called at the end of training."""
        total_time = time.time() - self.start_time
        print("\n" + "=" * 60)
        print("Training Completed")
        print("=" * 60)
        print(f"Total time: {self._format_time(total_time)}")
        print(f"Total steps: {state.global_step}")
        print(f"Best model checkpoint: {state.best_model_checkpoint}")
        print("=" * 60)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


class EarlyStoppingCallback(TrainerCallback):
    """
    Early stopping callback to stop training when validation loss stops improving.
    
    Args:
        early_stopping_patience: Number of evaluations with no improvement after which training stops
        early_stopping_threshold: Minimum change to qualify as an improvement
    """
    
    def __init__(
        self,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.001,
    ):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.best_metric = None
        self.counter = 0
    
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """Check if training should stop."""
        if metrics is None:
            return
        
        # Get the metric to monitor
        metric_key = args.metric_for_best_model
        if metric_key not in metrics:
            return
        
        metric_value = metrics[metric_key]
        
        # Initialize best metric
        if self.best_metric is None:
            self.best_metric = metric_value
            return
        
        # Check for improvement
        if args.greater_is_better:
            improved = metric_value > self.best_metric + self.early_stopping_threshold
        else:
            improved = metric_value < self.best_metric - self.early_stopping_threshold
        
        if improved:
            self.best_metric = metric_value
            self.counter = 0
            print(f"\nNew best {metric_key}: {metric_value:.4f}")
        else:
            self.counter += 1
            print(f"\nNo improvement for {self.counter} evaluation(s)")
            
            if self.counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {self.counter} evaluations without improvement")
                control.should_training_stop = True


class SaveCheckpointCallback(TrainerCallback):
    """
    Custom callback for saving checkpoints with additional metadata.
    """
    
    def __init__(
        self,
        save_every_n_steps: Optional[int] = None,
        save_every_n_epochs: Optional[int] = None,
    ):
        self.save_every_n_steps = save_every_n_steps
        self.save_every_n_epochs = save_every_n_epochs
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Save checkpoint at specified step intervals."""
        if self.save_every_n_steps and state.global_step % self.save_every_n_steps == 0:
            control.should_save = True
    
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Save checkpoint at specified epoch intervals."""
        if self.save_every_n_epochs and state.epoch % self.save_every_n_epochs == 0:
            control.should_save = True


class GradientNormCallback(TrainerCallback):
    """
    Callback to monitor gradient norms during training.
    """
    
    def __init__(self, log_every_n_steps: int = 100):
        self.log_every_n_steps = log_every_n_steps
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Log gradient norms."""
        if state.global_step % self.log_every_n_steps == 0:
            model = kwargs.get("model")
            if model:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"  Gradient norm: {total_norm:.4f}")


class MemoryMonitorCallback(TrainerCallback):
    """
    Callback to monitor GPU memory usage during training.
    """
    
    def __init__(self, log_every_n_steps: int = 100):
        self.log_every_n_steps = log_every_n_steps
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Log GPU memory usage."""
        if state.global_step % self.log_every_n_steps == 0:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
