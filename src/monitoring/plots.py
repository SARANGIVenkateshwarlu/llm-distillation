"""
Plotting utilities for training visualization.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    log_history: List[Dict],
    output_path: Optional[str] = None,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Plot training and validation curves from log history.
    
    Args:
        log_history: List of log dictionaries from Trainer
        output_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    # Extract metrics
    train_steps = []
    train_loss = []
    eval_steps = []
    eval_loss = []
    learning_rates = []
    
    for log in log_history:
        if "loss" in log and "step" in log:
            train_steps.append(log["step"])
            train_loss.append(log["loss"])
        if "eval_loss" in log and "step" in log:
            eval_steps.append(log["step"])
            eval_loss.append(log["eval_loss"])
        if "learning_rate" in log:
            learning_rates.append(log["learning_rate"])
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot training loss
    if train_loss:
        axes[0, 0].plot(train_steps, train_loss, label="Training Loss", color="blue")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot validation loss
    if eval_loss:
        axes[0, 1].plot(eval_steps, eval_loss, label="Validation Loss", color="orange")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_title("Validation Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot combined losses
    if train_loss and eval_loss:
        axes[1, 0].plot(train_steps, train_loss, label="Training Loss", color="blue", alpha=0.7)
        axes[1, 0].plot(eval_steps, eval_loss, label="Validation Loss", color="orange", alpha=0.7)
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].set_title("Training vs Validation Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot learning rate
    if learning_rates:
        axes[1, 1].plot(learning_rates, label="Learning Rate", color="green")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].set_title("Learning Rate Schedule")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Training curves saved to {output_path}")
    
    return fig


def plot_optuna_study(
    study,
    output_path: Optional[str] = None,
    figsize: tuple = (15, 10),
) -> plt.Figure:
    """
    Plot Optuna study results.
    
    Args:
        study: Optuna study object
        output_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot optimization history
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trial_numbers = [t.number for t in trials]
    values = [t.value for t in trials]
    
    if values:
        # Best value over time
        best_values = np.minimum.accumulate(values) if study.direction == optuna.study.StudyDirection.MINIMIZE else np.maximum.accumulate(values)
        
        axes[0, 0].plot(trial_numbers, values, "o", alpha=0.5, label="Trial Value")
        axes[0, 0].plot(trial_numbers, best_values, "r-", linewidth=2, label="Best Value")
        axes[0, 0].set_xlabel("Trial")
        axes[0, 0].set_ylabel("Objective Value")
        axes[0, 0].set_title("Optimization History")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot parameter importances
    try:
        importances = optuna.importance.get_param_importances(study)
        params = list(importances.keys())
        importance_values = list(importances.values())
        
        axes[0, 1].barh(params, importance_values)
        axes[0, 1].set_xlabel("Importance")
        axes[0, 1].set_title("Parameter Importances")
        axes[0, 1].grid(True, alpha=0.3, axis="x")
    except Exception as e:
        axes[0, 1].text(0.5, 0.5, f"Could not compute importances\n{str(e)}", 
                       ha="center", va="center", transform=axes[0, 1].transAxes)
    
    # Plot slice for most important parameter
    try:
        if params:
            param_name = params[0]
            param_values = [t.params.get(param_name) for t in trials if param_name in t.params]
            param_values_numeric = [v for v in param_values if isinstance(v, (int, float))]
            
            if param_values_numeric:
                trial_vals = [t.value for t in trials if param_name in t.params and isinstance(t.params[param_name], (int, float))]
                axes[1, 0].scatter(param_values_numeric, trial_vals, alpha=0.5)
                axes[1, 0].set_xlabel(param_name)
                axes[1, 0].set_ylabel("Objective Value")
                axes[1, 0].set_title(f"Slice: {param_name}")
                axes[1, 0].grid(True, alpha=0.3)
    except Exception as e:
        axes[1, 0].text(0.5, 0.5, f"Could not plot slice\n{str(e)}", 
                       ha="center", va="center", transform=axes[1, 0].transAxes)
    
    # Plot trial duration
    durations = [t.duration.total_seconds() / 60 for t in trials if t.duration]
    if durations:
        axes[1, 1].bar(trial_numbers[:len(durations)], durations, alpha=0.7)
        axes[1, 1].set_xlabel("Trial")
        axes[1, 1].set_ylabel("Duration (minutes)")
        axes[1, 1].set_title("Trial Durations")
        axes[1, 1].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Optuna plots saved to {output_path}")
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (8, 6),
    normalize: bool = False,
) -> plt.Figure:
    """
    Plot a confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        output_path: Optional path to save figure
        figsize: Figure size
        normalize: Whether to normalize the matrix
    
    Returns:
        Matplotlib figure
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks
    if class_names:
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               ylabel="True label",
               xlabel="Predicted label")
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {output_path}")
    
    return fig


def plot_metric_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot comparison of metrics across different models/experiments.
    
    Args:
        metrics_dict: Dictionary of {name: {metric: value}}
        metric_names: List of metrics to plot
        output_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    if metric_names is None:
        # Get all unique metric names
        metric_names = set()
        for metrics in metrics_dict.values():
            metric_names.update(metrics.keys())
        metric_names = sorted(list(metric_names))
    
    # Prepare data
    names = list(metrics_dict.keys())
    x = np.arange(len(metric_names))
    width = 0.8 / len(names)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, name in enumerate(names):
        values = [metrics_dict[name].get(m, 0) for m in metric_names]
        offset = width * i - (width * len(names) / 2)
        ax.bar(x + offset, values, width, label=name)
    
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Value")
    ax.set_title("Metric Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Metric comparison saved to {output_path}")
    
    return fig


# Import optuna for type hints
try:
    import optuna
except ImportError:
    optuna = None
