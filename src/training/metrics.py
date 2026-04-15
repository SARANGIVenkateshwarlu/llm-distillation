"""
Metrics computation utilities.
"""

import math
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


class MetricCalculator:
    """
    Calculator for various NLP metrics.
    
    Accumulates predictions and labels over batches and computes
    final metrics at the end.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.all_predictions = []
        self.all_labels = []
        self.total_loss = 0.0
        self.total_tokens = 0
        self.num_batches = 0
    
    def add_batch(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        loss: Optional[float] = None,
    ):
        """
        Add a batch of predictions and labels.
        
        Args:
            predictions: Model predictions (logits or token IDs)
            labels: Ground truth labels
            loss: Optional batch loss
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # Store
        self.all_predictions.append(predictions)
        self.all_labels.append(labels)
        
        if loss is not None:
            self.total_loss += loss
        
        self.num_batches += 1
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary of metrics
        """
        if not self.all_predictions:
            return {}
        
        # Concatenate all batches
        predictions = np.concatenate(self.all_predictions, axis=0)
        labels = np.concatenate(self.all_labels, axis=0)
        
        metrics = {}
        
        # Compute perplexity if we have logits
        if len(predictions.shape) == 3:  # [batch, seq_len, vocab_size]
            metrics.update(self._compute_language_modeling_metrics(predictions, labels))
        
        # Compute classification metrics if applicable
        if len(predictions.shape) == 2 and len(labels.shape) == 1:
            metrics.update(self._compute_classification_metrics(predictions, labels))
        
        return metrics
    
    def _compute_language_modeling_metrics(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """Compute metrics for language modeling."""
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        shift_labels = labels[:, 1:].reshape(-1)
        
        # Filter out ignored indices
        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]
        
        if len(shift_labels) == 0:
            return {}
        
        # Compute cross-entropy loss
        log_probs = np.log_softmax(shift_logits, axis=-1)
        nll_loss = -log_probs[np.arange(len(shift_labels)), shift_labels].mean()
        
        # Compute perplexity
        try:
            perplexity = math.exp(nll_loss)
        except OverflowError:
            perplexity = float("inf")
        
        # Compute accuracy
        predictions = np.argmax(shift_logits, axis=-1)
        accuracy = (predictions == shift_labels).mean()
        
        return {
            "loss": float(nll_loss),
            "perplexity": float(perplexity),
            "accuracy": float(accuracy),
        }
    
    def _compute_classification_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """Compute metrics for classification."""
        # Get predicted classes
        pred_classes = np.argmax(predictions, axis=-1)
        
        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(labels, pred_classes),
            "precision": precision_score(labels, pred_classes, average="weighted", zero_division=0),
            "recall": recall_score(labels, pred_classes, average="weighted", zero_division=0),
            "f1": f1_score(labels, pred_classes, average="weighted", zero_division=0),
        }
        
        return metrics


def compute_all_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    task_type: str = "language_modeling",
) -> Dict[str, float]:
    """
    Compute all applicable metrics.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        task_type: Type of task
    
    Returns:
        Dictionary of metrics
    """
    calculator = MetricCalculator()
    calculator.add_batch(predictions, labels)
    return calculator.compute()


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss
    
    Returns:
        Perplexity value
    """
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def compute_token_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute token-level accuracy.
    
    Args:
        predictions: Predicted token IDs
        labels: Ground truth token IDs
        ignore_index: Index to ignore
    
    Returns:
        Accuracy as a float
    """
    # Flatten
    predictions = predictions.view(-1)
    labels = labels.view(-1)
    
    # Mask ignored indices
    mask = labels != ignore_index
    predictions = predictions[mask]
    labels = labels[mask]
    
    if len(labels) == 0:
        return 0.0
    
    correct = (predictions == labels).sum().item()
    total = len(labels)
    
    return correct / total


def compute_sequence_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute sequence-level accuracy.
    
    Args:
        predictions: Predicted token IDs [batch, seq_len]
        labels: Ground truth token IDs [batch, seq_len]
        ignore_index: Index to ignore
    
    Returns:
        Accuracy as a float
    """
    batch_size = predictions.shape[0]
    correct_sequences = 0
    
    for i in range(batch_size):
        pred_seq = predictions[i]
        label_seq = labels[i]
        
        # Mask ignored indices
        mask = label_seq != ignore_index
        pred_seq = pred_seq[mask]
        label_seq = label_seq[mask]
        
        # Check if entire sequence matches
        if torch.equal(pred_seq, label_seq):
            correct_sequences += 1
    
    return correct_sequences / batch_size


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Format metrics for display.
    
    Args:
        metrics: Dictionary of metrics
        precision: Decimal precision
    
    Returns:
        Formatted string
    """
    lines = []
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.{precision}f}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        title: Title to display
    """
    print("=" * 50)
    print(title)
    print("=" * 50)
    print(format_metrics(metrics))
    print("=" * 50)
