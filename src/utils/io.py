"""
Input/Output utility functions for saving and loading artifacts.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


def save_json(data: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        path: Path to save file
        indent: JSON indentation
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        path: Path to JSON file
    
    Returns:
        Loaded dictionary
    """
    with open(path, "r") as f:
        return json.load(f)


def save_pickle(data: Any, path: Union[str, Path]) -> None:
    """
    Save object to pickle file.
    
    Args:
        data: Object to save
        path: Path to save file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: Union[str, Path]) -> Any:
    """
    Load object from pickle file.
    
    Args:
        path: Path to pickle file
    
    Returns:
        Loaded object
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    step: int,
    loss: float,
    path: Union[str, Path],
    additional_data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        epoch: Current epoch
        step: Current step
        loss: Current loss
        path: Path to save checkpoint
        additional_data: Additional data to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if additional_data is not None:
        checkpoint.update(additional_data)
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load checkpoint to
    
    Returns:
        Dictionary with checkpoint metadata
    """
    if device is None:
        device = torch.device("cpu")
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", float("inf"))
    }


def save_model_only(
    model: torch.nn.Module,
    path: Union[str, Path]
) -> None:
    """
    Save only model weights (not full checkpoint).
    
    Args:
        model: Model to save
        path: Path to save file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), path)


def load_model_only(
    model: torch.nn.Module,
    path: Union[str, Path],
    device: Optional[torch.device] = None
) -> None:
    """
    Load only model weights.
    
    Args:
        model: Model to load weights into
        path: Path to weights file
        device: Device to load to
    """
    if device is None:
        device = torch.device("cpu")
    
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """
    Find the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
    
    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.pth"))
    
    if not checkpoints:
        return None
    
    # Sort by modification time
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return latest


def format_size(size_bytes: int) -> str:
    """
    Format byte size to human-readable string.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def get_file_size(path: Union[str, Path]) -> str:
    """
    Get human-readable file size.
    
    Args:
        path: Path to file
    
    Returns:
        Formatted size string
    """
    size_bytes = Path(path).stat().st_size
    return format_size(size_bytes)
