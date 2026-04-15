"""
Random seed management for reproducibility.
"""

import os
import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: The random seed to use
        deterministic: If True, enables deterministic behavior in PyTorch
                      (may impact performance)
    
    Example:
        >>> set_seed(42)
        >>> # Now all random operations will be reproducible
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Environment variable for Hugging Face
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # PyTorch deterministic settings
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Enable cudnn.benchmark for faster training if input sizes don't vary
        torch.backends.cudnn.benchmark = True
    
    # For Hugging Face transformers
    os.environ["TRANSFORMERS_SEED"] = str(seed)


def get_seed() -> Optional[int]:
    """
    Get the current random seed from environment.
    
    Returns:
        The seed if set, None otherwise
    """
    seed_str = os.environ.get("PYTHONHASHSEED")
    return int(seed_str) if seed_str else None


def seed_worker(worker_id: int) -> None:
    """
    Worker initialization function for DataLoader workers.
    
    Use this with PyTorch DataLoader to ensure each worker has a unique seed:
    
    Example:
        >>> loader = DataLoader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=seed_worker
        ... )
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
