"""
Logging utilities for the project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to log to
        format_string: Custom format string
    
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Get logger
    logger = logging.getLogger("llm_distillation")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "llm_distillation") -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TensorBoardLogger:
    """
    Simple wrapper for TensorBoard logging.
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: dict, step: int):
        """Log multiple scalar values."""
        if self.enabled:
            self.writer.add_scalars(tag, values, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram."""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def close(self):
        """Close the writer."""
        if self.enabled:
            self.writer.close()


class MetricsTracker:
    """
    Track metrics during training.
    """
    
    def __init__(self):
        self.metrics = {}
    
    def add(self, name: str, value: float, step: int):
        """
        Add a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Step number
        """
        if name not in self.metrics:
            self.metrics[name] = {"steps": [], "values": []}
        
        self.metrics[name]["steps"].append(step)
        self.metrics[name]["values"].append(value)
    
    def get(self, name: str) -> dict:
        """Get metric data."""
        return self.metrics.get(name, {"steps": [], "values": []})
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.metrics
    
    def save(self, path: str):
        """Save metrics to JSON."""
        import json
        
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)
    
    def load(self, path: str):
        """Load metrics from JSON."""
        import json
        
        with open(path, "r") as f:
            self.metrics = json.load(f)
