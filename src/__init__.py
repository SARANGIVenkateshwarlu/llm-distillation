"""
LLM Knowledge Distillation Project

A production-grade implementation of teacher-student fine-tuning
with knowledge distillation for large language models.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from src.config import Config, load_config
from src.utils.seed import set_seed

__all__ = [
    "Config",
    "load_config",
    "set_seed",
]
