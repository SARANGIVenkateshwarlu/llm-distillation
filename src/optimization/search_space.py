"""
Hyperparameter search space definitions for Optuna.
"""

from typing import Any, Dict

import optuna

from src.config import OptunaSearchSpace


def get_search_space(config: OptunaSearchSpace) -> Dict[str, Dict[str, Any]]:
    """
    Get the search space as a dictionary.
    
    Args:
        config: Search space configuration
    
    Returns:
        Dictionary defining the search space
    """
    return {
        "learning_rate": {
            "type": "float",
            "low": config.learning_rate[0],
            "high": config.learning_rate[1],
            "log": True,
        },
        "weight_decay": {
            "type": "float",
            "low": config.weight_decay[0],
            "high": config.weight_decay[1],
            "log": False,
        },
        "lora_r": {
            "type": "categorical",
            "choices": config.lora_r,
        },
        "lora_alpha": {
            "type": "categorical",
            "choices": config.lora_alpha,
        },
        "lora_dropout": {
            "type": "float",
            "low": config.lora_dropout[0],
            "high": config.lora_dropout[1],
            "log": False,
        },
        "temperature": {
            "type": "float",
            "low": config.temperature[0],
            "high": config.temperature[1],
            "log": False,
        },
        "alpha": {
            "type": "float",
            "low": config.alpha[0],
            "high": config.alpha[1],
            "log": False,
        },
        "beta": {
            "type": "float",
            "low": config.beta[0],
            "high": config.beta[1],
            "log": False,
        },
        "per_device_train_batch_size": {
            "type": "categorical",
            "choices": config.per_device_train_batch_size,
        },
        "gradient_accumulation_steps": {
            "type": "categorical",
            "choices": config.gradient_accumulation_steps,
        },
        "num_train_epochs": {
            "type": "categorical",
            "choices": config.num_train_epochs,
        },
        "warmup_ratio": {
            "type": "float",
            "low": config.warmup_ratio[0],
            "high": config.warmup_ratio[1],
            "log": False,
        },
        "max_length": {
            "type": "categorical",
            "choices": config.max_length,
        },
    }


def suggest_parameters(
    trial: optuna.Trial,
    search_space: OptunaSearchSpace,
) -> Dict[str, Any]:
    """
    Suggest parameters for a trial based on the search space.
    
    Args:
        trial: Optuna trial object
        search_space: Search space configuration
    
    Returns:
        Dictionary of suggested parameters
    """
    space = get_search_space(search_space)
    params = {}
    
    for name, config in space.items():
        if config["type"] == "float":
            params[name] = trial.suggest_float(
                name,
                config["low"],
                config["high"],
                log=config.get("log", False),
            )
        elif config["type"] == "int":
            params[name] = trial.suggest_int(
                name,
                config["low"],
                config["high"],
                log=config.get("log", False),
            )
        elif config["type"] == "categorical":
            params[name] = trial.suggest_categorical(
                name,
                config["choices"],
            )
    
    # Ensure alpha + beta = 1.0
    if "alpha" in params and "beta" in params:
        # Normalize to ensure they sum to 1
        total = params["alpha"] + params["beta"]
        params["alpha"] = params["alpha"] / total
        params["beta"] = params["beta"] / total
    
    return params


def get_default_search_space() -> Dict[str, Any]:
    """
    Get a default search space for quick experimentation.
    
    Returns:
        Dictionary with default parameter ranges
    """
    return {
        "learning_rate": {"low": 1e-5, "high": 5e-4, "log": True},
        "weight_decay": {"low": 0.0, "high": 0.1, "log": False},
        "lora_r": {"choices": [8, 16, 32]},
        "lora_alpha": {"choices": [16, 32, 64]},
        "lora_dropout": {"low": 0.0, "high": 0.1, "log": False},
        "temperature": {"low": 1.0, "high": 4.0, "log": False},
        "alpha": {"low": 0.2, "high": 0.4, "log": False},
        "beta": {"low": 0.6, "high": 0.8, "log": False},
        "per_device_train_batch_size": {"choices": [1, 2]},
        "gradient_accumulation_steps": {"choices": [4, 8, 16]},
        "num_train_epochs": {"choices": [1, 2, 3]},
        "warmup_ratio": {"low": 0.0, "high": 0.1, "log": False},
        "max_length": {"choices": [256, 512]},
    }


def get_narrow_search_space(
    center_params: Dict[str, Any],
    variance: float = 0.5,
) -> Dict[str, Any]:
    """
    Get a narrow search space around known good parameters.
    
    This is useful for Round 2 optimization after Round 1 has
    identified a promising region.
    
    Args:
        center_params: Center point parameters from Round 1
        variance: How much to vary around center (0.5 = 50%)
    
    Returns:
        Narrow search space dictionary
    """
    space = {}
    
    for key, value in center_params.items():
        if isinstance(value, float):
            if key in ["learning_rate"]:
                # Log scale for learning rate
                log_value = np.log10(value)
                log_low = log_value - variance
                log_high = log_value + variance
                space[key] = {
                    "low": 10 ** log_low,
                    "high": 10 ** log_high,
                    "log": True,
                }
            else:
                # Linear scale
                low = max(0, value * (1 - variance))
                high = value * (1 + variance)
                space[key] = {"low": low, "high": high, "log": False}
        elif isinstance(value, int):
            # For integers, create a range around the value
            low = max(1, int(value * (1 - variance)))
            high = max(low + 1, int(value * (1 + variance)))
            space[key] = {"low": low, "high": high, "log": False}
    
    return space


# Import numpy for get_narrow_search_space
try:
    import numpy as np
except ImportError:
    pass
