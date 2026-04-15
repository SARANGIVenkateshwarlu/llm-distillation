"""
Optuna hyperparameter optimization for knowledge distillation.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from src.config import Config
from src.optimization.search_space import suggest_parameters


# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_optuna_study(
    study_name: str,
    direction: str = "minimize",
    seed: int = 42,
) -> optuna.Study:
    """
    Create an Optuna study for hyperparameter optimization.
    
    Args:
        study_name: Name of the study
        direction: "minimize" or "maximize"
        seed: Random seed for reproducibility
    
    Returns:
        Optuna study object
    """
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
    )
    
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )
    
    return study


def run_optuna_optimization(
    objective_func: Callable[[optuna.Trial], float],
    study_name: str,
    n_trials: int = 20,
    timeout: Optional[int] = None,
    direction: str = "minimize",
    seed: int = 42,
    storage: Optional[str] = None,
    show_progress_bar: bool = True,
) -> optuna.Study:
    """
    Run Optuna hyperparameter optimization.
    
    Args:
        objective_func: Objective function to optimize
        study_name: Name of the study
        n_trials: Number of trials to run
        timeout: Timeout in seconds (None for no timeout)
        direction: "minimize" or "maximize"
        seed: Random seed
        storage: Storage URL for study persistence
        show_progress_bar: Whether to show progress bar
    
    Returns:
        Completed Optuna study
    
    Example:
        >>> def objective(trial):
        ...     lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        ...     # ... train model and return validation loss
        ...     return val_loss
        >>> study = run_optuna_optimization(objective, "my_study", n_trials=50)
    """
    print("=" * 60)
    print(f"Starting Optuna Optimization: {study_name}")
    print("=" * 60)
    print(f"Number of trials: {n_trials}")
    print(f"Timeout: {timeout}s" if timeout else "Timeout: None")
    print(f"Direction: {direction}")
    print("=" * 60)
    
    # Create or load study
    if storage:
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=TPESampler(seed=seed),
            pruner=MedianPruner(),
            storage=storage,
            load_if_exists=True,
        )
    else:
        study = create_optuna_study(study_name, direction, seed)
    
    # Run optimization
    study.optimize(
        objective_func,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress_bar,
        catch=(Exception,),
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Optimization Complete")
    print("=" * 60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    return study


def create_kd_objective(
    train_dataset: Any,
    eval_dataset: Any,
    teacher_model: Any,
    student_model_name: str,
    config: Config,
    max_epochs: int = 1,
) -> Callable[[optuna.Trial], float]:
    """
    Create an objective function for knowledge distillation hyperparameter search.
    
    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        teacher_model: Teacher model
        student_model_name: Student model name
        config: Base configuration
        max_epochs: Maximum epochs per trial
    
    Returns:
        Objective function for Optuna
    """
    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        params = suggest_parameters(trial, config.optuna.search_space)
        
        # Update config with suggested parameters
        trial_config = Config.from_dict(config.to_dict())
        trial_config.training.learning_rate = params["learning_rate"]
        trial_config.training.weight_decay = params["weight_decay"]
        trial_config.training.num_train_epochs = params["num_train_epochs"]
        trial_config.training.warmup_ratio = params["warmup_ratio"]
        trial_config.training.per_device_train_batch_size = params["per_device_train_batch_size"]
        trial_config.training.gradient_accumulation_steps = params["gradient_accumulation_steps"]
        trial_config.lora.r = params["lora_r"]
        trial_config.lora.lora_alpha = params["lora_alpha"]
        trial_config.lora.lora_dropout = params["lora_dropout"]
        trial_config.distillation.temperature = params["temperature"]
        trial_config.distillation.alpha = params["alpha"]
        trial_config.distillation.beta = params["beta"]
        trial_config.tokenization.max_length = params["max_length"]
        
        # Train and evaluate
        try:
            from src.models.student_loader import load_student_model
            from src.models.distillation import create_distillation_trainer
            from src.data.collators import DataCollatorForCausalLM
            
            # Load student with suggested LoRA config
            student, tokenizer = load_student_model(
                config=trial_config.models["student"],
                lora_config=trial_config.lora,
            )
            
            # Create data collator
            data_collator = DataCollatorForCausalLM(
                tokenizer=tokenizer,
                max_length=trial_config.tokenization.max_length,
            )
            
            # Create trainer
            trainer = create_distillation_trainer(
                student_model=student,
                teacher_model=teacher_model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                output_dir=f"./artifacts/optuna/{trial.number}",
                temperature=trial_config.distillation.temperature,
                alpha=trial_config.distillation.alpha,
                beta=trial_config.distillation.beta,
                num_train_epochs=min(params["num_train_epochs"], max_epochs),
                per_device_train_batch_size=params["per_device_train_batch_size"],
                gradient_accumulation_steps=params["gradient_accumulation_steps"],
                learning_rate=params["learning_rate"],
                weight_decay=params["weight_decay"],
                warmup_ratio=params["warmup_ratio"],
                logging_steps=50,
                eval_steps=100,
                save_steps=500,
            )
            
            # Train
            trainer.train()
            
            # Evaluate
            metrics = trainer.evaluate()
            eval_loss = metrics.get("eval_loss", float("inf"))
            
            # Clean up
            import gc
            del student
            del trainer
            gc.collect()
            import torch
            torch.cuda.empty_cache()
            
            return eval_loss
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return float("inf")
    
    return objective


def save_best_params(
    study: optuna.Study,
    output_path: str,
) -> None:
    """
    Save best parameters from a study to a JSON file.
    
    Args:
        study: Completed Optuna study
        output_path: Path to save JSON file
    """
    best_params = {
        "study_name": study.study_name,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(best_params, f, indent=2)
    
    print(f"Best parameters saved to {output_path}")


def load_best_params(path: str) -> Dict[str, Any]:
    """
    Load best parameters from a JSON file.
    
    Args:
        path: Path to JSON file
    
    Returns:
        Dictionary with best parameters
    """
    with open(path, "r") as f:
        data = json.load(f)
    
    return data["best_params"]


def apply_best_params_to_config(
    config: Config,
    best_params: Dict[str, Any],
) -> Config:
    """
    Apply best parameters from Optuna to a configuration.
    
    Args:
        config: Base configuration
        best_params: Best parameters from Optuna
    
    Returns:
        Updated configuration
    """
    config.training.learning_rate = best_params.get("learning_rate", config.training.learning_rate)
    config.training.weight_decay = best_params.get("weight_decay", config.training.weight_decay)
    config.training.num_train_epochs = best_params.get("num_train_epochs", config.training.num_train_epochs)
    config.training.warmup_ratio = best_params.get("warmup_ratio", config.training.warmup_ratio)
    config.training.per_device_train_batch_size = best_params.get("per_device_train_batch_size", config.training.per_device_train_batch_size)
    config.training.gradient_accumulation_steps = best_params.get("gradient_accumulation_steps", config.training.gradient_accumulation_steps)
    config.lora.r = best_params.get("lora_r", config.lora.r)
    config.lora.lora_alpha = best_params.get("lora_alpha", config.lora.lora_alpha)
    config.lora.lora_dropout = best_params.get("lora_dropout", config.lora.lora_dropout)
    config.distillation.temperature = best_params.get("temperature", config.distillation.temperature)
    config.distillation.alpha = best_params.get("alpha", config.distillation.alpha)
    config.distillation.beta = best_params.get("beta", config.distillation.beta)
    config.tokenization.max_length = best_params.get("max_length", config.tokenization.max_length)
    
    return config
