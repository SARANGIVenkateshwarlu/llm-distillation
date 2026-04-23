#!/usr/bin/env python
"""
Round 1 Optuna optimization - Quick search for good hyperparameters.

Usage:
    python scripts/optimize_round1.py --config configs/default.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config, load_config
from src.data.collators import DataCollatorForCausalLM
from src.data.dataset_loader import load_and_prepare_dataset
from src.data.preprocessing import preprocess_dataset
from src.data.tokenization import tokenize_dataset, get_tokenizer
from src.models.distillation import create_distillation_trainer
from src.models.student_loader import load_student_model
from src.models.teacher_loader import load_teacher_model
from src.optimization.optuna_search import (
    run_optuna_optimization,
    save_best_params,
    apply_best_params_to_config,
)
from src.optimization.search_space import suggest_parameters
from src.utils.seed import set_seed
from src.utils.env import print_hardware_summary, set_gpu_memory_fraction

import optuna


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Round 1 Optuna optimization for knowledge distillation"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of trials (overrides config)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds (overrides config)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="kd_round1",
        help="Name of the Optuna study",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--max-epochs-per-trial",
        type=int,
        default=1,
        help="Maximum epochs per trial (for speed)",
    )
    
    return parser.parse_args()


def main():
    """Main optimization function."""
    args = parse_args()
    
    # Print hardware info
    print_hardware_summary()
    
    # Limit GPU memory to 80% to avoid OOM / disconnections
    set_gpu_memory_fraction(0.8)
    
    # Set seed
    set_seed(args.seed)
    
    # Load configuration
    print(f"\nLoading configuration from {args.config}...")
    config = load_config(args.config)
    
    # Override config
    n_trials = args.n_trials or config.optuna.round1.n_trials
    timeout = args.timeout or config.optuna.round1.timeout
    
    # Create output directories
    os.makedirs(config.artifacts.optuna_dir, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading dataset: {config.dataset.name}...")
    dataset = load_and_prepare_dataset(config.dataset, seed=args.seed)
    print(f"Dataset loaded: {len(dataset['train']):,} train, {len(dataset['validation']):,} validation")
    
    # Preprocess dataset
    print("\nPreprocessing dataset...")
    dataset = preprocess_dataset(dataset, config.dataset)
    
    # Load teacher model
    print(f"\nLoading teacher model: {config.models['teacher'].name}...")
    teacher, _ = load_teacher_model(config.models["teacher"])
    
    # Get student tokenizer
    print("\nLoading tokenizer...")
    student_tokenizer = get_tokenizer(config.models["student"].name)
    
    # Tokenize dataset
    print("\nTokenizing dataset...")
    tokenized_dataset = tokenize_dataset(
        dataset,
        student_tokenizer,
        config.tokenization,
    )
    
    # Create objective function
    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna."""
        # Suggest hyperparameters
        params = suggest_parameters(trial, config.optuna.search_space)
        
        print(f"\n--- Trial {trial.number} ---")
        print(f"Parameters: {params}")
        
        # Load student with suggested parameters
        trial_config = Config.from_dict(config.to_dict())
        trial_config.training.learning_rate = params["learning_rate"]
        trial_config.training.weight_decay = params["weight_decay"]
        trial_config.training.num_train_epochs = min(params["num_train_epochs"], args.max_epochs_per_trial)
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
        
        try:
            # Load student model
            student, _ = load_student_model(
                trial_config.models["student"],
                lora_config=trial_config.lora,
            )
            
            # Create data collator
            data_collator = DataCollatorForCausalLM(
                tokenizer=student_tokenizer,
                max_length=trial_config.tokenization.max_length,
            )
            
            # Create trainer
            trial_output_dir = os.path.join(config.artifacts.optuna_dir, f"trial_{trial.number}")
            trainer = create_distillation_trainer(
                student_model=student,
                teacher_model=teacher,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["validation"],
                tokenizer=student_tokenizer,
                data_collator=data_collator,
                output_dir=trial_output_dir,
                temperature=trial_config.distillation.temperature,
                alpha=trial_config.distillation.alpha,
                beta=trial_config.distillation.beta,
                num_train_epochs=trial_config.training.num_train_epochs,
                per_device_train_batch_size=trial_config.training.per_device_train_batch_size,
                gradient_accumulation_steps=trial_config.training.gradient_accumulation_steps,
                learning_rate=trial_config.training.learning_rate,
                weight_decay=trial_config.training.weight_decay,
                warmup_ratio=trial_config.training.warmup_ratio,
                logging_steps=50,
                eval_steps=100,
                save_steps=1000,
                bf16=trial_config.hardware.mixed_precision == "bf16",
                fp16=trial_config.hardware.mixed_precision == "fp16",
                dataloader_num_workers=trial_config.training.dataloader_num_workers,
                gradient_checkpointing=trial_config.hardware.gradient_checkpointing,
            )
            
            # Train
            trainer.train()
            
            # Evaluate
            metrics = trainer.evaluate()
            eval_loss = metrics.get("eval_loss", float("inf"))
            
            print(f"Trial {trial.number} completed. Validation loss: {eval_loss:.4f}")
            
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
            import traceback
            traceback.print_exc()
            return float("inf")
    
    # Run optimization
    study = run_optuna_optimization(
        objective_func=objective,
        study_name=args.study_name,
        n_trials=n_trials,
        timeout=timeout,
        direction="minimize",
        seed=args.seed,
        show_progress_bar=True,
    )
    
    # Save results
    print("\nSaving optimization results...")
    save_best_params(study, os.path.join(config.artifacts.optuna_dir, "round1_best_params.json"))
    
    # Plot results
    try:
        from src.monitoring.plots import plot_optuna_study
        from src.optimization.study_utils import save_study
        
        save_study(study, os.path.join(config.artifacts.optuna_dir, "round1"))
    except Exception as e:
        print(f"Could not save plots: {e}")
    
    print("\nRound 1 optimization complete!")
    print(f"Best validation loss: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")


if __name__ == "__main__":
    main()
