#!/usr/bin/env python
"""
Round 2 Optuna optimization - Deeper search around best region from Round 1.

Usage:
    python scripts/optimize_round2.py --config configs/default.yaml --round1-results artifacts/optuna/round1_best_params.json
"""

import argparse
import json
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
from src.optimization.optuna_search import run_optuna_optimization, save_best_params
from src.optimization.search_space import suggest_parameters
from src.utils.seed import set_seed
from src.utils.env import print_hardware_summary

import optuna


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Round 2 Optuna optimization for knowledge distillation"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--round1-results",
        type=str,
        required=True,
        help="Path to Round 1 best params JSON file",
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
        default="kd_round2",
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
        default=2,
        help="Maximum epochs per trial",
    )
    
    return parser.parse_args()


def create_narrow_search_space(best_params: dict, variance: float = 0.3):
    """Create a narrow search space around best parameters."""
    space = {}
    
    for key, value in best_params.items():
        if key == "learning_rate":
            # Log scale for learning rate
            import numpy as np
            log_value = np.log10(value)
            log_low = log_value - variance
            log_high = log_value + variance
            space[key] = [10 ** log_low, 10 ** log_high]
        elif key in ["weight_decay", "lora_dropout", "warmup_ratio"]:
            # Linear scale with bounds
            low = max(0, value * (1 - variance))
            high = min(1, value * (1 + variance))
            space[key] = [low, high]
        elif key in ["temperature"]:
            low = max(1.0, value * (1 - variance))
            high = value * (1 + variance)
            space[key] = [low, high]
        elif key in ["alpha", "beta"]:
            low = max(0.1, value * (1 - variance))
            high = min(0.9, value * (1 + variance))
            space[key] = [low, high]
        elif key in ["lora_r", "lora_alpha"]:
            # For integers, keep close to best value
            space[key] = [max(4, value - 8), value + 8]
        elif key in ["per_device_train_batch_size", "gradient_accumulation_steps", "num_train_epochs"]:
            # Keep the same options
            space[key] = [value]
        elif key == "max_length":
            space[key] = [256, 512, 1024]
    
    return space


def main():
    """Main optimization function."""
    args = parse_args()
    
    # Print hardware info
    print_hardware_summary()
    
    # Set seed
    set_seed(args.seed)
    
    # Load configuration
    print(f"\nLoading configuration from {args.config}...")
    config = load_config(args.config)
    
    # Load Round 1 results
    print(f"\nLoading Round 1 results from {args.round1_results}...")
    with open(args.round1_results, "r") as f:
        round1_data = json.load(f)
    
    best_params = round1_data["best_params"]
    print(f"Round 1 best params: {best_params}")
    print(f"Round 1 best value: {round1_data['best_value']:.4f}")
    
    # Create narrow search space
    narrow_space = create_narrow_search_space(best_params, variance=0.3)
    print(f"\nNarrow search space: {narrow_space}")
    
    # Override config
    n_trials = args.n_trials or config.optuna.round2.n_trials
    timeout = args.timeout or config.optuna.round2.timeout
    
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
    
    # Create objective function with narrow search space
    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna Round 2."""
        # Suggest parameters from narrow space
        params = {}
        for key, bounds in narrow_space.items():
            if len(bounds) == 1:
                params[key] = bounds[0]
            elif key in ["learning_rate"]:
                params[key] = trial.suggest_float(key, bounds[0], bounds[1], log=True)
            elif key in ["lora_r", "lora_alpha"]:
                params[key] = trial.suggest_int(key, bounds[0], bounds[1])
            elif key in ["per_device_train_batch_size", "gradient_accumulation_steps", "num_train_epochs", "max_length"]:
                params[key] = trial.suggest_categorical(key, bounds)
            else:
                params[key] = trial.suggest_float(key, bounds[0], bounds[1])
        
        # Ensure alpha + beta = 1.0
        if "alpha" in params and "beta" in params:
            total = params["alpha"] + params["beta"]
            params["alpha"] = params["alpha"] / total
            params["beta"] = params["beta"] / total
        
        print(f"\n--- Trial {trial.number} ---")
        print(f"Parameters: {params}")
        
        # Load student with suggested parameters
        trial_config = Config.from_dict(config.to_dict())
        trial_config.training.learning_rate = params["learning_rate"]
        trial_config.training.weight_decay = params.get("weight_decay", best_params.get("weight_decay", 0.01))
        trial_config.training.num_train_epochs = min(params.get("num_train_epochs", best_params.get("num_train_epochs", 2)), args.max_epochs_per_trial)
        trial_config.training.warmup_ratio = params.get("warmup_ratio", best_params.get("warmup_ratio", 0.03))
        trial_config.training.per_device_train_batch_size = params.get("per_device_train_batch_size", best_params.get("per_device_train_batch_size", 1))
        trial_config.training.gradient_accumulation_steps = params.get("gradient_accumulation_steps", best_params.get("gradient_accumulation_steps", 8))
        trial_config.lora.r = params.get("lora_r", best_params.get("lora_r", 16))
        trial_config.lora.lora_alpha = params.get("lora_alpha", best_params.get("lora_alpha", 32))
        trial_config.lora.lora_dropout = params.get("lora_dropout", best_params.get("lora_dropout", 0.05))
        trial_config.distillation.temperature = params.get("temperature", best_params.get("temperature", 2.0))
        trial_config.distillation.alpha = params.get("alpha", best_params.get("alpha", 0.3))
        trial_config.distillation.beta = params.get("beta", best_params.get("beta", 0.7))
        trial_config.tokenization.max_length = params.get("max_length", best_params.get("max_length", 512))
        
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
            trial_output_dir = os.path.join(config.artifacts.optuna_dir, f"round2_trial_{trial.number}")
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
            )
            
            # Train
            trainer.train()
            
            # Save training history for this trial
            history_path = os.path.join(trial_output_dir, "training_history.json")
            with open(history_path, "w") as f:
                json.dump(trainer.state.log_history, f)
            
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
    save_best_params(study, os.path.join(config.artifacts.optuna_dir, "round2_best_params.json"))
    
    # Plot results
    try:
        from src.optimization.study_utils import save_study
        save_study(study, os.path.join(config.artifacts.optuna_dir, "round2"))
    except Exception as e:
        print(f"Could not save plots: {e}")
    
    # ============================================================
    # POST-OPTIMIZATION VISUALIZATION BLOCK
    # ============================================================
    print("\n" + "=" * 60)
    print("Generating Post-Optimization Plots")
    print("=" * 60)
    
    plots_dir = os.path.join(config.artifacts.plots_dir, "round2")
    os.makedirs(plots_dir, exist_ok=True)
    
    # --- Plot 1 & 2: Load best trial history ---
    best_trial_num = study.best_trial.number
    best_trial_dir = os.path.join(config.artifacts.optuna_dir, f"round2_trial_{best_trial_num}")
    history_path = os.path.join(best_trial_dir, "training_history.json")
    
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            best_history = json.load(f)
        
        from src.monitoring.plots import plot_loss_vs_accuracy, plot_kl_divergence
        
        plot_loss_vs_accuracy(
            best_history,
            output_path=os.path.join(plots_dir, "plot1_training_loss_vs_accuracy.png"),
        )
        plot_kl_divergence(
            best_history,
            output_path=os.path.join(plots_dir, "plot2_kl_divergence.png"),
        )
    else:
        print(f"Warning: No training history found at {history_path}")
    
    # --- Plot 3 & 4: Attention map analysis ---
    print("\nExtracting attention maps from best model...")
    
    # Reconstruct best config
    best_trial_params = study.best_params
    best_config = Config.from_dict(config.to_dict())
    best_config.training.learning_rate = best_trial_params["learning_rate"]
    best_config.training.weight_decay = best_trial_params.get("weight_decay", best_params.get("weight_decay", 0.01))
    best_config.training.num_train_epochs = min(best_trial_params.get("num_train_epochs", best_params.get("num_train_epochs", 2)), args.max_epochs_per_trial)
    best_config.training.warmup_ratio = best_trial_params.get("warmup_ratio", best_params.get("warmup_ratio", 0.03))
    best_config.training.per_device_train_batch_size = best_trial_params.get("per_device_train_batch_size", best_params.get("per_device_train_batch_size", 1))
    best_config.training.gradient_accumulation_steps = best_trial_params.get("gradient_accumulation_steps", best_params.get("gradient_accumulation_steps", 8))
    best_config.lora.r = best_trial_params.get("lora_r", best_params.get("lora_r", 16))
    best_config.lora.lora_alpha = best_trial_params.get("lora_alpha", best_params.get("lora_alpha", 32))
    best_config.lora.lora_dropout = best_trial_params.get("lora_dropout", best_params.get("lora_dropout", 0.05))
    best_config.distillation.temperature = best_trial_params.get("temperature", best_params.get("temperature", 2.0))
    best_config.distillation.alpha = best_trial_params.get("alpha", best_params.get("alpha", 0.3))
    best_config.distillation.beta = best_trial_params.get("beta", best_params.get("beta", 0.7))
    best_config.tokenization.max_length = best_trial_params.get("max_length", best_params.get("max_length", 512))
    
    try:
        # Load best student model
        best_student, _ = load_student_model(
            best_config.models["student"],
            lora_config=best_config.lora,
        )
        best_student.eval()
        
        # Recreate data collator
        data_collator = DataCollatorForCausalLM(
            tokenizer=student_tokenizer,
            max_length=best_config.tokenization.max_length,
        )
        
        # Get a small sample batch
        from torch.utils.data import DataLoader
        sample_size = min(2, len(tokenized_dataset["validation"]))
        sample_dataset = tokenized_dataset["validation"].select(range(sample_size))
        sample_loader = DataLoader(sample_dataset, batch_size=sample_size, collate_fn=data_collator)
        sample_batch = next(iter(sample_loader))
        
        import torch
        
        # Extract attention maps
        with torch.no_grad():
            student_device = next(best_student.parameters()).device
            student_outputs = best_student(
                input_ids=sample_batch["input_ids"].to(student_device),
                attention_mask=sample_batch["attention_mask"].to(student_device),
                output_attentions=True,
            )
            student_attentions = student_outputs.attentions
            
            teacher_outputs = teacher(
                input_ids=sample_batch["input_ids"].to(teacher.device),
                attention_mask=sample_batch["attention_mask"].to(teacher.device),
                output_attentions=True,
            )
            teacher_attentions = teacher_outputs.attentions
        
        from src.monitoring.plots import plot_attention_similarity, plot_attention_embedding_by_layer
        
        plot_attention_similarity(
            teacher_attentions,
            student_attentions,
            output_path=os.path.join(plots_dir, "plot3_attention_similarity_tsne_pca.png"),
        )
        plot_attention_embedding_by_layer(
            teacher_attentions,
            student_attentions,
            output_path=os.path.join(plots_dir, "plot4_teacher_vs_student_attention_by_layer.png"),
        )
        
        # Cleanup
        del best_student
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Could not generate attention plots: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nAll plots saved to: {plots_dir}")
    print("=" * 60)
    
    print("\nRound 2 optimization complete!")
    print(f"Best validation loss: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    print(f"\nComparison with Round 1:")
    print(f"  Round 1 best: {round1_data['best_value']:.4f}")
    print(f"  Round 2 best: {study.best_value:.4f}")
    improvement = round1_data['best_value'] - study.best_value
    print(f"  Improvement: {improvement:.4f} ({improvement/round1_data['best_value']*100:.1f}%)")


if __name__ == "__main__":
    main()
