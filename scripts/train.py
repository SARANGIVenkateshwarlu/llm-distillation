#!/usr/bin/env python
"""
Main training script for knowledge distillation.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --teacher Qwen/Qwen2.5-7B-Instruct --student Qwen/Qwen2.5-1.5B-Instruct
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
from src.utils.seed import set_seed
from src.utils.env import print_hardware_summary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a student model with knowledge distillation"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default=None,
        help="Teacher model name (overrides config)",
    )
    parser.add_argument(
        "--student",
        type=str,
        default=None,
        help="Student model name (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 precision",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 precision",
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Print hardware info
    print_hardware_summary()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load configuration
    print(f"\nLoading configuration from {args.config}...")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.teacher:
        config.models["teacher"].name = args.teacher
    if args.student:
        config.models["student"].name = args.student
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.epochs:
        config.training.num_train_epochs = args.epochs
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    # Setup precision
    if args.bf16:
        config.hardware.mixed_precision = "bf16"
    elif args.fp16:
        config.hardware.mixed_precision = "fp16"
    
    # Create output directories
    os.makedirs(config.training.output_dir, exist_ok=True)
    os.makedirs(config.artifacts.checkpoint_dir, exist_ok=True)
    os.makedirs(config.artifacts.best_model_dir, exist_ok=True)
    os.makedirs(config.artifacts.plots_dir, exist_ok=True)
    os.makedirs(config.artifacts.logs_dir, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading dataset: {config.dataset.name}...")
    dataset = load_and_prepare_dataset(config.dataset, seed=args.seed)
    print(f"Dataset loaded:")
    print(f"  Train: {len(dataset['train']):,} examples")
    print(f"  Validation: {len(dataset['validation']):,} examples")
    if 'test' in dataset:
        print(f"  Test: {len(dataset['test']):,} examples")
    
    # Preprocess dataset
    print("\nPreprocessing dataset...")
    dataset = preprocess_dataset(dataset, config.dataset)
    
    # Load teacher model
    print(f"\nLoading teacher model: {config.models['teacher'].name}...")
    teacher, teacher_tokenizer = load_teacher_model(config.models["teacher"])
    
    # Load student model and tokenizer
    print(f"\nLoading student model: {config.models['student'].name}...")
    student, student_tokenizer = load_student_model(
        config.models["student"],
        lora_config=config.lora,
    )
    
    # Tokenize dataset
    print("\nTokenizing dataset...")
    tokenized_dataset = tokenize_dataset(
        dataset,
        student_tokenizer,
        config.tokenization,
    )
    
    # Create data collator
    data_collator = DataCollatorForCausalLM(
        tokenizer=student_tokenizer,
        max_length=config.tokenization.max_length,
    )
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = create_distillation_trainer(
        student_model=student,
        teacher_model=teacher,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=student_tokenizer,
        data_collator=data_collator,
        output_dir=config.training.output_dir,
        temperature=config.distillation.temperature,
        alpha=config.distillation.alpha,
        beta=config.distillation.beta,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        logging_steps=config.training.logging_steps,
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        bf16=config.hardware.mixed_precision == "bf16",
        fp16=config.hardware.mixed_precision == "fp16",
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    trainer.train()
    
    # Evaluate on test set if available
    if "test" in tokenized_dataset:
        print("\nEvaluating on test set...")
        test_metrics = trainer.evaluate(tokenized_dataset["test"])
        print(f"Test metrics: {test_metrics}")
    
    # Save final model
    print("\nSaving final model...")
    final_model_dir = os.path.join(config.artifacts.best_model_dir, "final")
    trainer.save_model(final_model_dir)
    student_tokenizer.save_pretrained(final_model_dir)
    
    # Save training config
    config_save_path = os.path.join(final_model_dir, "training_config.yaml")
    config.save_yaml(config_save_path)
    
    print(f"\nTraining complete! Model saved to {final_model_dir}")
    
    # Plot training curves
    try:
        from src.monitoring.plots import plot_training_curves
        
        plot_path = os.path.join(config.artifacts.plots_dir, "training_curves.png")
        plot_training_curves(trainer.state.log_history, plot_path)
    except Exception as e:
        print(f"Could not plot training curves: {e}")


if __name__ == "__main__":
    main()
