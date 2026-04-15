#!/usr/bin/env python
"""
Evaluation script for trained models.

Usage:
    python scripts/evaluate.py --model-path artifacts/best_model/final --dataset databricks/databricks-dolly-15k
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.data.dataset_loader import load_and_prepare_dataset
from src.data.preprocessing import preprocess_dataset
from src.data.tokenization import tokenize_dataset, get_tokenizer
from src.models.student_loader import load_student_for_inference
from src.training.evaluate import evaluate_model, evaluate_with_generation
from src.training.metrics import compute_all_metrics, print_metrics
from src.utils.env import get_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (for LoRA models)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="databricks/databricks-dolly-15k",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Also evaluate with text generation",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples for generation evaluation",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file for results",
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model, tokenizer = load_student_for_inference(
        model_path=args.model_path,
        base_model_name=args.base_model,
    )
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}...")
    from src.config import DatasetConfig
    dataset_config = DatasetConfig(name=args.dataset)
    dataset = load_and_prepare_dataset(dataset_config, seed=42)
    
    if args.split not in dataset:
        print(f"Split '{args.split}' not found. Available splits: {list(dataset.keys())}")
        return
    
    eval_dataset = dataset[args.split]
    print(f"Evaluating on {len(eval_dataset):,} examples")
    
    # Preprocess dataset
    print("\nPreprocessing dataset...")
    dataset = preprocess_dataset(dataset, dataset_config)
    
    # Tokenize dataset
    print("\nTokenizing dataset...")
    from src.config import TokenizationConfig
    tokenization_config = TokenizationConfig(max_length=args.max_length)
    tokenized_dataset = tokenize_dataset(
        dataset,
        tokenizer,
        tokenization_config,
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Running Evaluation")
    print("=" * 60)
    
    metrics = evaluate_model(
        model=model,
        dataset=tokenized_dataset[args.split],
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    
    print("\nEvaluation Results:")
    print_metrics(metrics, title="Model Evaluation")
    
    # Generation evaluation if requested
    if args.generate:
        print("\n" + "=" * 60)
        print("Running Generation Evaluation")
        print("=" * 60)
        
        gen_results = evaluate_with_generation(
            model=model,
            dataset=dataset[args.split],
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_new_tokens=128,
            device=device,
            num_samples=args.num_samples,
        )
        
        # Print some examples
        print("\nGeneration Examples:")
        for i, result in enumerate(gen_results[:5]):
            print(f"\n--- Example {i+1} ---")
            print(f"Prompt: {result['prompt'][:100]}...")
            print(f"Reference: {result['reference'][:100]}...")
            print(f"Generated: {result['generated'][:100]}...")
    
    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
