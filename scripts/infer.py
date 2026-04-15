#!/usr/bin/env python
"""
Inference script for running predictions with trained models.

Usage:
    python scripts/infer.py --model-path artifacts/best_model/final --prompt "What is AI?"
    python scripts/infer.py --model-path artifacts/best_model/final --input-file prompts.txt --output-file results.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.predict import InferencePipeline
from src.models.student_loader import load_student_for_inference
from src.utils.env import get_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with a trained model"
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
        "--prompt",
        type=str,
        default=None,
        help="Single prompt for inference",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="File with prompts (one per line)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file for results",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus sampling)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty",
    )
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=1,
        help="Number of sequences to return",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model, tokenizer = load_student_for_inference(
        model_path=args.model_path,
        base_model_name=args.base_model,
    )
    
    # Create inference pipeline
    pipeline = InferencePipeline(model, tokenizer, device)
    
    # Prepare generation kwargs
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "num_return_sequences": args.num_return_sequences,
    }
    
    # Run inference based on mode
    if args.interactive:
        # Interactive mode
        print("\n" + "=" * 60)
        print("Interactive Mode")
        print("Type 'quit' or 'exit' to stop")
        print("=" * 60)
        
        while True:
            prompt = input("\nPrompt: ").strip()
            
            if prompt.lower() in ["quit", "exit", "q"]:
                break
            
            if not prompt:
                continue
            
            print("\nGenerating...")
            result = pipeline.generate(prompt, **gen_kwargs)
            print(f"\nResponse:\n{result['generated_text']}")
    
    elif args.input_file:
        # Batch inference from file
        print(f"\nLoading prompts from {args.input_file}...")
        
        with open(args.input_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(prompts)} prompts")
        
        results = []
        for i, prompt in enumerate(prompts):
            print(f"\n[{i+1}/{len(prompts)}] Processing: {prompt[:50]}...")
            result = pipeline.generate(prompt, **gen_kwargs)
            results.append({
                "prompt": prompt,
                "generated_text": result["generated_text"],
                "num_tokens": result["num_tokens_generated"],
            })
            print(f"Generated: {result['generated_text'][:100]}...")
        
        # Save results
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            print(f"\nResults saved to {args.output_file}")
    
    elif args.prompt:
        # Single prompt
        print(f"\nPrompt: {args.prompt}")
        print("\nGenerating...")
        
        result = pipeline.generate(args.prompt, **gen_kwargs)
        
        print(f"\nGenerated Text:\n{result['generated_text']}")
        print(f"\nTokens generated: {result['num_tokens_generated']}")
    
    else:
        print("\nError: Please provide --prompt, --input-file, or use --interactive mode")
        sys.exit(1)


if __name__ == "__main__":
    main()
