#!/usr/bin/env python
"""
Standalone script to push a trained model to HuggingFace Hub.

Usage:
    python scripts/push_to_hub.py --model-path artifacts/best_model/final --repo-id username/my-distilled-model
    python scripts/push_to_hub.py --model-path artifacts/best_model/final --repo-id username/my-model --private --merge-lora
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.publishing.hub_uploader import push_model_to_hub


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Push a trained model to HuggingFace Hub"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved model directory",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace Hub repository ID (e.g., username/model-name)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to training config file (for model card generation)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )
    parser.add_argument(
        "--no-merge-lora",
        action="store_true",
        help="Do NOT merge LoRA weights before uploading (upload adapter only)",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload distilled student model",
        help="Commit message for the upload",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="Tags for the model (e.g., --tags qwen instruction-tuned)",
    )
    parser.add_argument(
        "--license",
        type=str,
        default="mit",
        help="License for the model (default: mit)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (overrides config)",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default=None,
        help="Teacher model name (overrides config)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (overrides config)",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate model path
    if not Path(args.model_path).exists():
        print(f"Error: Model path not found: {args.model_path}")
        sys.exit(1)
    
    # Load config for model card if available
    config = None
    if Path(args.config).exists():
        try:
            config = load_config(args.config)
            print(f"Loaded config from {args.config}")
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
    
    # Push to Hub
    url = push_model_to_hub(
        model_path=args.model_path,
        repo_id=args.repo_id,
        config=config,
        token=args.token,
        private=args.private,
        merge_lora=not args.no_merge_lora,
        commit_message=args.commit_message,
        tags=args.tags,
        license=args.license,
        base_model=args.base_model,
        teacher_model=args.teacher_model,
        dataset=args.dataset,
    )
    
    print(f"\n✅ Model published successfully!")
    print(f"🔗 URL: {url}")


if __name__ == "__main__":
    main()
