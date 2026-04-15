"""
Batch inference for processing large datasets.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.inference.predict import InferencePipeline


class BatchInference:
    """
    Batch inference for processing large datasets efficiently.
    
    This class handles:
    - Batching of inputs
    - Progress tracking
    - Result saving
    - Error handling
    
    Example:
        >>> batch_infer = BatchInference(model, tokenizer)
        >>> results = batch_infer.process_dataset(dataset, output_file="results.jsonl")
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[torch.device] = None,
        batch_size: int = 8,
    ):
        """
        Initialize batch inference.
        
        Args:
            model: Model to use
            tokenizer: Tokenizer
            device: Device to use
            batch_size: Batch size for inference
        """
        self.pipeline = InferencePipeline(model, tokenizer, device)
        self.batch_size = batch_size
    
    def process_dataset(
        self,
        dataset: Dataset,
        prompt_column: str = "prompt",
        output_file: Optional[str] = None,
        max_new_tokens: int = 128,
        **generation_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process a dataset and generate outputs.
        
        Args:
            dataset: Dataset to process
            prompt_column: Column containing prompts
            output_file: Optional file to save results
            max_new_tokens: Maximum new tokens to generate
            **generation_kwargs: Generation parameters
        
        Returns:
            List of results
        """
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(dataset), self.batch_size), desc="Processing"):
            batch = dataset[i:i+self.batch_size]
            prompts = batch[prompt_column]
            
            # Generate
            batch_results = self.pipeline.batch_generate(
                prompts,
                batch_size=self.batch_size,
                max_new_tokens=max_new_tokens,
                **generation_kwargs
            )
            
            # Add to results
            for j, (prompt, result) in enumerate(zip(prompts, batch_results)):
                result_dict = {
                    "prompt": prompt,
                    "generated_text": result["generated_text"],
                    "num_tokens": result["num_tokens_generated"],
                }
                
                # Add other columns from dataset
                for col in dataset.column_names:
                    if col != prompt_column:
                        result_dict[col] = batch[col][j]
                
                results.append(result_dict)
        
        # Save if output file specified
        if output_file:
            self._save_results(results, output_file)
        
        return results
    
    def process_file(
        self,
        input_file: str,
        output_file: str,
        prompt_column: str = "prompt",
        max_new_tokens: int = 128,
        **generation_kwargs
    ) -> None:
        """
        Process a JSONL file and save results.
        
        Args:
            input_file: Input JSONL file
            output_file: Output JSONL file
            prompt_column: Column containing prompts
            max_new_tokens: Maximum new tokens to generate
            **generation_kwargs: Generation parameters
        """
        # Load prompts
        prompts = []
        metadata = []
        
        with open(input_file, "r") as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data[prompt_column])
                metadata.append(data)
        
        # Process in batches
        results = []
        for i in tqdm(range(0, len(prompts), self.batch_size), desc="Processing"):
            batch_prompts = prompts[i:i+self.batch_size]
            batch_metadata = metadata[i:i+self.batch_size]
            
            # Generate
            batch_results = self.pipeline.batch_generate(
                batch_prompts,
                batch_size=self.batch_size,
                max_new_tokens=max_new_tokens,
                **generation_kwargs
            )
            
            # Combine with metadata
            for meta, result in zip(batch_metadata, batch_results):
                output = meta.copy()
                output["generated_text"] = result["generated_text"]
                output["num_tokens_generated"] = result["num_tokens_generated"]
                results.append(output)
        
        # Save results
        self._save_results(results, output_file)
    
    def _save_results(
        self,
        results: List[Dict[str, Any]],
        output_file: str,
    ) -> None:
        """
        Save results to a JSONL file.
        
        Args:
            results: List of results
            output_file: Output file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        print(f"Results saved to {output_file}")


def run_batch_inference(
    model_path: str,
    input_file: str,
    output_file: str,
    batch_size: int = 8,
    max_new_tokens: int = 128,
    **generation_kwargs
) -> None:
    """
    Run batch inference from a saved model.
    
    Args:
        model_path: Path to saved model
        input_file: Input JSONL file
        output_file: Output JSONL file
        batch_size: Batch size
        max_new_tokens: Maximum new tokens to generate
        **generation_kwargs: Generation parameters
    """
    from src.models.student_loader import load_student_for_inference
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, tokenizer = load_student_for_inference(model_path)
    
    # Create batch inference
    batch_infer = BatchInference(model, tokenizer, batch_size=batch_size)
    
    # Process file
    print(f"Processing {input_file}...")
    batch_infer.process_file(
        input_file,
        output_file,
        max_new_tokens=max_new_tokens,
        **generation_kwargs
    )
