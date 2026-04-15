"""
Evaluation utilities for trained models.
"""

import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.data.collators import DataCollatorForCausalLM


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute metrics for evaluation.
    
    This function is compatible with Hugging Face Trainer's compute_metrics.
    
    Args:
        eval_pred: EvalPrediction object with predictions and labels
    
    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Compute perplexity
    # For language models, predictions are logits
    # We need to compute cross-entropy loss first
    
    metrics = {}
    
    # If predictions are logits, compute loss
    if len(predictions.shape) == 3:  # [batch, seq_len, vocab_size]
        # Shift predictions and labels for next-token prediction
        shift_predictions = predictions[:, :-1, :].reshape(-1, predictions.shape[-1])
        shift_labels = labels[:, 1:].reshape(-1)
        
        # Filter out ignored indices
        mask = shift_labels != -100
        shift_predictions = shift_predictions[mask]
        shift_labels = shift_labels[mask]
        
        if len(shift_labels) > 0:
            # Compute cross-entropy loss
            log_probs = np.log_softmax(shift_predictions, axis=-1)
            nll_loss = -log_probs[np.arange(len(shift_labels)), shift_labels].mean()
            
            # Compute perplexity
            try:
                perplexity = math.exp(nll_loss)
            except OverflowError:
                perplexity = float("inf")
            
            metrics["perplexity"] = perplexity
            metrics["loss"] = nll_loss
    
    return metrics


def evaluate_model(
    model: PreTrainedModel,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Model to evaluate
        dataset: Evaluation dataset
        tokenizer: Tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        device: Device to use
    
    Returns:
        Dictionary of evaluation metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Create data collator
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        max_length=max_length,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            # Accumulate loss
            loss = outputs.loss
            total_loss += loss.item() * (labels != -100).sum().item()
            total_tokens += (labels != -100).sum().item()
    
    # Compute metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float("inf")
    
    metrics = {
        "loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
    }
    
    return metrics


def evaluate_with_generation(
    model: PreTrainedModel,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_new_tokens: int = 128,
    device: Optional[torch.device] = None,
    num_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate model with text generation.
    
    Args:
        model: Model to evaluate
        dataset: Evaluation dataset
        tokenizer: Tokenizer
        batch_size: Batch size
        max_new_tokens: Maximum new tokens to generate
        device: Device to use
        num_samples: Number of samples to evaluate (None for all)
    
    Returns:
        List of evaluation results with generated text
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Limit samples if specified
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    results = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Generating"):
            batch = dataset[i:i+batch_size]
            
            # Get prompts (assuming 'prompt' or 'text' column)
            if "prompt" in batch:
                prompts = batch["prompt"]
            elif "text" in batch:
                prompts = batch["text"]
            else:
                prompts = [""] * len(batch["input_ids"])
            
            # Get references
            references = batch.get("response", [""] * len(prompts))
            
            # Tokenize prompts
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode
            generated_texts = tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            # Store results
            for prompt, reference, generated in zip(prompts, references, generated_texts):
                results.append({
                    "prompt": prompt,
                    "reference": reference,
                    "generated": generated,
                })
    
    return results


def compare_models(
    models: Dict[str, PreTrainedModel],
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models on the same dataset.
    
    Args:
        models: Dictionary of model name to model
        dataset: Evaluation dataset
        tokenizer: Tokenizer
        batch_size: Batch size
    
    Returns:
        Dictionary of model name to metrics
    """
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        metrics = evaluate_model(
            model=model,
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )
        results[name] = metrics
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Perplexity: {metrics['perplexity']:.2f}")
    
    return results


def compute_bleu_score(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute BLEU score for generation evaluation.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
    
    Returns:
        Dictionary with BLEU scores
    """
    try:
        from evaluate import load
        bleu = load("bleu")
        
        # BLEU expects list of references for each prediction
        refs = [[ref] for ref in references]
        
        results = bleu.compute(
            predictions=predictions,
            references=refs
        )
        
        return {
            "bleu": results["bleu"],
            "bleu_1": results["precisions"][0],
            "bleu_2": results["precisions"][1],
            "bleu_3": results["precisions"][2],
            "bleu_4": results["precisions"][3],
        }
    except Exception as e:
        print(f"Error computing BLEU: {e}")
        return {"bleu": 0.0}


def compute_rouge_score(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute ROUGE score for generation evaluation.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
    
    Returns:
        Dictionary with ROUGE scores
    """
    try:
        from evaluate import load
        rouge = load("rouge")
        
        results = rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True,
        )
        
        return {
            "rouge1": results["rouge1"],
            "rouge2": results["rouge2"],
            "rougeL": results["rougeL"],
        }
    except Exception as e:
        print(f"Error computing ROUGE: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
