"""
Tokenization utilities for preparing data for model training.
"""

from typing import Dict, List, Optional, Union

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer, AutoTokenizer

from src.config import TokenizationConfig


def get_tokenizer(
    model_name: str,
    use_fast: bool = True,
    trust_remote_code: bool = True,
    padding_side: str = "left",
) -> PreTrainedTokenizer:
    """
    Load a tokenizer for a model.
    
    Args:
        model_name: Name or path of the model
        use_fast: Whether to use the fast tokenizer
        trust_remote_code: Whether to trust remote code
        padding_side: Side for padding ("left" or "right")
    
    Returns:
        Loaded tokenizer
    
    Example:
        >>> tokenizer = get_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
        >>> print(tokenizer.pad_token)
        <|endoftext|>
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
    )
    
    # Set padding side
    tokenizer.padding_side = padding_side
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Add a new pad token
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    return tokenizer


def tokenize_function(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    padding: str = "max_length",
    truncation: bool = True,
    text_column: str = "text",
) -> Dict[str, List]:
    """
    Tokenize a batch of examples.
    
    Args:
        examples: Batch of examples
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        padding: Padding strategy
        truncation: Whether to truncate
        text_column: Column containing text to tokenize
    
    Returns:
        Tokenized batch
    """
    texts = examples[text_column]
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=None,  # Return lists, not tensors
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    config: TokenizationConfig,
    text_column: str = "text",
    num_proc: int = 4,
) -> DatasetDict:
    """
    Tokenize a dataset.
    
    Args:
        dataset: Input dataset
        tokenizer: Tokenizer to use
        config: Tokenization configuration
        text_column: Column containing text
        num_proc: Number of processes
    
    Returns:
        Tokenized dataset
    """
    tokenized_datasets = {}
    
    for split_name, split_dataset in dataset.items():
        # Check if text column exists
        if text_column not in split_dataset.column_names:
            # Try alternatives
            alternatives = ["prompt", "input", "instruction", "content"]
            for alt in alternatives:
                if alt in split_dataset.column_names:
                    text_column = alt
                    break
        
        tokenized = split_dataset.map(
            lambda x: tokenize_function(
                x,
                tokenizer=tokenizer,
                max_length=config.max_length,
                padding=config.padding,
                truncation=config.truncation,
                text_column=text_column,
            ),
            batched=True,
            num_proc=num_proc if num_proc > 1 else None,
            remove_columns=split_dataset.column_names,
            desc=f"Tokenizing {split_name} split",
        )
        
        tokenized_datasets[split_name] = tokenized
    
    return DatasetDict(tokenized_datasets)


def tokenize_for_causal_lm(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    num_proc: int = 4,
) -> DatasetDict:
    """
    Tokenize dataset for causal language modeling.
    
    This function handles the specific requirements for training
    causal language models where labels are shifted by one position.
    
    Args:
        dataset: Input dataset with 'text' column
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        num_proc: Number of processes
    
    Returns:
        Tokenized dataset ready for training
    """
    def tokenize_batch(examples):
        # Tokenize texts
        result = tokenizer(
            examples["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        
        # For causal LM, labels are input_ids shifted by one
        # We keep them the same here - the loss function will handle shifting
        result["labels"] = result["input_ids"].copy()
        
        return result
    
    tokenized = {}
    for split_name, split_dataset in dataset.items():
        tokenized[split_name] = split_dataset.map(
            tokenize_batch,
            batched=True,
            num_proc=num_proc if num_proc > 1 else None,
            remove_columns=split_dataset.column_names,
            desc=f"Tokenizing {split_name}",
        )
    
    return DatasetDict(tokenized)


def prepare_labels_for_lm(
    tokenized_dataset: DatasetDict,
    ignore_index: int = -100,
) -> DatasetDict:
    """
    Prepare labels for language modeling by masking prompt tokens.
    
    This is useful for instruction tuning where we only want to
    compute loss on the response, not the prompt.
    
    Args:
        tokenized_dataset: Tokenized dataset
        ignore_index: Index to use for ignored tokens
    
    Returns:
        Dataset with prepared labels
    """
    # This is a placeholder - actual implementation would require
    # knowing where prompts end and responses begin
    # For now, we return the dataset as-is
    return tokenized_dataset


def compute_token_lengths(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    text_column: str = "text",
) -> Dict[str, float]:
    """
    Compute token length statistics for a dataset.
    
    Args:
        dataset: Input dataset
        tokenizer: Tokenizer to use
        text_column: Column containing text
    
    Returns:
        Dictionary with statistics
    """
    lengths = []
    
    for example in dataset:
        text = example.get(text_column, "")
        tokens = tokenizer.encode(text)
        lengths.append(len(tokens))
    
    import numpy as np
    
    return {
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "percentile_90": int(np.percentile(lengths, 90)),
        "percentile_95": int(np.percentile(lengths, 95)),
        "percentile_99": int(np.percentile(lengths, 99)),
    }


def print_tokenization_info(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
) -> None:
    """
    Print information about tokenized dataset.
    
    Args:
        dataset: Tokenized dataset
        tokenizer: Tokenizer used
    """
    print("=" * 60)
    print("Tokenization Information")
    print("=" * 60)
    print(f"Tokenizer: {tokenizer.name_or_path}")
    print(f"Vocab size: {len(tokenizer):,}")
    print(f"Pad token: {tokenizer.pad_token}")
    print(f"EOS token: {tokenizer.eos_token}")
    print()
    
    for split_name, split_dataset in dataset.items():
        print(f"{split_name.upper()} Split:")
        print(f"  Examples: {len(split_dataset):,}")
        
        if "input_ids" in split_dataset.features:
            # Compute token statistics
            lengths = [len(ids) for ids in split_dataset["input_ids"]]
            import numpy as np
            print(f"  Avg tokens: {np.mean(lengths):.1f}")
            print(f"  Max tokens: {max(lengths)}")
    
    print("=" * 60)
