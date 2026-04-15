"""
Dataset loading utilities for various NLP tasks.
"""

from typing import Dict, List, Optional, Tuple, Union

from datasets import Dataset, DatasetDict, load_dataset

from src.config import DatasetConfig


# Dataset configurations for popular datasets
DATASET_CONFIGS = {
    "databricks/databricks-dolly-15k": {
        "task_type": "instruction_following",
        "text_column": "instruction",
        "context_column": "context",
        "response_column": "response",
        "category_column": "category",
    },
    "tatsu-lab/alpaca": {
        "task_type": "instruction_following",
        "text_column": "instruction",
        "input_column": "input",
        "response_column": "output",
    },
    "HuggingFaceH4/ultrachat_200k": {
        "task_type": "conversation",
        "messages_column": "messages",
    },
    "Open-Orca/OpenOrca": {
        "task_type": "instruction_following",
        "text_column": "question",
        "response_column": "response",
    },
    "imdb": {
        "task_type": "text_classification",
        "text_column": "text",
        "label_column": "label",
        "num_labels": 2,
    },
    "emotion": {
        "task_type": "text_classification",
        "text_column": "text",
        "label_column": "label",
        "num_labels": 6,
    },
    "sst2": {
        "task_type": "sentiment_analysis",
        "text_column": "sentence",
        "label_column": "label",
        "num_labels": 2,
    },
    "cnn_dailymail": {
        "task_type": "summarization",
        "text_column": "article",
        "summary_column": "highlights",
    },
    "squad": {
        "task_type": "question_answering",
        "context_column": "context",
        "question_column": "question",
        "answer_column": "answers",
    },
}


def get_dataset_info(dataset_name: str) -> Dict:
    """
    Get configuration info for a known dataset.
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        Dictionary with dataset configuration
    """
    return DATASET_CONFIGS.get(dataset_name, {})


def load_dataset_by_name(
    dataset_name: str,
    config: Optional[str] = None,
    split: Optional[str] = None,
    cache_dir: Optional[str] = None,
    num_proc: int = 4,
    max_samples: Optional[int] = None,
) -> Union[Dataset, DatasetDict]:
    """
    Load a dataset by name with optional configuration.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face Hub
        config: Dataset configuration name (if applicable)
        split: Which split to load (if None, loads all splits)
        cache_dir: Cache directory for downloaded data
        num_proc: Number of processes for loading
        max_samples: Maximum number of samples to load (for debugging)
    
    Returns:
        Loaded dataset
    
    Example:
        >>> ds = load_dataset_by_name("databricks/databricks-dolly-15k")
        >>> print(ds)
        DatasetDict({
            train: Dataset({
                features: ['instruction', 'context', 'response', 'category'],
                num_rows: 15015
            })
        })
    """
    try:
        dataset = load_dataset(
            dataset_name,
            config,
            split=split,
            cache_dir=cache_dir,
            num_proc=num_proc if num_proc > 1 else None,
            trust_remote_code=True,
        )
    except Exception as e:
        # Try without num_proc if it fails
        dataset = load_dataset(
            dataset_name,
            config,
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    
    # Limit samples if specified
    if max_samples is not None:
        if isinstance(dataset, DatasetDict):
            for split_name in dataset.keys():
                dataset[split_name] = dataset[split_name].select(
                    range(min(max_samples, len(dataset[split_name])))
                )
        else:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    return dataset


def prepare_dataset_splits(
    dataset: Union[Dataset, DatasetDict],
    train_split: str = "train",
    validation_split: Optional[Union[str, float]] = 0.1,
    test_split: Optional[Union[str, float]] = 0.05,
    seed: int = 42,
) -> DatasetDict:
    """
    Prepare train/validation/test splits from a dataset.
    
    Args:
        dataset: Input dataset
        train_split: Name of training split or "train" for single dataset
        validation_split: Validation split name or fraction
        test_split: Test split name or fraction
        seed: Random seed for reproducibility
    
    Returns:
        DatasetDict with train/validation/test splits
    """
    # If already a DatasetDict with all splits
    if isinstance(dataset, DatasetDict):
        splits = list(dataset.keys())
        
        # Check if we already have the desired splits
        if "train" in splits:
            result = {"train": dataset["train"]}
            
            if "validation" in splits:
                result["validation"] = dataset["validation"]
            elif "valid" in splits:
                result["validation"] = dataset["valid"]
            elif isinstance(validation_split, float):
                # Create validation split from train
                split_dataset = result["train"].train_test_split(
                    test_size=validation_split,
                    seed=seed
                )
                result["train"] = split_dataset["train"]
                result["validation"] = split_dataset["test"]
            
            if "test" in splits:
                result["test"] = dataset["test"]
            elif isinstance(test_split, float) and "validation" in result:
                # Create test split from validation
                val_test_split = result["validation"].train_test_split(
                    test_size=test_split / (validation_split + test_split),
                    seed=seed
                )
                result["validation"] = val_test_split["train"]
                result["test"] = val_test_split["test"]
            
            return DatasetDict(result)
    
    # Single dataset - need to create all splits
    if isinstance(dataset, Dataset):
        # First split: train vs (val + test)
        val_test_ratio = 0.0
        if isinstance(validation_split, float):
            val_test_ratio += validation_split
        if isinstance(test_split, float):
            val_test_ratio += test_split
        
        if val_test_ratio > 0:
            split1 = dataset.train_test_split(test_size=val_test_ratio, seed=seed)
            train_ds = split1["train"]
            val_test_ds = split1["test"]
            
            # Second split: val vs test
            if isinstance(validation_split, float) and isinstance(test_split, float):
                test_ratio = test_split / val_test_ratio
                split2 = val_test_ds.train_test_split(test_size=test_ratio, seed=seed)
                
                return DatasetDict({
                    "train": train_ds,
                    "validation": split2["train"],
                    "test": split2["test"]
                })
            else:
                return DatasetDict({
                    "train": train_ds,
                    "validation": val_test_ds
                })
        else:
            return DatasetDict({"train": dataset})
    
    return dataset


def load_and_prepare_dataset(
    config: DatasetConfig,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> DatasetDict:
    """
    Load and prepare dataset according to configuration.
    
    Args:
        config: Dataset configuration
        cache_dir: Cache directory
        seed: Random seed
    
    Returns:
        Prepared DatasetDict
    """
    # Load dataset
    dataset = load_dataset_by_name(
        dataset_name=config.name,
        cache_dir=cache_dir,
        max_samples=config.max_samples,
    )
    
    # Prepare splits
    dataset = prepare_dataset_splits(
        dataset,
        train_split=config.train_split,
        validation_split=config.validation_split,
        test_split=config.test_split,
        seed=seed,
    )
    
    return dataset


def get_dataset_statistics(dataset: Union[Dataset, DatasetDict]) -> Dict:
    """
    Get statistics about a dataset.
    
    Args:
        dataset: Input dataset
    
    Returns:
        Dictionary with statistics
    """
    stats = {}
    
    if isinstance(dataset, DatasetDict):
        for split_name, split_dataset in dataset.items():
            stats[split_name] = {
                "num_examples": len(split_dataset),
                "features": list(split_dataset.features.keys()),
            }
            
            # Add column statistics
            for feature in split_dataset.features:
                if feature in split_dataset.column_names:
                    col_data = split_dataset[feature]
                    if isinstance(col_data[0], str):
                        lengths = [len(str(x)) for x in col_data if x]
                        stats[split_name][f"{feature}_stats"] = {
                            "avg_length": sum(lengths) / len(lengths) if lengths else 0,
                            "max_length": max(lengths) if lengths else 0,
                            "min_length": min(lengths) if lengths else 0,
                        }
    else:
        stats["num_examples"] = len(dataset)
        stats["features"] = list(dataset.features.keys())
    
    return stats


def print_dataset_info(dataset: Union[Dataset, DatasetDict]) -> None:
    """
    Print information about a dataset.
    
    Args:
        dataset: Input dataset
    """
    print("=" * 60)
    print("Dataset Information")
    print("=" * 60)
    
    if isinstance(dataset, DatasetDict):
        for split_name, split_dataset in dataset.items():
            print(f"\n{split_name.upper()} Split:")
            print(f"  Examples: {len(split_dataset):,}")
            print(f"  Features: {list(split_dataset.features.keys())}")
    else:
        print(f"\nExamples: {len(dataset):,}")
        print(f"Features: {list(dataset.features.keys())}")
    
    print("=" * 60)
