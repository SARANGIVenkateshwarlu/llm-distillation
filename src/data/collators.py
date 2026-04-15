"""
Data collators for batching during training.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizer


@dataclass
class DataCollatorForDistillation:
    """
    Data collator for knowledge distillation training.
    
    This collator handles dynamic padding and prepares batches
    for both standard training and knowledge distillation.
    
    Args:
        tokenizer: Tokenizer for padding
        padding: Padding strategy ("longest", "max_length", etc.)
        max_length: Maximum sequence length
        pad_to_multiple_of: Pad to multiple of this value
        return_tensors: Type of tensors to return
        mlm: Whether to use masked language modeling
        mlm_probability: Probability for MLM
    """
    tokenizer: PreTrainedTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    mlm: bool = False
    mlm_probability: float = 0.15
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features.
        
        Args:
            features: List of feature dictionaries
        
        Returns:
            Batched features as tensors
        """
        # Separate special fields if present
        teacher_logits = None
        if "teacher_logits" in features[0]:
            teacher_logits = [f.pop("teacher_logits") for f in features]
        
        # Use the tokenizer's pad method for standard fields
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        # Add teacher logits back if present
        if teacher_logits is not None:
            # Stack teacher logits
            batch["teacher_logits"] = torch.stack([
                torch.tensor(t, dtype=torch.float32) for t in teacher_logits
            ])
        
        # Prepare decoder_input_ids for encoder-decoder models if needed
        if "decoder_input_ids" in features[0]:
            decoder_features = [{"input_ids": f["decoder_input_ids"]} for f in features]
            decoder_batch = self.tokenizer.pad(
                decoder_features,
                padding=self.padding,
                max_length=self.max_length,
                return_tensors=self.return_tensors,
            )
            batch["decoder_input_ids"] = decoder_batch["input_ids"]
        
        return batch


@dataclass
class DataCollatorForCausalLM:
    """
    Data collator for causal language modeling.
    
    Handles proper label preparation for autoregressive models.
    """
    tokenizer: PreTrainedTokenizer
    padding: Union[bool, str] = "longest"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 8
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate features for causal LM training."""
        # Extract input_ids and labels
        input_ids = [f["input_ids"] for f in features]
        labels = [f.get("labels", f["input_ids"]) for f in features]
        
        # Pad sequences
        batch = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        # Pad labels separately (use -100 for padding)
        labels_batch = self._pad_labels(
            labels,
            max_length=batch["input_ids"].shape[1],
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        
        batch["labels"] = labels_batch
        
        # Create attention mask if not present
        if "attention_mask" not in batch:
            batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).long()
        
        return batch
    
    def _pad_labels(
        self,
        labels: List[List[int]],
        max_length: int,
        pad_to_multiple_of: Optional[int] = None,
    ) -> torch.Tensor:
        """Pad labels with -100 (ignored in loss)."""
        if pad_to_multiple_of is not None:
            max_length = (
                (max_length + pad_to_multiple_of - 1)
                // pad_to_multiple_of
                * pad_to_multiple_of
            )
        
        padded_labels = []
        for label in labels:
            padding_length = max_length - len(label)
            if padding_length > 0:
                # Pad with -100 (ignored index)
                padded_label = label + [-100] * padding_length
            else:
                padded_label = label[:max_length]
            padded_labels.append(padded_label)
        
        return torch.tensor(padded_labels, dtype=torch.long)


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator for sequence-to-sequence models.
    
    Handles both encoder inputs and decoder inputs.
    """
    tokenizer: PreTrainedTokenizer
    padding: Union[bool, str] = "longest"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    label_pad_token_id: int = -100
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate features for seq2seq training."""
        # Separate labels from inputs
        labels = [f.pop("labels", None) for f in features]
        
        # Pad input features
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        # Pad labels
        if labels[0] is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
            
            padded_labels = []
            for label in labels:
                padding_length = max_label_length - len(label)
                if padding_length > 0:
                    padded_label = label + [self.label_pad_token_id] * padding_length
                else:
                    padded_label = label[:max_label_length]
                padded_labels.append(padded_label)
            
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        
        return batch


def get_data_collator(
    tokenizer: PreTrainedTokenizer,
    task_type: str = "causal_lm",
    **kwargs
) -> Any:
    """
    Get the appropriate data collator for a task type.
    
    Args:
        tokenizer: Tokenizer to use
        task_type: Type of task ("causal_lm", "seq2seq", "distillation")
        **kwargs: Additional arguments for the collator
    
    Returns:
        Data collator instance
    """
    collators = {
        "causal_lm": DataCollatorForCausalLM,
        "seq2seq": DataCollatorForSeq2Seq,
        "distillation": DataCollatorForDistillation,
    }
    
    collator_class = collators.get(task_type, DataCollatorForCausalLM)
    return collator_class(tokenizer=tokenizer, **kwargs)
