"""
Data preprocessing utilities for various NLP tasks.
"""

from typing import Callable, Dict, List, Optional, Union

from datasets import Dataset, DatasetDict

from src.config import DatasetConfig


def format_instruction_example(
    example: Dict,
    instruction_col: str = "instruction",
    context_col: Optional[str] = "context",
    response_col: str = "response",
    include_context: bool = True,
) -> Dict:
    """
    Format an instruction-following example into a standard format.
    
    Args:
        example: Input example dictionary
        instruction_col: Column name for instruction
        context_col: Column name for context (optional)
        response_col: Column name for response
        include_context: Whether to include context in the prompt
    
    Returns:
        Formatted example with 'text' and 'target' fields
    """
    instruction = example.get(instruction_col, "")
    context = example.get(context_col, "") if context_col else ""
    response = example.get(response_col, "")
    
    if include_context and context:
        prompt = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    return {
        "prompt": prompt,
        "response": response,
        "text": prompt + response,
    }


def format_alpaca_example(example: Dict) -> Dict:
    """
    Format an Alpaca-style example.
    
    Args:
        example: Input example with 'instruction', 'input', 'output' fields
    
    Returns:
        Formatted example
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    if input_text:
        prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
    
    return {
        "prompt": prompt,
        "response": output,
        "text": prompt + output,
    }


def format_chat_example(
    example: Dict,
    messages_col: str = "messages",
) -> Dict:
    """
    Format a chat/conversation example.
    
    Args:
        example: Input example with messages
        messages_col: Column name for messages list
    
    Returns:
        Formatted example
    """
    messages = example.get(messages_col, [])
    
    formatted_text = ""
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        
        if role == "system":
            formatted_text += f"<|system|>\n{content}\n"
        elif role == "user":
            formatted_text += f"<|user|>\n{content}\n"
        elif role == "assistant":
            formatted_text += f"<|assistant|>\n{content}\n"
    
    return {
        "text": formatted_text,
        "messages": messages,
    }


def format_classification_example(
    example: Dict,
    text_col: str = "text",
    label_col: str = "label",
    label_names: Optional[List[str]] = None,
) -> Dict:
    """
    Format a text classification example.
    
    Args:
        example: Input example
        text_col: Column name for text
        label_col: Column name for label
        label_names: Optional list of label names
    
    Returns:
        Formatted example
    """
    text = example.get(text_col, "")
    label = example.get(label_col, 0)
    
    result = {
        "text": text,
        "label": label,
    }
    
    if label_names and label < len(label_names):
        result["label_name"] = label_names[label]
    
    return result


def format_summarization_example(
    example: Dict,
    text_col: str = "article",
    summary_col: str = "highlights",
) -> Dict:
    """
    Format a summarization example.
    
    Args:
        example: Input example
        text_col: Column name for source text
        summary_col: Column name for summary
    
    Returns:
        Formatted example
    """
    text = example.get(text_col, "")
    summary = example.get(summary_col, "")
    
    prompt = f"Summarize the following text:\n\n{text}\n\nSummary:\n"
    
    return {
        "prompt": prompt,
        "response": summary,
        "text": prompt + summary,
        "source": text,
        "target": summary,
    }


def format_qa_example(
    example: Dict,
    context_col: str = "context",
    question_col: str = "question",
    answer_col: str = "answers",
) -> Dict:
    """
    Format a question-answering example.
    
    Args:
        example: Input example
        context_col: Column name for context
        question_col: Column name for question
        answer_col: Column name for answers
    
    Returns:
        Formatted example
    """
    context = example.get(context_col, "")
    question = example.get(question_col, "")
    answers = example.get(answer_col, {})
    
    # Handle different answer formats
    if isinstance(answers, dict):
        answer_texts = answers.get("text", [])
        answer = answer_texts[0] if answer_texts else ""
    elif isinstance(answers, list):
        answer = answers[0] if answers else ""
    else:
        answer = str(answers)
    
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:\n"
    
    return {
        "prompt": prompt,
        "response": answer,
        "text": prompt + answer,
        "context": context,
        "question": question,
        "answer": answer,
    }


def get_formatter_for_task(task_type: str) -> Callable:
    """
    Get the appropriate formatter function for a task type.
    
    Args:
        task_type: Type of NLP task
    
    Returns:
        Formatter function
    """
    formatters = {
        "instruction_following": format_instruction_example,
        "alpaca": format_alpaca_example,
        "conversation": format_chat_example,
        "chat": format_chat_example,
        "text_classification": format_classification_example,
        "classification": format_classification_example,
        "sentiment_analysis": format_classification_example,
        "summarization": format_summarization_example,
        "question_answering": format_qa_example,
        "qa": format_qa_example,
    }
    
    return formatters.get(task_type, format_instruction_example)


def preprocess_dataset(
    dataset: DatasetDict,
    config: DatasetConfig,
    num_proc: int = 4,
) -> DatasetDict:
    """
    Preprocess dataset according to configuration.
    
    Args:
        dataset: Input dataset
        config: Dataset configuration
        num_proc: Number of processes for preprocessing
    
    Returns:
        Preprocessed dataset
    """
    # Get the appropriate formatter
    formatter = get_formatter_for_task(config.task_type)
    
    # Prepare formatter kwargs
    formatter_kwargs = {}
    if config.task_type == "instruction_following":
        formatter_kwargs = {
            "instruction_col": config.text_column,
            "context_col": config.context_column,
            "response_col": config.response_column,
        }
    
    # Apply formatting to each split
    processed = {}
    for split_name, split_dataset in dataset.items():
        processed[split_name] = split_dataset.map(
            lambda x: formatter(x, **formatter_kwargs),
            num_proc=num_proc if num_proc > 1 else None,
            desc=f"Formatting {split_name} split",
        )
    
    return DatasetDict(processed)


def filter_by_length(
    dataset: Dataset,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    text_column: str = "text",
) -> Dataset:
    """
    Filter dataset by text length.
    
    Args:
        dataset: Input dataset
        min_length: Minimum text length
        max_length: Maximum text length
        text_column: Column containing text
    
    Returns:
        Filtered dataset
    """
    def length_filter(example):
        text = example.get(text_column, "")
        length = len(text)
        
        if min_length is not None and length < min_length:
            return False
        if max_length is not None and length > max_length:
            return False
        return True
    
    return dataset.filter(length_filter)


def deduplicate_dataset(
    dataset: Dataset,
    text_column: str = "text",
) -> Dataset:
    """
    Remove duplicate examples from dataset.
    
    Args:
        dataset: Input dataset
        text_column: Column containing text to deduplicate on
    
    Returns:
        Deduplicated dataset
    """
    seen = set()
    
    def dedup_filter(example):
        text = example.get(text_column, "")
        if text in seen:
            return False
        seen.add(text)
        return True
    
    return dataset.filter(dedup_filter)


def clean_text(
    text: str,
    remove_extra_whitespace: bool = True,
    normalize_unicode: bool = True,
) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        remove_extra_whitespace: Whether to remove extra whitespace
        normalize_unicode: Whether to normalize unicode characters
    
    Returns:
        Cleaned text
    """
    if normalize_unicode:
        import unicodedata
        text = unicodedata.normalize("NFKC", text)
    
    if remove_extra_whitespace:
        # Replace multiple whitespace with single space
        import re
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
    return text
