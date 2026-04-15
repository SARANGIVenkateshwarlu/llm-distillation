"""
Teacher model loading utilities with quantization support.
"""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import ModelConfig
from src.utils.env import get_device, supports_bf16


def create_quantization_config(
    config: ModelConfig,
) -> Optional[BitsAndBytesConfig]:
    """
    Create BitsAndBytesConfig for quantization.
    
    Args:
        config: Model configuration
    
    Returns:
        BitsAndBytesConfig or None if no quantization
    """
    if config.quantization == "4bit" or config.load_in_4bit:
        # Determine compute dtype
        if config.bnb_4bit_compute_dtype == "bfloat16":
            compute_dtype = torch.bfloat16
        elif config.bnb_4bit_compute_dtype == "float16":
            compute_dtype = torch.float16
        else:
            compute_dtype = torch.float32
        
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    
    elif config.quantization == "8bit" or config.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    
    return None


def load_teacher_model(
    config: ModelConfig,
    device_map: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load teacher model with optional quantization.
    
    The teacher model is loaded in evaluation mode and frozen
    for knowledge distillation.
    
    Args:
        config: Model configuration
        device_map: Device mapping strategy
    
    Returns:
        Tuple of (model, tokenizer)
    
    Example:
        >>> config = ModelConfig(
        ...     name="Qwen/Qwen2.5-7B-Instruct",
        ...     quantization="4bit"
        ... )
        >>> teacher, tokenizer = load_teacher_model(config)
        >>> print(f"Teacher loaded on {teacher.device}")
    """
    print(f"Loading teacher model: {config.name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.name,
        use_fast=True,
        trust_remote_code=config.trust_remote_code,
    )
    
    # Set padding side for generation
    tokenizer.padding_side = "left"
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create quantization config
    quantization_config = create_quantization_config(config)
    
    # Determine torch dtype
    if config.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif config.torch_dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    # Load model
    model_kwargs = {
        "pretrained_model_name_or_path": config.name,
        "trust_remote_code": config.trust_remote_code,
        "device_map": device_map if device_map else config.device_map,
    }
    
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = torch_dtype
    
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    
    # Set to evaluation mode
    model.eval()
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"Teacher model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")
    
    return model, tokenizer


def get_teacher_logits(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    return_hidden_states: bool = False,
) -> torch.Tensor:
    """
    Get logits from teacher model without gradients.
    
    Args:
        model: Teacher model
        input_ids: Input token IDs
        attention_mask: Attention mask
        return_hidden_states: Whether to return hidden states
    
    Returns:
        Logits tensor
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_hidden_states,
        )
    
    return outputs.logits


def load_teacher_for_inference(
    model_path: str,
    quantization: Optional[str] = "4bit",
    device_map: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a teacher model for inference only.
    
    Args:
        model_path: Path or name of the model
        quantization: Quantization type ("4bit", "8bit", or None)
        device_map: Device mapping strategy
    
    Returns:
        Tuple of (model, tokenizer)
    """
    config = ModelConfig(
        name=model_path,
        quantization=quantization,
        device_map=device_map,
    )
    return load_teacher_model(config, device_map)
