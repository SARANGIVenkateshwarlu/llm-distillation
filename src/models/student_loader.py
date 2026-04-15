"""
Student model loading utilities with LoRA support.
"""

from typing import List, Optional, Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import LoRAConfig, ModelConfig
from src.utils.env import get_optimal_dtype, supports_bf16


def detect_lora_targets(model: torch.nn.Module) -> Tuple[List[str], bool]:
    """
    Automatically detect appropriate LoRA target modules for a model.
    
    Args:
        model: The model to analyze
    
    Returns:
        Tuple of (target_modules, fan_in_fan_out)
    """
    # Get all linear layer names
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append(name)
    
    # Check for known model architectures
    # LLaMA/Mistral/Qwen style
    if any("q_proj" in name for name in linear_layers):
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ], False
    
    # GPT-2 / DistilGPT2 style (Conv1D)
    if any("c_attn" in name for name in linear_layers):
        return ["c_attn", "c_fc", "c_proj"], True
    
    # BERT/RoBERTa style
    if any("query" in name for name in linear_layers):
        return ["query", "key", "value", "dense"], False
    
    # T5 style
    if any("q" in name and "k" in name for name in linear_layers):
        return ["q", "k", "v", "o"], False
    
    # Fallback: use unique last layer names
    unique_names = list(set(name.split(".")[-1] for name in linear_layers))
    return unique_names, False


def apply_lora(
    model: torch.nn.Module,
    config: LoRAConfig,
) -> PeftModel:
    """
    Apply LoRA (Low-Rank Adaptation) to a model.
    
    Args:
        model: Base model to apply LoRA to
        config: LoRA configuration
    
    Returns:
        Model with LoRA applied
    
    Example:
        >>> lora_config = LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.05)
        >>> student = apply_lora(base_model, lora_config)
        >>> student.print_trainable_parameters()
    """
    # Auto-detect targets if not specified
    if config.target_modules is None:
        target_modules, fan_in_fan_out = detect_lora_targets(model)
        print(f"Auto-detected LoRA targets: {target_modules}")
    else:
        target_modules = config.target_modules
        fan_in_fan_out = config.fan_in_fan_out
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        task_type=config.task_type,
        target_modules=target_modules,
        fan_in_fan_out=fan_in_fan_out,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    return model


def load_student_model(
    config: ModelConfig,
    lora_config: Optional[LoRAConfig] = None,
    device_map: str = "auto",
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load student model with optional LoRA.
    
    The student model is loaded in training mode and can be
    fine-tuned with or without LoRA.
    
    Args:
        config: Model configuration
        lora_config: Optional LoRA configuration
        device_map: Device mapping strategy
    
    Returns:
        Tuple of (model, tokenizer)
    
    Example:
        >>> config = ModelConfig(name="Qwen/Qwen2.5-1.5B-Instruct")
        >>> lora = LoRAConfig(r=16, lora_alpha=32)
        >>> student, tokenizer = load_student_model(config, lora)
    """
    print(f"Loading student model: {config.name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.name,
        use_fast=True,
        trust_remote_code=config.trust_remote_code,
    )
    
    # Set padding side for training
    tokenizer.padding_side = "left"
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine torch dtype
    if config.torch_dtype:
        if config.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif config.torch_dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    else:
        torch_dtype = get_optimal_dtype()
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.name,
        torch_dtype=torch_dtype,
        device_map=device_map if device_map else config.device_map,
        trust_remote_code=config.trust_remote_code,
    )
    
    # Disable cache for training
    model.config.use_cache = False
    
    # Apply LoRA if configured
    if lora_config is not None:
        print("Applying LoRA to student model...")
        model = apply_lora(model, lora_config)
        model.print_trainable_parameters()
    
    print(f"Student model loaded successfully")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params / 1e6:.1f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.1f}M")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    return model, tokenizer


def load_student_for_inference(
    model_path: str,
    base_model_name: Optional[str] = None,
    device_map: str = "auto",
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load a student model (potentially with LoRA weights) for inference.
    
    Args:
        model_path: Path to the saved model
        base_model_name: Base model name if loading LoRA adapter
        device_map: Device mapping strategy
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Determine if this is a LoRA adapter
    adapter_config_path = f"{model_path}/adapter_config.json"
    
    try:
        # Try to load as LoRA adapter
        from peft import PeftModel
        
        if base_model_name is None:
            # Try to infer from adapter config
            import json
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path")
        
        if base_model_name:
            print(f"Loading LoRA adapter from {model_path}")
            print(f"Base model: {base_model_name}")
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=get_optimal_dtype(),
                device_map=device_map,
                trust_remote_code=True,
            )
            
            # Load LoRA weights
            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.merge_and_unload()  # Merge LoRA weights for faster inference
            
            return model, tokenizer
    except (FileNotFoundError, KeyError):
        pass
    
    # Load as regular model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=get_optimal_dtype(),
        device_map=device_map,
        trust_remote_code=True,
    )
    
    return model, tokenizer


def merge_lora_weights(
    model: PeftModel,
) -> torch.nn.Module:
    """
    Merge LoRA weights into the base model.
    
    This creates a standalone model without LoRA adapters.
    
    Args:
        model: Model with LoRA applied
    
    Returns:
        Model with merged weights
    """
    return model.merge_and_unload()
