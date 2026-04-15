"""
Inference pipeline for text generation with trained models.
"""

from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.utils.env import get_device


class InferencePipeline:
    """
    Pipeline for running inference with trained models.
    
    This class provides a clean interface for text generation
    with proper handling of generation parameters.
    
    Example:
        >>> pipeline = InferencePipeline(model, tokenizer)
        >>> result = pipeline.generate("What is AI?", max_new_tokens=100)
        >>> print(result["generated_text"])
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the inference pipeline.
        
        Args:
            model: Trained model
            tokenizer: Tokenizer
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or get_device()
        
        # Move model to device if not already
        if hasattr(model, 'device') and model.device != self.device:
            self.model = self.model.to(self.device)
        
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to return
            pad_token_id: Pad token ID
            eos_token_id: End-of-sequence token ID
            **kwargs: Additional generation parameters
        
        Returns:
            Dictionary with generated text and metadata
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        
        # Set pad and eos tokens
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        
        # Generate
        with torch.no_grad():
            output_sequences = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **kwargs
            )
        
        # Decode generated text
        input_length = inputs["input_ids"].shape[1]
        generated_sequences = output_sequences[:, input_length:]
        
        generated_texts = self.tokenizer.batch_decode(
            generated_sequences,
            skip_special_tokens=True
        )
        
        # Calculate confidence scores if available
        confidence = None
        if num_return_sequences == 1:
            # Simple confidence based on sequence probability could be added here
            pass
        
        return {
            "prompt": prompt,
            "generated_text": generated_texts[0] if num_return_sequences == 1 else generated_texts,
            "generated_tokens": generated_sequences[0].tolist() if num_return_sequences == 1 else [seq.tolist() for seq in generated_sequences],
            "num_tokens_generated": len(generated_sequences[0]) if num_return_sequences == 1 else [len(seq) for seq in generated_sequences],
            "confidence": confidence,
        }
    
    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 8,
        **generation_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate text for multiple prompts in batches.
        
        Args:
            prompts: List of input prompts
            batch_size: Batch size for generation
            **generation_kwargs: Generation parameters
        
        Returns:
            List of generation results
        """
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                output_sequences = self.model.generate(
                    **inputs,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **generation_kwargs
                )
            
            # Decode
            input_lengths = inputs["input_ids"].shape[1]
            for j, prompt in enumerate(batch_prompts):
                generated = output_sequences[j, input_lengths:]
                generated_text = self.tokenizer.decode(
                    generated,
                    skip_special_tokens=True
                )
                
                results.append({
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "num_tokens_generated": len(generated),
                })
        
        return results
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **generation_kwargs
    ) -> str:
        """
        Generate a response in a chat context.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt
            **generation_kwargs: Generation parameters
        
        Returns:
            Generated response
        """
        # Format messages into a prompt
        formatted_prompt = ""
        
        if system_prompt:
            formatted_prompt += f"<|system|>\n{system_prompt}\n"
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            formatted_prompt += f"<|{role}|>\n{content}\n"
        
        formatted_prompt += "<|assistant|>\n"
        
        # Generate response
        result = self.generate(formatted_prompt, **generation_kwargs)
        
        return result["generated_text"]


def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    device: Optional[torch.device] = None,
    **generation_kwargs
) -> str:
    """
    Simple function to generate text from a prompt.
    
    Args:
        model: Model to use
        tokenizer: Tokenizer
        prompt: Input prompt
        device: Device to use
        **generation_kwargs: Generation parameters
    
    Returns:
        Generated text
    
    Example:
        >>> text = generate_text(model, tokenizer, "Once upon a time")
        >>> print(text)
    """
    pipeline = InferencePipeline(model, tokenizer, device)
    result = pipeline.generate(prompt, **generation_kwargs)
    return result["generated_text"]


def batch_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    **generation_kwargs
) -> List[str]:
    """
    Generate text for multiple prompts.
    
    Args:
        model: Model to use
        tokenizer: Tokenizer
        prompts: List of prompts
        batch_size: Batch size
        device: Device to use
        **generation_kwargs: Generation parameters
    
    Returns:
        List of generated texts
    """
    pipeline = InferencePipeline(model, tokenizer, device)
    results = pipeline.batch_generate(prompts, batch_size, **generation_kwargs)
    return [r["generated_text"] for r in results]
