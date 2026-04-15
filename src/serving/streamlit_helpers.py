"""
Helper functions for Streamlit app.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.models.student_loader import load_student_for_inference
from src.utils.env import get_device


@st.cache_resource
def load_model_for_app(
    model_path: str,
    base_model_name: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model for Streamlit app with caching.
    
    Args:
        model_path: Path to saved model
        base_model_name: Base model name for LoRA
    
    Returns:
        Tuple of (model, tokenizer)
    """
    device = get_device()
    
    with st.spinner("Loading model..."):
        model, tokenizer = load_student_for_inference(
            model_path=model_path,
            base_model_name=base_model_name,
        )
    
    return model, tokenizer


def format_chat_message(
    role: str,
    content: str,
) -> str:
    """
    Format a chat message for display.
    
    Args:
        role: Message role ("user", "assistant", "system")
        content: Message content
    
    Returns:
        Formatted message string
    """
    role_emojis = {
        "user": "👤",
        "assistant": "🤖",
        "system": "⚙️",
    }
    
    emoji = role_emojis.get(role, "💬")
    role_display = role.capitalize()
    
    return f"**{emoji} {role_display}:**\n{content}"


def get_model_info(model: PreTrainedModel) -> dict:
    """
    Get information about a model for display.
    
    Args:
        model: Model to get info for
    
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": f"{total_params / 1e6:.1f}M",
        "trainable_parameters": f"{trainable_params / 1e6:.1f}M",
        "model_class": model.__class__.__name__,
    }


def render_sidebar(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
) -> dict:
    """
    Render the sidebar with model info and settings.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
    
    Returns:
        Dictionary with generation parameters
    """
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model info
        st.subheader("Model Information")
        model_info = get_model_info(model)
        st.write(f"**Model:** {model.__class__.__name__}")
        st.write(f"**Total Parameters:** {model_info['total_parameters']}")
        st.write(f"**Tokenizer:** {tokenizer.name_or_path}")
        
        st.divider()
        
        # Generation parameters
        st.subheader("Generation Parameters")
        
        max_new_tokens = st.slider(
            "Max New Tokens",
            min_value=16,
            max_value=512,
            value=128,
            step=16,
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.1,
        )
        
        top_p = st.slider(
            "Top-p (Nucleus Sampling)",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.05,
        )
        
        top_k = st.slider(
            "Top-k",
            min_value=1,
            max_value=100,
            value=50,
            step=1,
        )
        
        repetition_penalty = st.slider(
            "Repetition Penalty",
            min_value=1.0,
            max_value=2.0,
            value=1.1,
            step=0.1,
        )
        
        st.divider()
        
        # About
        st.subheader("About")
        st.write("This demo showcases a distilled student model trained with knowledge distillation.")
        st.write("The model learns from a larger teacher model while being more efficient.")
    
    return {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }


def display_chat_history(messages: list):
    """
    Display chat history in Streamlit.
    
    Args:
        messages: List of message dictionaries
    """
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        
        if role == "user":
            with st.chat_message("user"):
                st.write(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                st.write(content)


def create_example_prompts() -> list:
    """
    Create example prompts for the demo.
    
    Returns:
        List of example prompt strings
    """
    return [
        "Explain the concept of machine learning in simple terms.",
        "Write a Python function to calculate the factorial of a number.",
        "What are the benefits of knowledge distillation in AI?",
        "Summarize the key points of effective prompt engineering.",
        "How does a transformer neural network work?",
    ]


def save_chat_history(messages: list, output_file: str):
    """
    Save chat history to a file.
    
    Args:
        messages: List of message dictionaries
        output_file: Output file path
    """
    import json
    
    with open(output_file, "w") as f:
        json.dump(messages, f, indent=2)


def load_chat_history(input_file: str) -> list:
    """
    Load chat history from a file.
    
    Args:
        input_file: Input file path
    
    Returns:
        List of message dictionaries
    """
    import json
    
    with open(input_file, "r") as f:
        return json.load(f)
