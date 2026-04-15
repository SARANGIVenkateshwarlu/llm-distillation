#!/usr/bin/env python
"""
Streamlit demo app for the distilled student model.

Usage:
    streamlit run app/streamlit_app.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

from src.inference.predict import InferencePipeline
from src.models.student_loader import load_student_for_inference
from src.serving.streamlit_helpers import (
    create_example_prompts,
    display_chat_history,
    format_chat_message,
    render_sidebar,
)


# Page configuration
st.set_page_config(
    page_title="LLM Knowledge Distillation Demo",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str, base_model: str = None):
    """Load model with caching."""
    with st.spinner("Loading model... This may take a minute."):
        model, tokenizer = load_student_for_inference(
            model_path=model_path,
            base_model_name=base_model,
        )
    return model, tokenizer


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None


def main():
    """Main Streamlit app."""
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">🎓 LLM Knowledge Distillation Demo</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Experience a distilled student model that learns from a larger teacher model</p>', unsafe_allow_html=True)
    
    st.divider()
    
    # Model selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_path = st.text_input(
            "Model Path",
            value=os.environ.get("MODEL_PATH", "artifacts/best_model/final"),
            help="Path to the trained student model",
        )
    
    with col2:
        base_model = st.text_input(
            "Base Model (for LoRA)",
            value=os.environ.get("BASE_MODEL", ""),
            help="Base model name if using LoRA (leave empty if not needed)",
        )
        base_model = base_model if base_model else None
    
    # Load model button
    if st.button("🚀 Load Model", type="primary"):
        try:
            model, tokenizer = load_model(model_path, base_model)
            st.session_state.pipeline = InferencePipeline(model, tokenizer)
            st.session_state.model_info = {
                "name": model.__class__.__name__,
                "total_params": sum(p.numel() for p in model.parameters()) / 1e6,
            }
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
    
    # Sidebar with settings
    if st.session_state.pipeline:
        gen_params = render_sidebar(
            st.session_state.pipeline.model,
            st.session_state.pipeline.tokenizer,
        )
    else:
        gen_params = {
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
        }
    
    st.divider()
    
    # Example prompts
    st.subheader("💡 Example Prompts")
    example_prompts = create_example_prompts()
    
    cols = st.columns(len(example_prompts))
    for i, (col, prompt) in enumerate(zip(cols, example_prompts)):
        with col:
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                st.session_state.current_prompt = prompt
    
    # Chat interface
    st.divider()
    st.subheader("💬 Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input
    default_prompt = st.session_state.get("current_prompt", "")
    prompt = st.chat_input("Enter your message...", key="chat_input")
    
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        if st.session_state.pipeline:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.pipeline.generate(
                            prompt,
                            **gen_params
                        )
                        response = result["generated_text"]
                        st.markdown(response)
                        
                        # Add to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                        })
                        
                        # Show metadata
                        with st.expander("Generation Details"):
                            st.write(f"Tokens generated: {result['num_tokens_generated']}")
                            st.write(f"Temperature: {gen_params['temperature']}")
                            st.write(f"Top-p: {gen_params['top_p']}")
                    
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
        else:
            with st.chat_message("assistant"):
                st.warning("⚠️ Please load a model first using the button above.")
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with ❤️ using <a href="https://streamlit.io">Streamlit</a> and 
        <a href="https://huggingface.co/transformers">Hugging Face Transformers</a></p>
        <p>Learn more about <a href="https://arxiv.org/abs/1503.02531">Knowledge Distillation</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
