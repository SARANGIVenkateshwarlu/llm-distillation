# Dockerfile for LLM Knowledge Distillation Project
# =================================================

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy project files
COPY . .

# Install project in editable mode
RUN pip3 install -e .

# Create necessary directories
RUN mkdir -p artifacts/checkpoints artifacts/best_model artifacts/plots artifacts/logs artifacts/optuna

# Expose ports for Jupyter and Streamlit
EXPOSE 8888 8501

# Set entrypoint
CMD ["/bin/bash"]
