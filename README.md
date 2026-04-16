# LLM Knowledge Distillation Project

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade implementation of **Knowledge Distillation** for Large Language Models (LLMs). This project enables you to transfer knowledge from a large "teacher" model to a smaller "student" model, achieving comparable performance with significantly reduced computational requirements.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training](#training)
- [Optimization](#optimization)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Docker](#docker)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

Knowledge Distillation (KD) is a technique where a smaller "student" model learns to mimic the behavior of a larger "teacher" model. The key insight is that the teacher's probability distributions ("soft targets") contain more information than one-hot labels, revealing relationships between different outputs.

### Why Knowledge Distillation?

- **Efficiency**: Smaller models run faster and use less memory
- **Deployment**: Easier to deploy on edge devices or with limited resources
- **Cost**: Reduced inference costs in production
- **Performance**: Student can match or exceed teacher on specific tasks

### Loss Function

```
loss = α × CE(student_logits, labels) + β × T² × KL(student_soft, teacher_soft)
```

Where:
- **α (alpha)**: Weight for cross-entropy with ground truth (typically 0.2-0.4)
- **β (beta)**: Weight for KL divergence with teacher (typically 0.6-0.8)
- **T (temperature)**: Softens probability distributions (typically 2.0-5.0)

## ✨ Features

- 🎓 **Teacher-Student Architecture**: Transfer knowledge from large to small models
- 🔧 **LoRA Integration**: Efficient fine-tuning with Low-Rank Adaptation
- 📊 **Two-Round Optuna Optimization**: Automated hyperparameter search
- 🐳 **Docker Support**: Containerized training and deployment
- 🌐 **Streamlit Demo**: Interactive web interface for model testing
- 📈 **Comprehensive Logging**: Training curves, metrics, and visualizations
- 🧪 **Evaluation Suite**: Multiple metrics for model comparison
- 📝 **Jupyter Notebook**: Educational walkthrough of the entire process

## 📁 Project Structure

```
llm-distillation-project/
├── finetune.ipynb              # Main educational notebook
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project configuration
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Docker Compose services
├── Makefile                   # Common commands
│
├── configs/                   # Configuration files
│   ├── default.yaml          # Default configuration
│   └── teacher_student.yaml  # Model pair configurations
│
├── src/                       # Source code
│   ├── config.py             # Configuration management
│   ├── data/                 # Data loading and preprocessing
│   ├── models/               # Model loading and distillation
│   ├── training/             # Training and evaluation
│   ├── optimization/         # Optuna hyperparameter search
│   ├── inference/            # Inference utilities
│   ├── serving/              # Deployment helpers
│   ├── monitoring/           # Logging and plotting
│   └── utils/                # Utility functions
│
├── scripts/                   # Executable scripts
│   ├── train.py              # Main training script
│   ├── optimize_round1.py    # Round 1 Optuna optimization
│   ├── optimize_round2.py    # Round 2 Optuna optimization
│   ├── evaluate.py           # Evaluation script
│   └── infer.py              # Inference script
│
├── app/                       # Streamlit application
│   └── streamlit_app.py      # Interactive demo
│
├── artifacts/                 # Generated artifacts
│   ├── checkpoints/          # Training checkpoints
│   ├── best_model/           # Best saved model
│   ├── plots/                # Training plots
│   ├── logs/                 # Training logs
│   └── optuna/               # Optuna study results
│
├── data/                      # Data directory
│   ├── raw/                  # Raw datasets
│   ├── processed/            # Processed datasets
│   └── external/             # External data
│
├── notebooks/                 # Additional notebooks
│   └── exploration.ipynb     # Data exploration
│
└── tests/                     # Unit tests
    ├── test_data.py
    ├── test_losses.py
    └── test_inference.py
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd llm-distillation-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Run Training

```bash
# Quick training (1 epoch)
make train

# Or with custom settings
python scripts/train.py --epochs 3 --batch-size 2
```

### 4. Launch Demo App

```bash
make app
# Open http://localhost:8501 in your browser
```

## 💻 Installation

### Requirements

- Python 3.9+
- CUDA-capable GPU (recommended: 24GB+ VRAM)
- 32GB+ system RAM

### Standard Installation

```bash
# Install from requirements
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

### Docker Installation

```bash
# Build Docker image
make docker-build

# Or manually
docker-compose build
```

## 🔧 Usage

### Training

```bash
# Basic training
python scripts/train.py --config configs/default.yaml

# With custom models
python scripts/train.py \
    --teacher Qwen/Qwen2.5-7B-Instruct \
    --student Qwen/Qwen2.5-1.5B-Instruct \
    --epochs 3

# Using Docker
make docker-train
```

### Hyperparameter Optimization

```bash
# Round 1: Quick search (20 trials, ~1 hour)
make optimize-round1

# Round 2: Deep search around best region (75 trials, ~8 hours)
make optimize-round2
```

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py \
    --model-path artifacts/best_model/final \
    --generate

# Compare with baseline
python scripts/evaluate.py \
    --model-path artifacts/baseline_model \
    --generate
```

### Inference

```bash
# Interactive mode
python scripts/infer.py --model-path artifacts/best_model/final --interactive

# Single prompt
python scripts/infer.py \
    --model-path artifacts/best_model/final \
    --prompt "Explain machine learning"

# Batch inference
python scripts/infer.py \
    --model-path artifacts/best_model/final \
    --input-file prompts.txt \
    --output-file results.jsonl
```

## ⚙️ Configuration

Configuration is managed through YAML files in `configs/`:

### Default Configuration (`configs/default.yaml`)

```yaml
# Model Configuration
models:
  teacher:
    name: "Qwen/Qwen2.5-7B-Instruct"
    quantization: "4bit"
  student:
    name: "Qwen/Qwen2.5-1.5B-Instruct"

# Dataset Configuration
dataset:
  name: "databricks/databricks-dolly-15k"
  task_type: "instruction_following"

# Training Configuration
training:
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4

# Distillation Configuration
distillation:
  temperature: 2.0
  alpha: 0.3
  beta: 0.7

# LoRA Configuration
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
```

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
TEACHER_MODEL=Qwen/Qwen2.5-7B-Instruct
STUDENT_MODEL=Qwen/Qwen2.5-1.5B-Instruct
NUM_EPOCHS=3
LEARNING_RATE=2e-4
TEMPERATURE=2.0
ALPHA=0.3
BETA=0.7
```

## 🎓 Training

### Using the Notebook

The `finetune.ipynb` notebook provides an educational walkthrough:

```bash
jupyter notebook finetune.ipynb
```

### Using Scripts

```bash
# Full training pipeline
python scripts/train.py --config configs/default.yaml

# With specific hyperparameters
python scripts/train.py \
    --teacher Qwen/Qwen2.5-7B-Instruct \
    --student Qwen/Qwen2.5-1.5B-Instruct \
    --epochs 3 \
    --batch-size 1 \
    --learning-rate 2e-4
```

### Training with Makefile

```bash
make train          # Run training
make evaluate       # Evaluate model
make infer          # Interactive inference
```

## 🔬 Optimization

### Two-Round Optuna Search

**Round 1**: Quick exploration across wide search space
- 20 trials
- ~1 hour runtime
- Identifies promising regions

**Round 2**: Deep search around best region from Round 1
- 75 trials
- ~8 hours runtime
- Fine-tunes hyperparameters

```bash
# Run both rounds
make optimize-round1
make optimize-round2

# Or use Docker
make docker-optimize-round1
make docker-optimize-round2
```

### Search Space

| Parameter | Range | Description |
|-----------|-------|-------------|
| learning_rate | 1e-5 - 5e-4 | Learning rate for optimizer |
| weight_decay | 0.0 - 0.1 | L2 regularization |
| lora_r | 8, 16, 32, 64 | LoRA rank |
| lora_alpha | 16, 32, 64, 128 | LoRA scaling factor |
| temperature | 1.0 - 5.0 | Distillation temperature |
| alpha | 0.1 - 0.5 | CE loss weight |
| beta | 0.5 - 0.9 | KD loss weight |

## 📊 Evaluation

### Metrics

- **Perplexity**: Measure of model confidence
- **Loss**: Training and validation loss
- **BLEU/ROUGE**: For generation tasks (optional)

### Running Evaluation

```bash
# Basic evaluation
python scripts/evaluate.py --model-path artifacts/best_model/final

# With generation
python scripts/evaluate.py \
    --model-path artifacts/best_model/final \
    --generate \
    --num-samples 100
```

## 🌐 Deployment

### Streamlit App

```bash
# Local deployment
make app

# Docker deployment
make docker-app
```

Access the app at `http://localhost:8501`

### Exporting for Production

```python
from src.models.student_loader import merge_lora_weights

# Merge LoRA weights for faster inference
student = merge_lora_weights(student)
student.save_pretrained("./production_model")
```

## 🐳 Docker

### Services

| Service | Description | Port |
|---------|-------------|------|
| `train` | Training container | - |
| `optimize-round1` | Round 1 optimization | - |
| `optimize-round2` | Round 2 optimization | - |
| `evaluate` | Evaluation container | - |
| `app` | Streamlit demo | 8501 |
| `jupyter` | Jupyter notebook | 8888 |

### Commands

```bash
# Build image
make docker-build

# Run training
docker-compose up train

# Run Streamlit app
docker-compose up app

# Run Jupyter
docker-compose up jupyter
```

## 📈 Results

### Expected Performance

With the default configuration (Qwen 7B → Qwen 1.5B):

| Metric | Teacher (7B) | Student (1.5B) | Improvement |
|--------|--------------|----------------|-------------|
| Perplexity | ~12.5 | ~14.2 | Baseline |
| With KD | - | ~13.1 | **~8% better** |
| Inference Time | 150ms | 45ms | **3x faster** |
| Memory Usage | 16GB | 4GB | **4x less** |

### Training Time

- Single epoch on Dolly-15k: ~2 hours (A100)
- Full 3-epoch training: ~6 hours
- Round 1 optimization: ~1 hour
- Round 2 optimization: ~8 hours

## 🔧 Troubleshooting

### Out of Memory

```python
# Reduce batch size
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16

# Use 8-bit quantization for teacher
quantization: "8bit"

# Reduce sequence length
max_length: 256
```

### Slow Training

```python
# Enable gradient checkpointing
gradient_checkpointing: true

# Use bf16 if available
mixed_precision: "bf16"
```

### Poor Results

- Increase temperature (try 3.0-5.0)
- Adjust alpha/beta ratio (try 0.2/0.8)
- Increase LoRA rank (try 32 or 64)
- Train for more epochs

## 📚 Recommended Model Pairs

### For 46GB GPU

| Teacher | Student | Total Memory |
|---------|---------|--------------|
| Qwen2.5-7B-Instruct | Qwen2.5-1.5B-Instruct | ~32GB |
| DeepSeek-R1-Distill-Qwen-7B | DeepSeek-R1-Distill-Qwen-1.5B | ~32GB |
| Mistral-7B-Instruct | SmolLM2-1.7B-Instruct | ~35GB |

### For 24GB GPU

| Teacher | Student | Total Memory |
|---------|---------|--------------|
| Qwen2.5-3B-Instruct | Qwen2.5-0.5B-Instruct | ~18GB |
| Phi-3-mini-4k-instruct | TinyLlama-1.1B | ~16GB |

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@misc{llm-distillation-2024,
  title={LLM Knowledge Distillation: A Production-Grade Implementation},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/llm-knowledge-distillation}}
}
```

Original Knowledge Distillation paper:

```bibtex
@article{hinton2015distilling,
  title={Distilling the knowledge in a neural network},
  author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  journal={arXiv preprint arXiv:1503.02531},
  year={2015}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers)
- [PEFT Library](https://github.com/huggingface/peft)
- [Optuna](https://optuna.org/)
- [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1) for demonstrating effective distillation

## 📞 Support

For questions or issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Open an issue on GitHub
3. Contact: your.email@example.com

---

**Happy Distilling! 🎓✨**
DeepSeek-R1
https://arxiv.org/pdf/2501.12948
Everything You Need to Know about Knowledge Distillation 
https://huggingface.co/blog/Kseniase/kd

https://www.geeksforgeeks.org/nlp/what-is-llm-distillation/
What is LLM Distillation

https://www.reddit.com/r/LocalLLaMA/comments/1et9ay6/understanding_llm_distillation_gemma_2_and_nvidia/
https://www.youtube.com/watch?v=riUYGZ-_fJY

https://www.youtube.com/watch?v=O1AR4iL30mg

https://www.youtube.com/watch?v=qZ10DO3F8uo
https://www.youtube.com/watch?v=gSzv6s9X5Ak
https://www.youtube.com/watch?v=PYU1uX5l29c
https://www.youtube.com/watch?v=FiG72a3hGtQ
https://www.youtube.com/watch?v=jrJKRYAdh7I
https://www.youtube.com/watch?v=FIOigevZdDU
https://www.youtube.com/watch?v=tf60owmwR-c
