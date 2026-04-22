# HuggingFace Hub Integration Guide

Complete guide for automatically publishing your distilled models to the HuggingFace Hub after training.

---

## Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [Auto-Push After Training](#auto-push-after-training)
4. [Manual Push](#manual-push)
5. [Model Cards](#model-cards)
6. [LoRA Handling](#lora-handling)
7. [Troubleshooting](#troubleshooting)
8. [Code Reference](#code-reference)

---

## Overview

This project can automatically push your trained student model to [HuggingFace Hub](https://huggingface.co/) when training completes. It supports:

- ✅ **Auto-push** after `scripts/train.py` finishes
- ✅ **Manual push** via `scripts/push_to_hub.py`
- ✅ **LoRA merging** before upload (creates a standalone model)
- ✅ **Auto-generated model cards** with training metadata
- ✅ **Private or public** repositories

---

## Setup

### 1. Get a HuggingFace Token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a **Write** token
3. Copy the token (starts with `hf_`)

### 2. Set the Token

**Option A: Environment Variable (Recommended)**

```bash
export HF_TOKEN=hf_your_token_here
```

On Windows:
```powershell
$env:HF_TOKEN="hf_your_token_here"
```

**Option B: `.env` File**

Add to your `.env` file (create from `.env.example`):
```bash
HF_TOKEN=hf_your_token_here
```

**Option C: Config YAML**

Set in `configs/default.yaml` (less secure, not recommended for shared machines):
```yaml
hub:
  token: "hf_your_token_here"
```

### 3. Log In (Optional)

```bash
huggingface-cli login
# Paste your token when prompted
```

---

## Auto-Push After Training

### Enable in Config

Edit `configs/default.yaml`:

```yaml
hub:
  enabled: true
  repo_id: "your-username/my-distilled-qwen"  # Required
  private: false                              # Public repo
  merge_lora: true                            # Merge adapters → standalone model
  tags:
    - "qwen"
    - "instruction-tuned"
  license: "mit"
```

### Run Training

```bash
make train
# or
python scripts/train.py --config configs/default.yaml
```

When training completes, you will see:

```
============================================================
Pushing to HuggingFace Hub
============================================================
Repository ready: https://huggingface.co/your-username/my-distilled-qwen
LoRA adapters detected. Merging weights before upload...
Uploading to your-username/my-distilled-qwen...
Upload complete! Model available at: https://huggingface.co/your-username/my-distilled-qwen
```

### Override via CLI

You can also override the repo ID from command line (add this flag support by editing `scripts/train.py` if needed, or just use the config):

```bash
python scripts/train.py --config configs/default.yaml
```

---

## Manual Push

Use the standalone script to push any saved model folder:

### Basic Push

```bash
python scripts/push_to_hub.py \
    --model-path artifacts/best_model/final \
    --repo-id your-username/my-distilled-model
```

### With All Options

```bash
python scripts/push_to_hub.py \
    --model-path artifacts/best_model/final \
    --repo-id your-username/my-distilled-model \
    --private \
    --no-merge-lora \
    --commit-message "Qwen 1.5B distilled from 7B, epoch 3" \
    --tags qwen instruction-tuned knowledge-distillation \
    --license apache-2.0 \
    --base-model "Qwen/Qwen2.5-1.5B-Instruct" \
    --teacher-model "Qwen/Qwen2.5-7B-Instruct" \
    --dataset "databricks/databricks-dolly-15k"
```

### Push LoRA Adapter Only (No Merge)

```bash
python scripts/push_to_hub.py \
    --model-path artifacts/best_model/final \
    --repo-id your-username/my-lora-adapter \
    --no-merge-lora
```

> **Note:** When pushing without merging, users must load both the base model + your adapter. The model card will still be generated.

---

## Model Cards

A model card (`README.md`) is automatically generated and uploaded with every push. It includes:

- **Base Model** and **Teacher Model** names
- **Training hyperparameters** (epochs, LR, LoRA config)
- **Distillation parameters** (temperature, alpha, beta)
- **Usage example** in Python
- **Citation** for Knowledge Distillation

### Customizing the Model Card

Edit `src/publishing/hub_uploader.py` → `generate_model_card()` method to change the template.

### Tags

Tags help users discover your model on HF Hub. Good tags for this project:

```yaml
tags:
  - "distillation"
  - "knowledge-distillation"
  - "lora"
  - "qwen"
  - "instruction-tuned"
  - "1.5B"
```

---

## LoRA Handling

| Mode | Behavior | Use Case |
|------|----------|----------|
| `merge_lora: true` (default) | Merges adapters into base model, uploads standalone model | Easiest for end users |
| `merge_lora: false` / `--no-merge-lora` | Uploads adapter only (~10-100 MB) | Users need base model + `peft` library |

### Loading a Merged Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-username/my-distilled-model")
tokenizer = AutoTokenizer.from_pretrained("your-username/my-distilled-model")
```

### Loading an Adapter (Non-Merged)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "your-username/my-lora-adapter")
tokenizer = AutoTokenizer.from_pretrained("your-username/my-lora-adapter")
```

---

## Troubleshooting

### "Repository not found" or 403 Error

- Check your token has **Write** permission
- Verify `HF_TOKEN` is set: `echo $HF_TOKEN`
- Try logging in manually: `huggingface-cli login`

### "Model path not found"

Ensure training completed successfully and `artifacts/best_model/final/` exists:
```bash
ls -la artifacts/best_model/final/
```

### Out of Memory During Upload

The uploader loads the model on **CPU** to merge LoRA, so VRAM isn't the issue. If system RAM is limited:

```bash
# Use swap or push without merging
python scripts/push_to_hub.py --model-path artifacts/best_model/final --repo-id <repo> --no-merge-lora
```

### Push Succeeds but Model Card is Missing

The model card (`README.md`) is generated in the temp upload folder. If missing, regenerate:

```python
from src.publishing.hub_uploader import HubUploader

uploader = HubUploader(repo_id="your-username/model")
uploader.upload(model_path="artifacts/best_model/final")
```

### Large Uploads Fail / Timeout

For large models (several GB), the upload may take time:

```bash
# Increase git buffer size
export GIT_LFS_SKIP_SMUDGE=0
export HF_HUB_ENABLE_HF_TRANSFER=1  # faster uploads (install: pip install hf-transfer)
```

---

## Code Reference

### Files Added/Modified

| File | Purpose |
|------|---------|
| `src/publishing/__init__.py` | Package init for publishing utilities |
| `src/publishing/hub_uploader.py` | Core `HubUploader` class and `push_model_to_hub()` function |
| `scripts/push_to_hub.py` | Standalone CLI script for manual uploads |
| `src/config.py` | Added `HubConfig` dataclass |
| `configs/default.yaml` | Added `hub:` configuration section |
| `scripts/train.py` | Auto-push integration after training completes |

### Key Classes & Functions

```python
from src.publishing.hub_uploader import HubUploader, push_model_to_hub

# Method 1: Using the class
uploader = HubUploader(
    repo_id="username/model-name",
    private=False,
    merge_lora=True,
    tags=["distillation", "lora"],
)
uploader.upload(model_path="artifacts/best_model/final", config=config)

# Method 2: Using the convenience function
push_model_to_hub(
    model_path="artifacts/best_model/final",
    repo_id="username/model-name",
    config=config,
    merge_lora=True,
)
```

### Config Schema

```yaml
hub:
  enabled: false              # Enable auto-push after training
  repo_id: null               # HF Hub repo (username/repo-name)
  private: false              # true = private repo
  token: null                 # HF token (or use HF_TOKEN env var)
  merge_lora: true            # Merge LoRA before upload
  commit_message: "..."       # Git commit message
  tags: ["distillation"]      # Model tags
  license: "mit"              # License
  base_model: null            # Override base model name
  teacher_model: null         # Override teacher model name
  dataset: null               # Override dataset name
```

---

## Quick Start Checklist

- [ ] Create HF account at [huggingface.co](https://huggingface.co)
- [ ] Generate a **Write** token at [settings/tokens](https://huggingface.co/settings/tokens)
- [ ] Set `HF_TOKEN` environment variable
- [ ] Edit `configs/default.yaml` → set `hub.enabled: true` and `hub.repo_id`
- [ ] Run training: `make train`
- [ ] Verify model appears at `https://huggingface.co/your-username/repo-id`
