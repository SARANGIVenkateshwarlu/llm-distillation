# Cloud GPU Training Guide

Best practices for running this project on rented cloud GPUs (RunPod, Vast.ai, Lambda Labs, Google Colab, AWS EC2, etc.).

---

## 1. Pick the Right Instance

### Minimum Specs for Default Config (7B → 1.5B)

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 40 GB | 80 GB (A100) |
| System RAM | 32 GB | 64 GB |
| Disk | 100 GB SSD | 200 GB+ NVMe |

**Good options:**
- **NVIDIA A100 40GB/80GB** — fits default config comfortably
- **NVIDIA A6000 48GB** — great price/performance on Vast/RunPod
- **NVIDIA RTX 4090 24GB** — use `configs/teacher_student.yaml` → `low_memory` config (3B → 0.5B)

### Quick GPU Cheat Sheet

| GPU | VRAM | Fits Default? | Fits Low-Memory? |
|-----|------|---------------|------------------|
| A100 80GB | 80 GB | ✅ Yes | ✅ Yes |
| A100 40GB | 40 GB | ✅ Yes (tight) | ✅ Yes |
| A6000 / RTX A6000 | 48 GB | ✅ Yes | ✅ Yes |
| RTX 4090 / 3090 | 24 GB | ❌ No | ✅ Yes |
| A10G / RTX 3080 | 16-24 GB | ❌ No | ✅ Yes |

---

## 2. Pre-Flight Checklist

Before starting training:

- [ ] **HF Token ready** — `export HF_TOKEN=hf_...` (for gated/private models)
- [ ] **Persistent volume mounted** — for model cache & artifacts (optional but recommended)
- [ ] **SSH access confirmed** — test connection before starting long jobs
- [ ] **tmux or screen installed** — never train inside a bare SSH session

---

## 3. First-Time Setup on a Fresh Cloud VM

```bash
# 1. Check CUDA version (should be 11.8+ or 12.1+)
nvidia-smi
nvcc --version

# 2. Clone repo
git clone <your-repo-url> llm-distillation
cd llm-distillation

# 3. Set up Python (use system python or conda)
python -m venv .venv
source .venv/bin/activate

# 4. Install dependencies
make install

# 5. Set env vars
export HF_TOKEN=hf_your_token_here
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/workspace/hf_cache   # if using persistent volume

# Optional: write to .env for persistence
cp .env.example .env
# edit .env and add HF_TOKEN, HF_HOME, etc.
```

> **Tip:** If the VM comes with PyTorch pre-installed, run `pip install -r requirements.txt` **without** reinstalling torch to avoid CUDA mismatches.

---

## 4. Use tmux / screen (Critical!)

Never run long training inside a plain SSH session. If your connection drops, training dies.

```bash
# Start a tmux session
tmux new -s distillation

# Inside tmux, run training
make train

# Detach: press Ctrl+B then D
# Re-attach later: tmux attach -t distillation
```

Or use `screen`:
```bash
screen -S distillation
make train
# Detach: Ctrl+A then D
# Re-attach: screen -r distillation
```

---

## 5. Run Training

### Default (A100 40GB+)
```bash
tmux new -s train
make train
# Detach with Ctrl+B, D
```

### Low-Memory (24GB GPU)
```bash
python scripts/train.py \
    --teacher Qwen/Qwen2.5-3B-Instruct \
    --student Qwen/Qwen2.5-0.5B-Instruct \
    --epochs 3 \
    --batch-size 1 \
    --learning-rate 2e-4 \
    --bf16
```

### With Custom Config
```bash
python scripts/train.py --config configs/default.yaml
```

---

## 6. Persist Your Artifacts

Cloud VM local disks are **ephemeral**. If the instance stops, your `artifacts/` folder disappears.

### Option A: Sync to Cloud Storage (Recommended)

Install `rclone` or AWS CLI, then sync checkpoints periodically:

```bash
# In a second tmux window or background job:
while true; do
    sleep 600  # every 10 minutes
    rsync -av --progress artifacts/ /mnt/persistent-volume/artifacts/
    # Or: aws s3 sync artifacts/ s3://your-bucket/llm-distillation/
done
```

### Option B: Mount a Persistent Volume

On RunPod / Vast.ai, mount a network volume to `/workspace` and symlink:

```bash
mkdir -p /workspace/llm-distillation
ln -s /workspace/llm-distillation/artifacts artifacts
ln -s /workspace/llm-distillation/hf_cache ~/.cache/huggingface
```

### Option C: Push to HuggingFace Hub

After training completes, push the model:
```bash
pip install huggingface_hub
huggingface-cli login

# In Python or a script:
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("artifacts/best_model/final")
tokenizer = AutoTokenizer.from_pretrained("artifacts/best_model/final")
model.push_to_hub("your-username/distilled-student")
tokenizer.push_to_hub("your-username/distilled-student")
```

---

## 7. Survive Spot / Preemptible Instances

If using spot instances (cheaper but can be interrupted):

1. **Save checkpoints frequently**
   ```yaml
   # In configs/default.yaml, reduce save_steps:
   training:
     save_steps: 50      # default is 100
     save_total_limit: 5  # keep last 5 to save disk
   ```

2. **Resume from checkpoint**
   ```bash
   python scripts/train.py \
       --config configs/default.yaml \
       --resume-from-checkpoint artifacts/checkpoints/checkpoint-500
   ```
   > Note: Add resume logic to `scripts/train.py` if not already present (pass `resume_from_checkpoint` to `trainer.train()`).

3. **Watch for termination warnings**
   Some clouds send a warning 30-60s before shutdown. Use a watchdog script:
   ```bash
   # Example for AWS Spot
   while true; do
       if curl -s http://169.254.169.254/latest/meta-data/spot/termination-time 2>/dev/null | grep -q .*T.*Z; then
           echo "Spot termination imminent! Saving checkpoint..."
           # Trigger a save or sync
           rsync -av artifacts/ /mnt/backup/
           exit 0
       fi
       sleep 5
   done
   ```

---

## 8. Monitor Training Remotely

### Check GPU utilization
```bash
watch -n 1 nvidia-smi
```

### Follow logs without tmux attach
```bash
tail -f artifacts/logs/latest.log  # if logging to file
```

### Optional: Simple webhook notifications
Add this to the end of `scripts/train.py` or wrap your run:

```bash
# Discord/Slack webhook when done
make train && curl -H "Content-Type: application/json" \
    -d '{"content":"Training finished ✅"}' \
    YOUR_WEBHOOK_URL
```

---

## 9. Docker on Cloud (Optional)

If the cloud provider has Docker + NVIDIA Container Toolkit:

```bash
# Build once
docker-compose build

# Run training in container
docker-compose up train

# Or run detached (like tmux)
docker-compose up -d train
docker logs -f llm-distillation-train
```

> **Note:** Ensure the host CUDA version is compatible with the image (CUDA 12.1 in Dockerfile).

---

## 10. Cost Optimization Tips

| Strategy | Savings |
|----------|---------|
| Use **spot/preemptible** instances | 50-70% cheaper |
| Use **low_memory** config on 24GB GPUs | Avoid expensive A100s |
| Cache models on **persistent volume** | Avoid re-downloading (~20GB) |
| Run **Round 1 Optuna only** on cloud, Round 2 locally | Round 1 = 1 hour, Round 2 = 8 hours |
| Shut down immediately after training | Don’t pay for idle GPU time |

---

## 11. Quick Reference: Full Cloud Setup Script

Copy-paste this into your cloud VM terminal:

```bash
#!/bin/bash
set -e

# CONFIG
export HF_TOKEN=hf_your_token_here
export REPO_URL=https://github.com/yourname/llm-distillation.git
export PROJECT_DIR=~/llm-distillation

# 1. Persistent cache (adjust path to your mount point)
mkdir -p /workspace/hf_cache /workspace/artifacts
export HF_HOME=/workspace/hf_cache

# 2. Clone & setup
git clone "$REPO_URL" "$PROJECT_DIR"
cd "$PROJECT_DIR"
python -m venv .venv
source .venv/bin/activate
make install

# 3. Symlink artifacts to persistent storage
rm -rf artifacts
ln -s /workspace/artifacts artifacts

# 4. Start training in tmux
tmux new -d -s train "python scripts/train.py --config configs/default.yaml"

echo "Training started in tmux session 'train'"
echo "Attach with: tmux attach -t train"
```

---

## 12. Shutdown Checklist

Before terminating the instance:

- [ ] `artifacts/best_model/final/` synced to persistent storage / S3 / HF Hub
- [ ] `artifacts/checkpoints/` synced (if you want to resume later)
- [ ] `artifacts/plots/` and `artifacts/logs/` copied
- [ ] Instance stopped (not just disconnected)
