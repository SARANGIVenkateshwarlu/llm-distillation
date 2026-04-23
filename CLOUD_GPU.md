# Cloud GPU Training Guide for Beginners

Simple step-by-step guide to train your model on cloud GPU (Lightning.AI, RunPod, Vast.ai, Google Colab, etc.).

---

## What You Will Do

1. **Step 1** — Connect to your cloud GPU
2. **Step 2** — Download this project from GitHub
3. **Step 3** — Install requirements (one command)
4. **Step 4** — Run training
5. **Step 5** — Check if training is working
6. **Step 6** — Save your model

---

## Step 1: Connect to Your Cloud GPU

After you rent a GPU instance, you will get:
- An **IP address** (like `192.168.1.100`)
- A **port** (like `22`)
- A **username** (usually `root` or `ubuntu`)

Open your computer terminal and connect:

```bash
ssh root@192.168.1.100 -p 22
```

> Replace `192.168.1.100` with your real IP address.

---

## Step 2: Download This Project

Once connected, run this command to copy the project to your cloud GPU:

```bash
git clone https://github.com/YOUR_USERNAME/llm-distillation.git
cd llm-distillation
```

> Replace `YOUR_USERNAME` with your GitHub username.

---

## Step 3: Install Requirements

Run this **one command**. It installs only what you need for cloud training:

```bash
pip install -r requirements-cloud.txt
```

This is a **lean** requirements file (no docs, no testing tools, no unused packages). It installs faster and uses less disk space on cloud GPUs.

If you also want to install the project package itself:

```bash
pip install -r requirements-cloud.txt
pip install -e .
```

Wait 2-5 minutes. It will download libraries.

---

## Step 4: Set Your HuggingFace Token (Important!)

Some models need a token to download. Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

Run this in the terminal:

```bash
export HF_TOKEN=hf_your_token_here
```

> Replace `hf_your_token_here` with your real token.

---

## Step 5: Run Training

We have **3 ways** to train. Pick one:

### Option A: Run Notebook (Recommended for Beginners)

Start Jupyter and open the beginner notebooks:

```bash
make jupyter
```

Then open your browser and go to the URL shown in the terminal. Open these notebooks in order:

| Notebook | What It Does | Time |
|----------|-------------|------|
| `notebooks/01_setup_and_train.ipynb` | Train your first model | ~30 min |
| `notebooks/02_optimize_round1.ipynb` | Find best settings quickly | ~1 hour |
| `notebooks/03_optimize_round2.ipynb` | Fine-tune the best settings | ~4 hours |
| `notebooks/04_check_progress.ipynb` | Check if training is running | instant |

### Option B: Run Script (Simple)

Run this one command to start training:

```bash
make train
```

This uses the default settings. Training takes about **30 minutes** on an A100 GPU.

### Option C: Run Script with Custom Settings

If you want to change the teacher or student model:

```bash
python scripts/train.py \
    --teacher Qwen/Qwen2.5-7B-Instruct \
    --student Qwen/Qwen2.5-1.5B-Instruct \
    --epochs 3 \
    --batch-size 1 \
    --bf16
```

---

## Step 6: Check If Training Is Working

### Method 1: Check GPU Usage

Open a **new terminal window** (keep the first one running) and run:

```bash
nvidia-smi
```

You should see:
- GPU temperature
- Memory usage going up
- A Python process using the GPU

If you see this, training is working! ✅

### Method 2: Check Training Output

While training runs, it prints messages like this:

```
Loading dataset: databricks/databricks-dolly-15k...
Dataset loaded: 15,011 train, 1,668 validation
Loading teacher model: Qwen/Qwen2.5-7B-Instruct...
Loading student model: Qwen/Qwen2.5-1.5B-Instruct...
Starting Training
Step 10/1000 | Loss: 2.345 | LR: 1.8e-4
Step 20/1000 | Loss: 2.123 | LR: 1.6e-4
```

The **Loss** number should go **down** over time. This means the model is learning. ✅

### Method 3: Use the Progress Notebook

Open `notebooks/04_check_progress.ipynb` and run the cells. It shows:
- GPU memory usage
- Latest training loss
- How many steps are done

---

## Step 7: Do Not Lose Your Work!

Cloud GPUs delete everything when they shut down. You must save your model!

### Save to HuggingFace Hub (Easiest)

```bash
python scripts/push_to_hub.py \
    --model-path artifacts/best_model/final \
    --repo-id your-username/my-distilled-model
```

### Save to Your Computer

Download the model folder to your local machine:

```bash
# On your LOCAL computer (not cloud), run:
scp -r root@192.168.1.100:/root/llm-distillation/artifacts/best_model/final ./my-model
```

---

## Common Problems for Beginners

| Problem | Why It Happens | How to Fix |
|---------|---------------|------------|
| `Out of Memory` | GPU is too small | Use smaller model: `--student Qwen/Qwen2.5-0.5B-Instruct` |
| `CUDA not available` | PyTorch not installed for GPU | Run `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `HF_TOKEN error` | Missing HuggingFace token | Run `export HF_TOKEN=hf_...` before training |
| Training stops when I close laptop | You ran it in normal SSH | Use `tmux` (see below) |

---

## Keep Training Running When You Disconnect

If you close your laptop, training stops. Use `tmux` to keep it running:

```bash
# Start a new session
tmux new -s train

# Now run your training inside tmux
make train

# Press Ctrl+B, then D to disconnect (training keeps running!)

# Later, reconnect to your cloud GPU, then run:
tmux attach -t train
```

---

## What Files Are Created?

After training finishes, you will see these folders:

```
artifacts/
├── best_model/final/     ← Your trained model! Save this!
├── checkpoints/          ← Backup checkpoints during training
├── plots/                ← Training graphs
├── logs/                 ← Training logs
└── optuna/               ← Best settings found (if you ran optimization)
```

---

## Full Pipeline for Best Results

For the best model, run these **in order**:

```bash
# 1. Find good settings (1 hour)
make optimize-round1

# 2. Fine-tune best settings (4 hours)
make optimize-round2

# 3. Train final model with best settings (30 min)
make train

# 4. Test your model
make evaluate
```

Or use the notebooks — they do the same thing with explanations!

---

## GPU Requirements

| GPU Type | Can Run? | Notes |
|----------|----------|-------|
| A100 80GB | ✅ Yes | Best choice, everything fits |
| A100 40GB | ✅ Yes | Default config works |
| A6000 48GB | ✅ Yes | Good price/performance |
| RTX 4090 24GB | ⚠️ Maybe | Use smaller models only |
| T4 16GB | ❌ No | Too small for default config |

---

## Need Help?

1. Check `notebooks/04_check_progress.ipynb` to see what's happening
2. Read `TRAINING.md` for more details
3. Check GPU memory with `nvidia-smi`
4. Look at the error message — it usually tells you what's wrong
